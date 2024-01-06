from typing import Any

import einops
import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.bert.configuration_bert import BertConfig


from lavis.models import load_model


class BLIP2Cir(nn.Module):
    def __init__(
        self,
        loss: Any,
        med_config="configs/med_config.json",
        image_size=384,
        vit="eva_clip_g",
        num_query_token=35,
        embed_dim=768,
        train_vit=False,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        precision=1e-2,
    ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.loss = loss
        boite = load_model("blip2", "pretrain")

        assert vit in ["eva_clip_g", "clip_L"], "vit parameter must be eva_clip_g or clip_L"
        if vit == "eva_clip_g":
            vision_width = 1048
        
        elif vit == "clip_L":
            vision_width = 1024
        
        else:
            raise NotImplementedError
            
        
        boite.init_vision_encoder(model_name=vit,img_size=image_size,drop_path_rate=0 or drop_path_rate,
                                  use_grad_checkpoint=use_grad_checkpoint,precision=precision)

        self.visual_encoder = boite.visual_encoder
        self.tokenizer = init_tokenizer(boite.init_tokenizer())

        boite.init_Qformer(num_query_token=num_query_token,vision_width=vision_width)

        self.Qformer = boite.Qformer
        self.Qformer.config.vocab_size = 30525
        text_width = self.Qformer.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.train_vit = train_vit
        if not self.train_vit:
            # Do not train visual encoder
            for p in self.visual_encoder.parameters():
                p.requires_grad = False

        for p in self.vision_proj.parameters():
            p.requires_grad = False

        self.temp = 0.07

    def forward(self, batch, fabric):
        ref_img, tar_feat, caption, _ = batch

        device = ref_img.device
        if self.train_vit:
            ref_img_embs = self.visual_encoder(ref_img)
        else:
            with torch.no_grad():
                ref_img_embs = self.visual_encoder(ref_img)

        # Encode the target image
        tar_feat = tar_feat.to(device)
        print("a")
        tar_img_feat = F.normalize(tar_feat, dim=-1)

        # Encode the reference image
        ref_img_atts = torch.ones(ref_img_embs.size()[:-1], dtype=torch.long).to(device)

        print("b")
        text = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(device)

        # Shift encoder
        encoder_input_ids = text.input_ids.clone()
        print("c")
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id
        query_embs = self.Qformer(
            encoder_input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=ref_img_embs,
            encoder_attention_mask=ref_img_atts,
            return_dict=True,
        )
        print("d")
        query_feat = query_embs.last_hidden_state[:, 0, :]
        query_feat = F.normalize(self.text_proj(query_feat), dim=-1)
        print(query_feat.shape)
        if fabric.world_size > 1:
            # d: devices, b: batch size, e: embedding dim
            query_feat = fabric.all_gather(query_feat, sync_grads=True)
            query_feat = einops.rearrange(query_feat, "d b e -> (d b) e")

            tar_img_feat = fabric.all_gather(tar_img_feat, sync_grads=True)
            tar_img_feat = einops.rearrange(tar_img_feat, "d b e -> (d b) e")

        print(query_feat.shape,tar_img_feat.shape)

        return self.loss(query_feat, tar_img_feat, self.temp)


def blip2_cir(model, **kwargs):
    return model


def init_tokenizer(tokenizer):
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer