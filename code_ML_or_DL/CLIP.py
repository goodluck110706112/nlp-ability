from turtle import forward
import torch
from torch import nn, Tensor
from torch.nn import functional as F

# 简单的实现一下openai的CLIP：Contrastive Language-Image Pre-training
# modified from：huggingface的源码

# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: Tensor) -> Tensor:
    target = torch.arange(len(logits), device=logits.device)
    return F.cross_entropy(input=logits, target=target)


def clip_loss(similarity: Tensor) -> Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.T)
    return (caption_loss + image_loss) / 2.0


class CLIPModel(nn.Module):
    def __init__(self, text_encoder, image_encoder, init_temperature):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.temperature = nn.Parameter(
            nn.Parameter(torch.ones([]) * init_temperature)
        )  # 就是对比学习的温度，是个可学习的标量，比如0.07

    def forward(self, txts: Tensor, images: Tensor):
        text_embeds = self.text_encoder(txts)  # (bsz, txt_emb_dim)
        image_embeds = self.image_encoder(images)  # (bsz, img_emb_dim)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.temperature.exp()
        logits_text = (
            torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        )  # （bsz, bsz）

        loss = clip_loss(logits_text)

        return loss
