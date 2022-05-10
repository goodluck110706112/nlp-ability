import torch
from torch import nn, Tensor
import math
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_size: int,
        n_heads: int,
        dropout: float = 0,
    ):
        super().__init__()
        self.embed_size = embed_size
        self.n_heads = n_heads
        self.head_size = embed_size // n_heads
        assert (
            self.head_size * n_heads == self.embed_size
        ), "embed_size must divided by n_heads"
        self.q_proj = nn.Linear(embed_size, embed_size)
        self.k_proj = nn.Linear(embed_size, embed_size)
        self.v_proj = nn.Linear(embed_size, embed_size)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.out_proj = nn.Linear(embed_size, embed_size)

    def split_into_diff_heads(self, x: Tensor):
        # x: (bsz, seq_len, embed_size) -> (bsz, n_heads, seq_len, head_size)
        bsz = x.shape[0]
        return x.reshape(bsz, -1, self.n_heads, self.head_size).transpose(1, 2)

    def scaled_dot_product_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask=None
    ) -> Tensor:
        attention_weights = torch.matmul(q, k.transpose(-2, -1))  #1
        attention_weights /= math.sqrt(k.shape[-1])
        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask, -1e9)
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout_layer(attention_weights)  # # 这个就是最后的权重
        out = torch.matmul(
            attention_weights, v
        )  # (bsz, n_heads, seq_len, head_size)
        return out

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask=None):
        # self attention: query = key = value
        # encoder-decoder attention: query from decoder, key = value ,from encoder
        # query.shape: (bsz, seq_len, embed_size)
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q, k, v = (
            self.split_into_diff_heads(x) for x in [q, k, v]
        )  # (bsz, n_heads, seq_len, head_size)
        scores = self.scaled_dot_product_attention(q, k, v, mask)
        # merge diff heads
        scores = (
            scores.transpose(1, 2)
            .contiguous()  # contiguous常加在view()之前
            .view(scores.shape[0], -1, self.embed_size)
        )  # (bsz, seq_len, embed_size)
        out = self.out_proj(scores)  # (bsz, seq_len, embed_size)
        return out
