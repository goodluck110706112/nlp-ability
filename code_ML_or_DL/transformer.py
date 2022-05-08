"""参考youtube的《Pytorch Transformers from Scratch》，
还有https://github.com/Whiax/BERT-Transformer-Pytorch/blob/main/train.py
从零开始，实现一个Transformer
"""

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from multi_head_attention import MultiHeadAttention
from positional_embedding import PositionalEmbedding

class FeedForward(nn.Module):
    def __init__(
        self, embed_size: int, dropout: float = 0, inner_size: int = None
    ):
        super().__init__()
        if inner_size is None:
            inner_size = embed_size
        self.linear1 = nn.Linear(embed_size, inner_size)
        self.linear2 = nn.Linear(inner_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_size: int,
        n_heads: int,
        dropout: float,
        ff_inner_size: int,
        normal_before=False,  # normal_before这个参数知道就行，为了精简不写了，有兴趣可以参考fairseq
    ):
        super().__init__()
        self.multihead_attention = MultiHeadAttention(
            embed_size, n_heads, dropout
        )
        self.ff = FeedForward(embed_size, dropout, ff_inner_size)
        self.self_attention_norm = nn.LayerNorm(embed_size)
        self.final_norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, encoder_padding_mask: Tensor = None):
        # first block: mha
        residual = x
        x = self.dropout(
            self.multihead_attention(
                query=x, key=x, value=x, mask=encoder_padding_mask
            )  # self-attention , q=k=v=x.
        )
        x = residual + x
        x = self.self_attention_norm(x)
        # second block: ff
        residual = x
        x = self.dropout(self.ff(x))
        x = residual + x
        x = self.final_norm(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        n_layers: int,
        vocab_size: int,
        max_len: int,
        embed_size: int,
        n_heads: int,
        ff_inner_size: int,
        dropout: float,
        normal_before: bool,
    ):
        super().__init__()  # 对继承自父类的属性初始化
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.pe = PositionalEmbedding(max_len, embed_size)
        self.layers = [
            TransformerEncoderLayer(
                embed_size, n_heads, dropout, ff_inner_size, normal_before
            )
            for _ in range(n_layers)
        ]
        self.layers = nn.ModuleList(self.layers)
        self.final_linear = nn.Linear(embed_size, embed_size)

    def forward(self, src_tokens: Tensor, encoder_padding_mask: Tensor = None):
        # src_tokens.shape (bsz, seq_len)
        x = self.embeddings(src_tokens)  # (bsz, seq_len, embed_size)
        x = x + self.pe(x)
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)
        x = self.final_linear(x)  # (bsz, seq_len, embed_size)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_size: int,
        n_heads: int,
        dropout: float,
        ff_inner_size: int,
        normal_before: bool = False,
    ):
        super().__init__()
        self.self_attention = MultiHeadAttention(embed_size, n_heads, dropout)
        self.encoder_decoder_attention = MultiHeadAttention(
            embed_size, n_heads, dropout
        )
        self.ff = FeedForward(embed_size, dropout, ff_inner_size)
        self.self_attention_norm = nn.LayerNorm(embed_size)
        self.encoder_decoder_attention_norm = nn.LayerNorm(embed_size)
        self.final_norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        encoder_padding_mask: Tensor,
        tgt_mask: Tensor,
    ):
        # self attention, 为了得到query
        residual = x
        x = self.dropout(
            self.self_attention(query=x, key=x, value=x, mask=tgt_mask)
        )
        query = residual + x
        query = self.self_attention_norm(query)
        # encoder-decoder-attention
        residual = query
        x = self.dropout(
            self.encoder_decoder_attention(
                query=query,
                key=encoder_out,
                value=encoder_out,
                mask=encoder_padding_mask,
            )  # k,v all from encoder
        )
        x = query + x
        x = self.encoder_decoder_attention_norm(x)
        # ff
        residual = x
        x = self.ff(x)
        x = residual + x
        x = self.final_norm(x)
        return x


if __name__ == "__main__":
    n_layers = 6
    vocab_size: int = 10000
    max_len: int = 512
    embed_size: int = 768
    n_heads: int = 8
    ff_inner_size: int = 512
    dropout: float = 0.2
    normal_before: bool = True
    transformer_encoder = TransformerEncoder(
        n_layers,
        vocab_size,
        max_len,
        embed_size,
        n_heads,
        ff_inner_size,
        dropout,
        normal_before,
    )
    bsz = 12
    seq_len = 56
    src_tokens = torch.randint(0, 679, (bsz, seq_len))
    print(src_tokens.shape)
    out = transformer_encoder(src_tokens)
    print(out.shape)