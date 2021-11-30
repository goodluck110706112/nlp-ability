"""参考youtube的《Pytorch Transformers from Scratch》，
还有https://github.com/Whiax/BERT-Transformer-Pytorch/blob/main/train.py
从零开始，实现一个Transformer
"""

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
        attention_weights = torch.matmul(q, k.transpose(-2, -1))
        attention_weights /= math.sqrt(k.shape[-1])
        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask, -1e9)
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout_layer(attention_weights)
        out = torch.matmul(
            attention_weights, v
        )  # (bsz, n_heads, seq_len, head_size)
        return out

    # @snoop
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
        out = self.out_proj(scores)
        return out


class PositionalEmbedding(nn.Module):
    # Transformer的PositionalEmbedding
    def __init__(self, max_len, d_model):
        # d_model一般情况下就是embed_size
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        # 根据论文，实现PositionalEmbedding：
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i) / d_model)))
        pe = pe.unsqueeze(
            0
        )  # 因为输入一般是(bsz,seq_len,embed_size)或者(bsz,seq_len,d_model)，所以pe.shape=(1,seq_len,embed_size)
        self.register_buffer(
            "pe", pe
        )  # 用register_buffer的好处是？答：定义一个不需要参与反向传播的常量，比如这里的pe，这样不会被视作模型的参数

    def forward(self, x):
        seq_len = x.shape[1]
        return self.pe[:, :seq_len, :]


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


# class PositionalEmbedding(nn.Module):
#     # Transformer的PositionalEmbedding
#     def __init__(self, d_model, max_seq_len=80):
#         super().__init__()
#         self.d_model = d_model
#         pe = torch.zeros(max_seq_len, d_model, requires_grad=False)
#         for pos in range(max_seq_len):
#             for i in range(0, d_model, 2):
#                 pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
#                 pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i) / d_model)))
#         pe = pe.unsqueeze(0)
#         self.register_buffer(
#             "pe", pe
#         )  # 用register_buffer的好处是？答：定义一个不需要参与反向传播的常量，比如这里的pe

#     def forward(self, x):
#         return self.pe[:, : x.size(1)]  # x.size(1) = seq_len


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
