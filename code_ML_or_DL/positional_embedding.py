import torch
from torch import nn, Tensor
import math

# 从0实现positional_embedding，from：https://zhuanlan.zhihu.com/p/398039366
# 还有参考了CLIP的代码


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, d_model: int, learned: bool = False):
        # max_len: 输入文本的可能的最大长度，比如512或者1024
        # d_model一般情况下就是embed_size
        # learned，True的话就是重新学习PositionalEmbedding，BERT用重新学的，False就是《attention is all you need》的默认的
        super().__init__()
        if learned:  # used in BERT
            pe = nn.Parameter(torch.randn(max_len, d_model))
            pe = pe.unsqueeze(0)  # （1,max_len, d_model）,这样的shape可以方便直接在forward的时候和x相加
            self.pe = pe
        else:  # used in transformer
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)，这样的shape可以方便直接在forward的时候和x相加
            self.register_buffer("pe", pe)
            # 用register_buffer的好处是？答：定义一个不需要参与反向传播的常量，比如这里的pe，这样不会被视作模型的参数

    def forward(self, x: Tensor):
        seq_len = x.shape[1]  # x.shape = (bsz, seq_len, emb_size)
        return self.pe[:, :seq_len, :]


if __name__ == "__main__":
    # 测试
    position_embedding = PositionalEmbedding(1024, 768)
    bsz, seq_len, d_model = 4, 12, 768
    x = torch.Tensor(bsz, seq_len, d_model)
    pe = position_embedding(x)
    print(f"x.shape={x.shape}")
    print(f"pe.shape={pe.shape}")
    result = x + pe
    print(f"result.shape={result.shape}")

    print("----" * 40)
    
    position_embedding = PositionalEmbedding(1024, 768, True)
    bsz, seq_len, d_model = 4, 12, 768
    x = torch.Tensor(bsz, seq_len, d_model)
    pe = position_embedding(x)
    print(f"x.shape={x.shape}")
    print(f"pe.shape={pe.shape}")
    result = x + pe
    print(f"result.shape={result.shape}")
