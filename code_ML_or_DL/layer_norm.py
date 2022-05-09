import torch
from torch import nn, Tensor

# 实现layer normalization，来自:https://zhuanlan.zhihu.com/p/398039366


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x: Tensor):
        # 这里的实现，和隔壁batch normalization的差不多，就是dim换一下
        mean = x.mean(dim=-1, keepdim=True)  # 如果是bn，那么这里dim都是0
        var = x.var(dim=-1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x + self.beta


if __name__ == "__main__":
    embed_size = 512
    ln_1 = nn.LayerNorm(embed_size)  # 官方的
    ln_2 = LayerNorm(embed_size)  # 自己实现的
    print(ln_1.weight.shape)
    print(ln_2.gamma.shape)
    bsz, seq_len = 2, 5
    x = torch.rand(bsz, seq_len, embed_size)
    x_cp = x.clone()
    print("----" * 30)
    x1 = ln_1(x)
    x2 = ln_2(x_cp)
    print(x1.mean(dim=0))
    print(x1.mean(dim=1))
    print(x1.mean(dim=2))
    print("----" * 30)
    print(x2.mean(dim=0))
    print(x2.mean(dim=1))
    print(x2.mean(dim=2))
