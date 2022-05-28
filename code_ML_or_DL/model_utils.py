from celery import xmap
import torch
from torch import nn, Tensor
from code_ML_or_DL.multi_head_attention import MultiHeadAttention
# 考察pytorch常用的组件的位置，形式服从：from torch.* imort *， 比如：from torch import nn,或者import torch.* as *
from torch.utils.data import DataLoader  # DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP  # ddp
import torch.nn.functional as F  # 大家常用的F， 比如F.softmax(attention_weights, dim=-1), 还有F.relu(x)
from torch.optim import Adam  # adam
from torch.utils.data import Dataset  # DataSet
from torch.optim.lr_scheduler import CosineAnnealingLR  # 余弦退火学习率（CosineAnnealingLR）
import snoop

# 下面是实现一些简短的代码，比如one_hot，还有dropout,
def one_hot(x: Tensor, n_class: int, dtype=torch.float32) -> Tensor:
    # X shape: (bsz), output shape: (bsz, n_class)
    # 实现one_hoh编码，假设输入x = torch.tensor([1, 0, 2])，n_class = 3,则输出为：
    # tensor([[0., 1., 0.],
    #         [1., 0., 0.],
    #         [0., 0., 1.]])
    x = x.long()
    result = torch.zeros(
        x.shape[0], n_class, dtype=dtype, device=x.device
    )  # (bsz, n_class)
    idx = x.view(-1, 1)  # (bsz, 1)
    # 送给scatter的index的shape的长度必须是2，也就是shape必须是（*， *）这种，比如（3，2）
    # 这里由于是ont-hot编码，每行只有一个1，所以此处idx.shape = (bsz, 1)
    result = result.scatter(dim=1, index=idx, value=1)  # (bsz, n_class)
    return result


def dropout(x: Tensor, dropout_prob: float = 0.5) -> Tensor:
    if dropout_prob == 0:  # 特殊情况：0的话直接返回x
        return x
    x = x.float()
    assert 0 <= dropout_prob <= 1
    keep_p = 1 - dropout_prob
    # 注意，#1和#2是等价的
    # mask = (torch.rand(x.shape) < keep_p).float()  #1
    mask = (torch.rand_like(x) < keep_p).float()  # 2
    # torch.rand_like()和torch.randn_like()的区别,前者是(0,1)的均匀分布，后者是标准正态分布
    # torch.rand_like(x)返回的是和x的shape一样的，服从(0,1)的均匀分布的tensor
    # torch.randn_like(x)返回的是和x的shape一样的，服从标准正态的tensor
    return mask * x / keep_p  # 这里除以1-p，是为了在训练和测试的时候，这一层的输出有相同的期望


def softmax(x: torch.Tensor) -> torch.Tensor:
    # 从0实现softmax，一维的向量，二维的都可以处理，参考了logsumexp trick，先减去最后那个维度(dim=-1)的最大值，防止溢出
    x_max = torch.max(
        x, dim=-1, keepdim=True  # 为了适配后面的广播操作，需要keepdim=True
    ).values  # 这种torch.max返回的是namedtuple，有两个key，分别是values, indices
    x -= x_max
    x = torch.exp(x)  # 也可以写x = x.exp()
    return x / torch.sum(x, dim=-1, keepdim=True)  # 为了适配后面的广播操作，需要keepdim=True


def attention_mask(
    length,
):  # transformer-decoder的mask，用来防止信息泄露，做文本生成用的到，这个mask也叫做decoder-mask
    # attention_mask(4)，输出：
    # tensor([[0., -inf, -inf, -inf],
    #         [0., 0., -inf, -inf],
    #         [0., 0., 0., -inf],
    #         [0., 0., 0., 0.]])
    mask = (torch.zeros([length, length])).float().fill_(float("-inf"))
    mask = torch.triu(input=mask, diagonal=1)  # 我们希望对角线及对角线以下（包括对角线）的部分全部置0，所以diagonal = 1
    return mask


class AttentionPooling(nn.Module):
    # 来源于CLIP的AttentionPool2d，用这个pooling方式取代了global average pooling
    def __init__(
        self, spacial_dim: int = 7, embed_size: int = 512, num_heads: int = 8
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim ** 2 + 1, embed_size) / embed_size ** 0.5
        )  # 保证std=embedding_dim ** -0.5，这样相当于方差=1/embed_size
        self.mha = MultiHeadAttention(
            embed_size=embed_size, n_heads=num_heads
        )  # mha就是一个multi-head-attention

    def forward(self, x: torch.Tensor):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(
            2, 0, 1
        )  # (N,C,H,W) -> (H*W,N,C)
        cls_emb = x.mean(
            dim=0, keepdim=True
        )  # (1, N, C)，注意这里keepdim=True，为了方面下一行进行cat操作
        x = torch.cat([cls_emb, x], dim=0)  # (1+H*W,N,C)
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (H*W+1,N,C)
        x = self.mha(query=x, key=x, value=x)  # (1+HW, N, C)，这里输入的qk和v都是x

        return x[0]  # 取cls_emb，最后的shape为（N, C）

if __name__ == "__main__":
    # demo：

    # dropout
    # x = torch.randn(4, 5)
    # drop_prob = 0.5
    # x_drop = dropout(x, drop_prob)
    # print(x)
    # print(x_drop)

    # softmax
    x = torch.tensor([[1, 6, 24, 12, 4], [5, -8, 1, 2, 3]]).float()
    # x = torch.tensor([1, 6, 2, 12, 4, 52, 5]).float()

    # attention_mask
    print(attention_mask(length=4))
