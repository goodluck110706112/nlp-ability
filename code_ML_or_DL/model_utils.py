import torch
from torch import nn, Tensor

# 考察pytorch常用的组件的位置，形式服从：from torch.* imort *， 比如：from torch import nn,或者import torch.* as *
from torch.utils.data import DataLoader  # DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP  # ddp
import torch.nn.functional as F  # 大家常用的F， 比如F.softmax(attention_weights, dim=-1), 还有F.relu(x)
from torch.optim import Adam  # adam
from torch.utils.data import Dataset  # DataSet
from torch.optim.lr_scheduler import CosineAnnealingLR  # 余弦退火学习率（CosineAnnealingLR）


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
    x = x.float()
    assert 0 <= dropout_prob <= 1
    keep_p = 1 - dropout_prob
    if keep_p == 0:
        return torch.zeros_like(x)
    # 注意，#1和#2是等价的
    # mask = (torch.rand(x.shape) < keep_p).float()  #1
    mask = (torch.rand_like(x) < keep_p).float()     #2
    # torch.rand_like()和torch.randn_like()的区别,前者是(0,1)的均匀分布，后者是标准正态分布
    # torch.rand_like(x)返回的是和x的shape一样的，服从(0,1)的均匀分布的tensor
    # torch.randn_like(x)返回的是和x的shape一样的，服从标准正态的tensor
    return mask * x / keep_p   # 这里除以1-p，是为了在训练和测试的时候，这一层的输出有相同的期望


if __name__ == "__main__":
    x = torch.randn(4, 5)
    drop_prob = 0.5
    x_drop = dropout(x, drop_prob)
    print(x)
    print(x_drop)
    