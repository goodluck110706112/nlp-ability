import torch
from torch import nn, Tensor

# 从0开始实现多层感知机，包括激活函数relu也要自己实现
# 参考动手深度学习：


def relu(x: Tensor):
    zeros_x = torch.zeros_like(x)
    return torch.max(x, zeros_x)


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


class MultiLayerPerception(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, out_dim: int, dropout_p: float = 0.0
    ):
        super().__init__()
        self.activation = relu
        self.dropout_p = dropout_p
        # in_layer
        self.w1 = nn.Parameter(
            torch.randn(input_dim, hidden_dim, requires_grad=True) * 0.01
        )
        self.b1 = nn.Parameter(torch.zeros(hidden_dim, requires_grad=True))
        # out_layer
        self.w2 = nn.Parameter(
            torch.randn(hidden_dim, out_dim, requires_grad=True) * 0.01
        )
        self.b2 = nn.Parameter(torch.zeros(out_dim, requires_grad=True))
        self.params = [self.w1, self.b1, self.w2, self.b2]
        self.losser = nn.CrossEntropyLoss()

    def forward(self, x: Tensor, target=None):
        x = x.reshape(-1, x.shape[-1])  # (-1, input_dim)
        x = relu(x @ self.w1 + self.b1)  # 这里“@”代表矩阵乘法
        if self.training:
            x = dropout(x, self.dropout_p)
        x = x @ self.w2 + self.b2
        if target is not None:
            loss = self.losser(x, target)
        else:
            loss = torch.tensor(0, device=x.device)
        return x, loss


if __name__ == "__main__":
    # 验证让mlp训起来以后，loss能否收敛
    # data
    input_dim = 256
    num_class = 4
    mlp = MultiLayerPerception(input_dim, 256, num_class)
    bsz = 4
    x = torch.randn(bsz, input_dim)
    target = torch.tensor([0, 2, 1, 3], dtype=torch.long)
    # train
    from torch.optim import Adam  # 引入adam

    optimizer = Adam(mlp.params, lr=1e-3)
    num_steps = 100
    for _ in range(num_steps):
        optimizer.zero_grad()
        pred, loss = mlp(x, target)
        print(f"pred = {pred}")
        print(f"loss = {loss}")
        loss.backward()    # 反向传播求解梯度
        optimizer.step()   # 更新权重参数。
    print(f'torch.is_grad_enabled() = {torch.is_grad_enabled()}')
    mlp.eval()
    print(f'torch.is_grad_enabled() = {torch.is_grad_enabled()}')

