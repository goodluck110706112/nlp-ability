import torch
from torch import nn, Tensor

# 基于torch，从0实现batch normalization，来自于《动手学深度学习》：https://zh.d2l.ai/chapter_convolutional-modern/batch-norm.html#id11


class BatchNorm(nn.Module):
    # num_features：完全连接层的输出数量或卷积层的输出通道数。
    # num_dims：2表示完全连接层，4表示卷积层
    def __init__(self, num_features: int, num_dims: int):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的gamma和beta参数，分别初始化成1和0，也就是bn的scale and shift操作为：Y = gamma * X + beta（X是标准化后的）
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型参数的变量初始化为0和1，下面这两个参数在预测的时候用的到，训练的时候会更新，但是用不到，
        # 也就是说预测的时候用的全局的均值和方差，训练的时候用的这个batch的均值和方差
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)
        # 定义eps 和 momentum
        self.eps = 1e-6
        self.momentum = 0.9

    def batch_norm(self, X: Tensor):
        # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式，这个操作有点骚
        if not torch.is_grad_enabled():
            # eval，直接使用传入的移动平均所得的均值和方差，也就是全局的均值和方差
            X_hat = (X - self.moving_mean) / torch.sqrt(self.moving_var + self.eps)
        else:  # training
            assert len(X.shape) in (2, 4)
            if len(X.shape) == 2:
                # 使用全连接层的情况，计算特征维上的均值和方差
                mean = X.mean(dim=0)
                var = ((X - mean) ** 2).mean(dim=0)
            else:
                # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。
                # 这里我们需要保持X的形状以便后面可以做广播运算
                mean = X.mean(dim=(0, 2, 3), keepdim=True)
                var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
            # 训练模式下，用当前batch的均值和方差做标准化
            X_hat = (X - mean) / torch.sqrt(var + self.eps)
            # 更新移动平均的均值和方差，也就是更新全局的均值和方差
            self.moving_mean = self.momentum * self.moving_mean + (1.0 - self.momentum) * mean
            self.moving_var = self.momentum * self.moving_var + (1.0 - self.momentum) * var
        Y = self.gamma * X_hat + self.beta  # rescale and shift
        return Y

    def forward(self, X: Tensor):
        # 如果X不在内存上，将moving_mean和moving_var复制到X所在显存上，这个写法也要学会
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 然后直接进行bn
        Y = self.batch_norm(X)
        return Y


if __name__ == "__main__":
    bn = BatchNorm(512, 4)
    bsz = 4
    C, H, W = 512, 7, 7
    X = torch.Tensor(bsz, C, H, W)
    print(bn(X).shape)
