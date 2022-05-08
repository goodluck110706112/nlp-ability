import torch
from torch import nn, Tensor
import numpy as np

# 可选内容：
# 从0开始实现包括max pooling，这个只适用于batch size为1，单通道（channel=1）的输入，
# 这个代码不一定全部要掌握，看时间是否来得及，优先掌握__call__()的实现
# from:https://zhuanlan.zhihu.com/p/188881658


class MaxPooling2D:
    def __init__(self, kernel_size=(2, 2), stride=1):
        self.kernel_size = kernel_size  # kernel的(h, w)
        self.kernel_height, self.kernel_width = kernel_size
        self.stride = stride

        self.x = None
        self.out_height = None
        self.out_width = None
        self.arg_max = None

    def __call__(self, x: np.ndarray):
        assert (len(x.shape) == 2) # 假设输入x的shape是(h,w)
        self.x = x  # (h, w)
        in_height, in_width = x.shape  # (h, w)

        # 下面两行和卷积的公式一样的，只是这里没有padding
        self.out_height = (int((in_height - self.kernel_height) / self.stride) + 1)
        self.out_width = int((in_width - self.kernel_width) / self.stride) + 1

        out = np.zeros((self.out_height, self.out_width))
        self.arg_max = np.zeros_like(out, dtype=np.int32)

        for i in range(self.out_height):
            for j in range(self.out_width):
                start_i = i * self.stride
                start_j = j * self.stride
                end_i = start_i + self.kernel_height
                end_j = start_j + self.kernel_width
                out[i, j] = np.max(x[start_i:end_i, start_j:end_j])
                self.arg_max[i, j] = np.argmax(x[start_i:end_i, start_j:end_j])  # 记录这个信息，反向传播的时候用的到

        return out

    def backward(self, d_loss):  # 反向传播，返回梯度，这个函数也可以命名为get_grad()，但是这个函数好像有点小问题
        dx = np.zeros_like(self.x)
        for i in range(self.out_height):
            for j in range(self.out_width):
                start_i = i * self.stride
                start_j = j * self.stride
                end_i = start_i + self.kernel_height
                end_j = start_j + self.kernel_width
                index = np.unravel_index(
                    indices=self.arg_max[i, j], shape=self.kernel_size
                )
                dx[start_i:end_i, start_j:end_j][index] = d_loss[i, j]  #
        return dx


if __name__ == "__main__":
    np.set_printoptions(precision=8, suppress=True, linewidth=120)
    x_numpy = np.random.random((1, 1, 4, 6))
    x_tensor = torch.tensor(x_numpy, requires_grad=True)

    max_pool_tensor = nn.MaxPool2d((2, 2), 2)
    max_pool_numpy = MaxPooling2D((2, 2), stride=2)

    out_numpy = max_pool_numpy(x_numpy[0, 0])
    out_tensor = max_pool_tensor(x_tensor)

    d_loss_numpy = np.random.random(out_tensor.shape)
    d_loss_tensor = torch.tensor(d_loss_numpy, requires_grad=True)
    out_tensor.backward(d_loss_tensor)

    dx_numpy = max_pool_numpy.backward(d_loss_numpy[0, 0])
    dx_tensor = x_tensor.grad
    # print('input \n', x_numpy)
    print("out_numpy \n", out_numpy)
    print("out_tensor \n", out_tensor.data.numpy())
    print("dx_numpy \n", dx_numpy)
    print("dx_tensor \n", dx_tensor.data.numpy())