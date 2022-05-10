from dataclasses import dataclass
import numpy as np
from typing import List, Union, Tuple, Any
import torch
from torch import rand_like
# 选修的，时间来不及不用掌握，可以看一眼就行
# from cupy.cuda import Device
# 这里实现一个轻量的torch框架，只要实现1，Tensor这个类， 2，加减乘除操作的forward()和backward()就行


class Tensor:
    # 注意这个不是对标torch.Tensor(shape)，其实对标的是torch.tensor(data)，
    # 比如torch.tensor([1,2])，返回的就是tensor([1, 2])，也就是说最开始的参数不是shape，而直接就是data
    def __init__(
        self,
        data: Union[np.ndarray, List],
        requires_grad: bool = False,
        dtype=None,
        device=None,
    ) -> None:
        if not isinstance(data, np.ndarray):  # 转化成np.array形式，方便取shape等操作
            data = np.array(
                data, dtype=dtype
            )  # 这样的话，type(data) = <class 'numpy.ndarray'>
        self.data = data
        self.requires_grad = requires_grad
        self.data.dtype = dtype
        self._device = device
        self.grad = None

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:  # 用于debug和print
        return f"tensor({self.data})"

    @property  # 只有@property表示self.shape不能修改，除非另写一个也叫shape的函数，加一个@shape.setter装饰器
    def shape(self) -> Tuple:
        """返回Tensor各维度大小的元素"""
        return self.data.shape

    @property
    def dtype(self):
        """返回Tensor中数据的类型"""
        return self.data.dtype

    @property
    def ndim(self):
        return self.data.ndim


class OpsBase:
    def __init__(self) -> None:
        # 保存需要在backward()中使用的Tensor或其他对象(如Shape)
        self.saved_tensors = []

    def save_for_backward(self, *x: Any) -> None:
        self.saved_tensors.extend(x)

    def unbroadcast(self, grad: Tensor, in_shape: Tuple) -> Tensor:
        """
        广播操作的逆操作，确保grad转换成in_shape的形状
        Args:
            grad: 梯度
            in_shape: 梯度要转换的形状
        Returns:
        """
        # 首先计算维度个数之差
        ndims_added = grad.ndim - len(in_shape)
        # 由于广播时，先从左边插入，再进行复制，所以逆操作时，也从左边开始，进行复制的逆操作（求和）
        for _ in range(ndims_added):
            # 在axis=0上进行求和，去掉第0个维度，如果ndims_added > 1，就需要不停的在第0个维度上面求和
            grad = grad.sum(axis=0)

        # 处理 (2,3) + (1,3) => (2,3) grad的情况
        # 看in_shape中有没有维度=1的情况
        for i, dim in enumerate(in_shape):
            if dim == 1:
                # 那么需要在该axis上求和，并且保持维度 这里(2,3) => (1,3) grad 就和输入维度保持一致了
                grad = grad.sum(axis=i, keepdims=True)

        return grad


class Add(OpsBase):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        self.save_for_backward(x.shape, y.shape)  # 加法的梯度为1，不需要保存x和y，只需要保存他们的shape
        return x + y

    def backward(self, grad):
        shape_x, shape_y = self.saved_tensors
        # 输入有两个，都是需要计算梯度的，因此输出也是两个
        return self.unbroadcast(grad, shape_x), self.unbroadcast(grad, shape_y)


class Mul(OpsBase):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        """
        实现 z = x * y
        """
        # 乘法需要保存输入x和y，用于反向传播,所以这里和Add（加法）那里不太一样，加法的梯度为1
        self.save_for_backward(x, y)
        return x * y

    def backward(self, grad: Tensor) -> Tuple[Tensor, Tensor]:
        x, y = self.saved_tensors
        # 分别返回∂L/∂x 和 ∂L/∂y
        return self.unbroadcast(grad * y, x.shape), self.unbroadcast(
            grad * x, y.shape
        )


if __name__ == "__main__":
    x = [1, 2, 3]
    y = Tensor(x)
    print(y)
    print(len(y))
    print(y.shape)
    print(y.dtype)
    print(y.ndim)
