from typing import Tuple

import torch
import torch.nn as nn


class Conv1D(nn.Conv1d):
    """
    1D conv in ConvTasNet
    """

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x


class Conv2dBlock(nn.Module):

    def __init__(
            self,
            in_dims: int = 16,
            out_dims: int = 32,
            kernel_size: Tuple[int] = (3, 3),
            stride: Tuple[int] = (1, 1),
            padding: Tuple[int] = (1, 1),
    ) -> None:
        super(Conv2dBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_dims, out_dims, kernel_size, stride,
                                padding)
        self.elu = nn.ELU()
        self.norm = nn.InstanceNorm2d(out_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv2d(x)
        x = self.elu(x)
        return self.norm(x)


class ConvTrans2dBlock(nn.Module):

    def __init__(
            self,
            in_dims: int = 32,
            out_dims: int = 16,
            kernel_size: Tuple[int] = (3, 3),
            stride: Tuple[int] = (1, 2),
            padding: Tuple[int] = (1, 0),
            output_padding: Tuple[int] = (0, 0),
    ) -> None:
        super(ConvTrans2dBlock, self).__init__()
        self.convtrans2d = nn.ConvTranspose2d(in_dims, out_dims, kernel_size,
                                              stride, padding, output_padding)
        self.elu = nn.ELU()
        self.norm = nn.InstanceNorm2d(out_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convtrans2d(x)
        x = self.elu(x)
        return self.norm(x)


class DenseBlock(nn.Module):

    def __init__(self, in_dims, out_dims, mode="enc", **kargs):
        super(DenseBlock, self).__init__()
        if mode not in ["enc", "dec"]:
            raise RuntimeError("The mode option must be 'enc' or 'dec'!")

        n = 1 if mode == "enc" else 2
        self.conv1 = Conv2dBlock(in_dims=in_dims * n,
                                 out_dims=in_dims,
                                 **kargs)
        self.conv2 = Conv2dBlock(in_dims=in_dims * (n + 1),
                                 out_dims=in_dims,
                                 **kargs)
        self.conv3 = Conv2dBlock(in_dims=in_dims * (n + 2),
                                 out_dims=in_dims,
                                 **kargs)
        self.conv4 = Conv2dBlock(in_dims=in_dims * (n + 3),
                                 out_dims=in_dims,
                                 **kargs)
        self.conv5 = Conv2dBlock(in_dims=in_dims * (n + 4),
                                 out_dims=out_dims,
                                 **kargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.conv1(x)
        y2 = self.conv2(torch.cat([x, y1], 1))
        y3 = self.conv3(torch.cat([x, y1, y2], 1))
        y4 = self.conv4(torch.cat([x, y1, y2, y3], 1))
        y5 = self.conv5(torch.cat([x, y1, y2, y3, y4], 1))
        return y5


class TCNBlock(nn.Module):
    """
    TCN block:
        IN - ELU - Conv1D - IN - ELU - Conv1D
    """

    def __init__(
        self,
        in_dims: int = 384,
        out_dims: int = 384,
        kernel_size: int = 3,
        dilation: int = 1,
        causal: bool = False,
    ) -> None:
        super(TCNBlock, self).__init__()
        self.norm1 = nn.InstanceNorm1d(in_dims)
        self.elu1 = nn.ELU()
        dconv_pad = ((dilation * (kernel_size - 1)) // 2 if not causal else
                     (dilation * (kernel_size - 1)))
        # dilated conv
        self.dconv1 = nn.Conv1d(
            in_dims,
            out_dims,
            kernel_size,
            padding=dconv_pad,
            dilation=dilation,
            groups=in_dims,
            bias=True,
        )

        self.norm2 = nn.InstanceNorm1d(in_dims)
        self.elu2 = nn.ELU()
        self.dconv2 = nn.Conv1d(in_dims, out_dims, 1, bias=True)

        # different padding way
        self.causal = causal
        self.dconv_pad = dconv_pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.elu1(self.norm1(x))
        y = self.dconv1(y)
        if self.causal:
            y = y[:, :, :-self.dconv_pad]
        y = self.elu2(self.norm2(y))
        y = self.dconv2(y)
        x = x + y
        return x
