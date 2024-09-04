import torch.nn as nn

from wesep.modules.common.norm import ChannelWiseLayerNorm
from wesep.modules.tasnet.convs import Conv1D


class ResBlock(nn.Module):
    """
    ref to
        https://github.com/fatchord/WaveRNN/blob/master/models/fatchord_version.py
        and
        https://github.com/Jungjee/RawNet/blob/master/PyTorch/model_RawNet.py
    """

    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dims, out_dims, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_dims, out_dims, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(out_dims)
        self.batch_norm2 = nn.BatchNorm1d(out_dims)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()

        self.mp = nn.MaxPool1d(3)
        if in_dims != out_dims:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(
                in_dims, out_dims, kernel_size=1, bias=False
            )
        else:
            self.downsample = False

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        if self.downsample:
            residual = self.conv_downsample(residual)
        x = x + residual
        x = self.prelu2(x)
        return self.mp(x)


class ResNet4SpExplus(nn.Module):
    def __init__(self, in_channel=256, C_embedding=256):
        super().__init__()
        self.aux_enc3 = nn.Sequential(
            ChannelWiseLayerNorm(3 * in_channel),
            Conv1D(3 * 256, 256, 1),
            ResBlock(256, 256),
            ResBlock(256, 512),
            ResBlock(512, 512),
            Conv1D(512, C_embedding, 1),
        )

    def forward(self, x):
        aux = self.aux_enc3(x)
        aux = aux.mean(dim=-1)
        return aux
