import torch as th
import torch.nn as nn
import torch.nn.functional as F

from wesep.modules.common import select_norm
from wesep.modules.tasnet.convs import Conv1D


class DeepEncoder(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DeepEncoder, self).__init__()
        self.sequential = nn.Sequential(
            Conv1D(in_channels, out_channels, kernel_size, stride=stride),
            Conv1D(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                dilation=1,
                padding=1,
            ),
            nn.PReLU(),
            Conv1D(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                dilation=2,
                padding=2,
            ),
            nn.PReLU(),
            Conv1D(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                dilation=4,
                padding=4,
            ),
            nn.PReLU(),
            Conv1D(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                dilation=8,
                padding=8,
            ),
            nn.PReLU(),
        )

    def forward(self, x):
        """
        :param  x: [B, T]
        :return: out: [B, N, T]
        """

        x = self.sequential(x)
        return x


class MultiEncoder(nn.Module):

    def __init__(
        self, in_channels, middle_channels, out_channels, kernel_size, stride
    ):
        super(MultiEncoder, self).__init__()
        self.L1 = kernel_size
        self.L2 = 80
        self.L3 = 160
        self.encoder_1d_short = Conv1D(
            in_channels, middle_channels, self.L1, stride=stride, padding=0
        )
        self.encoder_1d_middle = Conv1D(
            in_channels, middle_channels, self.L2, stride=stride, padding=0
        )
        self.encoder_1d_long = Conv1D(
            in_channels, middle_channels, self.L3, stride=stride, padding=0
        )
        # keep T not change
        # T = int((xlen - L) / (L // 2)) + 1
        # before repeat blocks, always cLN
        self.ln = select_norm(
            "cLN", 3 * middle_channels
        )  # ChannelWiseLayerNorm(3 * middle_channels)
        # n x N x T => n x B x T
        self.proj = Conv1D(3 * middle_channels, out_channels, 1)

    def forward(self, x):
        """
        :param  x: [B, T]
        :return: out: [B, N, T]
        """
        w1 = F.relu(self.encoder_1d_short(x))
        T = w1.shape[-1]
        xlen1 = x.shape[-1]
        xlen2 = (T - 1) * (self.L1 // 2) + self.L2
        xlen3 = (T - 1) * (self.L1 // 2) + self.L3
        w2 = F.relu(
            self.encoder_1d_middle(F.pad(x, (0, xlen2 - xlen1), "constant", 0))
        )
        w3 = F.relu(
            self.encoder_1d_long(F.pad(x, (0, xlen3 - xlen1), "constant", 0))
        )
        # n x 3N x T
        x = self.ln(th.cat([w1, w2, w3], 1))
        # n x B x T
        x = self.proj(x)
        return x, w1, w2, w3
