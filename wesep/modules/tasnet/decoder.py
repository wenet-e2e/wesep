import torch
import torch.nn as nn

from wesep.modules.tasnet.convs import Conv1D, ConvTrans1D


class DeepDecoder(nn.Module):

    def __init__(self, N, kernel_size=16, stride=16 // 2):
        super(DeepDecoder, self).__init__()
        self.sequential = nn.Sequential(
            nn.ConvTranspose1d(N,
                               N,
                               kernel_size=3,
                               stride=1,
                               dilation=8,
                               padding=8),
            nn.PReLU(),
            nn.ConvTranspose1d(N,
                               N,
                               kernel_size=3,
                               stride=1,
                               dilation=4,
                               padding=4),
            nn.PReLU(),
            nn.ConvTranspose1d(N,
                               N,
                               kernel_size=3,
                               stride=1,
                               dilation=2,
                               padding=2),
            nn.PReLU(),
            nn.ConvTranspose1d(N,
                               N,
                               kernel_size=3,
                               stride=1,
                               dilation=1,
                               padding=1),
            nn.PReLU(),
            nn.ConvTranspose1d(N,
                               1,
                               kernel_size=kernel_size,
                               stride=stride,
                               bias=True),
        )

    def forward(self, x):
        """
        x: N x L or N x C x L
        """
        x = self.sequential(x)
        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)

        return x


class MultiDecoder(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels, kernel_size,
                 stride):
        super(MultiDecoder, self).__init__()

        B = in_channels
        N = middle_channels
        L = kernel_size
        # n x B x T => n x 2N x T
        self.mask1 = Conv1D(B, N, 1)
        self.mask2 = Conv1D(B, N, 1)
        self.mask3 = Conv1D(B, N, 1)

        # using ConvTrans1D: n x N x T => n x 1 x To
        # To = (T - 1) * L // 2 + L
        self.decoder_1d_1 = ConvTrans1D(N,
                                        out_channels,
                                        kernel_size=L,
                                        stride=stride,
                                        bias=True)
        self.decoder_1d_2 = ConvTrans1D(N,
                                        out_channels,
                                        kernel_size=80,
                                        stride=stride,
                                        bias=True)
        self.decoder_1d_3 = ConvTrans1D(N,
                                        out_channels,
                                        kernel_size=160,
                                        stride=stride,
                                        bias=True)

    def forward(self, x, w1, w2, w3, actLayer):
        """
        x: N x L or N x C x L
        """
        m1 = actLayer(self.mask1(x))
        m2 = actLayer(self.mask2(x))
        m3 = actLayer(self.mask3(x))

        s1 = w1 * m1
        s2 = w2 * m2
        s3 = w3 * m3

        est1 = self.decoder_1d_1(s1, squeeze=True)
        xlen = est1.shape[-1]
        if est1.dim() > 1:
            est2 = self.decoder_1d_2(s2, squeeze=True)[:, :xlen]
            est3 = self.decoder_1d_3(s3, squeeze=True)[:, :xlen]
        else:
            est1 = est1.unsqueeze(0)
            est2 = self.decoder_1d_2(s2, squeeze=True).unsqueeze(0)[:, :xlen]
            est3 = self.decoder_1d_3(s3, squeeze=True).unsqueeze(0)[:, :xlen]
        s = [est1, est2, est3]
        return s
