import torch
import torch.nn as nn

from wesep.modules.common import select_norm

# from wesep.modules.common.spkadapt import SpeakerFuseLayer


class Conv1D(nn.Conv1d):

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        # x: N x C x L
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x


class ConvTrans1D(nn.ConvTranspose1d):

    def __init__(self, *args, **kwargs):
        super(ConvTrans1D, self).__init__(*args, **kwargs)

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


class Conv1DBlock(nn.Module):
    """
    Consider only residual links
    """

    def __init__(
        self,
        in_channels=256,
        out_channels=512,
        kernel_size=3,
        dilation=1,
        norm="gln",
        causal=False,
        skip_con=True,
    ):
        super(Conv1DBlock, self).__init__()
        # conv 1 x 1
        self.conv1x1 = Conv1D(in_channels, out_channels, 1)
        self.PReLU_1 = nn.PReLU()
        self.norm_1 = select_norm(norm, out_channels)
        # not causal don't need to padding, causal need to pad+1 = kernel_size
        self.pad = ((dilation * (kernel_size - 1)) // 2 if not causal else
                    (dilation * (kernel_size - 1)))
        # depthwise convolution
        # TODO: This is not depthwise seperable convolution
        self.dwconv = Conv1D(
            out_channels,
            out_channels,
            kernel_size,
            groups=out_channels,
            padding=self.pad,
            dilation=dilation,
        )
        self.PReLU_2 = nn.PReLU()
        self.norm_2 = select_norm(norm, out_channels)
        if skip_con:
            self.Sc_conv = nn.Conv1d(out_channels, in_channels, 1, bias=True)
        self.Output = nn.Conv1d(out_channels, in_channels, 1, bias=True)
        self.causal = causal
        self.skip_con = skip_con

    def forward(self, x):
        # x: N x C x L
        # N x O_C x L
        c = self.conv1x1(x)
        # N x O_C x L
        c = self.PReLU_1(c)
        c = self.norm_1(c)
        # causal: N x O_C x (L+pad)
        # noncausal: N x O_C x L
        c = self.dwconv(c)
        if self.causal:
            c = c[:, :, :-self.pad]
        c = self.PReLU_2(c)
        c = self.norm_2(c)
        # N x O_C x L
        if self.skip_con:
            Sc = self.Sc_conv(c)
            c = self.Output(c)
            return Sc, c + x
        c = self.Output(c)
        return x + c


class Conv1DBlock4Fuse(nn.Module):
    """
    1D convolutional block:
        Conv1x1 - PReLU - Norm - DConv - PReLU - Norm - SConv
    """

    def __init__(
        self,
        in_channels=256,
        spk_embed_dim=100,
        conv_channels=512,
        kernel_size=3,
        dilation=1,
        norm="cLN",
        causal=False,
    ):
        super(Conv1DBlock4Fuse, self).__init__()
        # 1x1 conv
        self.conv1x1 = Conv1D(in_channels + spk_embed_dim, conv_channels, 1)
        self.prelu1 = nn.PReLU()
        self.lnorm1 = select_norm(norm, conv_channels)
        dconv_pad = ((dilation * (kernel_size - 1)) // 2 if not causal else
                     (dilation * (kernel_size - 1)))
        # depthwise conv
        self.dconv = nn.Conv1d(
            conv_channels,
            conv_channels,
            kernel_size,
            groups=conv_channels,
            padding=dconv_pad,
            dilation=dilation,
            bias=True,
        )
        self.prelu2 = nn.PReLU()
        self.lnorm2 = select_norm(norm, conv_channels)
        # 1x1 conv cross channel
        self.sconv = nn.Conv1d(conv_channels, in_channels, 1, bias=True)
        # different padding way
        self.causal = causal
        self.dconv_pad = dconv_pad

    def forward(self, x, aux):
        T = x.shape[-1]
        aux = aux.repeat(1, 1, T)
        y = torch.cat([x, aux], 1)
        y = self.conv1x1(y)
        y = self.lnorm1(self.prelu1(y))
        y = self.dconv(y)
        if self.causal:
            y = y[:, :, :-self.dconv_pad]
        y = self.lnorm2(self.prelu2(y))
        y = self.sconv(y)
        x = x + y
        return x
