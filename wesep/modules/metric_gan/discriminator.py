import torch
import torch.nn as nn


# utility functions/classes used in the implementation of discriminators.
class LearnableSigmoid(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


# discriminators
class CMGAN_Discriminator(nn.Module):
    def __init__(
        self,
        n_fft=400,
        hop=100,
        in_channels=2,
        hid_chans=16,
        ksz=(4, 4),
        stride=(2, 2),
        padding=(1, 1),
        bias=False,
        num_conv_blocks=4,
        num_linear_layers=2,
    ):
        """discriminator used in CMGAN (Interspeech 2022)
            paper: https://arxiv.org/pdf/2203.15149.pdf
            code: https://github.com/ruizhecao96/CMGAN

        Args:
        n_fft (int, optional): the windows length of stft. Defaults to 400.
        hop (int, optional): the hop length of stft. Defaults to 100.
        in_channels (int, optional): num of input channels. Defaults to 2.
        hid_chans (int, optional): num of hidden channels. Defaults to 16.
        ksz (tuple, optional): kernel size. Defaults to (4, 4).
        stride (tuple, optional): stride. Defaults to (2, 2).
        padding (tuple, optional): padding. Defaults to (1, 1).
        bias (bool, optional): bias. Defaults to False.
        num_conv_blocks (int, optional): num of conv blocks. Defaults to 4.
        num_linear_layers (int, optional): num of linear layers. Defaults to 2.
        """
        super(CMGAN_Discriminator, self).__init__()
        assert num_conv_blocks >= num_linear_layers

        self.n_fft = n_fft
        self.hop = hop
        self.num_conv_blocks = num_conv_blocks
        self.num_linear_layers = num_linear_layers

        self.conv = nn.ModuleList([])
        in_chans = in_channels
        out_chans = hid_chans
        for i in range(num_conv_blocks):
            self.conv.append(
                nn.Sequential(
                    nn.utils.spectral_norm(
                        nn.Conv2d(
                            in_chans,
                            out_chans,
                            ksz,
                            stride,
                            padding,
                            bias=bias,
                        )
                    ),
                    nn.InstanceNorm2d(out_chans, affine=True),
                    nn.PReLU(out_chans),
                )
            )
            in_chans = out_chans
            out_chans = hid_chans * (2 ** (i + 1))

        self.pooling = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
        )

        self.fc = nn.ModuleList([])
        for i in range(num_linear_layers - 1):
            self.fc.append(
                nn.Sequential(
                    nn.utils.spectral_norm(
                        nn.Linear(
                            hid_chans * (2 ** (num_conv_blocks - 1 - i)),
                            hid_chans * (2 ** (num_conv_blocks - 2 - i)),
                        )
                    ),
                    nn.Dropout(0.3),
                    nn.PReLU(hid_chans * (2 ** (num_conv_blocks - 2 - i))),
                )
            )
        self.fc.append(
            nn.Sequential(
                nn.utils.spectral_norm(
                    nn.Linear(
                        hid_chans
                        * (2 ** (num_conv_blocks - num_linear_layers)),
                        1,
                    )
                ),
                LearnableSigmoid(1),
            )
        )

    def forward(self, ref_wav, est_wav):
        """

        Args:
            ref_wav (torch.Tensor): the reference signal. [B, T]
            est_wav (torch.Tensor): the estimated signal. [B, T]

        Return:
            estimated_scores (torch.Tensor): estimated scores, [B]
        """
        ref_spec = torch.stft(
            ref_wav,
            self.n_fft,
            self.hop,
            window=torch.hann_window(self.n_fft)
            .to(ref_wav.device)
            .type(ref_wav.type()),
            return_complex=True,
        ).transpose(-1, -2)
        est_spec = torch.stft(
            est_wav,
            self.n_fft,
            self.hop,
            window=torch.hann_window(self.n_fft)
            .to(est_wav.device)
            .type(est_wav.type()),
            return_complex=True,
        ).transpose(-1, -2)
        # input shape: (B, 2, T, F)
        input = torch.stack((abs(ref_spec), abs(est_spec)), dim=1)
        for i in range(self.num_conv_blocks):
            input = self.conv[i](input)

        input = self.pooling(input)
        for i in range(self.num_linear_layers):
            input = self.fc[i](input)
        return input


if __name__ == "__main__":
    # functions used to test discriminators
    def test_CMGAN_Discriminator():
        B, T = 2, 16000
        ref_spec = torch.randn(B, T)
        est_spec = torch.randn(B, T)
        D = CMGAN_Discriminator()
        metric = D(ref_spec, est_spec).detach()
        print(f"estimated metric score is {metric}")

    test_CMGAN_Discriminator()
