import torch.nn as nn

from wesep.modules.common import select_norm
from wesep.modules.common.speaker import SpeakerFuseLayer
from wesep.modules.tasnet.convs import Conv1DBlock, Conv1DBlock4Fuse


class Separation(nn.Module):

    def __init__(
        self,
        R,
        X,
        B,
        H,
        P,
        norm="gLN",
        causal=False,
        skip_con=True,
        start_dilation=0,
    ):
        """
        Args
        :param R: Number of repeats
        :param X: Number of convolutional blocks in each repeat
        :param B: Number of channels in bottleneck and the residual paths
        :param H: Number of channels in convolutional blocks
        :param P: Kernel size in convolutional blocks
        :param norm: The type of normalization(gln, cln, bn)
        :param causal: Two choice(causal or noncausal)
        :param skip_con: Whether to use skip connection
        """
        super(Separation, self).__init__()
        self.separation = nn.ModuleList([])
        for _ in range(R):
            for x in range(start_dilation, X):
                self.separation.append(
                    Conv1DBlock(B, H, P, 2**x, norm, causal, skip_con))
        self.skip_con = skip_con

    def forward(self, x):
        """
        x: [B, N, L]
        out: [B, N, L]
        """
        if self.skip_con:
            skip_connection = 0
            for i in range(len(self.separation)):
                skip, out = self.separation[i](x)
                skip_connection = skip_connection + skip
                x = out
            return skip_connection
        else:
            for i in range(len(self.separation)):
                out = self.separation[i](x)
                x = out
            return x


class FuseSeparation(nn.Module):

    def __init__(
        self,
        R,
        X,
        B,
        H,
        P,
        norm="gLN",
        causal=False,
        skip_con=False,
        C_embedding=256,
        spk_fuse_type="concatConv",
        multi_fuse=True,
    ):
        """

        :param R: Number of repeats
        :param X: Number of convolutional blocks in each repeat
        :param B: Number of channels in bottleneck and the residual paths
        :param H: Number of channels in convolutional blocks
        :param P: Kernel size in convolutional blocks
        :param norm: The type of normalization(gln, cln, bn)
        :param causal: Two choice(causal or noncausal)
        :param skip_con: Whether to use skip connection
        """
        super(FuseSeparation, self).__init__()
        self.multi_fuse = multi_fuse
        self.spk_fuse_type = spk_fuse_type
        self.separation = nn.ModuleList([])
        if self.multi_fuse:
            for _ in range(R):
                if spk_fuse_type == "concatConv":
                    self.separation.append(
                        Conv1DBlock4Fuse(
                            spk_embed_dim=C_embedding,
                            in_channels=B,
                            conv_channels=H,
                            kernel_size=P,
                            norm=norm,
                            causal=causal,
                            dilation=1,
                        ))
                    self.separation.append(
                        Separation(
                            1,
                            X,
                            B,
                            H,
                            P,
                            norm=norm,
                            causal=causal,
                            skip_con=skip_con,
                            start_dilation=1,
                        ))
                else:
                    self.separation.append(
                        SpeakerFuseLayer(
                            embed_dim=C_embedding,
                            feat_dim=B,
                            fuse_type=spk_fuse_type,
                        ))
                    self.separation.append(nn.PReLU())
                    self.separation.append(select_norm(norm, B))
                    self.separation.append(
                        Separation(
                            1,
                            X,
                            B,
                            H,
                            P,
                            norm=norm,
                            causal=causal,
                            skip_con=skip_con,
                        ))
        else:
            if spk_fuse_type == "concatConv":
                self.separation.append(
                    Conv1DBlock4Fuse(
                        spk_embed_dim=C_embedding,
                        in_channels=B,
                        conv_channels=H,
                        kernel_size=P,
                        norm=norm,
                        causal=causal,
                        dilation=1,
                    ))
            else:
                self.separation.append(
                    SpeakerFuseLayer(
                        embed_dim=C_embedding,
                        feat_dim=B,
                        fuse_type=spk_fuse_type,
                    ))
                self.separation.append(nn.PReLU())
                self.separation.append(select_norm(norm, B))
            self.separation = Separation(R,
                                         X,
                                         B,
                                         H,
                                         P,
                                         norm=norm,
                                         causal=causal,
                                         skip_con=skip_con)

    def forward(self, x, spk_embedding):
        """
        x: [B, N, L]
        out: [B, N, L]
        """

        if self.multi_fuse:
            if self.spk_fuse_type == "concatConv":
                round_num = 2
            else:
                round_num = 4
            for i in range(len(self.separation)):
                if i % round_num == 0:
                    x = self.separation[i](x, spk_embedding)
                else:
                    x = self.separation[i](x)
        else:
            x = self.separation[0](x, spk_embedding)
            for i in range(1, len(self.separation)):
                x = self.separation[i](x)
        return x
