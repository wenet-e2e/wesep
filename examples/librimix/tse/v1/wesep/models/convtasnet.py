import torch
import torch.nn as nn

from wesep.modules.common import select_norm
from wesep.modules.common.speaker import SpeakerTransform
from wesep.modules.tasnet import DeepEncoder, DeepDecoder
from wesep.modules.tasnet import MultiEncoder, MultiDecoder
from wesep.modules.tasnet import FuseSeparation
from wesep.modules.tasnet.convs import Conv1D, ConvTrans1D
from wesep.modules.tasnet.speaker import ResNet4SpExplus
from wespeaker.models.speaker_model import get_speaker_model


class ConvTasNet(nn.Module):
    def __init__(
        self,
        N=512,
        L=16,
        B=128,
        H=512,
        P=3,
        X=8,
        R=3,
        spk_emb_dim=256,
        norm="gLN",
        activate="relu",
        causal=False,
        skip_con=False,
        spk_fuse_type="concatConv",
        # "concat", "additive", "multiply", "FiLM", "None",
        # ("concatConv" only for convtasnet)
        multi_fuse=True,
        use_spk_transform=True,
        encoder_type="Multi",  # 'Multi', 'Deep', None
        decoder_type="Multi",
        joint_training=True,
        multi_task=False,
        spksInTrain=251,
        spk_model=None,
        spk_model_init=None,
        spk_model_freeze=False,
        spk_args=None,
        spk_feat=False,
        feat_type="consistent",
    ):
        """
        :param N: Number of filters in autoencoder
        :param L: Length of the filters (in samples)
        :param B: Number of channels in bottleneck and the residual paths
        :param H: Number of channels in convolutional blocks
        :param P: Kernel size in convolutional blocks
        :param X: Number of convolutional blocks in each repeat
        :param R: Number of repeats
        :param norm:
        :param activate:
        :param causal:
        :param skip_con:
        :param spk_fuse_type: concat/addition/FiLM
        :param use_spk_transform:
        :param use_deep_enc:
        :param use_deep_dec:
        """
        super(ConvTasNet, self).__init__()

        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        # n x 1 x T => n x N x T
        if encoder_type == "Multi":
            self.encoder = MultiEncoder(
                in_channels=1,
                middle_channels=N,
                out_channels=B,
                kernel_size=L,
                stride=L // 2,
            )
        elif encoder_type == "Deep":
            self.encoder = DeepEncoder(1, N, L, stride=L // 2)
            self.LayerN_S = select_norm(norm, N)
            self.BottleN_S = Conv1D(N, B, 1)
        else:
            self.encoder = nn.Sequential(
                Conv1D(1, N, L, stride=L // 2, padding=0), nn.ReLU()
            )
            self.LayerN_S = select_norm(norm, N)
            self.BottleN_S = Conv1D(N, B, 1)

        self.joint_training = joint_training
        self.spk_feat = spk_feat
        self.feat_type = feat_type
        self.spk_model_freeze = spk_model_freeze
        self.multi_task = multi_task

        if joint_training:
            if not self.spk_feat:
                if self.feat_type == "consistent":
                    self.spk_model = ResNet4SpExplus(
                        in_channel=N, C_embedding=spk_emb_dim
                    )  # The speaker model is fixed for SpEx+ currently
            else:
                self.spk_model = get_speaker_model(spk_model)(**spk_args)
                if spk_model_init:
                    pretrained_model = torch.load(spk_model_init)
                    state = self.spk_model.state_dict()
                    for key in state.keys():
                        if key in pretrained_model.keys():
                            state[key] = pretrained_model[key]
                            # print(key)
                        else:
                            print("not %s loaded" % key)
                    self.spk_model.load_state_dict(state)
                    if self.spk_model_freeze:
                        for param in self.spk_model.parameters():
                            param.requires_grad = False
            if multi_task:
                self.pred_linear = nn.Linear(spk_emb_dim, spksInTrain)

        if not use_spk_transform:
            self.spk_transform = nn.Identity()
        else:
            self.spk_transform = SpeakerTransform()

        # Separation block
        # n x B x T => n x B x T
        self.separation = FuseSeparation(
            R,
            X,
            B,
            H,
            P,
            norm=norm,
            causal=causal,
            skip_con=skip_con,
            C_embedding=spk_emb_dim,
            spk_fuse_type=spk_fuse_type,
            multi_fuse=multi_fuse,
        )

        # n x N x T => n x 1 x L
        if decoder_type == "Multi":
            self.decoder = MultiDecoder(
                in_channels=B,
                middle_channels=N,
                out_channels=1,
                kernel_size=L,
                stride=L // 2,
            )
        elif decoder_type == "Deep":
            self.decoder = DeepDecoder(N, L, stride=L // 2)
            self.gen_masks = Conv1D(B, N, 1)
        else:
            self.decoder = ConvTrans1D(N, 1, L, stride=L // 2)
            self.gen_masks = Conv1D(B, N, 1)
        # activation function
        active_f = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "softmax": nn.Softmax(dim=0),
        }
        # self.activation_type = activate
        self.activation = active_f[activate]

    def forward(self, x, embeddings):
        if x.dim() >= 3:
            raise RuntimeError(
                "{} accept 1/2D tensor as input, but got {:d}".format(
                    self.__name__, x.dim()
                )
            )
        if x.dim() == 1:
            x = torch.unsqueeze(x, 0)
        # x: n x 1 x L => n x N x T
        if self.encoder_type == "Multi":
            e, w1, w2, w3 = self.encoder(x)
            x = e  # replace x with e, for asymmetric encoder-decoder
        else:
            x = self.encoder(x)
            e = self.LayerN_S(x)
            e = self.BottleN_S(
                e
            )  # Embedding fuse after dimension changed fro N to B

        if (
            self.joint_training
        ):  # Only support sharing Encoder and ResNet in SpEx+ currently
            # Speaker Encoder
            if not self.spk_feat and self.feat_type == "consistent":
                if self.encoder_type == "Multi":
                    _, aux_w1, aux_w2, aux_w3 = self.encoder(embeddings)
                    embeddings = torch.cat([aux_w1, aux_w2, aux_w3], 1)
                else:
                    aux_x = self.encoder(embeddings)
                    aux_e = self.LayerN_S(aux_x)
                    embeddings = self.BottleN_S(aux_e)
            embeddings = self.spk_model(embeddings)
            if isinstance(embeddings, tuple):
                embeddings = embeddings[-1]
            if self.multi_task:
                predict_speaker_lable = self.pred_linear(embeddings)

        spk_embeds = self.spk_transform(embeddings.unsqueeze(-1))
        e = self.separation(e, spk_embeds)

        # decoder part  n x L
        if self.decoder_type == "Multi":
            s = self.decoder(
                e, w1, w2, w3, actLayer=self.activation
            )  # s is a tuple by using multiDecoder
        else:
            # n x B x L => n x N x L
            m = self.gen_masks(e)
            # n x N x L
            m = self.activation(m)
            x = x * m
            s = self.decoder(x)

        if self.joint_training and self.multi_task:
            if not isinstance(s, list):
                s = [
                    s,
                ]
            s.append(predict_speaker_lable)

        return s  # s: N x Len Or List(N  x Len,x3/x4)


def check_parameters(net):
    """
    Returns module parameters. Mb
    """
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10**6


def test_convtasnet():
    x = torch.randn(4, 32000)
    spk_embeddings = torch.randn(4, 256)
    net = ConvTasNet(use_spk_transform=False, spk_fuse_type="FiLM")
    s = net(x, spk_embeddings)
    print(str(check_parameters(net)) + " Mb")
    print(s[1].shape)


if __name__ == "__main__":
    test_convtasnet()
