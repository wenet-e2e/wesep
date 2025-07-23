from __future__ import print_function

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from wespeaker.models.speaker_model import get_speaker_model

from wesep.modules.common.speaker import PreEmphasis
from wesep.modules.common.speaker import SpeakerFuseLayer
from wesep.modules.common.speaker import SpeakerTransform


class ResRNN(nn.Module):

    def __init__(self, input_size, hidden_size, bidirectional=True):
        super(ResRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.eps = torch.finfo(torch.float32).eps

        self.norm = nn.GroupNorm(1, input_size, self.eps)
        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            1,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # linear projection layer
        self.proj = nn.Linear(hidden_size * 2,
                              input_size)  # hidden_size = feature_dim * 2

    def forward(self, input):
        # input shape: batch, dim, seq

        rnn_output, _ = self.rnn(self.norm(input).transpose(1, 2).contiguous())
        rnn_output = self.proj(rnn_output.contiguous().view(
            -1, rnn_output.shape[2])).view(input.shape[0], input.shape[2],
                                           input.shape[1])

        return input + rnn_output.transpose(1, 2).contiguous()


"""
TODO : attach the speaker embedding to each input
Input shape:(B,feature_dim + spk_emb_dim , T)
"""


class BSNet(nn.Module):

    def __init__(self, in_channel, nband=7, bidirectional=True):
        super(BSNet, self).__init__()

        self.nband = nband
        self.feature_dim = in_channel // nband
        self.band_rnn = ResRNN(self.feature_dim,
                               self.feature_dim * 2,
                               bidirectional=bidirectional)
        self.band_comm = ResRNN(self.feature_dim,
                                self.feature_dim * 2,
                                bidirectional=bidirectional)

    def forward(self, input, dummy: Optional[torch.Tensor] = None):
        # input shape: B, nband*N, T
        B, N, T = input.shape

        band_output = self.band_rnn(
            input.view(B * self.nband, self.feature_dim,
                       -1)).view(B, self.nband, -1, T)

        # band comm
        band_output = (band_output.permute(0, 3, 2, 1).contiguous().view(
            B * T, -1, self.nband))
        output = (self.band_comm(band_output).view(
            B, T, -1, self.nband).permute(0, 3, 2, 1).contiguous())

        return output.view(B, N, T)


class FuseSeparation(nn.Module):

    def __init__(
        self,
        nband=7,
        num_repeat=6,
        feature_dim=128,
        spk_emb_dim=256,
        spk_fuse_type="concat",
        multi_fuse=True,
    ):
        """

        :param nband : len(self.band_width)
        """
        super(FuseSeparation, self).__init__()
        self.multi_fuse = multi_fuse
        self.nband = nband
        self.feature_dim = feature_dim
        self.separation = nn.ModuleList([])
        if self.multi_fuse:
            for _ in range(num_repeat):
                self.separation.append(
                    SpeakerFuseLayer(
                        embed_dim=spk_emb_dim,
                        feat_dim=feature_dim,
                        fuse_type=spk_fuse_type,
                    ))
                self.separation.append(BSNet(nband * feature_dim, nband))
        else:
            self.separation.append(
                SpeakerFuseLayer(
                    embed_dim=spk_emb_dim,
                    feat_dim=feature_dim,
                    fuse_type=spk_fuse_type,
                ))
            for _ in range(num_repeat):
                self.separation.append(BSNet(nband * feature_dim, nband))

    def forward(self, x, spk_embedding, nch: torch.Tensor = torch.tensor(1)):
        """
        x: [B, nband, feature_dim, T]
        out: [B, nband, feature_dim, T]
        """
        batch_size = x.shape[0]

        if self.multi_fuse:
            for i, sep_func in enumerate(self.separation):
                x = sep_func(x, spk_embedding)
                if i % 2 == 0:
                    x = x.view(batch_size * nch, self.nband * self.feature_dim,
                               -1)
                else:
                    x = x.view(batch_size * nch, self.nband, self.feature_dim,
                               -1)
        else:
            x = self.separation[0](x, spk_embedding)
            x = x.view(batch_size * nch, self.nband * self.feature_dim, -1)
            for idx, sep in enumerate(self.separation):
                if idx > 0:
                    x = sep(x, spk_embedding)
            x = x.view(batch_size * nch, self.nband, self.feature_dim, -1)
        return x


class BSRNN(nn.Module):
    # self, sr=16000, win=512, stride=128, feature_dim=128, num_repeat=6,
    # use_bidirectional=True
    def __init__(
        self,
        spk_emb_dim=256,
        sr=16000,
        win=512,
        stride=128,
        feature_dim=128,
        num_repeat=6,
        use_spk_transform=True,
        use_bidirectional=True,
        spk_fuse_type="concat",
        multi_fuse=True,
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
        super(BSRNN, self).__init__()

        self.sr = sr
        self.win = win
        self.stride = stride
        self.group = self.win // 2
        self.enc_dim = self.win // 2 + 1
        self.feature_dim = feature_dim
        self.eps = torch.finfo(torch.float32).eps
        self.spk_emb_dim = spk_emb_dim
        self.joint_training = joint_training
        self.spk_feat = spk_feat
        self.feat_type = feat_type
        self.spk_model_freeze = spk_model_freeze
        self.multi_task = multi_task

        # 0-1k (100 hop), 1k-4k (250 hop),
        # 4k-8k (500 hop), 8k-16k (1k hop),
        # 16k-20k (2k hop), 20k-inf

        # 0-8k (1k hop), 8k-16k (2k hop), 16k
        bandwidth_100 = int(np.floor(100 / (sr / 2.0) * self.enc_dim))
        bandwidth_200 = int(np.floor(200 / (sr / 2.0) * self.enc_dim))
        bandwidth_500 = int(np.floor(500 / (sr / 2.0) * self.enc_dim))
        bandwidth_2k = int(np.floor(2000 / (sr / 2.0) * self.enc_dim))

        # add up to 8k
        self.band_width = [bandwidth_100] * 15
        self.band_width += [bandwidth_200] * 10
        self.band_width += [bandwidth_500] * 5
        self.band_width += [bandwidth_2k] * 1

        self.band_width.append(self.enc_dim - int(np.sum(self.band_width)))
        self.nband = len(self.band_width)

        if use_spk_transform:
            self.spk_transform = SpeakerTransform()
        else:
            self.spk_transform = nn.Identity()

        if joint_training:
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
            if spk_model_freeze:
                for param in self.spk_model.parameters():
                    param.requires_grad = False
            if not spk_feat:
                if feat_type == "consistent":
                    self.preEmphasis = PreEmphasis()
                    self.spk_encoder = torchaudio.transforms.MelSpectrogram(
                        sample_rate=sr,
                        n_fft=win,
                        win_length=win,
                        hop_length=stride,
                        f_min=20,
                        window_fn=torch.hamming_window,
                        n_mels=spk_args["feat_dim"],
                    )
            else:
                self.preEmphasis = nn.Identity()
                self.spk_encoder = nn.Identity()

            if multi_task:
                self.pred_linear = nn.Linear(spk_emb_dim, spksInTrain)
            else:
                self.pred_linear = nn.Identity()

        self.BN = nn.ModuleList([])
        for i in range(self.nband):
            self.BN.append(
                nn.Sequential(
                    nn.GroupNorm(1, self.band_width[i] * 2, self.eps),
                    nn.Conv1d(self.band_width[i] * 2, self.feature_dim, 1),
                ))

        self.separator = FuseSeparation(
            nband=self.nband,
            num_repeat=num_repeat,
            feature_dim=feature_dim,
            spk_emb_dim=spk_emb_dim,
            spk_fuse_type=spk_fuse_type,
            multi_fuse=multi_fuse,
        )

        # self.proj =  nn.Linear(hidden_size*2, input_size)

        self.mask = nn.ModuleList([])
        for i in range(self.nband):
            self.mask.append(
                nn.Sequential(
                    nn.GroupNorm(1, self.feature_dim,
                                 torch.finfo(torch.float32).eps),
                    nn.Conv1d(self.feature_dim, self.feature_dim * 4, 1),
                    nn.Tanh(),
                    nn.Conv1d(self.feature_dim * 4, self.feature_dim * 4, 1),
                    nn.Tanh(),
                    nn.Conv1d(self.feature_dim * 4, self.band_width[i] * 4, 1),
                ))

    def pad_input(self, input, window, stride):
        """
        Zero-padding input according to window/stride size.
        """
        batch_size, nsample = input.shape

        # pad the signals at the end for matching the window/stride size
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(input.type())
            input = torch.cat([input, pad], 1)
        pad_aux = torch.zeros(batch_size, stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 1)

        return input, rest

    def forward(self, input, embeddings):
        # input shape: (B, C, T)

        wav_input = input
        spk_emb_input = embeddings
        batch_size, nsample = wav_input.shape
        nch = 1

        # frequency-domain separation
        spec = torch.stft(
            wav_input,
            n_fft=self.win,
            hop_length=self.stride,
            window=torch.hann_window(self.win).to(wav_input.device).type(
                wav_input.type()),
            return_complex=True,
        )

        # concat real and imag, split to subbands
        spec_RI = torch.stack([spec.real, spec.imag], 1)  # B*nch, 2, F, T
        subband_spec = []
        subband_mix_spec = []
        band_idx = 0
        for i in range(len(self.band_width)):
            subband_spec.append(spec_RI[:, :, band_idx:band_idx +
                                        self.band_width[i]].contiguous())
            subband_mix_spec.append(spec[:, band_idx:band_idx +
                                         self.band_width[i]])  # B*nch, BW, T
            band_idx += self.band_width[i]

        # normalization and bottleneck
        subband_feature = []
        for i, bn_func in enumerate(self.BN):
            subband_feature.append(
                bn_func(subband_spec[i].view(batch_size * nch,
                                             self.band_width[i] * 2, -1)))
        subband_feature = torch.stack(subband_feature, 1)  # B, nband, N, T
        # print(subband_feature.size(), spk_emb_input.size())

        predict_speaker_lable = torch.tensor(0.0).to(
            spk_emb_input.device)  # dummy
        if self.joint_training:
            if not self.spk_feat:
                if self.feat_type == "consistent":
                    with torch.no_grad():
                        spk_emb_input = self.preEmphasis(spk_emb_input)
                        spk_emb_input = self.spk_encoder(spk_emb_input) + 1e-8
                        spk_emb_input = spk_emb_input.log()
                        spk_emb_input = spk_emb_input - torch.mean(
                            spk_emb_input, dim=-1, keepdim=True)
                        spk_emb_input = spk_emb_input.permute(0, 2, 1)

            tmp_spk_emb_input = self.spk_model(spk_emb_input)
            if isinstance(tmp_spk_emb_input, tuple):
                spk_emb_input = tmp_spk_emb_input[-1]
            else:
                spk_emb_input = tmp_spk_emb_input
            predict_speaker_lable = self.pred_linear(spk_emb_input)

        spk_embedding = self.spk_transform(spk_emb_input)
        spk_embedding = spk_embedding.unsqueeze(1).unsqueeze(3)

        sep_output = self.separator(subband_feature, spk_embedding,
                                    torch.tensor(nch))

        sep_subband_spec = []
        for i, mask_func in enumerate(self.mask):
            this_output = mask_func(sep_output[:, i]).view(
                batch_size * nch, 2, 2, self.band_width[i], -1)
            this_mask = this_output[:, 0] * torch.sigmoid(
                this_output[:, 1])  # B*nch, 2, K, BW, T
            this_mask_real = this_mask[:, 0]  # B*nch, K, BW, T
            this_mask_imag = this_mask[:, 1]  # B*nch, K, BW, T
            est_spec_real = (subband_mix_spec[i].real * this_mask_real -
                             subband_mix_spec[i].imag * this_mask_imag
                             )  # B*nch, BW, T
            est_spec_imag = (subband_mix_spec[i].real * this_mask_imag +
                             subband_mix_spec[i].imag * this_mask_real
                             )  # B*nch, BW, T
            sep_subband_spec.append(torch.complex(est_spec_real,
                                                  est_spec_imag))
        est_spec = torch.cat(sep_subband_spec, 1)  # B*nch, F, T
        output = torch.istft(
            est_spec.view(batch_size * nch, self.enc_dim, -1),
            n_fft=self.win,
            hop_length=self.stride,
            window=torch.hann_window(self.win).to(wav_input.device).type(
                wav_input.type()),
            length=nsample,
        )

        output = output.view(batch_size, nch, -1)
        s = torch.squeeze(output, dim=1)

        return s, predict_speaker_lable


if __name__ == "__main__":
    from thop import profile, clever_format

    model = BSRNN(
        spk_emb_dim=256,
        sr=16000,
        win=512,
        stride=128,
        feature_dim=128,
        num_repeat=6,
        spk_fuse_type="additive",
        joint_training=True,
        spk_model="ResNet34",
        spk_model_init=False,
        spk_args={"feat_dim":80, "embed_dim": 256, "pooling_func": "TSTP", "two_emb_layer": False},
        spk_feat=True,
    )

    s = 0
    for param in model.parameters():
        s += np.product(param.size())
    print("# of parameters: " + str(s / 1024.0 / 1024.0))
    x = torch.randn(4, 32000)
    #spk_embeddings = torch.randn(4, 256)
    #output = model(x, spk_embeddings)
    spk_feats = torch.randn(4, 300, 80)
    output = model(x, spk_feats)
    print(output[0].shape)

    #macs, params = profile(model, inputs=(x, spk_embeddings))
    macs, params = profile(model, inputs=(x, spk_feats))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)
