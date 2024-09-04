# The implementation is based on:
# https://github.com/espnet/espnet/blob/master/espnet2/enh/separator/tfgridnetv2_separator.py
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torchaudio
from packaging.version import parse as V

from wespeaker.models.speaker_model import get_speaker_model

from wesep.modules.common.speaker import PreEmphasis
from wesep.modules.common.speaker import SpeakerFuseLayer, SpeakerTransform
from wesep.modules.tfgridnet.gridnet_block import GridNetBlock

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")


class TFGridNet(nn.Module):
    """Offline TFGridNetV2. Compared with TFGridNet, TFGridNetV2 speeds up the code
        by vectorizing multiple heads in self-attention, and better dealing with
        Deconv1D in each intra- and inter-block when emb_ks == emb_hs.

    Reference:
    [1] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
    "TF-GridNet: Integrating Full- and Sub-Band Modeling for Speech Separation",
    in TASLP, 2023.
    [2] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
    "TF-GridNet: Making Time-Frequency Domain Models Great Again for Monaural
    Speaker Separation", in ICASSP, 2023.

    NOTES:
    As outlined in the Reference, this model works best when trained with
    variance normalized mixture input and target, e.g., with mixture of
    shape [batch, samples, microphones], you normalize it by dividing
    with torch.std(mixture, (1, 2)). You must do the same for the target
    signals. It is encouraged to do so when not using
    scale-invariant loss functions such as SI-SDR.
    Specifically, use:
        std_ = std(mix)
        mix = mix / std_
        tgt = tgt / std_

    Args:
        n_srcs: number of output sources/speakers.
        n_fft: stft window size.
        stride: stft stride.
        window: stft window type choose between 'hamming', 'hanning' or None.
        n_imics: num of channels (only fixed-array geometry supported).
        n_layers: number of TFGridNetV2 blocks.
        lstm_hidden_units: number of hidden units in LSTM.
        attn_n_head: number of heads in self-attention
        attn_approx_qk_dim: approximate dim of frame-level key/value tensors
        emb_dim: embedding dimension
        emb_ks: kernel size for unfolding and deconv1D
        emb_hs: hop size for unfolding and deconv1D
        activation: activation function to use in the whole TFGridNetV2 model,
            you can use any torch supported activation e.g. 'relu' or 'elu'.
        eps: small epsilon for normalization layers.
        spk_emb_dim: the dimension of target speaker embeddings.
        use_spk_transform: whether use networks to transfer the speaker embeds.
        spk_fuse_type: the fusion method of speaker embeddings.
    """

    def __init__(
        self,
        n_srcs=1,
        sr=16000,
        n_fft=128,
        stride=64,
        window="hann",
        n_imics=1,
        n_layers=6,
        lstm_hidden_units=192,
        attn_n_head=4,
        attn_approx_qk_dim=512,
        emb_dim=48,
        emb_ks=4,
        emb_hs=1,
        activation="prelu",
        eps=1.0e-5,
        spk_emb_dim=256,
        use_spk_transform=False,
        spk_fuse_type="multiply",
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
        super().__init__()
        self.n_srcs = n_srcs
        self.n_fft = n_fft
        self.stride = stride
        self.window = window
        self.n_imics = n_imics
        self.n_layers = n_layers
        self.spk_emb_dim = spk_emb_dim
        self.joint_training = joint_training
        self.spk_feat = spk_feat
        self.feat_type = feat_type
        self.spk_model_freeze = spk_model_freeze
        self.multi_task = multi_task

        assert n_fft % 2 == 0
        n_freqs = n_fft // 2 + 1

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
                        n_fft=n_fft,
                        win_length=n_fft,
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

        self.spk_fuse = SpeakerFuseLayer(
            embed_dim=spk_emb_dim,
            feat_dim=n_freqs,
            fuse_type=spk_fuse_type,
        )

        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2 * n_imics, emb_dim, ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )

        self.blocks = nn.ModuleList([])
        for _ in range(n_layers):
            self.blocks.append(
                GridNetBlock(
                    emb_dim,
                    emb_ks,
                    emb_hs,
                    n_freqs,
                    lstm_hidden_units,
                    n_head=attn_n_head,
                    approx_qk_dim=attn_approx_qk_dim,
                    activation=activation,
                    eps=eps,
                ))

        self.deconv = nn.ConvTranspose2d(emb_dim,
                                         n_srcs * 2,
                                         ks,
                                         padding=padding)

    def forward(
        self,
        input: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Forward.

        Args:
            input (torch.Tensor): batched multi-channel audio tensor with
                    M audio channels and N samples [B, N, M]
            embeddings (torch.Tensor): batched target speaker embeddings [B, D]

        Returns:
            enhanced (List[Union(torch.Tensor)]):
                    [(B, T), ...] list of len n_srcs
                    of mono audio tensors with T samples.
        """
        batch_size, n_samples = input.shape[0], input.shape[1]
        spk_emb_input = embeddings
        if self.n_imics == 1:
            assert len(input.shape) == 2
            input = input[..., None]  # [B, N, M]

        mix_std_ = torch.std(input, dim=(1, 2), keepdim=True)  # [B, 1, 1]
        input = input / mix_std_  # RMS normalization

        input = input.transpose(1, 2).reshape(
            -1, input.size(1))  # [B, N, M] -> [B*M, N]
        window_func = getattr(torch, f"{self.window}_window")
        window = window_func(self.n_fft,
                             dtype=input.dtype,
                             device=input.device)

        batch = torch.stft(
            input,
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.stride,
            window=window,
            return_complex=True,
            onesided=True,
        )  # [B, F, T]
        batch = batch.transpose(1, 2)  # [B, T, F]

        batch0 = batch.view(batch_size, -1, batch.size(1),
                            batch.size(2))  # [B, M, T, F]
        # ilens = torch.full((batch_size,), n_samples, dtype=torch.long)
        batch = torch.cat((batch0.real, batch0.imag), dim=1)  # [B, 2*M, T, F]
        n_batch, _, n_frames, n_freqs = batch.shape

        batch = self.conv(batch)  # [B, -1, T, F]

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

        spk_embedding = self.spk_transform(spk_emb_input)  # [B, D]
        spk_embedding = spk_embedding.unsqueeze(1).unsqueeze(3)  # [B, 1, D, 1]

        for ii in range(self.n_layers):
            batch = torch.transpose(
                self.spk_fuse(batch.transpose(2, 3), spk_embedding), 2,
                3)  # [B, -1, T, F]
            batch = self.blocks[ii](batch)  # [B, -1, T, F]

        batch = self.deconv(batch)  # [B, n_srcs*2, T, F]

        batch = batch.view([n_batch, self.n_srcs, 2, n_frames, n_freqs])
        assert is_torch_1_9_plus, "Require torch 1.9.0+."
        batch = torch.complex(batch[:, :, 0], batch[:, :, 1])

        batch = torch.istft(
            torch.transpose(batch.view(-1, n_frames, n_freqs), 1, 2),
            n_fft=self.n_fft,
            hop_length=self.stride,
            win_length=self.n_fft,
            window=window,
            onesided=True,
            length=n_samples,
            return_complex=False,
        )  # [B, n_srcs]

        batch = self.pad2(batch.view([n_batch, self.num_spk, -1]), n_samples)

        batch = batch * mix_std_  # reverse the RMS normalization

        # batch = [batch[:, src] for src in range(self.num_spk)]
        batch = batch.squeeze(1)

        return batch, predict_speaker_lable

    @property
    def num_spk(self):
        return self.n_srcs

    @staticmethod
    def pad2(input_tensor, target_len):
        input_tensor = torch.nn.functional.pad(
            input_tensor, (0, target_len - input_tensor.shape[-1]))
        return input_tensor
