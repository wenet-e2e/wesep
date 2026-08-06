import os
import yaml
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence #, pack_sequence, pad_packed_sequence

import torchaudio.compliance.kaldi as kaldi

import wesep.models.kws_modules as NM
from wesep.models.aed_kws_asr_phone import AEDKWSASRPhone

# model from Ying Shi, the primary auther of https://arxiv.org/abs/2310.05352
class KWTSEncoder(nn.Module):
    def __init__(self, checkpoint_dir: str, fnn: dict):
        super().__init__()

        # wav -> fbank
        self.frontend = kaldi.fbank
        self.frontend_conf = self.load_frontend(checkpoint_dir)

        # load & freeze TC-ASR model
        self.tcasr_encoder = self.load_checkpoint(checkpoint_dir)

        # adapter
        num_layer = fnn['num_layer']
        if num_layer == 1:
            self.fnn = nn.Sequential(*[
                nn.Linear(fnn['input_dim'], fnn['output_dim']),
            ])
        else:
            self.fnn = []
            for _ in range(num_layer - 1):
                self.fnn.extend([
                    nn.Linear(fnn['input_dim'], fnn['input_dim']),
                ])
            self.fnn.extend([
                nn.Linear(fnn['input_dim'], fnn['output_dim']),
            ])
            self.fnn = nn.Sequential(*self.fnn)

    def load_frontend(self, checkpoint_dir) -> dict:
        checkpoint_dir = os.path.expanduser(checkpoint_dir)
        exp_dir = os.path.dirname(checkpoint_dir)

        config_file = '{}/data.yaml'.format(exp_dir)
        data_config = yaml.load(open(config_file), Loader=yaml.FullLoader)

        return data_config['speech_config']['feats_config']

    def load_checkpoint(self, checkpoint_dir) -> nn.Module:
        checkpoint_dir = os.path.expanduser(checkpoint_dir)
        exp_dir = os.path.dirname(checkpoint_dir)

        config_file = '{}/model.yaml'.format(exp_dir)
        model_config = yaml.load(open(config_file), Loader=yaml.FullLoader)

        model = AEDKWSASRPhone(**model_config)
        state_dict = torch.load(
            checkpoint_dir,
            weights_only=True,
            map_location='cuda' if torch.cuda.is_available() else 'cpu',
        )['model']
        model.load_state_dict(state_dict)

        return model

    def forward(self, wav_input: torch.Tensor, text_input: list):
        fbank = [self.frontend(_.unsqueeze(0), **self.frontend_conf) for _ in wav_input]
        fbank_len = torch.tensor([_.shape[0] for _ in fbank])
        fbank = pad_sequence(fbank, batch_first=True)
        fbank_len = fbank_len.to(fbank.device)

        sph_emb = self.forward_tcasr((fbank, fbank_len), text_input)     # B, D

        speaker_emb = self.fnn(sph_emb)  # hopefully speaker embedding

        return speaker_emb

    @torch.no_grad()
    def forward_tcasr(self, wav_input, text_input):
        sph_input, sph_len = wav_input  # mix_enroll

        kw_label, kw_len = text_input
        b, t, d = sph_input.size()

        sph_len = NM.BaseConv.compute_dim_reduction(sph_len, 3, 2, 0, 1)
        sph_len = NM.BaseConv.compute_dim_reduction(sph_len, 3, 2, 0, 1)
        sph_mask = ~NM.make_mask(sph_len).unsqueeze(1)
        kw_mask = ~NM.make_mask(kw_len).unsqueeze(1)
        cross_mask = ~NM.combine_mask(sph_mask.squeeze(1), kw_mask.squeeze(1), 1)

        # embedding
        sph_emb = self.tcasr_encoder.au_conv(sph_input.unsqueeze(1))
        b, c, t, d = sph_emb.size()
        sph_emb = self.tcasr_encoder.au_conv_trans(sph_emb.transpose(1,2).contiguous().view(b, t, c * d))
        sph_emb = self.tcasr_encoder.speech_input_projection(sph_emb)
        kw_emb = self.tcasr_encoder.phn_emb(kw_label.to(torch.long))
        kw_emb = self.tcasr_encoder.keyword_input_projection(kw_emb)

        # add position embedding
        sph_emb = self.tcasr_encoder.speech_pe_module(sph_emb)
        kw_emb = self.tcasr_encoder.keyword_pe_module(kw_emb)

        kw_emb = self.tcasr_encoder.forward_transformer(
            self.tcasr_encoder.keyword_transformer,
            kw_emb,
            attention_mask=kw_mask,
        )
        sph_emb, (sph_att_scores, sph_emb_list) = self.tcasr_encoder.forward_transformer(
            self.tcasr_encoder.speech_transformer,
            sph_emb,
            attention_mask=sph_mask,
            cross_hidden_states=(
                kw_emb, kw_emb, cross_mask
            ),
            analyse=True,
        )

        sph_emb_list = sph_emb_list[0]  # omit repeated sph_emb, now (N, B, T, D)

        # layer pooling, followd by time-axis pooling
        if self.tcasr_encoder.use_sv:
            sph_emb = self.tcasr_encoder.sv_ce_crit.forward_pooling(sph_emb_list, sph_len)
        else:
            sph_emb = sph_emb_list[-2].mean(dim=1)      # really only `-2`?

        return sph_emb
