import torch
import torch.nn as nn
import wesep.models.kws_modules as NM

att_dict = {
    'MultiHeadCrossAtt': NM.MultiHeadCrossAtt,
    'MultiHeadAtt': NM.MultiHeadAtt,
}

class AEDKWSASRPhone(nn.Module):
    def __init__(
        self,
        audio_net_config: dict,
        kw_net_config: dict,
        num_audio_block: int = 8,
        num_kw_block: int = 3,
        loss_weight: dict = None,
        sv_net_config: dict = None,
        **kwargs,
    ):
        super(AEDKWSASRPhone, self).__init__()

        # audio config
        au_input_trans_config = audio_net_config['input_trans']
        au_transformer_config = audio_net_config['transformer_config']
        au_self_att = att_dict[au_transformer_config['self_att']]
        au_self_att_cofing = au_transformer_config['self_att_config']
        au_cross_att = att_dict[au_transformer_config['cross_att']]
        au_cross_att_config = au_transformer_config['corss_att_config']
        au_feed_forward_config = au_transformer_config['feed_forward_config']
        au_hidden_dim = au_transformer_config['size']
        au_conv_config = au_transformer_config['conv_config']

        # vocab config
        kw_input_trans_config = kw_net_config['input_trans']
        num_phn_token = kw_net_config['num_phn_token']
        kw_transformer_config = kw_net_config['transformer_config']
        kw_self_att = att_dict[kw_transformer_config['self_att']]
        kw_self_att_cofing = kw_transformer_config['self_att_config']
        kw_feed_forward_config = kw_transformer_config['feed_forward_config']
        kw_hidden_dim = kw_transformer_config['size']

        # audio net
        self.au_conv = nn.Sequential(
            nn.Conv2d(1, au_hidden_dim, 3, 2),
            nn.ReLU(),
            nn.Conv2d(au_hidden_dim, au_hidden_dim, 3, 2),
            nn.ReLU(),
        )
        self.num_mel_bins = audio_net_config.get('num_mel_bins', 80)
        self.au_conv_trans = nn.Linear(
            au_hidden_dim * (((self.num_mel_bins - 1) // 2 - 1) // 2), au_hidden_dim)

        self.speech_input_projection = NM.FNNBlock(**au_input_trans_config)
        self.speech_pe_module = NM.PositionalEncoding(au_hidden_dim)
        self.speech_transformer = nn.ModuleList([
            NM.TransformerLayer(
                size=au_hidden_dim,
                self_att=au_self_att(**au_self_att_cofing),
                cross_att=au_cross_att(**au_cross_att_config),
                feed_forward=NM.FNNBlock(**au_feed_forward_config),
                macaron_layer=NM.FNNBlock(**au_feed_forward_config),
                conv_layer=NM.DepthWiseConv(**au_conv_config),
            ) for _ in range(num_audio_block)
        ])

        # kw net
        self.phn_emb = NM.WordEmbedding(
            num_tokens=num_phn_token, dim=kw_transformer_config['size'])
        self.keyword_pe_module = NM.PositionalEncoding(kw_hidden_dim)
        self.keyword_input_projection = NM.FNNBlock(**kw_input_trans_config)
        self.keyword_transformer = nn.ModuleList([
            NM.TransformerLayer(
                size=kw_hidden_dim,
                self_att=kw_self_att(**kw_self_att_cofing),
                feed_forward=NM.FNNBlock(**kw_feed_forward_config),
            ) for _ in range(num_kw_block)
        ])
        if kw_hidden_dim != au_hidden_dim:
            self.kw_au_link = nn.Linear(kw_hidden_dim, au_hidden_dim)
        else:
            self.kw_au_link = nn.Identity()

        self.loss_weight = loss_weight

        phn_ctc_conf = {
            'num_tokens': num_phn_token,
            'front_output_size': au_hidden_dim,
        }
        self.asr_phn_criterion = NM.CTC(**phn_ctc_conf)

        self.use_sv = False
        if sv_net_config:
            self.sv_ce_crit = NM.CE(**sv_net_config)
            self.use_sv = True

    def forward_transformer(self, transformer_module, hidden_states,
                            attention_mask=None, cross_hidden_states=None,
                            analyse=False, print_mask=False):
        if analyse:
            batch_size = hidden_states.size(0)
            attentions = {i: [] for i in range(batch_size)}
            all_hidden_states = {i: [] for i in range(batch_size)}

        for _, transformer_layer in enumerate(transformer_module):
            hidden_states, attention = transformer_layer(
                hidden_states, attention_mask,
                cross_input=cross_hidden_states, print_mask=print_mask,
            )
            if not analyse:
                continue
            for batch_idx, att in enumerate(attention):
                attentions[batch_idx].append(att)
                all_hidden_states[batch_idx].append(hidden_states)

        if analyse:
            return hidden_states, (attentions, all_hidden_states)
        return hidden_states

    def forward(self, input_data):
        sph_input, sph_len, phn_label, phn_len, kw_label, kw_len, _, _, target, sv_target = input_data

        sph_len = NM.BaseConv.compute_dim_reduction(sph_len, 3, 2, 0, 1)
        sph_len = NM.BaseConv.compute_dim_reduction(sph_len, 3, 2, 0, 1)
        sph_mask = ~NM.make_mask(sph_len).unsqueeze(1)
        kw_mask = ~NM.make_mask(kw_len).unsqueeze(1)
        cross_mask = ~NM.combine_mask(sph_mask.squeeze(1), kw_mask.squeeze(1), 1)

        speech_feature = self.au_conv(sph_input.unsqueeze(1))
        b, c, t, d = speech_feature.size()
        speech_feature = self.au_conv_trans(
            speech_feature.transpose(1, 2).contiguous().view(b, t, c * d))
        speech_feature = self.speech_input_projection(speech_feature)
        keyword_feature = self.phn_emb(kw_label.to(torch.long))
        keyword_feature = self.keyword_input_projection(keyword_feature)

        speech_feature = self.speech_pe_module(speech_feature)
        keyword_feature = self.keyword_pe_module(keyword_feature)

        target = target.squeeze()

        keyword_feature = self.forward_transformer(
            self.keyword_transformer, keyword_feature, attention_mask=kw_mask)

        speech_feature, (att_scores, sph_emb_list) = self.forward_transformer(
            self.speech_transformer, speech_feature,
            attention_mask=sph_mask,
            cross_hidden_states=(keyword_feature, keyword_feature, cross_mask),
            analyse=True)

        phn_ctc_loss, phn_asr_hyp = self.asr_phn_criterion(
            speech_feature, phn_label, sph_len, phn_len, return_hyp=True)

        total_loss = phn_ctc_loss * self.loss_weight['phn_ctc']
        detail_loss = {'phn_ctc_loss': phn_ctc_loss.clone().detach()}

        if self.use_sv:
            sph_emb_list = sph_emb_list[0]
            sv_ce_loss = self.sv_ce_crit(sph_emb_list, sph_len, sv_target, target)
            total_loss += sv_ce_loss * self.loss_weight['sv_ce']
            detail_loss['sv_ce_loss'] = sv_ce_loss

        return total_loss, detail_loss

    @torch.no_grad()
    def evaluate(self, input_data, return_embedding=False):
        sph_input, sph_len, phn_label, phn_len, kw_label, kw_len, kws_target = input_data

        sph_len = NM.BaseConv.compute_dim_reduction(sph_len, 3, 2, 0, 1)
        sph_len = NM.BaseConv.compute_dim_reduction(sph_len, 3, 2, 0, 1)
        b, t, d = sph_input.size()
        sph_mask = ~NM.make_mask(sph_len).unsqueeze(1)
        kw_mask = ~NM.make_mask(kw_len).unsqueeze(1)
        cross_mask = ~NM.combine_mask(sph_mask.squeeze(1), kw_mask.squeeze(1), 1)

        speech_feature = self.au_conv(sph_input.unsqueeze(1))
        b, c, t, d = speech_feature.size()
        speech_feature = self.au_conv_trans(
            speech_feature.transpose(1, 2).contiguous().view(b, t, c * d))
        speech_feature = self.speech_input_projection(speech_feature)
        keyword_feature = self.phn_emb(kw_label.to(torch.long))
        keyword_feature = self.keyword_input_projection(keyword_feature)

        speech_feature = self.speech_pe_module(speech_feature)
        keyword_feature = self.keyword_pe_module(keyword_feature)

        keyword_feature = self.forward_transformer(
            self.keyword_transformer, keyword_feature, attention_mask=kw_mask)

        speech_feature, (attention_scores, sph_emb_list) = self.forward_transformer(
            self.speech_transformer, speech_feature,
            attention_mask=sph_mask,
            cross_hidden_states=(keyword_feature, keyword_feature, cross_mask),
            analyse=True)

        additional_stats = {
            'keyword_label': kw_label,
            'keyword_len': kw_len,
            'speech_len': sph_len,
            'keyword_target': kws_target,
            'attention_maps': attention_scores,
        }

        # Extract speaker embeddings from per-layer hidden states
        if return_embedding:
            raw_emb_list = sph_emb_list[0]  # List of 9 tensors, each (B, T, 256)

            # Per-layer time-pooled embeddings: 9 × (B, 256)
            per_layer_emb = []
            for layer_hidden in raw_emb_list:
                # layer_hidden: (B, T, 256)
                pooled = torch.stack([
                    layer_hidden[i, :sph_len[i], :].mean(0)
                    for i in range(b)
                ])  # (B, 256)
                per_layer_emb.append(pooled)
            additional_stats['per_layer_emb'] = per_layer_emb

            # SV-pooled speaker embedding (learnable_weights + time mean)
            if self.use_sv:
                speaker_emb = self.sv_ce_crit.forward_pooling(raw_emb_list, sph_len)
            else:
                # Fallback: mean-pool last layer over time
                speaker_emb = per_layer_emb[-1]
            additional_stats['speaker_emb'] = speaker_emb

        ctc_alis = self.asr_phn_criterion.get_hyp(speech_feature)
        ctc_alis = ctc_alis.argmax(dim=-1)

        gts, alis = [], []
        for i in range(b):
            alis.append(ctc_alis[i][:sph_len[i]].tolist())
            gts.append(phn_label[i][:phn_len[i]].tolist())

        return gts, alis, kws_target.tolist(), additional_stats
