# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0
"""Frozen keyword-conditioned encoder used by DAE-TSE.

The architecture follows GnafiY/DAE-TSE and its KCE implementation:
https://github.com/GnafiY/DAE-TSE
"""

import logging
import math
import os
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi
import yaml
from torch.nn.utils.rnn import pad_sequence

from wesep.utils.torch_compat import register_load_state_dict_pre_hook

DEFAULT_KCE_MODEL_CONFIG = {
    "audio_net_config": {
        "num_mel_bins": 80,
        "input_trans": {
            "dim": [256, 256]
        },
        "transformer_config": {
            "size": 256,
            "self_att": "MultiHeadAtt",
            "self_att_config": {
                "n_head": 4,
                "n_feats": 256
            },
            "cross_att": "MultiHeadCrossAtt",
            "corss_att_config": {
                "n_head": 4,
                "n_feats": 256,
                "norm": "LayerNorm",
            },
            "feed_forward_config": {
                "dim": [256, 1024, 256]
            },
            "conv_config": {
                "channels": 256,
                "kernel_size": 15
            },
        },
    },
    "kw_net_config": {
        "num_phn_token": 73,
        "input_trans": {
            "dim": [256, 256]
        },
        "transformer_config": {
            "size": 256,
            "self_att": "MultiHeadAtt",
            "self_att_config": {
                "n_head": 4,
                "n_feats": 256
            },
            "feed_forward_config": {
                "dim": [256, 1024, 256]
            },
        },
    },
    "num_audio_block": 8,
    "num_kw_block": 3,
    "sv_net_config": {
        "num_classes": 251,
        "front_output_size": 256,
        "pooling_conf": {
            "type": "learnable_weights",
            "layer_indices": list(range(8)),
            "use_softmax": True,
        },
    },
}

DEFAULT_KCE_FBANK_CONFIG = {
    "num_mel_bins": 80,
    "frame_length": 25.0,
    "frame_shift": 10.0,
    "dither": 0.0,
    "sample_frequency": 16000.0,
    "window_type": "hamming",
    "use_energy": False,
}


class WordEmbedding(nn.Module):

    def __init__(self, num_tokens, dim, padding_idx=0):
        super().__init__()
        self.emb = nn.Embedding(num_tokens, dim, padding_idx=padding_idx)

    def forward(self, x):
        return self.emb(x)


class PositionalEncoding(nn.Module):

    def __init__(self, model_dim, max_len=5000):
        super().__init__()
        self.d_model = model_dim
        self.xcale = math.sqrt(model_dim)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2, dtype=torch.float32) *
            -(math.log(10000.0) / model_dim))
        pe = torch.zeros(max_len, model_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x * self.xcale + self.pe[:, :x.shape[1]]


class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, n_feats):
        super().__init__()
        if n_feats % n_head:
            raise ValueError("Attention dimension must be divisible by heads")
        self.d_k = n_feats // n_head
        self.n_head = n_head
        self.q = nn.Linear(n_feats, n_feats)
        self.k = nn.Linear(n_feats, n_feats)
        self.v = nn.Linear(n_feats, n_feats)
        self.linear_out = nn.Linear(n_feats, n_feats)

    def forward(self, query, key, value, mask=None, **kwargs):
        batch = query.shape[0]
        query = self.q(query).view(batch, -1, self.n_head,
                                   self.d_k).transpose(1, 2)
        key = self.k(key).view(batch, -1, self.n_head,
                               self.d_k).transpose(1, 2)
        value = self.v(value).view(batch, -1, self.n_head,
                                   self.d_k).transpose(1, 2)
        score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            self.d_k)
        if mask is not None:
            invalid = mask.unsqueeze(1).eq(0) if mask.ndim < 4 else mask.eq(0)
            score = score.masked_fill(invalid, -torch.inf)
        weight = torch.softmax(score, dim=-1)
        if mask is not None:
            weight = weight.masked_fill(invalid, 0.0)
        context = torch.matmul(weight, value).transpose(1, 2).contiguous()
        context = context.view(batch, query.shape[2], -1)
        return self.linear_out(context), weight


class MultiHeadCrossAttention(nn.Module):

    def __init__(self, n_head, n_feats, norm=None):
        super().__init__()
        normalizer = nn.LayerNorm if norm == "LayerNorm" else nn.Identity
        self.q = nn.Sequential(normalizer(n_feats),
                               nn.Linear(n_feats, n_feats))
        self.k = nn.Sequential(normalizer(n_feats),
                               nn.Linear(n_feats, n_feats))
        self.v = nn.Sequential(normalizer(n_feats),
                               nn.Linear(n_feats, n_feats))
        self.linear_out = nn.Linear(n_feats, n_feats)
        self.n_head = n_head
        self.n_feats = n_feats

    def forward(self, query, key, value, mask=None, aux_score=None, **kwargs):
        query, key, value = self.q(query), self.k(key), self.v(value)
        batch, query_len = query.shape[:2]
        query = query.view(batch, query_len, self.n_head, -1).transpose(1, 2)
        key = key.view(batch, key.shape[1], self.n_head, -1).transpose(1, 2)
        value = value.view(batch, value.shape[1], self.n_head,
                           -1).transpose(1, 2)
        score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            query.shape[-1])
        if aux_score is not None:
            score = score * aux_score[:, None, None, :]
        if mask is not None:
            invalid = mask.unsqueeze(1).eq(0) if mask.ndim < 4 else mask.eq(0)
            score = score.masked_fill(invalid, -torch.inf)
        weight = torch.softmax(score, dim=-1)
        if mask is not None:
            weight = weight.masked_fill(invalid, 0.0)
        context = torch.matmul(weight, value).transpose(1, 2).contiguous()
        context = context.view(batch, query_len, -1)
        return self.linear_out(context), weight


class FeedForward(nn.Module):

    def __init__(self, dim, num_block=1, bias=True, norm=None, act="ReLU"):
        super().__init__()
        if isinstance(dim, list):
            input_dim = dim[0]
            hidden_dim = dim[1] if len(dim) == 3 else dim[0]
            output_dim = dim[-1]
        else:
            input_dim = hidden_dim = output_dim = dim
        # Keep the public KCE parameterization, which always uses linear bias.
        self.w1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.w2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))


class DepthWiseConv(nn.Module):

    def __init__(self,
                 channels,
                 kernel_size=15,
                 activation=None,
                 norm="batch_norm",
                 causal=False,
                 bias=True):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, 1, bias=bias)
        padding = 0 if causal else (kernel_size - 1) // 2
        self.lorder = kernel_size - 1 if causal else 0
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            groups=channels,
            bias=bias,
        )
        self.use_layer_norm = norm == "layer_norm"
        self.norm = (nn.LayerNorm(channels)
                     if self.use_layer_norm else nn.BatchNorm1d(channels))
        self.pointwise_conv2 = nn.Conv1d(channels, channels, 1, bias=bias)
        self.activation = activation or nn.ReLU()

    def forward(self, x):
        x = F.glu(self.pointwise_conv1(x.transpose(1, 2)), dim=1)
        if self.lorder:
            x = F.pad(x, (self.lorder, 0))
        x = self.depthwise_conv(x)
        if self.use_layer_norm:
            x = self.activation(self.norm(x.transpose(1, 2))).transpose(1, 2)
        else:
            x = self.activation(self.norm(x))
        return self.pointwise_conv2(x).transpose(1, 2)


class TransformerLayer(nn.Module):

    def __init__(self,
                 self_att,
                 feed_forward,
                 macaron_layer=None,
                 conv_layer=None,
                 cross_att=None,
                 decoder_att=None,
                 size=256):
        super().__init__()
        self.self_att = self_att
        self.feed_forward = feed_forward
        self.macaron_layer = macaron_layer
        self.cross_att = cross_att
        self.decoder_att = decoder_att
        self.input_norm = nn.LayerNorm(size, eps=1e-5)
        self.fnn_norm = nn.LayerNorm(size, eps=1e-5)
        self.size = size
        self.conv_layer = conv_layer
        if conv_layer is not None:
            self.conv_norm = nn.LayerNorm(size, eps=1e-5)
        if macaron_layer is not None:
            self.macaron_norm = nn.LayerNorm(size, eps=1e-5)
            self.macaron_factor = 0.5

    def forward(self, x, mask, cross_input=None, **kwargs):
        if self.macaron_layer is not None:
            x = x + self.macaron_factor * self.macaron_layer(
                self.macaron_norm(x))
        residual = x
        context, attention = self.self_att(self.input_norm(x),
                                           self.input_norm(x),
                                           self.input_norm(x), mask)
        x = residual + context
        if self.cross_att is not None:
            key, value, cross_mask = cross_input
            context, attention = self.cross_att(x, key, value, cross_mask)
            x = x + context
        if self.conv_layer is not None:
            x = x + self.conv_layer(self.conv_norm(x))
        return x + self.feed_forward(self.fnn_norm(x)), attention


class CTC(nn.Module):

    def __init__(self, num_tokens, front_output_size, reduce=True):
        super().__init__()
        self.linear_project = nn.Linear(front_output_size, num_tokens)
        self.ctc_loss = nn.CTCLoss(reduction="sum" if reduce else "none")


class SpeakerPooling(nn.Module):

    def __init__(self,
                 num_classes,
                 front_output_size,
                 pooling_conf,
                 use_reg_loss=True,
                 reg_loss_weight=0.01,
                 reduce=True):
        super().__init__()
        self.linear_project = nn.Linear(front_output_size, num_classes)
        self.ce_loss = nn.CrossEntropyLoss(
            reduction="sum" if reduce else "none", ignore_index=-100)
        self.pooling_type = pooling_conf["type"]
        self.layer_indices = torch.tensor(pooling_conf.get(
            "layer_indices", []),
                                          dtype=torch.long)
        self.use_softmax = pooling_conf["use_softmax"]
        self.use_reg_loss = use_reg_loss
        self.reg_loss_weight = reg_loss_weight
        if self.pooling_type == "learnable_weights":
            count = len(self.layer_indices)
            self.weight = nn.Parameter(torch.full((count, ), 1.0 / count))

    def forward_pooling(self, hidden_states, lengths):
        states = torch.stack(hidden_states)[self.layer_indices.to(
            hidden_states[0].device)]
        if self.pooling_type == "mean":
            states = states.mean(dim=0)
        else:
            weight = F.softmax(self.weight, dim=-1) if self.use_softmax \
                else self.weight
            states = sum(w * state for w, state in zip(weight, states))
        return torch.stack(
            [state[:length].mean(0) for state, length in zip(states, lengths)])


def make_mask(lengths, max_len=None):
    max_len = int(max_len or lengths.max().item())
    positions = torch.arange(max_len, device=lengths.device)
    return positions.unsqueeze(0) >= lengths.unsqueeze(1)


def combine_mask(mask1, mask2):
    return (~(mask1.unsqueeze(2) & mask2.unsqueeze(1))).unsqueeze(1)


class KCEBackbone(nn.Module):
    """Checkpoint-compatible inference subset of AEDKWSASRPhone."""

    def __init__(self,
                 audio_net_config,
                 kw_net_config,
                 num_audio_block=8,
                 num_kw_block=3,
                 loss_weight=None,
                 sv_net_config=None,
                 **kwargs):
        super().__init__()
        audio_transformer = audio_net_config["transformer_config"]
        keyword_transformer = kw_net_config["transformer_config"]
        audio_dim = audio_transformer["size"]
        keyword_dim = keyword_transformer["size"]

        self.au_conv = nn.Sequential(
            nn.Conv2d(1, audio_dim, 3, 2),
            nn.ReLU(),
            nn.Conv2d(audio_dim, audio_dim, 3, 2),
            nn.ReLU(),
        )
        self.num_mel_bins = audio_net_config.get("num_mel_bins", 80)
        reduced_mels = ((self.num_mel_bins - 1) // 2 - 1) // 2
        self.au_conv_trans = nn.Linear(audio_dim * reduced_mels, audio_dim)
        self.speech_input_projection = FeedForward(
            **audio_net_config["input_trans"])
        self.speech_pe_module = PositionalEncoding(audio_dim)
        self.speech_transformer = nn.ModuleList([
            TransformerLayer(
                size=audio_dim,
                self_att=MultiHeadAttention(
                    **audio_transformer["self_att_config"]),
                cross_att=MultiHeadCrossAttention(
                    **audio_transformer["corss_att_config"]),
                feed_forward=FeedForward(
                    **audio_transformer["feed_forward_config"]),
                macaron_layer=FeedForward(
                    **audio_transformer["feed_forward_config"]),
                conv_layer=DepthWiseConv(**audio_transformer["conv_config"]),
            ) for _ in range(num_audio_block)
        ])

        self.phn_emb = WordEmbedding(kw_net_config["num_phn_token"],
                                     keyword_dim)
        self.keyword_pe_module = PositionalEncoding(keyword_dim)
        self.keyword_input_projection = FeedForward(
            **kw_net_config["input_trans"])
        self.keyword_transformer = nn.ModuleList([
            TransformerLayer(
                size=keyword_dim,
                self_att=MultiHeadAttention(
                    **keyword_transformer["self_att_config"]),
                feed_forward=FeedForward(
                    **keyword_transformer["feed_forward_config"]),
            ) for _ in range(num_kw_block)
        ])
        self.kw_au_link = (nn.Linear(keyword_dim, audio_dim)
                           if keyword_dim != audio_dim else nn.Identity())
        self.loss_weight = loss_weight
        self.asr_phn_criterion = CTC(kw_net_config["num_phn_token"], audio_dim)
        self.use_sv = sv_net_config is not None
        if self.use_sv:
            self.sv_ce_crit = SpeakerPooling(**sv_net_config)

    @staticmethod
    def forward_transformer(layers,
                            hidden,
                            mask=None,
                            cross=None,
                            collect=False):
        states = []
        for layer in layers:
            hidden, _ = layer(hidden, mask, cross_input=cross)
            if collect:
                states.append(hidden)
        return (hidden, states) if collect else hidden

    def forward(self, speech, speech_lengths, keyword, keyword_lengths):
        speech_lengths = torch.div(
            speech_lengths - 3, 2, rounding_mode="floor") + 1
        speech_lengths = torch.div(
            speech_lengths - 3, 2, rounding_mode="floor") + 1
        speech_mask = ~make_mask(speech_lengths).unsqueeze(1)
        keyword_mask = ~make_mask(keyword_lengths).unsqueeze(1)
        cross_mask = ~combine_mask(speech_mask.squeeze(1),
                                   keyword_mask.squeeze(1))

        speech = self.au_conv(speech.unsqueeze(1))
        batch, channels, frames, bins = speech.shape
        speech = speech.transpose(1,
                                  2).contiguous().view(batch, frames,
                                                       channels * bins)
        speech = self.speech_pe_module(
            self.speech_input_projection(self.au_conv_trans(speech)))
        keyword = self.keyword_pe_module(
            self.keyword_input_projection(self.phn_emb(keyword.long())))
        keyword = self.forward_transformer(self.keyword_transformer, keyword,
                                           keyword_mask)
        keyword = self.kw_au_link(keyword)
        speech, states = self.forward_transformer(
            self.speech_transformer,
            speech,
            speech_mask,
            cross=(keyword, keyword, cross_mask),
            collect=True,
        )
        if self.use_sv:
            return self.sv_ce_crit.forward_pooling(states, speech_lengths)
        return torch.stack([
            value[:length].mean(0)
            for value, length in zip(speech, speech_lengths)
        ])


class KCE(nn.Module):
    """Encode mixture waveforms and phoneme IDs into adapted DAE cues."""

    def __init__(self, config=None, defer_pretrained=False):
        super().__init__()
        config = config or {}
        self.pretrained = config.get("pretrained")
        self.pretrained_loaded = False
        self.fallback_reported = False
        model_config = config.get("model_config")
        fbank_config = config.get("fbank")
        eps = config.get(
            "eps",
            torch.finfo(torch.float32).eps)  # Add eps to avoid NaN in fbank

        # Read available sidecars even when loading the external weights later.
        if self.pretrained:
            directory = os.path.dirname(os.path.expanduser(self.pretrained))
            if model_config is None:
                model_path = os.path.join(directory, "model.yaml")
                if os.path.isfile(model_path):
                    with open(model_path, encoding="utf-8") as stream:
                        model_config = yaml.safe_load(stream)
            if fbank_config is None:
                data_path = os.path.join(directory, "data.yaml")
                if os.path.isfile(data_path):
                    with open(data_path, encoding="utf-8") as stream:
                        data_config = yaml.safe_load(stream)
                    fbank_config = data_config["speech_config"]["feats_config"]

        model_config = model_config or deepcopy(DEFAULT_KCE_MODEL_CONFIG)
        self.fbank_config = fbank_config or deepcopy(DEFAULT_KCE_FBANK_CONFIG)
        backbone_dim = model_config["audio_net_config"]["transformer_config"][
            "size"]
        self.model = KCEBackbone(**model_config)
        self.model.requires_grad_(False)

        adapter = config.get("adapter", {})
        input_dim = adapter.get("input_dim", backbone_dim)
        self.output_dim = adapter.get("output_dim", 128)
        """
            key fix: add layer norm after linear layer to stabilize training
            During official training, norm of projected embedding stays stable; yet abnormal training
            appears frequently during migration.
        """
        hidden_layers = [
            layer for _ in range(adapter.get("num_layers", 1) - 1)
            for layer in (nn.Linear(input_dim, input_dim),
                          nn.LayerNorm(input_dim, eps=eps))
        ]
        self.adapter = nn.Sequential(
            *hidden_layers,
            nn.Linear(input_dim, self.output_dim),
            nn.LayerNorm(self.output_dim, eps=eps),
        )

        self.padding_id = config.get("padding_id", 0)
        self.bos_id = config.get("bos_id", 71)
        self.eos_id = config.get("eos_id", 72)
        if self.pretrained and not defer_pretrained:
            self.load_pretrained()
        register_load_state_dict_pre_hook(self, self._prepare_pretrained_state)
        self.model.eval()

    def load_pretrained(self):
        """Load the external KCE backbone checkpoint at most once."""
        if self.pretrained_loaded:
            return
        if not self.pretrained:
            raise RuntimeError(
                "KCE is absent from the TSE checkpoint and no external "
                "dae_kce.pretrained path is configured.")
        checkpoint = torch.load(
            os.path.expanduser(self.pretrained),
            map_location="cpu",
            weights_only=True,
        )
        self.model.load_state_dict(checkpoint.get("model", checkpoint))
        self.pretrained_loaded = True

    def _prepare_pretrained_state(
        self,
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Use external weights when the TSE checkpoint omits the backbone."""
        model_prefix = prefix + "model."
        if any(key.startswith(model_prefix) for key in state_dict):
            return
        self.load_pretrained()
        state_dict.update({
            model_prefix + key: value
            for key, value in self.model.state_dict().items()
        })
        if not self.fallback_reported:
            logging.warning(
                "KCE is absent from the TSE checkpoint; loaded %s.",
                self.pretrained,
            )
            self.fallback_reported = True

    def train(self, mode=True):
        """Train the adapter while keeping the KCE backbone frozen."""
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, mix, textual_aux):
        """Return adapted cues [B,D] from mix [B,T] and IDs [B,L]."""
        keywords = []
        for row in textual_aux:
            valid = row[row.ne(self.padding_id)].long()
            boundary = row.new_tensor([self.bos_id, self.eos_id],
                                      dtype=torch.long)
            keywords.append(torch.cat([boundary[:1], valid, boundary[1:]]))
        keyword_lengths = torch.tensor([value.numel() for value in keywords],
                                       device=mix.device)
        keywords = pad_sequence(keywords,
                                batch_first=True,
                                padding_value=self.padding_id)

        fbanks = [
            kaldi.fbank(wav.unsqueeze(0), **self.fbank_config)
            for wav in mix.float()
        ]
        speech_lengths = torch.tensor([value.shape[0] for value in fbanks],
                                      device=mix.device)
        fbanks = pad_sequence(fbanks, batch_first=True)
        with torch.no_grad():
            cue = self.model(fbanks, speech_lengths, keywords, keyword_lengths)
        return self.adapter(cue)
