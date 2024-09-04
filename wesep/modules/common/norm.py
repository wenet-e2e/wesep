import numbers

import torch
import torch.nn as nn


class GlobalChannelLayerNorm(nn.Module):
    """
    Calculate Global Layer Normalization
    dim: (int or list or torch.Size) –
         input shape from an expected input of size
    eps: a value added to the denominator for numerical stability.
    elementwise_affine: a boolean value that when set to True,
        this module has learnable per-element affine parameters
        initialized to ones (for weights) and zeros (for biases).
    """

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalChannelLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.dim, 1))
            self.bias = nn.Parameter(torch.zeros(self.dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        # x = N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x L
        # gln: mean,var N x 1 x 1
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))

        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x - mean)**2, (1, 2), keepdim=True)
        # N x C x L
        if self.elementwise_affine:
            x = (self.weight * (x - mean) / torch.sqrt(var + self.eps) +
                 self.bias)
        else:
            x = (x - mean) / torch.sqrt(var + self.eps)
        return x


class ChannelWiseLayerNorm(nn.LayerNorm):
    """
    Channel wise layer normalization
    """

    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: N x C x T
        """
        x = torch.transpose(x, 1, 2)
        x = super().forward(x)
        x = torch.transpose(x, 1, 2)
        return x


def select_norm(norm, dim):
    """
    Build normalize layer
    LN cost more memory than BN
    """
    if norm not in ["cLN", "gLN", "BN"]:
        raise RuntimeError("Unsupported normalize layer: {}".format(norm))
    if norm == "cLN":
        return ChannelWiseLayerNorm(dim, elementwise_affine=True)
    elif norm == "BN":
        return nn.BatchNorm1d(dim)
    else:
        return GlobalChannelLayerNorm(dim, elementwise_affine=True)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation (FiLM) layer
    https://github.com/HuangZiliAndy/fairseq/blob/multispk/fairseq/models/wavlm/WavLM.py#L1160  # noqa
    """

    def __init__(self,
                 feat_size,
                 embed_size,
                 num_film_layers=1,
                 layer_norm=False):
        super(FiLM, self).__init__()
        self.feat_size = feat_size
        self.embed_size = embed_size
        self.num_film_layers = num_film_layers
        self.layer_norm = nn.LayerNorm(embed_size) if layer_norm else None
        gamma_fcs, beta_fcs = [], []
        for i in range(num_film_layers):
            if i == 0:
                gamma_fcs.append(nn.Linear(embed_size, feat_size))
                beta_fcs.append(nn.Linear(embed_size, feat_size))
            else:
                gamma_fcs.append(nn.Linear(feat_size, feat_size))
                beta_fcs.append(nn.Linear(feat_size, feat_size))
        self.gamma_fcs = nn.ModuleList(gamma_fcs)
        self.beta_fcs = nn.ModuleList(beta_fcs)
        self.init_weights()

    def init_weights(self):
        for i in range(self.num_film_layers):
            nn.init.zeros_(self.gamma_fcs[i].weight)
            nn.init.zeros_(self.gamma_fcs[i].bias)
            nn.init.zeros_(self.beta_fcs[i].weight)
            nn.init.zeros_(self.beta_fcs[i].bias)

    def forward(self, embed, x):
        gamma, beta = None, None
        for i in range(len(self.gamma_fcs)):
            if i == 0:
                gamma = self.gamma_fcs[i](embed)
                beta = self.beta_fcs[i](embed)
            else:
                gamma = self.gamma_fcs[i](gamma)
                beta = self.beta_fcs[i](beta)

        if len(gamma.shape) < len(x.shape):
            gamma = gamma.unsqueeze(-1).expand_as(x)
            beta = beta.unsqueeze(-1).expand_as(x)
        else:
            gamma = gamma.expand_as(x)
            beta = beta.expand_as(x)

        # print(gamma.size(), beta.size())
        x = (1 + gamma) * x + beta
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return x


class ConditionalLayerNorm(nn.Module):
    """
    https://github.com/HuangZiliAndy/fairseq/blob/multispk/fairseq/models/wavlm/WavLM.py#L1160
    """

    def __init__(self,
                 normalized_shape,
                 embed_dim,
                 modulate_bias=False,
                 eps=1e-5):
        super(ConditionalLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape, )
        self.normalized_shape = tuple(normalized_shape)

        self.embed_dim = embed_dim
        self.eps = eps

        self.weight = nn.Parameter(torch.empty(*normalized_shape))
        self.bias = nn.Parameter(torch.empty(*normalized_shape))
        assert len(normalized_shape) == 1
        self.ln_weight_modulation = FiLM(normalized_shape[0], embed_dim)
        self.modulate_bias = modulate_bias
        if self.modulate_bias:
            self.ln_bias_modulation = FiLM(normalized_shape[0], embed_dim)
        else:
            self.ln_bias_modulation = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input, embed):
        mean = torch.mean(input, -1, keepdim=True)
        var = torch.var(input, -1, unbiased=False, keepdim=True)
        weight = self.ln_weight_modulation(
            embed, self.weight.expand(embed.size(0), -1))
        if self.ln_bias_modulation is None:
            bias = self.bias
        else:
            bias = self.ln_bias_modulation(embed,
                                           self.bias.expand(embed.size(0), -1))
        res = (input - mean) / torch.sqrt(var + self.eps) * weight + bias
        return res

    def extra_repr(self):
        return "{normalized_shape}, {embed_dim}, \
            modulate_bias={modulate_bias}, eps={eps}".format(**self.__dict__)
