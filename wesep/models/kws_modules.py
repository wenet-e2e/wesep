# Most of the code are referenced from Wenet
# Thanks wenet
import math
import logging
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F

norm_dict = {
    'LayerNorm': nn.LayerNorm,
    'BatchNorm1d': nn.BatchNorm1d, 'BatchNorm2d': nn.BatchNorm2d, 'BatchNorm3d': nn.BatchNorm3d,
    'InstanceNorm1d': nn.InstanceNorm1d, 'InstanceNorm2d': nn.InstanceNorm2d, 'InstanceNorm3d': nn.InstanceNorm3d
}

act_dict = {
    'ReLU': nn.ReLU, 'LeakyReLU': nn.LeakyReLU,
    'Sigmoid': nn.Sigmoid, 'Tanh': nn.Tanh
}

conv_dict = {
    'Conv1D': nn.Conv1d, 'Conv2D': nn.Conv2d, 'Conv3D': nn.Conv3d,
    'TConv1D': nn.ConvTranspose1d, 'TConv2D': nn.ConvTranspose2d, 'TConv3D': nn.ConvTranspose3d
}

pool_dict = {
    'max2d': nn.MaxPool2d, 'admax2d': nn.AdaptiveMaxPool2d,
}

CE_IGNORE_INDEX = -100

# classes 
class CTC(nn.Module):
    def __init__(
        self,
        num_tokens,
        front_output_size,
        reduce=True,
    ):
        super(CTC, self).__init__()
        self.linear_project = nn.Linear(front_output_size, num_tokens)
        reduction_type = "sum" if reduce else "none"
        self.ctc_loss = nn.CTCLoss(reduction=reduction_type)

    def forward(self, logit, label, hyp_len, label_len, return_hyp=False):
        logit = self.linear_project(logit)
        logit = logit.transpose(0,1)
        hyp = logit.log_softmax(2) # the final dim
        loss = self.ctc_loss(hyp, label, hyp_len, label_len)
        loss = loss / hyp.size(1)
        if return_hyp:
            return loss, hyp
        else:
            return loss

    @torch.no_grad()
    def get_hyp(self, logit):
        return self.linear_project(logit)

class LabelSmoothingLoss(nn.Module):

    def __init__(self,
                 size: int,
                 padding_idx: int,
                 smoothing: float,
                 normalize_length: bool = False):
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="none")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.normalize_length = normalize_length

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert x.size(2) == self.size
        batch_size = x.size(0)
        x = x.view(-1, self.size)
        target = target.view(-1)
        # use zeros_like instead of torch.no_grad() for true_dist,
        # since no_grad() can not be exported by JIT
        true_dist = torch.zeros_like(x)
        true_dist.fill_(self.smoothing / (self.size - 1))
        ignore = target == self.padding_idx  # (B,)
        total = len(target) - ignore.sum().item()
        target = target.masked_fill(ignore, 0)  # avoid -1 index
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)
        denom = total if self.normalize_length else batch_size
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom

class CTCPIT(nn.Module):
    # /Users/shiying/Research/speechbrain/speechbrain/nnet/losses.py
    def __init__(
        self,
        num_tokens,
        front_output_size,
        reduce=False,
        PIT=True
    ):
        super(CTCPIT, self).__init__()
        self.linear_project = nn.Linear(front_output_size, num_tokens)
        reduction_type = "sum" if reduce else "none"
        self.ctc_loss = nn.CTCLoss(reduction=reduction_type)
        if PIT:
            self.linear_project2 = nn.Linear(front_output_size, num_tokens)
            self.PIT = PIT 
            self.proj = [self.linear_project, self.linear_project2]
        
    def compute_pit_permute(self, hyp_tuple, label_tuple, hyp_len, label_len_tuple):
        idx1 = [[0,0], [1,1]]
        idx2 = [[0,1], [1,0]]
        #loss = self.ctc_loss(hyp, label, hyp_len, label_len)
        for i, p in enumerate(idx1):
            l = self.ctc_loss(
                hyp_tuple[p[0]], label_tuple[p[1]], hyp_len, label_len_tuple[p[1]]
            )
            if i == 0:
                loss1 = l
            else:
                loss1 = loss1 + l

        for i, p in enumerate(idx2):
            l = self.ctc_loss(
                hyp_tuple[p[0]], label_tuple[p[1]], hyp_len, label_len_tuple[p[1]]
            )
            if i == 0:
                loss2 = l
            else:
                loss2 = loss2 + l
        mask = loss1 > loss2 
        mask = mask.to(loss1.device)
        loss1.masked_fill(mask, 0)
        loss2.masked_fill(~mask, 0)
        #print (loss1.size(), loss2.size())
        loss = loss1 + loss2
        return loss, mask
    
    def forward(self, logit, label, hyp_len, label_len, return_hyp=False):
        # 0 mean the label which contain keyword sequence
        if self.PIT:
            logit1 = self.linear_project(logit[0])
            logit2 = self.linear_project2(logit[1])
            logit1 = logit1.transpose(0,1)
            logit2 = logit2.transpose(0,1)
            hyp1 = logit1.log_softmax(2)
            hyp2 = logit2.log_softmax(2)
            hyp_tuple = (hyp1, hyp2)
            loss, mask = self.compute_pit_permute(hyp_tuple, label, hyp_len, label_len)
            loss = loss.sum() / logit1.size(1)
            
            if return_hyp:
                return loss, mask
            else:
                return loss
        else:
            logit = self.linear_project(logit)
            logit = logit.transpose(0,1)
            hyp = logit.log_softmax(2) # the final dim
            loss = self.ctc_loss(hyp, label, hyp_len, label_len)
            loss = loss / hyp.size(1)
            if return_hyp:
                return loss, hyp
            else:
                return loss

    @torch.no_grad()
    def get_hyp(self, logit):
        return self.linear_project(logit) 

class CE(nn.Module):
    def __init__(
        self,
        num_classes: int,
        front_output_size: int,
        pooling_conf: dict,
        use_reg_loss: bool = True,
        reg_loss_weight: float = 0.01,
        reduce: bool = True,
    ):
        super(CE, self).__init__()
        self.linear_project = nn.Linear(front_output_size, num_classes)
        reduction_type = "sum" if reduce else "none"
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction_type, ignore_index=CE_IGNORE_INDEX)
        
        self.pooling_type = pooling_conf['type']
        self.layer_indices = torch.tensor(pooling_conf.get('layer_indices', []), dtype=torch.long)
        self.use_softmax = pooling_conf['use_softmax']
        self.use_reg_loss = use_reg_loss
        self.reg_loss_weight = reg_loss_weight
        if self.pooling_type == 'learnable_weights':
            self.weight = nn.Parameter(
                torch.tensor([1 / len(self.layer_indices) for _ in range(len(self.layer_indices))])
            )

    def forward_pooling(self, logit_list: list, sph_len: torch.Tensor) -> torch.Tensor:
        device = logit_list[0].device

        logit = torch.stack(logit_list)
        logit = logit[self.layer_indices.to(device).long()]
        if self.pooling_type == 'mean':
            pooled_logit = logit.mean(dim=0)  # mean along layer_indices axis
        elif self.pooling_type == 'learnable_weights':
            if self.use_softmax:
                _weight = F.softmax(self.weight, dim=-1)
            else:
                _weight = self.weight
            pooled_logit = sum(w * v for (w, v) in zip(_weight, logit))
            # pooled_logit = sum(w * v for (w, v) in zip(self.weight, logit))

        pooled_logit = torch.stack([
            _[:length, :].mean(0) for (_, length) in zip(pooled_logit, sph_len)
        ])

        # pooled_logit = pooled_logit.mean(dim=1) # mean along T axis
        return pooled_logit

    def forward(self, logit_list: list, sph_len: torch.Tensor, label: torch.Tensor, is_target: torch.Tensor):
        pooled_logit = self.forward_pooling(logit_list, sph_len)
        pooled_logit = self.linear_project(pooled_logit)
        label[is_target == 0] = CE_IGNORE_INDEX      # ignore keyword out of speech, where phn is predicted as 73 
        loss = self.ce_loss(pooled_logit, label)
        loss = loss / pooled_logit.size(0)

        if self.use_reg_loss:
            loss += self.reg_loss_weight * (self.weight.norm(2) - 1) ** 2     # regularization

        return loss

class WordEmbedding(nn.Module):
    def __init__(
        self,
        num_tokens,
        dim,
        padding_idx=0
    ):
        super(WordEmbedding, self).__init__()
        self.emb = nn.Embedding(
            num_tokens,
            dim,
            padding_idx=padding_idx 
        )
    def forward(self, input):
        return self.emb(input)


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        model_dim,
        max_len=5000,
    ):
        super(PositionalEncoding, self).__init__()
        self.d_model = model_dim
        self.xcale = math.sqrt(self.d_model)
        self.max_len = max_len

        self.pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(
            0, 
            self.max_len,
            dtype=torch.float32,
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) *
            -(math.log(10000.0) / self.d_model)
        )
        self.pe[0:, 0::2] = torch.sin(position * div_term)
        self.pe[0:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def position_encoding(self, size, device):
        index = torch.arange(0, size).to(device)
        pos_emb = F.embedding(index, self.pe[0])
        return pos_emb
    
    def forward(
        self,
        input
    ):
        self.pe = self.pe.to(input.device)
        pos_emb = self.position_encoding(input.size(1), input.device)
        input = input * self.xcale + pos_emb
        return input

class MultiHeadAtt(nn.Module):
    def __init__(
        self,
        n_head,
        n_feats,
    ):
        super(MultiHeadAtt, self).__init__()
        assert n_feats % n_head == 0
        self.d_k = n_feats // n_head
        self.n_head = n_head
        self.q = nn.Linear(n_feats, n_feats)
        self.k = nn.Linear(n_feats, n_feats)
        self.v = nn.Linear(n_feats, n_feats)
        self.linear_out = nn.Linear(n_feats, n_feats)
    
    def forward_qkv(self, q, k, v):
        batch = q.size(0) 
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)
        q = q.view(batch, -1, self.n_head, self.d_k).transpose(1,2)
        k = k.view(batch, -1, self.n_head, self.d_k).transpose(1,2)
        v = v.view(batch, -1, self.n_head, self.d_k).transpose(1,2)
        return q, k, v

    def forward_selfatt(
        self, 
        q, k, v,
        mask,
        softmax=True,
    ):
        batch, nhead, tq, d = q.size()
        score = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(d)
        if mask != None:
            if len(mask.size()) < len(q.size()):
                mask = mask.unsqueeze(1).eq(0)
            else:
                mask = mask.eq(0)
        if mask != None:
            score = score.masked_fill(mask, -float('inf'))
            att_weight = torch.softmax(score, dim=-1).masked_fill(mask, 0.0)
        else:
            att_weight = torch.softmax(score, dim=-1)
        context = torch.matmul(att_weight, v)
        context = context.transpose(1,2).contiguous().view(batch, tq, -1)
        return context, \
                att_weight if softmax else score
    
    def forward(
        self, 
        q, k, v,
        mask,
        softmax=True
    ):
        q, k, v = self.forward_qkv(q, k, v)
        context, score = self.forward_selfatt(q, k, v, mask)
        return self.linear_out(context), score

class MultiHeadCrossAtt(nn.Module):
    def __init__(
        self,
        n_head,
        n_feats,
        norm=None,
    ):
        super(MultiHeadCrossAtt, self).__init__()
        self.q = nn.Sequential(norm_dict[norm](n_feats), nn.Linear(n_feats, n_feats))
        self.k = nn.Sequential(norm_dict[norm](n_feats), nn.Linear(n_feats, n_feats))
        self.v = nn.Sequential(norm_dict[norm](n_feats), nn.Linear(n_feats, n_feats))
        self.linear_out = nn.Linear(n_feats, n_feats)
        self.n_head = n_head
        self.n_feats = n_feats

    def forward_selfatt(
        self,
        q, k, v,
        mask,
        aux_score=None,
        softmax=True,
        print_mask=False
    ):
        batch, nhead, tq, d = q.size()
        score = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(d)
        if mask != None:
            if len(mask.size()) < len(q.size()):
                mask = mask.unsqueeze(1).eq(0)
            else:
                mask = mask.eq(0)
        if print_mask:
            logging.debug("score.size=%s  mask.size=%s", score.size(), mask.size())
        if aux_score != None:
            score = score * aux_score[:,None,None,:]
        if mask != None:
            score = score.masked_fill(mask, -float('inf'))
            att_weight = torch.softmax(score, dim=-1).masked_fill(mask, 0.0)
        else:
            att_weight = torch.softmax(score, dim=-1)
        context = torch.matmul(att_weight, v)
        context = context.transpose(1,2).contiguous().view(batch, tq, -1)
        return context, \
                att_weight if softmax else score
        
    def forward(
        self, 
        q, k, v,
        mask=None,
        aux_score=None,
        softmax=True,
        print_mask=False
    ):
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)
        batch = k.size(0)
        tq = q.size(1)
        tk = k.size(1)
        tv = v.size(1)
        q = q.view(batch, tq, self.n_head, -1).transpose(1,2)
        k = k.view(batch, tk, self.n_head, -1).transpose(1,2)
        v = v.view(batch, tv, self.n_head, -1).transpose(1,2)
        context, score = self.forward_selfatt(
            q,k,v,mask, aux_score=aux_score, softmax=softmax, print_mask=print_mask
        )
        return self.linear_out(context), score

class BaseConv(nn.Module):
    def __init__(
        self,
        i_channel, o_channel,
        conv=nn.Conv2d,
        kernel_size=(3,3), stride=(2,2),
        norm='BatchNorm2d', input_dim=None,
        norm_dim=None,
        act='ReLU',
        padding=0, dilation=1,
        norm_before=False,
        pooling=None
    ):
        super(BaseConv, self).__init__()
        self.conv = conv(
            i_channel, o_channel, kernel_size, stride, 
            dilation=dilation, padding=padding
        )
        if isinstance(input_dim, list):
            input_dim = tuple(input_dim)
        if isinstance(input_dim, tuple):
            self._t = self.compute_dim_reduction(input_dim, kernel_size, stride, padding, dilation, 0)
            self._d = self.compute_dim_reduction(input_dim, kernel_size, stride, padding, dilation, 1)
        elif isinstance(input_dim, int):
            self._d = self.compute_dim_reduction(input_dim, kernel_size, stride, padding, dilation, 1)
            self._t = -1
        else:
            self._d = -1, 
            self._t = -1
        if norm == 'LayerNorm':
            self.norm = norm_dict[norm]([self._t, self._d.item() if norm_dim == None else norm_dim])
        elif 'BatchNorm' in norm:
            self.norm = norm_dict[norm](o_channel)
        elif norm == 'LayerNormNonePara': 
            self.norm = norm_dict['LayerNorm']([self._t, self._d.item() if norm_dim==None else norm_dim], elementwise_affine=False)
        else:
            self.norm = nn.Identity()
        self.conv_param = { 
            'kernel': kernel_size, 'stride': stride, 'padding': padding, 'dilation': dilation
        }
        self.pooling = pool_dict[pooling]() if pooling != None else nn.Identity()
        self.act = act_dict[act]() if act != None else nn.Identity()
        self.norm_before = norm_before

    @property
    def t(self):
        return self._t.item()
    
    @property
    def d(self):
        return self._d.item()

    def get_conv_para(self):
        return self.conv_param
    
    def forward(self, input):
        input = self.conv(input)
        if self.norm_before:
            input = self.norm(input)
            input = self.act(input)
            act = input.clone()
        else:
            input = self.act(input)
            act = input.clone()
            input = self.norm(input)
        input = self.pooling(input)
        return input

    @staticmethod
    def compute_dim_reduction(
        input_dim,
        kernel, stride, padding, dilation,
        dim=0
    ):
        '''
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv2d#torch.nn.Conv2d
        org_dim: dim before conv
        dim: 0/1 width=>0 hight=>1
        p,d,k,s => padding, diliation, kernel, stride
        '''
        p = padding if isinstance(padding, int) else padding[dim]
        d = dilation if isinstance(dilation, int) else dilation[dim]
        k = kernel if isinstance(kernel, int) else kernel[dim]
        s = stride if isinstance(stride, int) else stride[dim]
        idim = input_dim if not isinstance(input_dim, tuple) else input_dim[dim]
        rdim = idim + 2 * p - d * (k-1) - 1
        rdim = torch.div(rdim, s, rounding_mode='floor') + 1
        return rdim


class DepthWiseConv(nn.Module):
    def __init__(self,
        channels,
        kernel_size = 15,
        activation = nn.ReLU(),
        norm = "batch_norm",
        causal = False,
        bias = True
):
        super().__init__()

        self.pointwise_conv1 = nn.Conv1d(
            channels, 2 * channels,
            kernel_size=1, stride=1, padding=0,
            bias=bias,
        )
        if causal: # keep this as false
            padding = 0
            self.lorder = kernel_size - 1
        else:
            # kernel_size should be an odd number for none causal convolution
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0
        self.depthwise_conv = nn.Conv1d(
            channels, channels,
            kernel_size, stride=1, padding=padding,
            groups=channels, bias=bias,
        )

        assert norm in ['batch_norm', 'layer_norm']
        if norm == "batch_norm":
            self.use_layer_norm = False
            self.norm = nn.BatchNorm1d(channels)
        else:
            self.use_layer_norm = True
            self.norm = nn.LayerNorm(channels)

        self.pointwise_conv2 = nn.Conv1d(
            channels,  channels,
            kernel_size=1, stride=1, padding=0,
            bias=bias,
        )
        self.activation = activation

    #TODO: add  analysis args to return activation values
    def forward(self, input
    ):
        # exchange the temporal dimension and the feature dimension
        input = input.transpose(1, 2)  # (#batch, channels, time)

        # GLU mechanism
        input = self.pointwise_conv1(input)  # (batch, 2*channel, dim)
        input = nn.functional.glu(input, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        input = self.depthwise_conv(input)
        if self.use_layer_norm:
            input = input.transpose(1, 2)
        input = self.activation(self.norm(input))
        if self.use_layer_norm:
            input = input.transpose(1, 2)
        input = self.pointwise_conv2(input)
        return input.transpose(1, 2)
    
class BaseFNN(nn.Module):
    def __init__(
        self,
        indim,
        dim,
        norm='id',
        norm_before=True
    ):
        super(BaseFNN, self).__init__()
        self.l = nn.Linear(indim, dim) 
        if norm == 'layernorm':
            self.norm = nn.LayerNorm(dim)
        elif norm == 'bn':
            self.norm = nn.BatchNorm1d(dim)
        else:
            self.norm = nn.Identity()
        self.norm_before = norm_before
        self.act = nn.ReLU()
    
    def forward(self, input):
        input = self.l(input)
        if self.norm_before:
            input = self.norm(input)
            input = self.act(input)
            act = input.clone()
        else:
            input = self.act(input)
            act = input.clone()
            input = self.norm(input)
        return input, act


class RNNBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_block=1,
        rnn_type='lstm',
        full_output=False,
        pack_pad=True
    ):
        super(RNNBlock, self).__init__()
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=dim,
                hidden_size=dim,
                num_layers=num_block,
                batch_first=True
            )
        self.pack_pad = pack_pad
        self.full_output = full_output

    def forward(self, input_data, input_length):
        if self.pack_pad:
            input_data = nn_utils.rnn.pack_padded_sequence(
                input_data, input_length,
                batch_first=True,
                enforce_sorted=True
            )
        self.rnn.flatten_parameters()
        packed_output, (ht, ct) = self.rnn(input_data)

        if self.full_output:
            output = nn_utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            output = (ht, ct)
        return output
        
class CNNBlock(nn.Module):
    def __init__(
        self,
        input_dim=40,
        output_channel=[128, 256, 512],
        kernel_size=[[3,3], [3,3], [3,3]],
        stride=[[1,1], [2,2], [2,1]],
    ):
        super(CNNBlock, self).__init__()
        cnn_list = []
        for x in range(len(kernel_size)):
            if x == 0:
                input_channel = 1
                output_d = input_dim
            else:
                input_channel = output_channel[x-1]

            #output_t = (output_t - (kernel_size[x][0]-1) -1) // stride[x][0] + 1
            output_d = (output_d - (kernel_size[x][1]-1) -1) // stride[x][1] + 1

            cnn_list += [
                nn.Conv2d(
                    input_channel,
                    output_channel[x],
                    kernel_size[x],
                    stride[x]
                ),
                nn.ReLU(),
                nn.LayerNorm(output_d),
            ]
        self.output_d = output_d
        self.cnn_list = nn.Sequential(*cnn_list)

    def forward(self, input):
        return self.cnn_list(input)

    def get_output_dim(self):
        return self.output_d

class FNNBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_block=1,
        bias=True,
        norm=None,
        act='ReLU'
    ):
        super(FNNBlock, self).__init__()
        if isinstance(dim, list):
            assert(len(dim)<=3)
            if len(dim) == 2:
                idim, odim = dim
                hdim = None
            else:
                idim, hdim, odim = dim
        else:
            assert(isinstance(dim, int))
            idim = hdim = odim = dim

        if norm != None:
            assert norm in norm_dict 
        if act != None:
            assert act in act_dict 

        #net = [
        #    nn.Sequential(
        #        nn.Linear(
        #            idim if i == 0 else hdim,
        #            hdim if i != num_block-1 else odim,
        #            bias=bias
        #        ),
        #        act_dict[act]() if act != None else nn.Identity(),
        #        norm_dict[norm](hdim if i != num_block-1 else odim) \
        #            if norm != None else nn.Identity() 
        #    ) for i in range(num_block)
        #] 
        #self.net = nn.Sequential(*net)
        self.w1 = nn.Linear(idim, hdim if hdim else idim)
        self.act = nn.ReLU()
        self.w2 = nn.Linear(hdim if hdim  else idim, odim)
    
    def forward(self, input):
        #return self.net(input)
        return self.w2(self.act(self.w1(input)))
    
class TransformerLayer(nn.Module):
    def __init__(
        self,
        self_att,
        feed_forward,
        macaron_layer=None,
        conv_layer=None,
        cross_att=None,
        decoder_att=None,
        size=256,
    ):
        super(TransformerLayer, self).__init__()
        self.self_att = self_att
        self.feed_forward = feed_forward
        self.macaron_layer = macaron_layer
        self.cross_att = cross_att
        self.decoder_att = decoder_att
        self.input_norm = nn.LayerNorm(size, eps=1e-5)
        self.fnn_norm = nn.LayerNorm(size, eps=1e-5)
        #@if self.cross_att != None:
        #@    self.cross_att_norm = nn.LayerNorm(size, eps=1e-5)
        #if self.decoder_att != None:
        #    self.decode_att_norm = nn.LayerNorm(size, eps=1e-5)
        
        self.size = size
        self.conv_layer = conv_layer
        if self.conv_layer != None:
            self.conv_norm = nn.LayerNorm(size, eps=1e-5)
        if self.macaron_layer != None:
            self.macaron_norm = nn.LayerNorm(size, eps=1e-5)
            self.macaron_factor = 0.5
        ###!!TODO 
    def forward(self, input, mask, cross_input=None, aux_score=None, args=None, print_mask=False):

        if self.macaron_layer != None:
            residual = input
            input = self.macaron_norm(input)
            input = residual + self.macaron_factor * self.macaron_layer(input)

        residual = input
        input = self.input_norm(input)

        context, att_score = self.self_att(
            input, input, input, mask
        )
        input = residual + context
        residual = input
        if self.cross_att != None:
            #input = self.cross_att_norm1(input)
            k, v, cross_mask = cross_input 
            #@cross_context, cross_score = self.cross_att(context, k, v, cross_mask, aux_score) 
            cross_context, cross_score = self.cross_att(input, k, v, cross_mask, aux_score) 
            input = cross_context + residual
            residual = input
            #cross_residual = context
            #input = self.cross_att_norm1(input)

        if self.decoder_att != None:
            memory, memory, memory_mask = cross_input
            decoder_context, decoder_score = self.decoder_att(input, memory, memory, memory_mask, print_mask=print_mask)
            input = decoder_context + residual
            residual = input

        if self.conv_layer != None:
            conv_out = self.conv_norm(input)
            conv_out = self.conv_layer(conv_out)
            input = conv_out + residual
            residual = input

        input = self.fnn_norm(input) 
        if args != None:
            fnn_output, act = self.feed_forward(input, **args)
        else:
            fnn_output = self.feed_forward(input)

        input = fnn_output + residual
        #if self.cross_att != None:
        #    input = self.cross_att_norm(input) + cross_context

        if args != None:
            return input, act
        elif self.cross_att != None:
            return input, cross_score
        else:
            return input, att_score
# si-sdr loss
def sisdr_loss(estimate, target, eps=1e-8):
    estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)
    target = target - torch.mean(target, dim=-1, keepdim=True)

    estimate_scale = torch.sum(target * estimate, dim=-1, keepdim=True) / (torch.sum(estimate ** 2, dim=-1, keepdim=True) + eps)
    estimate = estimate_scale * estimate

    target_energy = torch.sum(target ** 2, dim=-1) + eps
    noise_energy = torch.sum((target - estimate) ** 2, dim=-1) + eps

    sisdr = 10 * torch.log10(target_energy / noise_energy)
    return -sisdr.mean()

# functions
def make_mask(length, max_len=None):
    assert isinstance(length, torch.Tensor)
    batch_size = length.size(0)
    max_len = max_len if max_len != None else length.max().item()
    seq_range = torch.arange(
        0, max_len, dtype=torch.int64, device=length.device
    )
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len) # 2d
    seq_length_expand = length.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask # 2d mask batch x time 

def combine_mask(mask1, mask2, tidx):
    t1 = mask1.size(tidx)
    t2 = mask2.size(tidx)
    mask1 = mask1.unsqueeze(1).repeat(1,t2,1).transpose(-2,-1)
    mask2 = mask2.unsqueeze(1).repeat(1,t1,1)
    #print (mask1[1], mask2[1])
    #mask1 = mask1.unsqueeze(1).repeat(1,1,t2,1).transpose(-2,-1)
    #mask2 = mask2.unsqueeze(1).repeat(1,1,t1,1)
    cmask = mask1 & mask2
    cmask = ~cmask
    #print (cmask[1])
    #exit()
    #print (cmask)
    ##cmask = torch.matmul(mask1, mask2)
    return cmask.unsqueeze(1)

def make_mix_target(target, ratio, mute_class=None, soft=False):
    bt, nt, _ = target.size() # target batch size, num target
    br, nr = ratio.size() # ratio batch size, num ratio
    assert(bt == br)
    assert(nt == nr)
    if soft:
        target = target * ratio.unsqueeze(-1)
        target = target.sum(dim=-2)
    else:
        target = target.sum(dim=-2)
    target = torch.where(target>1, 1, target)
    if mute_class:
        target[:,mute_class] *= 0
    return target
