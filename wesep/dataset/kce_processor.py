# ref: wenet processor.py
import re
import random
import torch
import torchaudio
import json
import copy
import os
import librosa

import wesep.utils.kaldi_io as kaldi_io
import torchaudio.compliance.kaldi as kaldi
import numpy as np

from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from scipy.io import wavfile
from scipy import signal
from typing import Generator

# spectrugram
def spectrum(wav, n_fft=512, hop_length=160):
    extractor = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80)
    spec = extractor(wav)
    spec = spec.squeeze(0)
    return spec.transpose(0,1).contiguous()  

def mel_padding_wav(waveform, window_size=400, window_shift=160):
    num_samples = waveform.size(0)
    reversed_waveform = torch.flip(waveform, [0])
    m = (num_samples + (window_shift // 2)) // window_shift
    pad = window_size // 2 - window_shift // 2
    pad_right = reversed_waveform[:pad]
    if pad > 0:
        # torch.nn.functional.pad returns [2,1,0,1,2] for 'reflect'
        # but we want [2, 1, 0, 0, 1, 2]
        pad_left = reversed_waveform[-pad:]
        waveform = torch.cat((pad_left, waveform, pad_right), dim=0)
    else:
        # pad is negative so we want to trim the waveform at the front
        waveform = torch.cat((waveform[-pad:], pad_right), dim=0)
    return waveform.view(1,-1)

def get_mel_scale(n_mels=80, sample_rate=16000, f_min=0, f_max=None, n_stft=201, norm=None, mel_scale='htk'):
    f_max = sample_rate // 2
    filter_bank = torchaudio.functional.melscale_fbanks(
        n_stft, f_min, f_max, n_mels, sample_rate, norm, mel_scale
    )
    return filter_bank

# def mel_spectrum(
#         waveform, 
#         win_length=400, 
#         hop_length=160, 
#         n_fft=400, 
#         win_fn='hamm',
#         pad_mode='reflect',
#         pow=2,
#         center=False, onesided=True, 
#     ):
#     waveform = mel_padding_wav(waveform.view(-1), win_length, hop_length)
#     win_fn = torch.hamming_window
#     window = win_fn(win_length)
#     spec_f = torch.stft(
#         input=waveform,
#         n_fft=n_fft,
#         hop_length=hop_length,
#         win_length=win_length,
#         window=window,
#         center=center,
#         pad_mode=pad_mode,
#         normalized=False,
#         onesided=onesided,
#         return_complex=True,
#     )
#     filter_bank = get_mel_scale(n_mels=80)
#     spec_f = spec_f.abs().pow(pow)
#     mel_spec = torch.matmul(spec_f.transpose(-1,-2), filter_bank).transpose(-1,-2)
#     return mel_spec.squeeze(0).transpose(0,1)

def mel_spectrum(
        audio: torch.Tensor,
        sr=16000,
        n_fft=1024,
        hop_size=512,
        f_max=8000,
        n_mels=128,
        pad_mode='constant',
        mel_scale='slaney',
        norm='slaney',
        log_mel=False,
        **kwargs,
    ):
        mel_processor = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_size,
            n_mels=n_mels,
            f_max=f_max,
            pad_mode=pad_mode,
            mel_scale=mel_scale,
            norm=norm,
            **kwargs,
        )

        mel = mel_processor(audio).squeeze(0).transpose(0, 1)
        if log_mel:
            mel = librosa.power_to_db(mel, ref=np.max)  # alignment with original librosa implement
        return torch.from_numpy(mel)

# Pre-Defined None-Tensor Key & CTC Tag
NONE_TENSOR_KEY = [
    'wav', 'key', 'sph', 'corruption_material', 'segment', 'segment_idx',
    'n_scorrupt', 'n_ncorrupt', 'num_corrupt', 'rirs', 'neg_candidate',
    'corrupt', 'speech_interference_candidates', 'noise_interference_candidates', 'nframes',
    'predefined_keyword_info', 'text', 'match_case', 'phn_label_list', # 'word_boundary'
]
CTC_KEY = [
    'label', 'crpt_label',
    'phn_label', 'bpe_label',     # CTC label
    'c_phn_label', 'c_bpe_label', # corruption label
] # to be extended


# Pre-Defined Special Token
TEXT_SPEC_TOKEN = {
    'sos': None, 'eos': None, 'sok': None, 'eok': None, 'unk': None, 'with_trans': None,
    'psok': None, 'peok': None, 'punk': None
}

# input loader and feats extractor mapping
INPUT_DATA_LOADER = {
    'raw': torchaudio.load, 'kaldi': kaldi_io.read_mat, 'torch': torch.load,
    'rm_sr': lambda x: x[0], 'copy': copy.deepcopy, 'empty': lambda x: x
}

# mfcc fbank spectrum factory
FEATS_EXTRACTOR = {
    'mfcc': kaldi.mfcc, 'fbank': kaldi.fbank, 
    #'spectrum': torchaudio.functional.spectrogram, 'empty': lambda x: x 
    'spectrum': spectrum, 'mel_spec': mel_spectrum
}

# some defualt setting for feats extractor:
# mfcc, fbank, spectrum
MFCC_DEFAULT_SETTING = {
    'num_mel_bins': 23, 'num_ceps': 13, 'frame_length': 25, 'frame_shift': 10,
    'energy_floor': 0.0, 'low_freq': 20
}
FBANK_DEFAULT_SETTING = {
    'num_mel_bins': 40, 'frame_length': 25, 'frame_shift': 10
}
SPECTRUM_DEFAULT_SETTING = {
    'window': torch.hann_window(400), 'normalized': False, 'pad': 0 # the parameter in window is win_length
}

# random factory
RANDOM_FACTOR = {
    'beta':np.random.beta, 'uniform': np.random.uniform, 'int': np.random.randint, 'random': np.random.random
}

# transform np arrary as python list
TRANSFORM_FACTOR ={
    'nptolist': lambda x: x.tolist() if isinstance(x, np.ndarray) else x
}

# DITHER_RANGE used to trim wav 10 means the speech feats are in frame(1s=>100 frame) level; 
# 1600 means feats are waveforme level(sample point 1s=>16000 samples)
DITHER_RANGE = [10, 1600]

# Recompile pattern
RE_PATTERN = {'space': re.compile(r" +"), 'dot': re.compile(r"\.")}

# read data list
def read_list(list_file):
    d_list = []
    with open(list_file, encoding='utf-8') as lf:
        for line in lf.readlines():
            d_list.append(line.strip())
    lf.close()
    return d_list


# split str "0.1 0.2 0.3" to float [0.1 0.2 0.3] 
# str to int; int to sym; tensor to str
def sym2float(sym_list):
    int_list = list(map(lambda x: float(x), sym_list.split(" ")))
    return int_list
def int2sym(int_list):
    if not isinstance(int_list, list):
        int_list = [int_list]
    int_list = [str(x) for x in int_list]
    return int_list
def sym2int(sym_list):
    int_list = list(map(lambda x: int(x), sym_list.split(" ")))
    return int_list
def tensor2str(t):
    if isinstance(t, torch.Tensor):
        t = t.numpy()
    t = list(t)
    t = list(map(lambda x: str(x), t))
    return t

# save wav as PCM_S 16bit 16k: always use to test code
def save_wav(wav, names):
    if isinstance(names, list):
        names = "_".join(names)
    torchaudio.save(
        "{}.wav".format(names), wav, sample_rate=16000, encoding="PCM_S", bits_per_sample=16
    )

# Splice feats: append context 
def splice_feats(
        feats: list[torch.Tensor],
        left_context: int,
        right_context: int,
        seq=False,
    ):
    frames, nmel = feats.size()
    l_padding = torch.ones_like(torch.rand(left_context, nmel))
    r_padding = torch.ones_like(torch.rand(right_context, nmel))
    l_padding *= feats[0]
    r_padding *= feats[-1]
    feats = torch.cat([l_padding, feats, r_padding], dim=0)
    if seq:
        return feats
    else:
        splice_v = []
        for i in range(left_context + right_context + 1):
            v = feats[i:frames + i]
            splice_v.append(v)
        feats = torch.cat([v for v in splice_v], dim=-1)
        return feats

# max_energy: find the max energy frames: TODO: partially duplicated with time shifting 
def max_energy(wav, frame_length=400, hop_length=160, wav_shift_type='mid'):
    wav = wav.view(-1) # assume wav is singal channel speech [1, num_samples] multi channel is not supportTODO:
    num_frames = 1 + (len(wav) - frame_length) // hop_length
    energy = torch.zeros(num_frames, dtype=torch.float32)
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = wav[start:end]
        energy[i] = torch.sum(frame ** 2)
    # max frame energy
    idx_frame = torch.argmax(energy) 
    dice_frame = random.randint(0,5)
    if dice_frame % 2 == 0:
        idx_frame = idx_frame - dice_frame
        idx_frame = idx_frame if idx_frame > 0 else 0
    else:
        idx_frame = idx_frame + dice_frame
    #idx_frame = idx_frame - dice_frame
    idx_sample = idx_frame * hop_length + frame_length 
    return idx_sample

# flatten list [[1,2,3],[4,5,6,[7]]] => [1,2,3,4,5,6,7]
def unfold_list(lst):
    if not isinstance(lst, list):
        lst = [lst]
    l = _unfold_list(lst)
    trans = int if re.search(RE_PATTERN['dot'], l) == None else float
    l = [trans(i) for i in l.split(" ") if i !=""]
    return l

# sub method of unfold_list 
def _unfold_list(lst):
    new = ""
    for x in lst:
        if isinstance(x, list):
            x = _unfold_list(x)
        x = str(x)
        new = new + x + " "
    return new

# random one sample from a pool
def random_one(pools, pool_len):
    return (pools[random.randint(0, pool_len-1)])

# spec augmentation
def spec_augment(spec, num_t_mask=2, num_f_mask=2, max_t=20, max_f=10):
    assert isinstance(spec, torch.Tensor)
    aug_spec = spec.clone().detach()
    max_frames = aug_spec.size(0)
    max_freq = aug_spec.size(1)
    # time mask
    for i in range(num_t_mask):
        start = np.random.randint(0, max_frames - 1)
        length = np.random.randint(1, max_t)
        end = min(max_frames, start + length)
        aug_spec[start:end, :] = 0
    # freq mask
    for i in range(num_f_mask):
        start = np.random.randint(0, max_freq - 1)
        length = np.random.randint(1, max_f)
        end = min(max_freq, start + length)
        aug_spec[:, start:end] = 0
    return aug_spec

# Speech augmentation: reverb, add noise, change speed
def wav_augment(waveform, config, rirs=None):
    # add noise reverb change speech NOTE: noise here means none humanc speech voice!!!!
    # TODO: if change speech alignment and segment should change too!!!
    if config.get("volume", False):
        volume_sampler = config['volume']['sampler']
        volume_sampler_config = config['volume']['config']
        ratio = RANDOM_FACTOR[volume_sampler](**volume_sampler_config)
        waveform = ratio * waveform

    if config.get("white_niose", False):
        noise_config = config['white_noise']['config']
        gua = torch.normal(**noise_config, size=waveform.size())
        waveform = waveform + gua
    
    if rirs:
        rirs_prob = config.get('rirs_prob', 0.4)
        if random.uniform(0,1) < rirs_prob:
            _, rirs_src = wavfile.read(rirs)
            rirs_src = rirs_src / np.sqrt(np.sum(rirs_src**2))
            rirs_src = rirs_src.astype(np.float32)
            l = waveform.size(1)
            waveform = waveform[0].numpy()
            waveform = signal.convolve(waveform, rirs_src)[:l]
            waveform = torch.from_numpy(waveform)
            waveform = waveform.view(1,-1)
    return waveform


# make mix wav
'''
    wavs: [Tensor, Tensor], spk1, spk2
'''
def make_mix_wav(
        wavs: list[torch.Tensor],
        ratios: list[float],
        skip_idx=None,
        # parameters below are from mix_config
        max_or_min: str ='max',   # max/min, default max
        wav_shift_type: str = 'fixed_shift_duration',    
        wav_shift_config: dict = {},
        major=False,
    ):
    if skip_idx is not None:
        skip_feats = wavs[-skip_idx:]
        skip_ratios = ratios[-skip_idx:]
        wavs = wavs[:-skip_idx]
        ratios = ratios[:-skip_idx]
    else:
        skip_feats, skip_ratios = [],[]

    # compute rms before padding wave
    rms = [w.norm(p=2).item() for w in wavs] # compute energy
    rms = list(map(lambda x: x if x > 0.01 else 1, rms)) # avoid devide very small value 0
    max_rms = max(rms) # find the max energy
    
    # make wav delay
    # if random.randint(0, 10) % 3 == 0:  # hard coded 30% of 3 second delay for waveform with 16000 sample rate
    wavs, overlap_ratio = shift_wav(wavs, wav_shift_type, wav_shift_config)

    # compute wav size
    wav_size = [wav.size(1) for wav in wavs]
    if max_or_min == 'max':
        wav_len = max(wav_size) if max(wav_size) >= 16000 else 16000
    else:
        wav_len = wav_size[0]
    
    # padding wav
    wavs = [
        padding_wav(wavs[i], length=wav_len) for i in range(len(wavs))
    ]

    # control the target wav always lounder one or weak one
    if major:
        assert len(wavs) > 1
        if major == 1:
            ratios[0], ratios[1] = (ratios[1], ratios[0]) if ratios[0]<ratios[1] else (ratios[0], ratios[1])
        else:
            ratios[1], ratios[0] = (ratios[1], ratios[0]) if ratios[0]<ratios[1] else (ratios[0], ratios[1])
    
    # scale wavs 
    wavs: list[torch.Tensor] = [wavs[i]*(max_rms/rms[i]) for i in range(len(rms))] # make wav1 wav2 to equal energy
    scaled_wav: list[torch.Tensor] = [wavs[i]*ratios[i] for i in range(len(wavs))]
    mix_wav: torch.Tensor = sum(scaled_wav)

    # apply clip
    if torch.max(mix_wav) > 1 or torch.min(mix_wav) < -1:
        ratios = [r/sum(ratios) for r in ratios]
        scaled_wav = [wavs[i]*ratios[i] for i in range(len(wavs))]
        mix_wav = sum(scaled_wav)
    return [mix_wav] + scaled_wav + skip_feats, ratios + skip_ratios

def shift_wav(wavs: list[torch.Tensor], wav_shift_type: str, wav_shift_config: dict):
    
    assert wav_shift_type in ['fixed_shift_duration', 'flexiable_shift_duration']
    
    if wav_shift_type == 'fixed_shift_duration':
        shift_seconds = wav_shift_config.get('shift_seconds', 3)
        if (random.random() < wav_shift_config.get('wav_shift_prob', 0.3)) and (len(wavs) > 1):
            apply_shift_index = random.randint(0, len(wavs) - 1)  # spk1, spk2
            delay_len =  random.randint(16000, 16000 * shift_seconds)

            # assume that we have only two speech 
            if apply_shift_index == 1:
                ts = [0, delay_len]  
                te = [wavs[0].shape[-1], wavs[1].shape[-1] + delay_len]
            else:
                ts = [delay_len, 0]
                te = [wavs[0].shape[-1] + delay_len, wavs[1].shape[-1]]

            zero_padding = torch.zeros(1, delay_len)
            wavs[apply_shift_index] = torch.cat([zero_padding, wavs[apply_shift_index]], dim=1)
        else:
            ts = [0, 0]
            te = [wav.shape[-1] for wav in wavs]
        overlap_ratio = max(0, min(te) - max(ts)) / (max(te) - min(ts))
        
        return wavs, overlap_ratio
    
    elif wav_shift_type == 'flexiable_shift_duration':
        assert len(wavs) == 2, "Only support 2-speaker overlap simulation"
        
        # wavs[0] is always the target waveform, so we cannot swap here
        wav_lens = [_.shape[-1] for _ in wavs]
        min_len = min(wav_lens)
        max_len = max(wav_lens)
        max_avail_overlap_ratio = min_len / max_len
    
        attempted_overlap_ratio = random.random()
        overlap_ratio = min(attempted_overlap_ratio, max_avail_overlap_ratio)
        # note that the maximum overlap ratio we can achieve is min(len(wavs)) / max(len(wavs))
        if attempted_overlap_ratio > max_avail_overlap_ratio:
            apply_shift_index = np.argmin([_.size(-1) for _ in wavs])
            delay_len = random.randint(0, max_len - min_len)
        else:
            apply_shift_index = random.randint(0, len(wavs) - 1)  # spk1, spk2
            dice: float = random.random()
            if dice < wav_shift_config.get('non_overlap_prob', 0.2):    # randomly set completely non-overlapped shift
                delay_len = wav_lens[1-apply_shift_index] + random.randint(8000, 24000)  # make a random gap of 0.5~1.5 seconds
                overlap_ratio = 0
            else:
                # this calculation may seem a little hard to understand
                # now that we know the attempted overlap ratio **can be achieved**, which allows for **exactly the following format** of delay,
                # where 0 is silence padding valud
                #          | delay_len (x) | 
                # wavs[0]: 0 0 0 0 0 0 0 0 0 1 1 1 1 1
                # wavs[1]: 1 1 1 1 1 1 1 1 1 1 1 1
                # let x denote delay_len, and assume that we make shift on wavs[0] (apply_shift_index = 0)
                # then overlap_ratio = (wav_lens[1] - x) / (x + wav_lens[0])
                # i.e. overlap_ratio = (wav_lens[1 - apply_shift_index] - x) / (x + wav_lens[apply_shift_index])
                # solve x and we get the following answer
                delay_len = int((wav_lens[1 - apply_shift_index] - overlap_ratio * wav_lens[apply_shift_index]) / (1 + overlap_ratio))

        zero_padding = torch.zeros(1, delay_len)
        wavs[apply_shift_index] = torch.cat([zero_padding, wavs[apply_shift_index]], dim=1)

        return wavs, overlap_ratio

# padding wav with zeros
def padding_wav(wav, length, length_idx=1):
    if wav.size(length_idx) < length:
        wav_shape = wav.size()
        res_num = length - wav.size(length_idx) 
        if length_idx == 0:
            d = wav_shape[1]
            padding_shape = (res_num, d)
        else:
            d = wav_shape[0]
            padding_shape = (d, res_num)
        padding = torch.zeros(padding_shape)
        wav = torch.cat([wav, padding], dim=length_idx)
    else:
        wav = wav[:,:length] if length_idx == 1 else wav[:length,:]
    return wav


# trim wav by lenth and segment
def trim_wav(wav, segment):
    head, tail = segment
    if head == -1:
        # when head = -1 tail is the target windows length
        head = 0
        wav_size = wav.size(1)
        res = wav_size - tail
        head = 0 if res <= 0 else int(random.uniform(0, res))
        tail = head + tail
    wav = wav[:,head: tail]
    return wav


# get segment idx by time stamp
def got_seg(v, segment):
    diff = float('inf')
    nearest_value, nearest_index = None, None
    for i, value in enumerate(segment):
        if abs(value - v) < diff:
            diff = abs(value - v)
            nearest_value = value
            nearest_index = i
    return nearest_value, nearest_index

# convert segment info as alignment
def convert_mfa_to_align(mfa_obj):
    ali = []
    # mfa_obj zip([p1, p2], [p1_head, p1_tail, p2_head, p2_tail]) => (phn_list, positio
    # for idx, p in [p1, p2] i is index in phone list it also can be used to
    # find the position of the corresponding phone as index*2, index*2+1 which is p1_head, p1_tail
    for (phn, pos) in mfa_obj:
        one_ali = [
            p for i, p in enumerate(phn)
            for _ in range(pos[i * 2], pos[i * 2 + 1] + 1)
        ]
        ali.extend(one_ali)
    return torch.tensor(ali)


# make corruption pairs, maybe triple or more
# detach_f: detach_functions, for self corruption the datalist will format as json object so detach_f will be json.loads
# for none target corruption detach_f is lambda x: x
def make_corrupt_party(
        corrupt_list: list,
        corrupt_list_len: int,
        config: dict,
        corruption_material: list, detach_f=lambda x: x, prob=1.2, 
    ):
    ratios = []
    n = config.get('num_corrupt')
    if config.get('random_num', False):
        assert(n > 1)
        n = RANDOM_FACTOR['int'](1, n+1)
    if config.get('prob', 1.2) < 1:
        dice = random.uniform(0,1)
        if dice > config.get('prob'):
            n = 0

    sampler = config.get('sampler')
    assert (sampler in RANDOM_FACTOR.keys())
    sampler_config = config.get('sampler_config')
    if len(corruption_material) == 0:
        ratio = RANDOM_FACTOR[sampler](**sampler_config)
        ratio = ratio.tolist()[0] if isinstance(ratio, np.ndarray) else ratio
        ratios.append(ratio)
        c_idx = 0
    else:
        c_idx = len(corruption_material)
    
    for i in (range(n)):
        one_corrupt = corrupt_list[np.random.randint(0, corrupt_list_len)]
        one_corrupt = detach_f(one_corrupt)
        ratio = RANDOM_FACTOR[sampler](**sampler_config)
        ratio = ratio.tolist()[0] if isinstance(ratio, np.ndarray) else ratio
        ratios.append(ratio)
        if isinstance(one_corrupt, dict):
            corruption_material[c_idx+i+1] = one_corrupt
        elif isinstance(one_corrupt, str):
            corruption_material[c_idx+i+1] = {'sph': one_corrupt}
        else:
            raise NotImplementedError("corrupt list error")
    return ratios, corruption_material, n


# time shiftting: shiftting keywords from waveforme to make sure that: when mix tow keywords they will completely overlaped
def time_shifting(wav, frame_length=400, hop_length=160, wav_shift_type='mid'):
    wav = wav.view(-1) # assume wav is singal channel speech [1, num_samples] multi channel is not supportTODO:
    num_frames = 1 + (len(wav) - frame_length) // hop_length
    energy = torch.zeros(num_frames, dtype=torch.float32)
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = wav[start:end]
        energy[i] = torch.sum(frame ** 2)
    # max frame energy
    idx_frame = torch.argmax(energy) 
    idx_sample = idx_frame * hop_length + frame_length 
    if wav_shift_type == 'mid':
        dest_pos = int(wav.size(0) / 2)
    shift = int(dest_pos-idx_sample)
    wav = torch.roll(wav, shift)
    return wav.view(1, -1) # NOTE: convert back to singal channel [1, num_sample]


# make segment: make segment head and tail => [0.5, 1.5] mains: wav will trimed from the 0.5s to 1.5s (1s) 
def make_segment(segments, win_len, trim_type='raw', dither=False, sample_rate=16000):
    seg_head_tail = []
    idx_head_tail = []
    for seg in segments:
        if seg == -1:
            seg_head_tail.append([-1, int(win_len*sample_rate)])
            idx_head_tail.append([-1, -1])
            continue
        if trim_type == 'raw':
            seg_head = seg[0]
            if dither:
                seg_head = seg_head - random.uniform(0, 0.1)
                seg_head = seg_head if seg_head > 0 else 0
            seg_tail = seg_head + win_len
            seg_head_tail.append([int(seg_head*sample_rate), int(seg_tail*sample_rate)])
            idx_head_tail.append([0, len(seg)])
        elif trim_type == 'completetrim': 
            # cut setence from one utterance; the sub-utterance will contain a complete content
            leng_wav = seg[-1][-1]
            if leng_wav > win_len:
                seg_head_range = leng_wav - win_len
                _, seg_idx = got_seg(seg_head_range, [s[0] for s in seg])
                seg_head_idx = seg_idx if seg_idx <= 1 else np.random.randint(0, seg_idx-1)
                seg_head = seg[seg_head_idx][0]
                seg_tail = seg_head + win_len
                _, seg_tail_idx = got_seg(seg_tail, [s[1] for s in seg])
                seg_tail_idx = seg_tail_idx + 1 if seg_tail_idx < len(seg) - 1 else seg_tail_idx
                seg_tail = seg[seg_tail_idx][1]
            else:
                seg_head = 0
                seg_tail = seg[-1][-1]
                seg_head_idx = 0 
                seg_tail_idx = len(seg)
            seg_head_tail.append([int(seg_head*sample_rate), int(seg_tail*sample_rate)])
            idx_head_tail.append([seg_head_idx, seg_tail_idx])
        else:
            raise NotImplementedError("Only support raw, xxx")
    return seg_head_tail, idx_head_tail


# detach corruption
def detach_corruption(
        material: dict,
        segment_idx: list = None,
    ):
    keywords = [] 
    phn_labels = []
    bpe_labels = []
    labels = []
    for i, (_idx, info) in enumerate(material.items()):
        if 'word_keyword' in info:
            keywords.append(info['word_keyword'])
        if 'keyword' in info:
            keywords.append(info['keyword'])
        if 'label' in info:
            label = info['label']
            if segment_idx is not None:
                segment_head, segment_tail = segment_idx[i]
                label = label[segment_head: segment_tail] 
            labels.append(label)
        if 'phn_label' in info:
            phn_label = info['phn_label']
            if segment_idx is not None:
                segment_head, segment_tail = segment_idx[i]
                phn_label = phn_label[segment_head: segment_tail] 
            phn_labels.append(phn_label)
        if 'bpe_label' in info:
            bpe_label = info['bpe_label']
            if segment_idx is not None:
                segment_head, segment_tail = segment_idx[i]
                bpe_label = bpe_label[segment_head: segment_tail] 
            bpe_labels.append(bpe_label)
    return keywords, labels, phn_labels, bpe_labels

# insert special token in label sequence such as SOS: 0(start of sentence) 
# 1 2 3 4 5 -> "0" 1 2 3 4 5
def inject_special_token(
        keyword, keyword_length, label, 
        positive=True, keyword_pos=None, special_token={}, bpe_label=None, bpe_candidate=None, 
        use_filler_label_for_neg = False,
    ):
    TEXT_SPEC_TOKEN.update(special_token)
    new_phn_label = copy.deepcopy(label)
    new_bpe_label = copy.deepcopy(bpe_label)
    new_keyword = copy.deepcopy(keyword)
    # # # # import ipdb; ipdb.set_trace()
    # # # # '''
    # # # #     这里，如果keyword和speech是mismatch，也就是说positive == False，phn_label会被直接修改成punk，这样做是否合理需要讨论
    # # # # '''
    if use_filler_label_for_neg:
        if (not positive) and (TEXT_SPEC_TOKEN['punk'] is not None):
            new_phn_label = torch.tensor([TEXT_SPEC_TOKEN['punk'] for x in range(len(new_phn_label)//3)])
            new_bpe_label = torch.tensor([TEXT_SPEC_TOKEN['unk']  for x in range(len(new_bpe_label)//3)])
    


    if TEXT_SPEC_TOKEN['sos'] is not None: # start of sentence
        new_phn_label = [TEXT_SPEC_TOKEN['sos']] + new_phn_label
        keyword_pos = keyword_pos + 1  if keyword_pos is not None else keyword_pos # one token insert before the keyword
        
    if TEXT_SPEC_TOKEN['eos'] is not None: # end of sentence
        new_phn_label = new_phn_label + [TEXT_SPEC_TOKEN['eos']] 

    if TEXT_SPEC_TOKEN['psok'] is not None: # start of keyword
        new_keyword.insert(0, [TEXT_SPEC_TOKEN['psok']])

    if TEXT_SPEC_TOKEN['peok'] is not None: # end of keyword
        new_keyword.insert(len(new_keyword), [TEXT_SPEC_TOKEN['peok']])

    if (TEXT_SPEC_TOKEN['with_trans']) and (positive): # modify keyword in label
        new_phn_label[keyword_pos: keyword_pos+keyword_length] = new_keyword
        # bpe_kw_head = bpe_candidate[keyword_pos]
        # bpe_kw_tail = bpe_candidate[keyword_pos+keyword_length]
        # bpe_kw = bpe_label[bpe_kw_head: bpe_kw_tail]
        # bpe_kw.insert(0, [TEXT_SPEC_TOKEN['sok']])
        # bpe_kw.insert(len(bpe_kw), [TEXT_SPEC_TOKEN['eok']])
        # new_bpe_label[bpe_kw_head: bpe_kw_tail] = bpe_kw

    return new_keyword, new_phn_label, new_bpe_label, keyword_pos

# snipe_edges for waveform
def snipe_edge(waveform, hop_length=160):
    num_samples = waveform.size(1)
    edges = num_samples % hop_length
    return waveform[:,0:num_samples-edges]

# process raw json line
# data list is aranged in json format, in this function convert json into dict
# NOTE:  egs_format is a test feature i.e. read data from egs file just same like kaldi
# But egs_format didn't boost the training speed yet. just keep it and waiting for tuning
def process_raw(data: Generator[dict, None, None]):
    for sample in data:
        # one_sample: dict = json.loads(sample['src'])
        # one_sample.update({'epoch': sample['epoch']})
        # if one_sample.get('infer_wav', False):
        #     yield one_sample

        # if 'speech_interference_candidates' in sample:     # speech noise
        #     speech_interference_candidates = sample['speech_interference_candidates']
        #     speech_interference_candidates = [json.loads(d) for d in speech_interference_candidates]
        #     one_sample.update({'speech_interference_candidates': speech_interference_candidates})
        # if 'noise_interference_candidates' in sample:  # non-speech noise
        #     noise_interference_candidates = sample['noise_interference_candidates']
        #     one_sample.update({'noise_interference_candidates': noise_interference_candidates})
        # if 'rirs' in sample:
        #     rirs_src = sample['rirs']
        #     one_sample.update({'rirs':rirs_src})
        # if 'neg_candidate' in sample:
        #     neg_candidate = sample['neg_candidate']
        #     one_sample.update({'neg_candidate': neg_candidate})

        # # yield one_sample
        # print('debuggggggggggggggggggggggg:::::::::::::::init dataset successfully')
        # breakpoint()
        sample.update(json.loads(sample['src']))
        sample.pop('src')

        if sample.get('infer_wav', False):
            pass
        else:
            if 'speech_interference_candidates' in sample:     # speech noise
                speech_interference_candidates = sample['speech_interference_candidates']
                speech_interference_candidates = [json.loads(d) for d in speech_interference_candidates]
                sample.update({'speech_interference_candidates': speech_interference_candidates})
            if 'noise_interference_candidates' in sample:  # non-speech noise
                noise_interference_candidates = sample['noise_interference_candidates']
                sample.update({'noise_interference_candidates': noise_interference_candidates})
            if 'rirs' in sample:
                rirs_src = sample['rirs']
                sample.update({'rirs':rirs_src})
            if 'neg_candidate' in sample:
                neg_candidate = sample['neg_candidate']
                sample.update({'neg_candidate': neg_candidate})

        yield sample

# make corruption
# mix wav: 
#    - self corrution means mix two target speech: such as keyword1 speech + keyword2 speech
#    - none_target corruption means target speech with noise: such as keyword1 speech + none target inteferance
# NOTE: in this function, waveform are not mixed !!!  Just extract mix mmaterials !!! e.g.:
# corruption_material:{1: keyword1 speech FILE, 2: keyword2 speech FILE, 3: niose speech FILE}
# corruption ratios: [0.1, 0.6, 0.5]    
# the function is process_speech_feats:mix_wav will employ corruption material and corruption ratios to make
# the real mix waveform!!!!
def process_interference(data: Generator[dict, None, None], addition_noise_config: dict, egs_format=False):
    for sample in data:
        if sample.get('infer_wav', False):
            pass
        else:
            corruption_material = {}
            corruption_ratios = []
            num_corrupt = n_scorrupt = n_ncorrupt = 0
            if addition_noise_config['speech_interference']['use']: # make self corruption materials, i.e. speech
                assert 'speech_interference_candidates' in sample
                corrupt_list = sample['speech_interference_candidates']
                corrupt_list_len = len(corrupt_list) 
                ratios, corruption_material, n_scorrupt = make_corrupt_party(
                    corrupt_list, corrupt_list_len, addition_noise_config['speech_interference'], corruption_material, 
                )
                corruption_ratios.extend(ratios)
                num_corrupt += n_scorrupt
            
            if addition_noise_config['noise_interference']['use']: # make none target corruption materials, i.e. non-speech noise
                assert 'noise_interference_candidates' in sample
                corrupt_list = sample['noise_interference_candidates']
                corrupt_list_len = len(corrupt_list)
                ratios, corruption_material, n_ncorrupt = make_corrupt_party(
                    corrupt_list, corrupt_list_len, addition_noise_config['noise_interference'], corruption_material
                )
                corruption_ratios.extend(ratios)
                num_corrupt += n_ncorrupt

            # save the metarial into sample dict
            sample.update({
                'corruption_ratios': corruption_ratios,
                'corruption_material': corruption_material,
                'n_scorrupt': n_scorrupt,   # number of speech noise
                'n_ncorrupt': n_ncorrupt,   # number of non-speech noise
                'num_corrupt': num_corrupt, # num_corrupt = n_scorrupt + n_ncorrupt
            })
        yield sample


# process speech feats
# load wav -> corrupt wav -> destroy a positive sample to negative (made for FA) -> extract fbank
# NOTE: this function can support load kaldi ark feats, torch pt file and read wavefrom from raw wav file
# NOTE: But kaldi feat, torch pt have not been verified in training process be carefull that.
def process_speech_feats(data: Generator[dict, None, None], config: dict, egs_format=False):
    for sample in data:
        # print(f'debugggggggggggggggg: keys number: {len(sample.keys())}')
        input_data_type = config.get('data_type', 'raw') # feats type includes: (1)raw waveform, (2)kaidl: kaldi ark, (3)pt: torch.pt
        if (input_data_type != 'raw') and ('corruption_material' in sample): # corruption only support performed on waveform
            raise NotImplementedError("Only support corruption on waveforme")
        feats: list[str] = [sample['sph']]

        if 'corruption_material' in sample: # corruption: self corruption=>mix training none target corruption=> data augmentation
            corruption_material: dict[int, dict] = sample['corruption_material']
            corruption_feats = [corruption_material[x]['sph'] for x in corruption_material.keys()]
            corruption_segments = [
                corruption_material[x]['segment'] if 'segment' in corruption_material[x] else -1
                for x in corruption_material.keys()
            ]
            feats.extend(corruption_feats)
        
        if egs_format:
            input_data_type = 'empty'
        feats: list[tuple[torch.Tensor, int]] = [INPUT_DATA_LOADER[input_data_type](x) for x in feats]
        feats: list[torch.Tensor] = [INPUT_DATA_LOADER['rm_sr'](x) for x in feats] if input_data_type == 'raw' else feats #remove sample rate


        # Wav augment: volume change and add white noise
        if config.get('wav_augment', False):
            if 'rirs' in sample:
                rirs_src = sample['rirs']
            feats = [wav_augment(f, config.get('wav_augment'), rirs_src) for f in feats]

        # Trim wav according to segment 
        if config.get('trim_config', False):
            if 'segment' in sample:
                segments = sample['segment']
            else:
                segments = [[0,0]]
            if 'corruption_segments' in locals().keys():
                segments += corruption_segments
            segments, segment_idx = make_segment(segments, **config.get("trim_config", {}))
            sample.update({"segment_idx": segment_idx})
            feats = [trim_wav(feats[i], segments[i]) for i in range(len(feats))]
        
        # Mix wav feats
        if 'corruption_material' in sample:
            mix_config: dict = config.get('mix_config', {})
            if (sample['n_scorrupt'] != 0) and (sample['n_ncorrupt'] != 0): 
                # target1 speech + target2 speech + corruption speech
                # random select a target and mix it with noise: target1 + corruption speech || target2 + corruption speech
                n_ncorrupt = sample['n_ncorrupt']
                crpt_idx = random.randint(0, sample['n_scorrupt'])
                target_feats = feats[crpt_idx]
                noise_feats, noise_ratio = feats[-n_ncorrupt:], sample['corruption_ratios'][-n_ncorrupt:]
                tmp_ratio = [sample['corruption_ratios'][crpt_idx]] + noise_ratio
                mix_feats, _ = make_mix_wav([target_feats]+noise_feats, tmp_ratio, **mix_config)
                feats[crpt_idx] = mix_feats[0] # replace the selected target speech by mixed noisy speech
                #feats, sample['corruption_ratios'] = feats[:-n_ncorrupt], sample['corruption_ratios'][:-n_ncorrupt]
                skip_idx = n_ncorrupt # here noise has been mix into one of the target speech so in the following mix
                                      # mixing process skip noise speech !!!!NOTE!!!!
            else:
                skip_idx = None

            feats, sample['corruption_ratios'] = make_mix_wav(
                feats, sample['corruption_ratios'], skip_idx=skip_idx, **mix_config
            )   # feats[0]: mixture, feats[1:1 + sample['num_corrupt'] + 1]: scaled waveform
        if config.get('snipe_edge', False):
            hop_length = config['snipe_edge'].get('hop_length', 160)
            feats = [snipe_edge(f, hop_length) for f in feats]
        
        if config.get('return_raw', False):
            if 'corruption_material' in sample:
                raw_feats = copy.deepcopy(feats[1])
            else:
                raw_feats = copy.deepcopy(feats[0])
            sample.update({'raw_wav': raw_feats.squeeze(0)}) # only keep the target speech
        
        # whether to sample 4s fixed window speech
        if config.get('random_raw', False):
            sample_len1 = feats[1].size(1)
            dur = 16000 * 4 
            sample_head = random.randint(0, sample_len1-dur-1) if sample_len1 > dur else 0
            raw_wav1 = copy.deepcopy(feats[1][:, sample_head:sample_head+dur])
            sample.update({'raw_wav1': raw_wav1.squeeze(0)})  

            sample_len2 = feats[2].size(1)
            dur = 16000 * 4 
            sample_head = random.randint(0, sample_len2-dur-1) if sample_len2 > dur else 0
            raw_wav2 = copy.deepcopy(feats[2][:, sample_head:sample_head+dur])
            sample.update({'raw_wav2': raw_wav2.squeeze(0)})  

        # Extract feature: MFCC / FBANK 
        feats_type: str = config.get('feats_type', 'fbank')
        feats_config: dict = config.get('feats_config', FBANK_DEFAULT_SETTING)
        # print(feats[0].shape)
        feats: list[torch.Tensor] = [FEATS_EXTRACTOR[feats_type](f, **feats_config) for f in feats]
        # print(feats[0].shape)

        if config.get('post_feats_config').get('use', False):
            # Splice Feature: add context
            splice_config = config.get('splice_config')
            feats = [splice_feats(f, **splice_config) for f in feats]
            
            # Subsample Feature: skip frame
            if config.get('subsample_rate'):
                feats = [f[::config.get('subsample_rate')] for f in feats]
        # print(feats[0].shape)
        # Load feats into torch Tensor
        start_idx = 0
        if 'corruption_material' in sample:
            mix_feats = feats[start_idx]
            start_idx += 1
            sample.update({"mixspeech": mix_feats})
        else: # if no corruption meterail the 1th feats is clean feats
            sample.update({"speech": feats[0]})
        
        # keep clean feats e.g. mix_wav = wav1 + wav2 the following code will 
        # concat wav1, wav2 into one matrix and load to {speech: [wav1; wav2]}
        if sample.get("n_scorrupt", 0) > 0:
            clean_feats = feats[start_idx: start_idx+sample['n_scorrupt']+1]
            ratios = sample['corruption_ratios']
            #TODO: consider about add noise augment ratios
            ratios = ratios[0:sample['n_scorrupt']+1]
            clean_feats = torch.cat([x.unsqueeze(0) for x in clean_feats], dim=0)
            start_idx = sample['n_scorrupt'] + 1
            sample.update({"speech": clean_feats[0]}) # here only keep the first wav
            sample.update({"ratios": ratios})
        # Same with the code upper noise wav will load to {noise: niose_wav} 
        # NOTE: noise wav have not involved in training process, in practice noise wav only 
        # applied in former HOMO method, we comment this code to save the memory resource
        # if sample.get("n_ncorrupt", 0) > 0:
        #     noise_feats = feats[start_idx: start_idx+sample['n_ncorrupt']]
        #     if len(noise_feats) > 1:
        #         noise_feats = torch.cat([x.unsqueeze(0) for x in noise_feats], dim=0)
        #     else:
        #         noise_feats = noise_feats[0]
        #     sample.update({"niose_speech": noise_feats})
        yield sample 

# Process text feats, mainly deal with segment:
# e.g. in process_speech_feats wav has been trimed by segment, and the text label will 
# be cutted in this function acorrding to segment also.
def process_text_feats(data: Generator[dict, None, None], id2k=None):
    for sample in data:
        if ('label' in sample) and ('segment_idx' in sample):
            label = sample['label']
            segment_idx = sample['segment_idx']
            m_head, m_tail = segment_idx[0]
            label = label[m_head: m_tail]
            sample.update({'label': label})
        
        if 'corruption_material' in sample:
            # 'c_' indicates 'corruption'
            if 'segment_idx' in sample:
                c_segment_idx = sample['segment_idx'][1:]
            else:
                c_segment_idx = None
            c_keyword, c_label, c_phn_label, c_bpe_label = detach_corruption(sample['corruption_material'], c_segment_idx)
            if len(c_keyword) != 0:
                sample.update({"mix_keyword": sample['word_keyword'] + unfold_list(c_keyword)})
            if len(c_label) != 0:
                sample.update({"crpt_label": c_label})
            if len(c_phn_label) != 0:
                sample.update({"c_phn_label": c_phn_label})
            if len(c_bpe_label) != 0:
                sample.update({"c_bpe_label": c_bpe_label})

        yield sample


# Process: sample keyword from continues label
def process_sampled_keyword_from_label(
        data: Generator[dict, None, None],
        positive_prob: float = 0.5,
        special_token: dict = {},
        sample_config: dict = {},
        neg_len = None, 
        use_filler_label_for_neg = False,
):
    # TEXT_SPEC_TOKEN = {'sos','eos','sok', 'eok', 'unk'}
    # sos: start of setence, eos: end of setence, sok: start of keyword, eok, end of keyword, unk: unknow token
    TEXT_SPEC_TOKEN.update(special_token)
    for sample in data:
        new_phn_label = copy.deepcopy(sample['phn_label'])
        new_bpe_label = copy.deepcopy(sample['bpe_label']) if 'bpe_label' in sample else None
        bpe_candidate = copy.deepcopy(sample['b_kw_candidate']) if 'b_kw_candidate' in sample else None

        keyword, kw_pos, kw_length, pos, target = generate_keyword_sample(
            data_sample=sample, positive_prob=positive_prob, 
            sample_config=sample_config, neg_len=neg_len,
        )
        kw, new_phn_label, new_bpe_label, kw_pos = inject_special_token(
            keyword=keyword, keyword_length=kw_length, positive=pos, label=new_phn_label, 
            keyword_pos=kw_pos, special_token=special_token, bpe_label=new_bpe_label, bpe_candidate=bpe_candidate,
            use_filler_label_for_neg = use_filler_label_for_neg
        )
        # 
        # sample['match_case']: zero_or_one_match

        if sample['match_case'] == 'zero_or_one_match':
            if pos:     # one_match
                sample['match_case'] = 'one_match'
            else:
                sample['match_case'] = 'zero_match'

        sample.update({'keyword': kw, 'phn_label': new_phn_label, 'bpe_label': new_bpe_label, 'target': target,
                       'phn_label_list': copy.deepcopy(sample['phn_label'])})
        yield sample

# process fix keyword from segment
def process_fix_keyword(data: Generator[dict, None, None], special_token={}, use_filler_label_for_neg = False, sample_config=None, positive_prob=1.0):
    for sample in data:
        if len(special_token) != 0:
            kw: list[int] = sample['keyword']
            phn_label: list[int] = sample['phn_label']
            sample['phn_label_list'] = copy.deepcopy(sample['phn_label'])
            # target: torch.Tensor = torch.tensor([sample['target']])   # sample['target']: int
            target: list[int] = [sample['target']]
            pos = (sample['target'] == 1)
            kw, label, _, _ = inject_special_token(
                keyword=kw, keyword_length=len(kw),
                label=phn_label, special_token=special_token,
                positive=pos, 
                use_filler_label_for_neg=use_filler_label_for_neg 
            )
            sample.update({'keyword': kw, 'label': label, 'target': target})
        yield sample

def process_sot_label(data: Generator[dict, None, None], special_token=None):
    #for i, sample in enumerate(data):
    for sample in data:
        new_label = copy.deepcopy(sample['bpe_label'])
        #t = copy.deepcopy(new_label)
        if 'sc' not in special_token:
            print('sc should be specify for sot methods')
        crpt_bpe_label = sample['c_bpe_label']
        new_label = new_label + [special_token['sc']] + crpt_bpe_label
        #new_label = unfold_list(new_label)
        #sym = [id2s[i] for i in new_label]
        #sym = "".join(sym).replace("▁"," ")
        #key = sample['key']
        #one_crpt = sample['corruption_material'][1]
        #ckey = one_crpt['key']
        #crpt_wav = sample['raw_wav']
        #torchaudio.save("{}_{}.wav".format(key, ckey), crpt_wav, encoding="PCM_S", bits_per_sample=16, sample_rate=16000)
        #w = open("{}_{}.txt".format(key, ckey), 'w')
        #w.write(sym.replace("sc", " <sc> ")+"\n")
        #if i > 10:
        #    exit()
        #print (sample['corruption_material'].keys())
        sample.update({'bpe_label': new_label})
        yield sample

def process_cohort_label(data: Generator[dict, None, None]):
    for sample in data:
        bpe_label = copy.deepcopy(sample['bpe_label'])
        phn_label = copy.deepcopy(sample['phn_label'])
        if 'c_bpe_label' not in sample:
            sample.update({"c_bpe_label": bpe_label})
        if 'c_phn_label' not in sample:
            sample.update({"c_phn_label": phn_label})
        yield sample

# sample keyword from asr label, actually sample positive and make a negative
def generate_keyword_sample(data_sample: dict, positive_prob: float, sample_config: dict, neg_len=None):
    random_value = random.uniform(0, 1)
    full_phoneme_label = copy.deepcopy(data_sample['phn_label'])
    if random_value > positive_prob and not data_sample.get('is_multiple_match', False): # negative sample
        # if corruption phn_label is in sample, i.e. overlapped speech,
        # then the negative keyword sequence should contain neither of speeches,
        # regardless of target or corruption
        if 'c_phn_label' in data_sample:     # corruption phn_label
            crpt_label = data_sample['c_phn_label']
            merged_phoneme_label = full_phoneme_label + crpt_label
        else:
            merged_phoneme_label = full_phoneme_label
        keyphrase = generate_negative_keyword_sample(
            neg_list=data_sample['neg_candidate'], neg_len=neg_len, 
            pos_label=merged_phoneme_label, spk_id=data_sample['key'].split('-')[0],
            sample_config=sample_config,
        )
        keyphrase_start_position = -1
        is_positive_keyword = False
        target_label = [0]
    else: 
        # positvie sample
        # standard sample: src, sample_info, speech_interference_candidates, neg_candidate
        # multiple_match keys: src, sample_info, speech_interference_candidates, multiple_match, predefined_keyword_info 
        kw_candidate = data_sample.get('kw_candidate', None)
        if data_sample.get('is_multiple_match', False):
            keyphrase = data_sample['predefined_keyword_info']['keyword']
            keyphrase_start_position = data_sample['predefined_keyword_info']['keyword_start_position']
        else:
            keyphrase, keyphrase_start_position = sample_keyword_from_whole_label(
                label=full_phoneme_label, kw_candidate=kw_candidate,
                sample_config=sample_config,
            )
        is_positive_keyword = True
        target_label = [1]
    return keyphrase, keyphrase_start_position, len(keyphrase), is_positive_keyword, target_label

# sample positive keyword from asr label
# Do we always need keywords to be continous?
def sample_keyword_from_whole_label(label, sample_config: dict, kw_candidate=None):
    keyword_length_range = sample_config.get('keyword_length_range', [2,6])
    continuous: bool = sample_config.get('continuous', True)
    
    if continuous:
        kw_len = random.randint(*keyword_length_range)
        kw_len = min(kw_len, len(label))

        kw_pos = random.randint(0, len(label) - kw_len)
        kw = label[kw_pos: kw_pos + kw_len]
    else:
        raise NotImplementedError
    return kw, kw_pos

# sample negative keyword from the whole corpus
# this function looks a little bit confusing, but is correctly implemented
# pos_label: [[1,2,3], [4,5,6], [7, 8, 9]]
def generate_negative_keyword_sample(neg_list: list, neg_len, pos_label: list, sample_config: dict, spk_id=None):
    negative_keyword = pos_label[0]

    flatten_label = unfold_list(pos_label)
    flatten_neg = unfold_list(negative_keyword)

    flatten_label = int2sym(flatten_label)
    flatten_neg = int2sym(flatten_neg)

    neg_spk = spk_id if spk_id is not None else -1
    
    while (" ".join(flatten_neg) in " ".join(flatten_label)) or (neg_spk == spk_id):
        one_neg_list: str = neg_list[random.randint(0, neg_len - 1)]
        one_neg_list: dict = json.loads(one_neg_list)
        if spk_id is not None:
            neg_spk = one_neg_list['key'].split('-')[0]
        one_neg_label = one_neg_list['phn_label']
        kw_candidate = one_neg_list.get('kw_candidate', None) 
        negative_keyword, _ = sample_keyword_from_whole_label(
            label=one_neg_label, kw_candidate=kw_candidate, 
            sample_config=sample_config,
        )
        flatten_neg = unfold_list(negative_keyword)
        flatten_neg = int2sym(flatten_neg)
    return negative_keyword

# world_size: 8
# num_workers: 9
def process_save_meta(data: Generator[dict, None, None], meta_save_config: dict):
    save_dir: str = meta_save_config['save_dir']
    meta_kept_keys = [
        'match_case', 'target', 'keyword', 'key', 'phn_label', 'text', 'sph', 
        'corruption_ratios', 'ratios',
        # rather useless for debug
        'corruption_material', 'kw_candidate', 'b_kw_candidate', 'bpe_label',
        'n_scorrupt', 'n_ncorrupt', 'num_corrupt',
    ]
    '''
        dict_keys(['match_case', 'key', 'phn_label', 'text', 'sph', 'kw_candidate', 'b_kw_candidate', 'bpe_label', 'corruption_ratios', 'corruption_material', 'n_scorrupt', 'n_ncorrupt', 'num_corrupt', 'ratios', 'target', 'keyword'])
    '''
    first: bool = True
    for sample in data:
        if first:
            epoch, rank, worker_id = sample['epoch'], sample['rank'], sample['worker_id']
            f = open(os.path.join(save_dir, f"epoch{epoch}.rank{rank}.worker{worker_id}.jsonl"), 'w')
            first = False

        output_jsonl: dict = {}
        for key in meta_kept_keys:
            output_jsonl[key] = sample[key]
        output_jsonl['infer_wav'] = True

        output_line: str = json.dumps(output_jsonl) 
        f.write(output_line + '\n')
        yield sample
        

# all the label information are [[...], [...]] unfold them.
# NOTE: the label in raw json file is aranged in word format especially for chinese characters such as :
# [[1,2],[3],[4,5]] is this example [1,2] is a chinese word such as [你好], this kind of arrangement is usefull
# for sample keyword as when sample index 0 [1,2] will be the candidate keyword but not [1]
# [process_list_data] is aim to unfold the label sequence, use the example above again. [[1,2],[3],[4,5]] ->
# [1,2,3,4,5], this function is applied after sample keywords
def process_list_data(data: Generator[dict, None, None],):
    for sample in data:
        for key, value in sample.items():
            if key in NONE_TENSOR_KEY:
                continue
            if isinstance(value, list):
                value = unfold_list(value)
            if not isinstance(value, torch.Tensor):
                sample.update({key: torch.tensor(value)})
        yield sample


# such as ctc loss and rnn-t loss need speech length and target length
# but after make batch. these data will be append as the same length.
# so we compute length information before make batch
def make_length(data: Generator[dict, None, None]):
    for sample in data:
        length_info = {}
        for key, value in sample.items():
            if key in NONE_TENSOR_KEY:
                continue
            if not isinstance(value, torch.Tensor):
                continue
            if value.dim() == 0:
                continue
            new_key = "{}_len".format(key)
            length = value.size(0)
            length_info.update({new_key: torch.tensor(length)})
        sample.update(length_info)
        yield sample


# concat into one batch
def concat_tensor(data_list, seq_padding=False, padding_value=0):
    if seq_padding:
        tensor = pad_sequence(data_list, batch_first=True, padding_value=padding_value)
    else:
        tensor = torch.cat([x.unsqueeze(0) for x in data_list], dim=0)
    return tensor


# fetch keys
# there are a lot of inter material is data processing, however most of them are not 
# training ingredients so we fetch the training data by keys, more detail can be found in
# config files fetch_keys: 
def fetch_tensor(data: Generator[dict, None, None], fetch_key):
    for sample in data:
        '''
            # data: dict
            filtered_data: dict = {}
            # fetch_key: list[str]
            for key in fetch_key:
                filtered_data[key] = data[key]
            yield data
        '''
        if fetch_key[0] == 'key':
            sort_key = fetch_key[1]
        else:
            sort_key = fetch_key[0]
        index = torch.tensor([x[sort_key].size(0) for x in sample])
        index = torch.argsort(index, descending=True)
        return_feats = []
        for k in fetch_key: #TODO: this code in not safe ...
            if (k == 'key') or (k == 'tag'):
                return_feats.append([sample[i][k] for i in index])
                continue
            if k in NONE_TENSOR_KEY:
                continue
            if sample[0][k].dim() != 0:
                seq_padding = True
            else:
                seq_padding = False
            if k in CTC_KEY:
                padding_value = -1
            else:
                padding_value = 0
            return_feats.append(
                concat_tensor(
                    [sample[i][k] for i in index], seq_padding=seq_padding, padding_value=padding_value
                )
            )
        yield tuple(return_feats)

def fetch_inference_data(data: Generator[dict, None, None], fetch_key, fetch_meta_key=None):
   for sample in data:
        # original fetch_key
        if fetch_key[0] == 'key':
            sort_key = fetch_key[1]
        else:
            sort_key = fetch_key[0]
        index = torch.tensor([x[sort_key].size(0) for x in sample])
        index = torch.argsort(index, descending=True)
        return_feats = []
        eval_tensor_feats = {}
        for sample_key in fetch_key: #TODO: this code in not safe ...
            if (sample_key == 'key') or (sample_key == 'tag'):
                return_feats.append([sample[i][sample_key] for i in index])
                continue
            if sample_key in NONE_TENSOR_KEY:
                continue
            if sample[0][sample_key].dim() != 0:
                seq_padding = True
            else:
                seq_padding = False
            if sample_key in CTC_KEY:
                padding_value = -1
            else:
                padding_value = 0

            concated_tensor = concat_tensor(
                    [sample[i][sample_key] for i in index], 
                    seq_padding=seq_padding, 
                    padding_value=padding_value
                )
            
            return_feats.append(concated_tensor)
            eval_tensor_feats[sample_key] = concated_tensor
            
        meta_feats = defaultdict(list)
        for idx in index:
            for key in fetch_meta_key:
                meta_feats[key].append(sample[idx][key])
        
        yield tuple(return_feats), meta_feats, eval_tensor_feats

def process_speaker_label(data: Generator[dict, None, None], spk2cls: str):
    #for i, sample in enumerate(data):
    spk2cls = json.load(open(spk2cls))
    for sample in data:
        spk_key = sample['key'].split('-')[0]
        spk_target = torch.tensor(spk2cls[spk_key])
        sample.update({'spk_label': spk_target})
        yield sample

# make batch
# TODO: make it support batch bce
def filter_max_length(data: Generator[dict, None, None], max_frames=3000):
    """Drop samples where mixspeech (fbank features) exceeds max_frames.
    This prevents OOM from overly long mixed-speech sequences in transformer attention."""
    dropped = 0
    kept = 0
    for sample in data:
        mixspeech_len = sample.get('mixspeech_len', None)
        if mixspeech_len is not None and mixspeech_len > max_frames:
            dropped += 1
            continue
        kept += 1
        yield sample
    if dropped > 0:
        import logging
        logging.getLogger(__name__).info(
            f"filter_max_length (max_frames={max_frames}): dropped {dropped}, kept {kept}"
        )


def make_batch(data: Generator[dict, None, None], batch_size=256):
    buf = []
    # batch_count = {}
    for sample in data:
        # breakpoint()
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


# # NOTE: this function is Deprecated temporarly
# def process_corruption_dump(data, config):
#     for sample in data:
#         corruption_material = {}
#         corruption_ratios = []
#         num_corrupt = n_scorrupt = n_ncorrupt = 0
#         if config.get('speech_interference', False): # make self corruption materials
#             corrupt_list = config['s_corrupt_list']
#             corrupt_list_len = config['num_scorrupt_samples']
#             ratios, corruption_material, n_scorrupt = make_corrupt_party(
#                 corrupt_list, corrupt_list_len, config['speech_interference'], corruption_material, 
#                 detach_f=json.loads, 
#             )
#             corruption_ratios.extend(ratios)
#             num_corrupt += n_scorrupt
        
#         if config.get('noise_interference', False): # make none target corruption materials
#             corrupt_list = config['n_corrupt_list']
#             corrupt_list_len = config['num_ncorrupt_samples']
#             ratios, corruption_material, n_ncorrupt = make_corrupt_party(
#                 corrupt_list, corrupt_list_len, config['noise_interference'], corruption_material
#             )
#             corruption_ratios.extend(ratios)
#             num_corrupt += n_ncorrupt

#         # save the metarial into sample dict
#         sample.update({
#             'corruption_ratios': corruption_ratios,
#             'corruption_material': corruption_material,
#             'n_scorrupt': n_scorrupt,
#             'n_ncorrupt': n_ncorrupt,
#             'num_corrupt': num_corrupt,
#         })
#         yield sample


# old version, partly flawed
# def sample_keyword_from_label(label, kw_candidate=None):
#     kw_len = random.randint(2, 6)
#     if kw_candidate: #TODO: a little bit confuse ...  optim it latter
#         kw_len = kw_len if kw_len < len(kw_candidate) else 1
#         kw_pos_idx = random.randint(0, len(kw_candidate) - kw_len - 1) if len(kw_candidate) > kw_len + 1 else 0
#         kw_pos = kw_candidate[kw_pos_idx]
#         if kw_pos_idx + kw_len >= len(kw_candidate):
#             kw_len -= 1 
#         kw_len = kw_candidate[kw_pos_idx+kw_len] - kw_pos
#     else:
#         kw_pos = random.randint(0, len(label)-kw_len) if len(label) > kw_len else 0
#        #kw_pos = random.randint(0, len(label)-kw_len) if len(label) > kw_len else 0
#     kw = label[kw_pos: kw_pos + kw_len]
#     return kw, kw_pos

# # previous reliable version of sample_keyword_from_label
# def sample_keyword_from_label(label, kw_candidate=None):
#     kw_len = random.randint(2, 6)
#     if kw_candidate: #TODO: a little bit confuse ...  optim it latter
#         kw_len = kw_len if kw_len < len(kw_candidate) else 1
#         kw_pos_idx = random.randint(0, len(kw_candidate) - kw_len) if len(kw_candidate) >= kw_len else 0
#         kw_pos = kw_candidate[kw_pos_idx]
#         # kw_len = kw_candidate[kw_pos_idx+kw_len] - kw_pos
#     else:
#         kw_pos = random.randint(0, len(label) - kw_len) if len(label) >= kw_len else 0
#        #kw_pos = random.randint(0, len(label)-kw_len) if len(label) > kw_len else 0
#     kw = label[kw_pos: kw_pos + kw_len]
#     return kw, kw_pos