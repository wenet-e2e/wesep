import io
import json
import logging
import random
import tarfile
from subprocess import PIPE, Popen
from urllib.parse import urlparse

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from scipy import signal

from wesep.dataset.FRAM_RIR import single_channel as RIR_sim
from wesep.dataset.lmdb_data import LmdbData


AUDIO_FORMAT_SETS = {"flac", "mp3", "m4a", "ogg", "opus", "wav", "wma"}

# set the simulation configuration
simu_config = {
    "min_max_room": [[3, 3, 2.5], [10, 6, 4]],
    "rt60": [0.1, 0.7],
    "sr": 16000,
    "mic_dist": [0.2, 5.0],
    "num_src": 1,
}


def url_opener(data):
    """Give url or local file, return file descriptor
    Inplace operation.

    Args:
        data(Iterable[str]): url or local file list

    Returns:
        Iterable[{src, stream}]
    """
    for sample in data:
        assert "src" in sample
        # TODO(Binbin Zhang): support HTTP
        url = sample["src"]
        try:
            pr = urlparse(url)
            # local file
            if pr.scheme == "" or pr.scheme == "file":
                stream = open(url, "rb")
            # network file, such as HTTP(HDFS/OSS/S3)/HTTPS/SCP
            else:
                cmd = f"wget -q -O - {url}"
                process = Popen(cmd, shell=True, stdout=PIPE)
                sample.update(process=process)
                stream = process.stdout
            sample.update(stream=stream)
            yield sample
        except Exception as ex:
            logging.warning("Failed to open {}".format(url))


def tar_file_and_group(data):
    """Expand a stream of open tar files into a stream of tar file contents.
    And groups the file with same prefix

    Args:
        data: Iterable[{src, stream}]

    Returns:
        Iterable[{key, mix_wav, spk1_wav, spk2_wav, ..., sample_rate}]
    """
    for sample in data:
        assert "stream" in sample
        stream = tarfile.open(fileobj=sample["stream"], mode="r:*")
        # TODO: The mode need to be validated
        # In order to be compatible with the torch 2.x version,
        # the file reading method here does not use streaming.
        prev_prefix = None
        example = {}
        num_speakers = 0
        valid = True
        for tarinfo in stream:
            name = tarinfo.name
            pos = name.rfind(".")
            assert pos > 0
            prefix, postfix = name[:pos], name[pos + 1 :]
            if prev_prefix is not None and prev_prefix not in prefix:
                example["key"] = prev_prefix
                if valid:
                    example["num_speaker"] = num_speakers
                    num_speakers = 0
                    yield example
                example = {}
                valid = True
            with stream.extractfile(tarinfo) as file_obj:
                try:
                    if "spk" in postfix:
                        example[postfix] = (
                            file_obj.read().decode("utf8").strip()
                        )
                        num_speakers += 1
                    elif postfix in AUDIO_FORMAT_SETS:
                        waveform, sample_rate = torchaudio.load(file_obj)
                        if prefix[-5:-1] == "_spk":
                            example["wav" + prefix[-5:]] = waveform
                            prefix = prefix[:-5]
                        else:
                            example["wav_mix"] = waveform
                            example["sample_rate"] = sample_rate
                    else:
                        example[postfix] = file_obj.read()
                except Exception as ex:
                    valid = False
                    logging.warning("error to parse {}".format(name))
            prev_prefix = prefix

        if prev_prefix is not None:
            example["key"] = prev_prefix
            example["num_speaker"] = num_speakers
            num_speakers = 0
            yield example
        stream.close()
        if "process" in sample:
            sample["process"].communicate()
        sample["stream"].close()


def tar_file_and_group_single_spk(data):
    """Expand a stream of open tar files into a stream of tar file contents.
    And groups the file with same prefix

    Args:
        data: Iterable[{src, stream}]

    Returns:
        Iterable[{key, wav, spk, sample_rate}]
    """
    for sample in data:
        assert "stream" in sample
        stream = tarfile.open(
            fileobj=sample["stream"], mode="r|*"
        )  # Only support pytorch version <2.0
        prev_prefix = None
        example = {}
        valid = True
        for tarinfo in stream:
            name = tarinfo.name
            pos = name.rfind(".")
            assert pos > 0
            prefix, postfix = name[:pos], name[pos + 1 :]
            if prev_prefix is not None and prefix != prev_prefix:
                example["key"] = prev_prefix
                if valid:
                    yield example
                example = {}
                valid = True
            with stream.extractfile(tarinfo) as file_obj:
                try:
                    if postfix in ["spk"]:
                        example[postfix] = (
                            file_obj.read().decode("utf8").strip()
                        )
                    elif postfix in AUDIO_FORMAT_SETS:
                        waveform, sample_rate = torchaudio.load(file_obj)
                        example["wav"] = waveform
                        example["sample_rate"] = sample_rate
                    else:
                        example[postfix] = file_obj.read()
                except Exception as ex:
                    valid = False
                    logging.warning("error to parse {}".format(name))
            prev_prefix = prefix
        if prev_prefix is not None:
            example["key"] = prev_prefix
            yield example
        stream.close()
        if "process" in sample:
            sample["process"].communicate()
        sample["stream"].close()


def parse_raw_single_spk(data):
    """Parse key/wav/spk from json line

    Args:
        data: Iterable[str], str is a json line has key/wav/spk

    Returns:
        Iterable[{key, wav, spk, sample_rate}]
    """
    for sample in data:
        assert "src" in sample
        json_line = sample["src"]
        obj = json.loads(json_line)
        assert "key" in obj
        assert "wav" in obj
        assert "spk" in obj
        key = obj["key"]
        wav_file = obj["wav"]
        spk = obj["spk"]
        try:
            waveform, sample_rate = torchaudio.load(wav_file)
            example = dict(
                key=key, spk=spk, wav=waveform, sample_rate=sample_rate
            )
            yield example
        except Exception as ex:
            logging.warning("Failed to read {}".format(wav_file))


def mix_speakers(data, num_speaker=2, shuffle_size=1000):
    """Dynamic mixing speakers when loading data,
    shuffle is not needed if this function is used
    Args:
        :param data: Iterable[{key, wavs, spks}]
        :param num_speaker:
        :param use_random_snr:
        :param shuffle_size:
    Returns:
        Iterable[{key, wav_mix, wav_spk1, wav_spk2, ..., spk1, spk2, ...}]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                cur_spk = x["spk"]
                example = {
                    "key": x["key"],
                    "wav_spk1": x["wav"],
                    "spk1": x["spk"],
                    "sample_rate": x["sample_rate"],
                }
                key = "mix_" + x["key"]
                interference_idx = 1
                while interference_idx < num_speaker:
                    interference = random.choice(buf)
                    while interference["spk"] == cur_spk:
                        interference = random.choice(buf)
                    key = key + "_" + interference["key"]
                    interference_idx += 1
                    example["wav_spk" + str(interference_idx)] = interference[
                        "wav"
                    ]
                    example["spk" + str(interference_idx)] = interference["spk"]
                example["key"] = key
                example["num_speaker"] = num_speaker
                yield example

            buf = []

    # The samples left over
    random.shuffle(buf)
    for x in buf:
        cur_spk = x["spk"]
        example = {
            "key": x["key"],
            "wav_spk1": x["wav"],
            "spk1": x["spk"],
            "sample_rate": x["sample_rate"],
        }
        key = "mix_" + x["key"]
        interference_idx = 1
        while interference_idx < num_speaker:
            interference = random.choice(buf)
            while interference["spk"] == cur_spk:
                interference = random.choice(buf)
            key = key + "_" + interference["key"]
            interference_idx += 1
            example["wav_spk" + str(interference_idx)] = interference["wav"]
            example["spk" + str(interference_idx)] = interference["spk"]
        example["key"] = key
        example["num_speaker"] = num_speaker
        yield example


def snr_mixer(data, use_random_snr: bool = False):
    """Dynamic mixing speakers when loading data, shuffle is not needed if this function is used.

    Args:
        data: Iterable[{key, wav_spk1, wav_spk2, ..., spk1, spk2, ...}]
        use_random_snr (bool, optional): Whether use random SNR to mix speeches. Defaults to False.

    Returns:
        Iterable[{key, wav_mix, wav_spk1, wav_spk2, ..., spk1, spk2, ...}]
    """
    for sample in data:
        assert "num_speaker" in sample.keys()
        if "wav_spk1_reverb" in sample.keys():
            suffix = "_reverb"
        else:
            suffix = ""
        num_speaker = sample["num_speaker"]
        wavs_to_mix = [sample["wav_spk1" + suffix]]
        target_energy = torch.sum(wavs_to_mix[0] ** 2, dim=-1, keepdim=True)
        for i in range(1, num_speaker):
            interference = sample[f"wav_spk{i + 1}" + suffix]
            if use_random_snr:
                snr = random.uniform(-10, 10)
            else:
                snr = 0
            energy = torch.sum(interference**2, dim=-1, keepdim=True)
            interference *= torch.sqrt(target_energy / energy) * 10 ** (
                snr / 20
            )
            wavs_to_mix.append(interference)
        wavs_to_mix = torch.stack(wavs_to_mix)
        sample["wav_mix"] = torch.sum(wavs_to_mix, 0)
        max_amp = max(
            torch.abs(sample["wav_mix"]).max().item(),
            *[x.item() for x in torch.abs(wavs_to_mix).max(dim=-1)[0]],
        )
        if max_amp != 0:
            mix_scaling = 1 / max_amp
        else:
            mix_scaling = 1

        sample["wav_mix"] = sample["wav_mix"] * mix_scaling
        for i in range(0, num_speaker):
            sample[f"wav_spk{i + 1}" + suffix] *= mix_scaling

        yield sample


def shuffle(data, shuffle_size=2500):
    """Local shuffle the data

    Args:
        data: Iterable[{key, wavs, spks}]
        shuffle_size: buffer size for shuffle

    Returns:
        Iterable[{key, wavs, spks}]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x


def spk_to_id(data, spk2id):
    """Parse spk id

    Args:
        data: Iterable[{key, wav/feat, spk}]
        spk2id: Dict[str, int]

    Returns:
        Iterable[{key, wav/feat, label}]
    """
    for sample in data:
        assert "spk" in sample
        if sample["spk"] in spk2id:
            label = spk2id[sample["spk"]]
        else:
            label = -1
        sample["label"] = label
        yield sample


def resample(data, resample_rate=16000):
    """Resample data.
    Inplace operation.
    Args:
        data: Iterable[{key, wavs, spks, sample_rate}]
        resample_rate: target resample rate
    Returns:
        Iterable[{key, wavs, spks, sample_rate}]
    """
    for sample in data:
        assert "sample_rate" in sample
        sample_rate = sample["sample_rate"]
        if sample_rate != resample_rate:
            all_keys = list(sample.keys())
            sample["sample_rate"] = resample_rate
            for key in all_keys:
                if "wav" in key:
                    waveform = sample[key]
                    sample[key] = torchaudio.transforms.Resample(
                        orig_freq=sample_rate, new_freq=resample_rate
                    )(waveform)
        yield sample


def sample_spk_embedding(data, spk_embeds):
    """sample reference speaker embeddings for the target speaker
    Args:
        data: Iterable[{key, wav, label, sample_rate}]
        spk_embeds: dict which stores all potential embeddings for the speaker
    Returns:
        Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        all_keys = list(sample.keys())
        for key in all_keys:
            if key.startswith("spk"):
                sample["embed_" + key] = random.choice(spk_embeds[sample[key]])
        yield sample


def sample_fix_spk_embedding(data, spk2embed_dict, spk1_embed, spk2_embed):
    """sample reference speaker embeddings for the target speaker
    Args:
        data: Iterable[{key, wav, label, sample_rate}]
        spk_embeds: dict which stores all potential embeddings for the speaker
    Returns:
        Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        all_keys = list(sample.keys())
        for key in all_keys:
            if key.startswith("spk"):
                if key == "spk1":
                    sample["embed_" + key] = spk2embed_dict[
                        spk1_embed[sample["key"]]
                    ]
                else:
                    sample["embed_" + key] = spk2embed_dict[
                        spk2_embed[sample["key"]]
                    ]
        yield sample


def sample_enrollment(data, spk_embeds, dict_spk):
    """sample reference speech for the target speaker
    Args:
        data: Iterable[{key, wav, label, sample_rate}]
        spk_embeds: dict which stores all potential enrollment utterance files(/.wav) for the speaker
        dict_spk: dict of speakers in the enrollment sets [Order: spkID]
    Returns:
        Iterable[{key, wav, label, sample_rate, spk_embed(raw waveform of enrollment),
                  spk_lable(when multi-task training)}]
    """
    for sample in data:
        all_keys = list(sample.keys())
        for key in all_keys:
            if key.startswith("spk"):
                enrollment, _ = sf.read(
                    random.choice(spk_embeds[sample[key]])[1]
                )
                sample["embed_" + key] = np.expand_dims(enrollment, axis=0)
                if dict_spk:
                    sample[key + "_label"] = dict_spk[sample[key]]
        yield sample


def sample_fix_spk_enrollment(
    data, spk2embed_dict, spk1_embed, spk2_embed, dict_spk=None
):
    """sample reference speaker embeddings for the target speaker
    Args:
        data: Iterable[{key, wav, label, sample_rate}]
        spk_embeds: dict which stores all potential enrollment utterance files(/.wav) for the speaker
        dict_spk: dict of speakers in the enrollment sets [Order: spkID]
    Returns:
        Iterable[{key, wav, label, sample_rate, spk_embed(raw waveform of enrollment),
                  spk_lable(when multi-task training)}]
    """
    for sample in data:
        all_keys = list(sample.keys())
        for key in all_keys:
            if key.startswith("spk"):
                if key == "spk1":
                    enrollment, _ = sf.read(
                        spk2embed_dict[spk1_embed[sample["key"]]]
                    )
                else:
                    enrollment, _ = sf.read(
                        spk2embed_dict[spk2_embed[sample["key"]]]
                    )
                sample["embed_" + key] = np.expand_dims(enrollment, axis=0)
                if dict_spk:
                    sample[key + "_label"] = dict_spk[sample[key]]
        yield sample


def compute_fbank(
    data, num_mel_bins=80, frame_length=25, frame_shift=10, dither=1.0
):
    """Extract fbank

    Args:
        data: Iterable['spk1', 'spk2', 'wav_mix', 'sample_rate', 'wav_spk1', 'wav_spk2', 'key', 'num_speaker', 'embed_spk1', 'embed_spk2']

    Returns:
        Iterable['spk1', 'spk2', 'wav_mix', 'sample_rate', 'wav_spk1', 'wav_spk2', 'key', 'num_speaker', 'embed_spk1', 'embed_spk2']
    """
    for sample in data:
        assert "sample_rate" in sample
        sample_rate = sample["sample_rate"]
        all_keys = list(sample.keys())
        for key in all_keys:
            if key.startswith("embed"):
                waveform = torch.from_numpy(sample[key])
                waveform = waveform * (1 << 15)
                mat = kaldi.fbank(
                    waveform,
                    num_mel_bins=num_mel_bins,
                    frame_length=frame_length,
                    frame_shift=frame_shift,
                    dither=dither,
                    sample_frequency=sample_rate,
                    window_type="hamming",
                    use_energy=False,
                )
                sample[key] = mat
        yield sample


def apply_cmvn(data, norm_mean=True, norm_var=False):
    """Apply CMVN

    Args:
        data: Iterable['spk1', 'spk2', 'wav_mix', 'sample_rate', 'wav_spk1', 'wav_spk2', 'key', 'num_speaker', 'embed_spk1', 'embed_spk2']

    Returns:
        Iterable['spk1', 'spk2', 'wav_mix', 'sample_rate', 'wav_spk1', 'wav_spk2', 'key', 'num_speaker', 'embed_spk1', 'embed_spk2']
    """
    for sample in data:
        all_keys = list(sample.keys())
        for key in all_keys:
            if key.startswith("embed"):
                mat = sample[key]
                if norm_mean:
                    mat = mat - torch.mean(mat, dim=0)
                if norm_var:
                    mat = mat / torch.sqrt(torch.var(mat, dim=0) + 1e-8)
                mat = mat.unsqueeze(0)
                sample[key] = mat.detach().numpy()
        yield sample


def get_random_chunk(data_list, chunk_len):
    """Get random chunk

    Args:
        data_list: [torch.Tensor: 1XT] (random len)
        chunk_len: chunk length

    Returns:
        [torch.Tensor] (exactly chunk_len)
    """
    # Assert all entries in the list share the same length
    assert False not in [len(i) == len(data_list[0]) for i in data_list]
    data_list = [data[0] for data in data_list]

    data_len = len(data_list[0])

    # random chunk
    if data_len >= chunk_len:
        chunk_start = random.randint(0, data_len - chunk_len)
        for i in range(len(data_list)):
            temp_data = data_list[i][chunk_start : chunk_start + chunk_len]
            while torch.equal(temp_data, torch.zeros_like(temp_data)):
                chunk_start = random.randint(0, data_len - chunk_len)
                temp_data = data_list[i][chunk_start : chunk_start + chunk_len]
            data_list[i] = temp_data
            # re-clone the data to avoid memory leakage
            if type(data_list[i]) == torch.Tensor:
                data_list[i] = data_list[i].clone()
            else:  # np.array
                data_list[i] = data_list[i].copy()
    else:
        # padding
        repeat_factor = chunk_len // data_len + 1
        for i in range(len(data_list)):
            if type(data_list[i]) == torch.Tensor:
                data_list[i] = data_list[i].repeat(repeat_factor)
            else:  # np.array
                data_list[i] = np.tile(data_list[i], repeat_factor)
            data_list[i] = data_list[i][:chunk_len]
    data_list = [data.unsqueeze(0) for data in data_list]
    return data_list


def filter_len(
    data,
    min_num_seconds=1,
    max_num_seconds=1000,
):
    """Filter the utterance with very short duration and random chunk the
    utterance with very long duration.

    Args:
        data: Iterable[{key, wav, label, sample_rate}]
        min_num_seconds: minimum number of seconds of wav file
        max_num_seconds: maximum number of seconds of wav file
    Returns:
        Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert "key" in sample
        assert "sample_rate" in sample
        assert "wav" in sample
        sample_rate = sample["sample_rate"]
        wav = sample["wav"]
        min_len = min_num_seconds * sample_rate
        max_len = max_num_seconds * sample_rate
        if wav.size(1) < min_len:
            continue
        elif wav.size(1) > max_len:
            wav = get_random_chunk([wav], max_len)[0]
        sample["wav"] = wav
        yield sample


def random_chunk(data, chunk_len):
    """Random chunk the data into chunk_len

    Args:
        data: Iterable[{key, wav/feat, label}]
        chunk_len: chunk length for each sample

    Returns:
        Iterable[{key, wav/feat, label}]
    """
    for sample in data:
        assert "key" in sample
        wav_keys = [key for key in list(sample.keys()) if "wav" in key]
        wav_data_list = [sample[key] for key in wav_keys]
        wav_data_list = get_random_chunk(wav_data_list, chunk_len)
        sample.update(zip(wav_keys, wav_data_list))
        yield sample


def fix_chunk(data, chunk_len):
    """Random chunk the data into chunk_len

    Args:
        data: Iterable[{key, wav/feat, label}]
        chunk_len: chunk length for each sample

    Returns:
        Iterable[{key, wav/feat, label}]
    """
    for sample in data:
        assert "key" in sample
        all_keys = list(sample.keys())
        for key in all_keys:
            if key.startswith("wav"):
                sample[key] = sample[key][:, :chunk_len]
        yield sample


def add_noise(
    data,
    noise_lmdb_file,
    noise_prob: float = 0.0,
    noise_db_low: int = -5,
    noise_db_high: int = 25,
    single_channel: bool = True,
):
    """Add noise to mixture

    Args:
        data: Iterable[{key, wav_mix, wav_spk1, wav_spk2, ..., spk1, spk2, ...}]
        noise_lmdb_file: noise LMDB data source.
        noise_db_low (int, optional): SNR lower bound. Defaults to -5.
        noise_db_high (int, optional): SNR upper bound. Defaults to 25.
        single_channel (bool, optional): Whether to force the noise file to be single channel.
                                         Defaults to True.

    Returns:
        Iterable[{key, wav_mix, wav_spk1, wav_spk2, ..., spk1, spk2, ..., noise, snr}]
    """
    noise_source = LmdbData(noise_lmdb_file)
    for sample in data:
        if noise_prob > random.random():
            assert "sample_rate" in sample.keys()
            tgt_fs = sample["sample_rate"]
            speech = sample["wav_mix"].numpy()  # [1, nsamples]
            nsamples = speech.shape[1]
            power = (speech**2).mean()
            noise_key, noise_data = noise_source.random_one()
            if noise_key.startswith(
                "speech"
            ):  # using interference speech as additive noise
                snr_range = [10, 30]
            else:
                snr_range = [noise_db_low, noise_db_high]
            noise_db = np.random.uniform(snr_range[0], snr_range[1])
            with sf.SoundFile(io.BytesIO(noise_data)) as f:
                fs = f.samplerate
                if tgt_fs and fs != tgt_fs:
                    nsamples_ = int(nsamples / tgt_fs * fs) + 1
                else:
                    nsamples_ = nsamples
                if f.frames == nsamples_:
                    noise = f.read(dtype=np.float64, always_2d=True)
                elif f.frames < nsamples_:
                    offset = np.random.randint(0, nsamples_ - f.frames)
                    # noise: (Time, Nmic)
                    noise = f.read(dtype=np.float64, always_2d=True)
                    # Repeat noise
                    noise = np.pad(
                        noise,
                        [(offset, nsamples_ - f.frames - offset), (0, 0)],
                        mode="wrap",
                    )
                else:
                    offset = np.random.randint(0, f.frames - nsamples_)
                    f.seek(offset)
                    # noise: (Time, Nmic)
                    noise = f.read(nsamples_, dtype=np.float64, always_2d=True)
                    if len(noise) != nsamples_:
                        raise RuntimeError(
                            f"Something wrong: {noise_lmdb_file}"
                        )

            if single_channel:
                num_ch = noise.shape[1]
                chs = [np.random.randint(num_ch)]
                noise = noise[:, chs]
            # noise: (Nmic, Time)
            noise = noise.T
            if tgt_fs and fs != tgt_fs:
                logging.warning(
                    f"Resampling noise to match the sampling rate ({fs} -> {tgt_fs} Hz)"
                )
                noise = librosa.resample(
                    noise, orig_sr=fs, target_sr=tgt_fs, res_type="kaiser_fast"
                )
                if noise.shape[1] < nsamples:
                    noise = np.pad(
                        noise,
                        [(0, 0), (0, nsamples - noise.shape[1])],
                        mode="wrap",
                    )
                else:
                    noise = noise[:, :nsamples]
            noise_power = (noise**2).mean()
            scale = (
                10 ** (-noise_db / 20)
                * np.sqrt(power)
                / np.sqrt(max(noise_power, 1e-10))
            )
            scaled_noise = scale * noise
            speech = speech + scaled_noise
            sample["wav_mix"] = torch.from_numpy(speech)
            sample["noise"] = torch.from_numpy(scaled_noise)
            sample["snr"] = noise_db
        yield sample


def add_reverb(data, reverb_prob=0):
    """
    Args:
        data: Iterable[{key, wav_spk1, wav_spk2, ..., spk1, spk2, ...}]

    Returns:
        Iterable[{key, wav_spk1, wav_spk2, ..., spk1, spk2, ...}]

    Note: This function is implemented with reference to
    Fast Random Appoximation of Multi-channel Room Impulse Response (FRAM-RIR)
    https://arxiv.org/pdf/2304.08052
        This function is only used when online_mixing.
    """
    for sample in data:
        assert "num_speaker" in sample.keys()
        assert "sample_rate" in sample.keys()
        simu_config["num_src"] = sample["num_speaker"]
        simu_config["sr"] = sample["sample_rate"]
        rirs, _ = RIR_sim(simu_config)  # [n_mic, nsource, nsamples]
        rirs = rirs[0]  # [nsource, nsamples]

        for i in range(sample["num_speaker"]):
            if reverb_prob > random.random():
                # [1, audio_len], currently only support single-channel audio
                audio = sample[f"wav_spk{i + 1}"].numpy()
                rir = rirs[i : i + 1, :]  # [1, nsamples]
                rir_audio = signal.convolve(audio, rir, mode="full")[
                    :, : audio.shape[1]
                ]  # [1, audio_len]

                max_scale = np.max(np.abs(rir_audio))
                out_audio = rir_audio / max_scale * 0.9
                # Note: Here, we do not replace the dry audio with the reverberant audio,
                # which means we hope the model to perform dereverberation and
                # TSE simultaneously.
                sample[f"wav_spk{i + 1}"] = torch.from_numpy(out_audio)
        yield sample


def add_noise_on_enroll(
    data,
    noise_lmdb_file,
    noise_enroll_prob: float = 0.0,
    noise_db_low: int = 0,
    noise_db_high: int = 25,
    single_channel: bool = True,
):
    """Add noise to mixture

    Args:
        data: Iterable[{key, wav_mix, wav_spk1, wav_spk2, ..., spk1, spk2, ...}]
        noise_lmdb_file: noise LMDB data source.
        noise_db_low (int, optional): SNR lower bound. Defaults to 0.
        noise_db_high (int, optional): SNR upper bound. Defaults to 25.
        single_channel (bool, optional): Whether to force the noise file to be single channel.
                                         Defaults to True.

    Returns:
        Iterable[{key, wav_mix, wav_spk1, wav_spk2, ..., spk1, spk2, ..., noise, snr}]
    """

    noise_source = LmdbData(noise_lmdb_file)
    for sample in data:
        assert "sample_rate" in sample.keys()
        tgt_fs = sample["sample_rate"]
        all_keys = list(sample.keys())
        for key in all_keys:
            if key.startswith("spk") and "label" not in key:
                if noise_enroll_prob > random.random():
                    speech = sample["embed_" + key]
                    nsamples = speech.shape[1]
                    power = (speech**2).mean()
                    noise_key, noise_data = noise_source.random_one()
                    if noise_key.startswith(
                        "speech"
                    ):  # using interference speech as additive noise
                        snr_range = [10, 30]
                    else:
                        snr_range = [noise_db_low, noise_db_high]
                    noise_db = np.random.uniform(snr_range[0], snr_range[1])
                    _, noise_data = noise_source.random_one()
                    with sf.SoundFile(io.BytesIO(noise_data)) as f:
                        fs = f.samplerate
                        if tgt_fs and fs != tgt_fs:
                            nsamples_ = int(nsamples / tgt_fs * fs) + 1
                        else:
                            nsamples_ = nsamples
                        if f.frames == nsamples_:
                            noise = f.read(dtype=np.float64, always_2d=True)
                        elif f.frames < nsamples_:
                            offset = np.random.randint(0, nsamples_ - f.frames)
                            # noise: (Time, Nmic)
                            noise = f.read(dtype=np.float64, always_2d=True)
                            # Repeat noise
                            noise = np.pad(
                                noise,
                                [
                                    (offset, nsamples_ - f.frames - offset),
                                    (0, 0),
                                ],
                                mode="wrap",
                            )
                        else:
                            offset = np.random.randint(0, f.frames - nsamples_)
                            f.seek(offset)
                            # noise: (Time, Nmic)
                            noise = f.read(
                                nsamples_, dtype=np.float64, always_2d=True
                            )
                            if len(noise) != nsamples_:
                                raise RuntimeError(
                                    f"Something wrong: {noise_lmdb_file}"
                                )

                    if single_channel:
                        num_ch = noise.shape[1]
                        chs = [np.random.randint(num_ch)]
                        noise = noise[:, chs]
                    # noise: (Nmic, Time)
                    noise = noise.T
                    if tgt_fs and fs != tgt_fs:
                        logging.warning(
                            f"Resampling noise to match the sampling rate ({fs} -> {tgt_fs} Hz)"
                        )
                        noise = librosa.resample(
                            noise,
                            orig_sr=fs,
                            target_sr=tgt_fs,
                            res_type="kaiser_fast",
                        )
                        if noise.shape[1] < nsamples:
                            noise = np.pad(
                                noise,
                                [(0, 0), (0, nsamples - noise.shape[1])],
                                mode="wrap",
                            )
                        else:
                            noise = noise[:, :nsamples]
                    noise_power = (noise**2).mean()
                    scale = (
                        10 ** (-noise_db / 20)
                        * np.sqrt(power)
                        / np.sqrt(max(noise_power, 1e-10))
                    )
                    scaled_noise = scale * noise
                    speech = speech + scaled_noise
                    sample["embed_" + key] = speech
        yield sample


def add_reverb_on_enroll(data, reverb_enroll_prob=0):
    """
    Args:
        data: Iterable[{key, wav_spk1, wav_spk2, ..., spk1, spk2, ...}]

    Returns:
        Iterable[{key, wav_spk1, wav_spk2, ..., spk1, spk2, ...}]

    """
    for sample in data:
        assert "num_speaker" in sample.keys()
        assert "sample_rate" in sample.keys()
        for i in range(sample["num_speaker"]):
            simu_config["sr"] = sample["sample_rate"]
            simu_config["num_src"] = 1
            rirs, _ = RIR_sim(simu_config)  # [n_mic, nsource, nsamples]
            rirs = rirs[0]  # [nsource, nsamples]
            if reverb_enroll_prob > random.random():
                # [1, audio_len], currently only support single-channel audio
                audio = sample[f"embed_spk{i+1}"]
                # rir = rirs[i : i + 1, :]  # [1, nsamples]
                rir = rirs
                rir_audio = signal.convolve(audio, rir, mode="full")[
                    :, : audio.shape[1]
                ]  # [1, audio_len]

                max_scale = np.max(np.abs(rir_audio))
                out_audio = rir_audio / max_scale * 0.9
                # Note: Here, we do not replace the dry audio with the reverberant audio,
                # which means we hope the model to perform dereverberation and
                # TSE simultaneously.
                sample[f"embed_spk{i+1}"] = out_audio

        yield sample


def spec_aug(data, num_t_mask=1, num_f_mask=1, max_t=10, max_f=8, prob=0):
    """Do spec augmentation
    Inplace operation

    Args:
        data: Iterable[{key, feat, label}]
        num_t_mask: number of time mask to apply
        num_f_mask: number of freq mask to apply
        max_t: max width of time mask
        max_f: max width of freq mask
        prob: prob of spec_aug

    Returns
        Iterable[{key, feat, label}]
    """
    for sample in data:
        if random.random() < prob:
            all_keys = list(sample.keys())
            for key in all_keys:
                if key.startswith("embed"):
                    y = sample[key]
                    max_frames = y.shape[1]
                    max_freq = y.shape[2]
                    # time mask
                    for i in range(num_t_mask):
                        start = random.randint(0, max_frames - 1)
                        length = random.randint(1, max_t)
                        end = min(max_frames, start + length)
                        y[:, start:end, :] = 0
                    # freq mask
                    for i in range(num_f_mask):
                        start = random.randint(0, max_freq - 1)
                        length = random.randint(1, max_f)
                        end = min(max_freq, start + length)
                        y[:, :, start:end] = 0
                    sample[key] = y
        yield sample
