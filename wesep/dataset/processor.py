import io
import json
import logging
import random
import re
import tarfile
from itertools import islice
from subprocess import PIPE, Popen
from urllib.parse import urlparse

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy import signal

from wesep.dataset.FRAM_RIR import single_channel as RIR_sim
from wesep.dataset.lmdb_data import LmdbData
from wesep.dataset.timeline import sample_num_speakers, timeline_generator, parse_timeline, parse_overlap_ratio

AUDIO_FORMAT_SETS = {"flac", "mp3", "m4a", "ogg", "opus", "wav", "wma"}


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


def _parse_shard_member(name):
    """Parse a shard member name generated from samples.jsonl.

    Returns:
        tuple(sample_key, kind, index, suffix) or None.
        kind is one of "spk", "mix", "src".
    """
    pos = name.rfind(".")
    if pos <= 0:
        return None

    prefix, postfix = name[:pos], name[pos + 1:]

    if postfix.startswith("spk") and postfix[3:].isdigit():
        return prefix, "spk", int(postfix[3:]), None

    if postfix not in AUDIO_FORMAT_SETS:
        return None

    spk_pos = prefix.rfind("_spk")
    if spk_pos >= 0 and prefix[spk_pos + 4:].isdigit():
        sample_key = prefix[:spk_pos]
        spk_idx = int(prefix[spk_pos + 4:])
        return sample_key, "src", spk_idx, postfix

    return prefix, "mix", None, postfix


def _standardize_source_sample(
    key,
    spk_ids,
    mix_items,
    src_items,
    source_len_policy="strict",
):
    """Convert loaded source fields into the standard non-online sample."""
    if source_len_policy not in ("strict", "trim_min"):
        raise ValueError(f"Unsupported source_len_policy: {source_len_policy}")

    # Validate and normalize speaker slots.
    if isinstance(spk_ids, dict):
        spk_map = dict(spk_ids)
    else:
        spk_map = dict(enumerate(spk_ids, start=1))
    if not spk_map:
        raise RuntimeError(f"sample has no speaker ids: {key}")

    num_speaker = len(spk_map)
    expected_spk_ids = list(range(1, num_speaker + 1))
    if sorted(spk_map.keys()) != expected_spk_ids:
        raise RuntimeError(f"non-contiguous speaker ids in sample {key}: "
                           f"{sorted(spk_map.keys())}")

    if not mix_items:
        raise RuntimeError(f"sample has no mix wav: {key}")

    # Normalize mixture items and concatenate all channels.
    wav_list = []
    sample_rate = None
    for item_idx, (waveform, sr, name) in enumerate(mix_items):
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() != 2:
            raise RuntimeError(
                f"Unsupported number of channels in mix wav: {name}, "
                f"shape={tuple(waveform.shape)}")

        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise RuntimeError(
                f"Sample rate mismatch in {key}: "
                f"mix={sample_rate}, item{item_idx}={sr}, name={name}")

        wav_list.append(waveform)

    mix_lens = [wav.size(-1) for wav in wav_list]
    if len(set(mix_lens)) != 1:
        if source_len_policy == "strict":
            raise RuntimeError(
                f"mix waveforms have different lengths in {key}: {mix_lens}")
        min_mix_len = min(mix_lens)
        wav_list = [wav[..., :min_mix_len] for wav in wav_list]
    wav_mix = torch.cat(wav_list, dim=0)

    example = {
        "key": key,
        "num_speaker": num_speaker,
        "wav_mix": wav_mix,
        "sample_rate": sample_rate,
    }

    # Normalize target sources; use the first source item and first channel.
    src_wavs = {}
    for i in expected_spk_ids:
        if i not in src_items or not src_items[i]:
            raise RuntimeError(f"missing source wav for spk{i} in {key}")

        waveform, sr, name = src_items[i][0]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() != 2:
            raise RuntimeError(
                f"Unsupported number of channels in source wav: {name}, "
                f"shape={tuple(waveform.shape)}")

        if sr != sample_rate:
            raise RuntimeError(f"Sample rate mismatch in {key}: "
                               f"mix={sample_rate}, src={sr}, name={name}")

        example[f"spk{i}"] = spk_map[i]
        src_wavs[f"wav_spk{i}"] = waveform[:1, :]

    # Verify mixture and target lengths share the same time axis.
    wav_lens = {"wav_mix": example["wav_mix"].size(-1)}
    wav_lens.update(
        {wav_key: wav.size(-1)
         for wav_key, wav in src_wavs.items()})
    if len(set(wav_lens.values())) != 1:
        if source_len_policy == "strict":
            raise RuntimeError(
                f"wav fields are not time-aligned in sample {key}: {wav_lens}")
        min_len = min(wav_lens.values())
        example["wav_mix"] = example["wav_mix"][..., :min_len]
        src_wavs = {
            wav_key: wav[..., :min_len]
            for wav_key, wav in src_wavs.items()
        }

    # Pack the final sample dictionary.
    example.update(src_wavs)
    return example


def tar_file_and_group(data, source_len_policy="strict"):
    """Expand a stream of open tar files into a stream of tar file contents.
    And groups the file with same prefix

    Args:
        data: Iterable[{src, stream}]

    Returns:
        Iterable[{key, wav_mix, wav_spk1, wav_spk2, ..., sample_rate}]
    """
    for sample in data:
        stream = tarfile.open(fileobj=sample["stream"], mode="r:*")

        # Current sample state. Shards generated by make_shards_from_samples.py
        # store all members of one sample contiguously.
        cur_key = None
        spk_ids = {}
        mix_items = []
        src_items = {}
        # If one member fails to parse, skip this sample when reaching its end.
        valid = True

        for tarinfo in stream:
            name = tarinfo.name
            # Map names like key.spk1, key.wav, key_spk1.wav to one sample key
            # and one field kind.
            parsed = _parse_shard_member(name)
            if parsed is None:
                continue

            sample_key, kind, idx, _ = parsed

            # A new key means the previous sample is complete and can be
            # normalized into the same output structure as parse_raw.
            if cur_key is not None and sample_key != cur_key:
                if valid:
                    yield _standardize_source_sample(
                        cur_key,
                        spk_ids,
                        mix_items,
                        src_items,
                        source_len_policy,
                    )

                cur_key = None
                spk_ids = {}
                mix_items = []
                src_items = {}
                valid = True

            if cur_key is None:
                cur_key = sample_key

            try:
                file_obj = stream.extractfile(tarinfo)
                if file_obj is None:
                    continue

                with file_obj as f:
                    # The loop only classifies and loads fields. Channel,
                    # length, and sample-rate normalization happens in finalize.
                    if kind == "spk":
                        spk_ids[idx] = f.read().decode("utf8").strip()
                    else:
                        waveform, sr = torchaudio.load(f)
                        item = (waveform, sr, name)
                        if kind == "mix":
                            mix_items.append(item)
                        else:
                            src_items.setdefault(idx, []).append(item)

            except Exception as ex:
                valid = False
                logging.warning(f"Failed to parse {name}: {ex}")

        if cur_key is not None and valid:
            yield _standardize_source_sample(
                cur_key,
                spk_ids,
                mix_items,
                src_items,
                source_len_policy,
            )

        stream.close()
        if "process" in sample:
            sample["process"].communicate()
        sample["stream"].close()


def tar_file_and_group_single_spk(data):
    """Read online single-speaker shards.

    Args:
        data: Iterable[{src, stream}]

    Returns:
        Iterable[{key, wav: Tensor [1, T], spk, sample_rate}]
    """
    for sample in data:
        assert "stream" in sample
        stream = tarfile.open(fileobj=sample["stream"], mode="r:*")
        prev_prefix = None
        example = {}
        valid = True
        for tarinfo in stream:
            name = tarinfo.name
            pos = name.rfind(".")
            assert pos > 0
            prefix, postfix = name[:pos], name[pos + 1:]
            if prev_prefix is not None and prefix != prev_prefix:
                example["key"] = prev_prefix
                required = {"spk", "wav", "sample_rate"}
                if valid and required.issubset(example):
                    yield example
                else:
                    logging.warning(
                        f"Invalid online shard sample: {prev_prefix}")
                example = {}
                valid = True
            with stream.extractfile(tarinfo) as file_obj:
                try:
                    if postfix in ["spk"]:
                        example[postfix] = (
                            file_obj.read().decode("utf8").strip())
                    elif postfix in AUDIO_FORMAT_SETS:
                        waveform, sample_rate = torchaudio.load(file_obj)
                        if waveform.size(0) != 1:
                            raise ValueError(
                                f"Online mixing requires mono audio: {name}, "
                                f"shape={tuple(waveform.shape)}")
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
            required = {"spk", "wav", "sample_rate"}
            if valid and required.issubset(example):
                yield example
            else:
                logging.warning(f"Invalid online shard sample: {prev_prefix}")
        stream.close()
        if "process" in sample:
            sample["process"].communicate()
        sample["stream"].close()


def parse_raw(data, source_len_policy="strict"):
    """Parse samples.jsonl line into wav tensors.

    Args:
        data: Iterable[dict], each item like:
          {
            "src": "<json string>",
            "rank": ...,
            "world_size": ...,
            ...
          }

    Yields:
        dict with fields aligned to shards:
          {
            key,
            num_speaker,
            spk1, spk2, ...
            wav_mix,
            wav_spk1, wav_spk2, ...
            sample_rate
          }
    """
    for sample in data:
        assert "src" in sample
        json_line = sample["src"]

        try:
            obj = json.loads(json_line)
        except Exception:
            logging.warning(f"Bad json line: {json_line}")
            continue

        try:
            key = obj["key"]
            spk_ids = obj["spk"]
            mix_dict = obj["mix"]
            src_dict = obj["src"]

            # Read mixture files; normalization happens in standardization.
            mix_items = []
            mix_paths = mix_dict.get("default", [])
            if not mix_paths:
                raise RuntimeError(f"No mix path for sample {key}")
            for mix_path in mix_paths:
                waveform, sr = torchaudio.load(mix_path)
                mix_items.append((waveform, sr, mix_path))

            # Read the first target file for each speaker.
            src_items = {}
            for i, spk_id in enumerate(spk_ids, start=1):
                if spk_id not in src_dict or not src_dict[spk_id]:
                    raise RuntimeError(
                        f"No src path for speaker {spk_id} in sample {key}")
                src_path = src_dict[spk_id][0]
                waveform, sr = torchaudio.load(src_path)
                src_items[i] = [(waveform, sr, src_path)]

        except Exception as ex:
            logging.warning(f"Failed to parse raw sample: {ex}")
            continue

        # Standardization errors describe a dataset contract violation.
        yield _standardize_source_sample(
            key,
            spk_ids,
            mix_items,
            src_items,
            source_len_policy,
        )


def expand_target_samples(data):
    """Expand each mixture into target-level samples with one speaker slot."""
    speaker_key = re.compile(r"^(.*spk)(\d+)(.*)$")

    for sample in data:
        num_speaker = int(sample["num_speaker"])

        # Keep mix-level fields and remap one target's fields to speaker slot 1.
        common = {
            key: value
            for key, value in sample.items()
            if speaker_key.fullmatch(key) is None
        }
        for target_slot in range(1, num_speaker + 1):
            target = dict(common)
            target["num_speaker"] = 1
            target["target_slot"] = target_slot
            for key, value in sample.items():
                match = speaker_key.fullmatch(key)
                if match is not None and int(match.group(2)) == target_slot:
                    target[f"{match.group(1)}1{match.group(3)}"] = value
            yield target


def add_speaker_labels(data, spk2label):
    """Attach zero-based class labels to every speaker slot."""
    for sample in data:
        for i in range(1, int(sample["num_speaker"]) + 1):
            sample[f"spk{i}_label"] = torch.tensor(
                spk2label[sample[f"spk{i}"]], dtype=torch.long)
        yield sample


def parse_raw_single_spk(data):
    """
    Parse raw single-speaker samples for online mix.

    Input sample schema (from samples.jsonl):
    {
      "key": "...",
      "spk": ["id10001"],
      "src": {
        "id10001": [".../00001.wav"]
      }
    }

    Yields:
    {
      "key": str,
      "spk": str,
      "wav": Tensor [1, T],
      "sample_rate": int
    }
    """

    for sample in data:
        # ---- FIX: decode samples.jsonl line if needed ----
        if "spk" not in sample:
            if "src" in sample and isinstance(sample["src"], str):
                sample = json.loads(sample["src"])
            else:
                raise ValueError(
                    f"Unexpected sample format: keys={sample.keys()}")
        # -------- sanity checks --------
        spk_list = sample.get("spk", [])
        if len(spk_list) != 1:
            raise ValueError(
                f"parse_raw_single_spk expects single speaker, "
                f"got {len(spk_list)} in sample {sample.get('key')}")

        spk = spk_list[0]

        src_map = sample.get("src", {})
        if spk not in src_map:
            raise KeyError(
                f"Speaker {spk} missing in src map for sample {sample.get('key')}"
            )

        wav_list = src_map[spk]

        # -------- explicitly forbid multi-audio (future multi-channel) --------
        if len(wav_list) != 1:
            raise NotImplementedError(
                f"Multiple audio files per speaker are not supported yet "
                f"(got {len(wav_list)}) in sample {sample.get('key')}")

        wav_path = wav_list[0]

        # -------- load audio --------
        try:
            wav_ch, sr = torchaudio.load(wav_path)  # (C, T) or (T,)
        except Exception:
            logging.warning(f"Failed to read wav: {wav_path}")
            continue

        # -------- normalize shape to [1, T] --------
        if wav_ch.dim() == 1:
            wav = wav_ch.unsqueeze(0)
        else:
            if wav_ch.size(0) != 1:
                raise NotImplementedError(
                    f"Multi-channel wav is not supported yet: "
                    f"{wav_path}, shape={tuple(wav_ch.shape)}")
            wav = wav_ch

        yield {
            "key": sample["key"],
            "spk": spk,
            "wav": wav,  # [1, T]
            "sample_rate": sr,
        }


def sample_speaker_group(data,
                         num_speakers=None,
                         shuffle_size=1000,
                         timeline_conf=None,
                         whole_utt=False,
                         rng=random):
    """Group single-speaker samples for online mixing.

    Args:
        data: Iterable[{key, spk, wav, sample_rate, chunk_ratio?}]
        num_speakers: Speaker-count sampling configuration.
        shuffle_size: Number of source samples buffered for grouping.
        timeline_conf: Relative activity-pattern configuration.
        whole_utt: Preserve complete sources and pad them to the group maximum.

    Returns:
        Iterable[{key, num_speaker, spk{i}, source_key_spk{i},
                  wav_spk{i}, chunk_ratio_spk{i}?, timeline_spk{i},
                  sample_rate}]
    """
    assert num_speakers is not None
    if shuffle_size <= 0:
        raise ValueError("online_buffer_size must be positive")

    source = iter(data)
    while True:
        buf = list(islice(source, shuffle_size))
        if not buf:
            break
        rng.shuffle(buf)

        # Group the current buffer by speaker for efficient unique sampling.
        speaker_pool = {}
        for candidate in buf:
            spk = candidate["spk"]
            if spk not in speaker_pool:
                speaker_pool[spk] = []
            speaker_pool[spk].append(candidate)
        speaker_ids = list(speaker_pool)
        speaker_index = {spk: index for index, spk in enumerate(speaker_ids)}

        for x in buf:
            num_speaker = sample_num_speakers(num_speakers, rng)

            # Sample different interference speakers while skipping the target.
            target_index = speaker_index[x["spk"]]
            available = len(speaker_ids) - 1
            num_speaker = min(num_speaker, available + 1)
            selected_indices = rng.sample(range(available), num_speaker - 1)
            selected_spks = [
                speaker_ids[index if index < target_index else index + 1]
                for index in selected_indices
            ]

            # Generate the acoustic activity pattern for the selected speakers.
            if timeline_conf is not None:
                timeline, overlap_ratio = timeline_generator(
                    timeline_conf, num_speaker, rng)
            else:
                timeline = [{
                    "speaker": i,
                    "start": 0.0,
                    "end": 1.0
                } for i in range(num_speaker)]
                overlap_ratio = {"overlap_ratio": 1.0}

            # Initialize the online sample with the current target utterance.
            example = {
                "key": x["key"],
                "wav_spk1": x["wav"],
                "spk1": x["spk"],
                "source_key_spk1": x["key"],
                "sample_rate": x["sample_rate"],
                "num_speaker": num_speaker,
                "overlap_ratio_2spk": parse_overlap_ratio(overlap_ratio),
            }
            if "chunk_ratio" in x:
                example["chunk_ratio_spk1"] = x["chunk_ratio"]
            example["timeline_spk1"] = parse_timeline(
                [t for t in timeline if t["speaker"] == 0])

            # Attach one utterance for every sampled interference speaker.
            key = "mix_" + x["key"]
            for interference_idx, spk in enumerate(selected_spks, start=2):
                interference = rng.choice(speaker_pool[spk])
                key = key + "_" + interference["key"]
                example[f"timeline_spk{interference_idx}"] = parse_timeline([
                    t for t in timeline if t["speaker"] == interference_idx - 1
                ])
                example[f"wav_spk{interference_idx}"] = interference["wav"]
                example[f"spk{interference_idx}"] = interference["spk"]
                example[f"source_key_spk{interference_idx}"] = interference[
                    "key"]
                if "chunk_ratio" in interference:
                    example[f"chunk_ratio_spk{interference_idx}"] = (
                        interference["chunk_ratio"])

            # Whole-utterance sources share a right-padded group timeline.
            if whole_utt:
                max_len = max(example[f"wav_spk{i}"].size(-1)
                              for i in range(1, num_speaker + 1))
                for i in range(1, num_speaker + 1):
                    wav_key = f"wav_spk{i}"
                    source_len = example[wav_key].size(-1)
                    if source_len < max_len:
                        pad = torch.zeros(
                            example[wav_key].size(0),
                            max_len - source_len,
                            dtype=example[wav_key].dtype,
                            device=example[wav_key].device,
                        )
                        example[wav_key] = torch.cat([example[wav_key], pad],
                                                     dim=-1)
                    example[f"chunk_ratio_spk{i}"] = {
                        "start_ratio": 0.0,
                        "end_ratio": max_len / source_len,
                        "orig_len": source_len,
                        "chunk_len": max_len,
                    }

            example["key"] = key
            yield example


def apply_timeline(data):
    """
    Apply activity masks to online-mixed speaker waveforms.

    Args:
        data: Iterable[example], where example contains:
              wav_spk{i}: Tensor [1, T]
              timeline_spk{i}: list of [start, end] in [0,1]
              num_speaker: int

    Yields:
        example with wav_spk{i} masked by timeline

    Timeline only controls acoustic activity. Cue processors keep the
    source-aligned cue unchanged.
    """
    for sample in data:
        K = sample["num_speaker"]

        for i in range(1, K + 1):
            wav_key = f"wav_spk{i}"
            tl_key = f"timeline_spk{i}"

            wav = sample[wav_key]  # [1, T]
            timeline = sample[tl_key]  # list of [s, e]

            assert wav.dim() == 2 and wav.size(0) == 1, \
                f"{wav_key} must be [1, T]"

            T = wav.size(1)
            device = wav.device

            # Build one activity mask from the normalized timeline segments.
            mask = torch.zeros(T, device=device)

            for seg in timeline:
                s, e = seg
                s = max(0.0, min(1.0, float(s)))
                e = max(0.0, min(1.0, float(e)))
                if e <= s:
                    continue

                start = int(round(s * T))
                end = int(round(e * T))
                start = max(0, min(T, start))
                end = max(0, min(T, end))

                if end > start:
                    mask[start:end] = 1.0

            # Apply the activity pattern without changing waveform length.
            wav = wav * mask.unsqueeze(0)  # [1, T]
            sample[wav_key] = wav
        yield sample


def snr_mixer(data, snr_conf=None, rng=random):
    """Mix mono online sources with sampled target-to-interference ratios.

    Args:
        data: Iterable[{key, wav_spk1: [1, T], wav_spk2: [1, T], ...}]
        snr_conf:
            range: range of target-to-interference ratio in dB (after reverb)
            gain: adjust the overall energy of mix, spk1, spk2.

    Returns:
        Iterable[{key, wav_mix, wav_spk1, wav_spk2, ..., spk1, spk2, ...}]
    """
    for sample in data:
        assert "num_speaker" in sample.keys()
        if snr_conf is None:
            snr_conf = {
                "range": [-5, 10],
                "gain": [-12, 0],
            }
        if "wav_spk1_reverb" in sample.keys():
            suffix = "_reverb"  # Reserved when maintian the dry wav
        else:
            suffix = ""
        num_speaker = sample["num_speaker"]
        source_keys = [f"wav_spk{i + 1}{suffix}" for i in range(num_speaker)]
        sources = [sample[key] for key in source_keys]
        if any(wav.dim() != 2 or wav.size(0) != 1 for wav in sources):
            raise ValueError("Online SNR mixing currently requires [1, T] "
                             f"sources: {sample['key']}")

        energies = [torch.sum(wav**2) for wav in sources]
        if any(energy.item() <= 1e-12 for energy in energies):
            logging.warning("Skip zero-energy online mixture: %s",
                            sample["key"])
            continue

        wavs_to_mix = [sources[0]]
        target_energy = energies[0]
        for i in range(1, num_speaker):
            snr = rng.uniform(*snr_conf["range"])
            interference = sources[i] * torch.sqrt(
                target_energy / energies[i]) * 10**(-snr / 20)
            sample[source_keys[i]] = interference
            wavs_to_mix.append(interference)
        wavs_to_mix = torch.stack(wavs_to_mix)
        sample["wav_mix"] = torch.sum(wavs_to_mix, 0)

        # ---------- Peak normalization ----------
        max_amp = max(sample["wav_mix"].abs().max().item(),
                      *(wav.abs().max().item() for wav in wavs_to_mix))
        if max_amp > 0:
            peak_scale = 1.0 / max_amp
        else:
            peak_scale = 1.0
        sample["wav_mix"] *= peak_scale
        for i in range(num_speaker):
            sample[f"wav_spk{i + 1}{suffix}"] *= peak_scale
        # ---------- Random global gain (after peak norm) ----------
        if snr_conf.get("gain", None) is not None:
            gain_db = rng.uniform(*snr_conf["gain"])  # e.g. [-12, 0]
            gain = 10**(gain_db / 20)
            sample["wav_mix"] *= gain
            for i in range(num_speaker):
                sample[f"wav_spk{i + 1}{suffix}"] *= gain
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
            # Release buffer ownership before downstream mutation.
            for i in range(len(buf)):
                x = buf[i]
                buf[i] = None
                yield x
            buf = []
    # The sample left over
    random.shuffle(buf)
    # Release leftover samples using the same ownership rule.
    for i in range(len(buf)):
        x = buf[i]
        buf[i] = None
        yield x


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
                        orig_freq=sample_rate,
                        new_freq=resample_rate)(waveform)
        yield sample


def get_random_chunk(data_list,
                     chunk_len,
                     random_start=True,
                     short_policy="pad",
                     mix_index=None,
                     max_zero_retry=3):
    """
    Args:
        data_list: list[Tensor], shapes like mix/targets: [C, T] or [..., T]
        chunk_len: int
        random_start: whether to randomly select the chunk start. If False,
            crop from the beginning.
        short_policy: how to handle an input shorter than chunk_len. ``pad``
            right-pads it to chunk_len; ``keep`` preserves its true length.
        mix_index: index of the mixture tensor in data_list for all-zero
            random chunk retry. Set to None to disable this check.
        max_zero_retry: maximum number of retries when the selected mixture
            chunk is all zero.

    Returns:
        list[Tensor], same leading dims, last dim = chunk_len
    """
    chunk_len = int(chunk_len)
    if len(data_list) == 0:
        raise ValueError("get_random_chunk requires at least one tensor")
    if chunk_len <= 0:
        raise ValueError(f"chunk_len must be positive, got {chunk_len}")
    if short_policy not in ("pad", "keep"):
        raise ValueError("short_policy must be 'pad' or 'keep', "
                         f"got {short_policy!r}")
    if mix_index is not None and not 0 <= mix_index < len(data_list):
        raise ValueError(
            f"mix_index out of range: {mix_index}, len={len(data_list)}")

    for idx, d in enumerate(data_list):
        if d.dim() < 2:
            raise ValueError(
                f"Expected tensor with at least 2 dims, got {d.dim()} "
                f"at data_list[{idx}]")

    # Check the time length and get T.
    T = data_list[0].size(-1)
    if T <= 0:
        raise ValueError("Input tensors must have non-zero time length")
    assert all(d.size(-1) == T for d in data_list)

    # Random/fixed crop if possible.
    if T >= chunk_len:
        if random_start:
            chunk_start = random.randint(0, T - chunk_len)
            if mix_index is not None:
                for retry_idx in range(max_zero_retry):
                    mix_chunk = data_list[mix_index][...,
                                                     chunk_start:chunk_start +
                                                     chunk_len]
                    if not torch.all(mix_chunk == 0):
                        break
                    chunk_start = random.randint(0, T - chunk_len)
                mix_chunk = data_list[mix_index][..., chunk_start:chunk_start +
                                                 chunk_len]
                if torch.all(mix_chunk == 0):
                    logging.warning(
                        "Selected all-zero mixture chunk after %d retries.",
                        max_zero_retry,
                    )
        else:
            chunk_start = 0

        out = []
        for d in data_list:
            chunk = d[..., chunk_start:chunk_start + chunk_len]
            out.append(chunk.clone())

        meta = {
            "start_ratio": chunk_start / T,
            "end_ratio": (chunk_start + chunk_len) / T,
            "orig_len": T,
            "chunk_len": chunk_len,
        }
        return out, meta

    # Preserve a short utterance and expose its full valid time range.
    if short_policy == "keep":
        return [d.clone() for d in data_list], {
            "start_ratio": 0.0,
            "end_ratio": 1.0,
            "orig_len": T,
            "chunk_len": T,
        }

    # Pad short utterances with zeros.
    out = []
    for d in data_list:
        pad_shape = list(d.shape)
        pad_shape[-1] = chunk_len - T
        pad = torch.zeros(*pad_shape, dtype=d.dtype, device=d.device)
        out.append(torch.cat([d, pad], dim=-1))

    meta = {
        "start_ratio": 0.0,
        "end_ratio": chunk_len / T,  # > 1.0, indicating padded context
        "orig_len": T,
        "chunk_len": chunk_len,
    }

    return out, meta


def filter_len(
    data,
    min_num_seconds=1,
    max_num_seconds=1000,
):
    """Filter utterances by duration without changing waveform timing.

    Args:
        data: Iterable[{key, wav/wav_mix/wav_spk*, sample_rate}]
        min_num_seconds: minimum number of seconds of wav file
        max_num_seconds: maximum number of seconds of wav file
    Returns:
        Iterable[{key, wav/wav_mix/wav_spk*, sample_rate}]
    """
    for sample in data:
        assert "key" in sample
        assert "sample_rate" in sample
        sample_rate = sample["sample_rate"]

        wav_keys = [
            key for key in list(sample.keys()) if key.startswith("wav")
        ]
        if not wav_keys:
            raise KeyError(f"sample has no wav fields: {sample['key']}")

        utt_len = sample[wav_keys[0]].size(-1)
        for key in wav_keys[1:]:
            if sample[key].size(-1) != utt_len:
                raise RuntimeError(
                    f"wav fields are not time-aligned in sample {sample['key']}: "
                    f"{ {k: sample[k].size(-1) for k in wav_keys} }")

        min_len = int(round(float(min_num_seconds) * sample_rate))
        max_len = int(round(float(max_num_seconds) * sample_rate))

        if utt_len < min_len:
            continue

        if utt_len > max_len:
            continue

        yield sample


def random_chunk(data, chunk_len, random_start=True, short_policy="pad"):
    """Random chunk the data into chunk_len

    Args:
        data: Iterable[{key, wav/feat, label}]
        chunk_len: chunk length for each sample
        random_start: randomly select the chunk start when true; otherwise
            always crop from the beginning.
        short_policy: ``pad`` short samples to chunk_len or ``keep`` their
            true length.

    Returns:
        Iterable[{key, wav/feat, label}]
    """
    for sample in data:
        assert "key" in sample
        wav_keys = [key for key in list(sample.keys()) if "wav" in key]
        wav_data_list = [sample[key] for key in wav_keys]
        mix_index = (wav_keys.index("wav_mix")
                     if "wav_mix" in wav_keys else None)
        wav_data_list, ratio = get_random_chunk(
            wav_data_list,
            chunk_len,
            random_start=random_start,
            short_policy=short_policy,
            mix_index=mix_index,
        )
        sample.update(zip(wav_keys, wav_data_list))
        sample["chunk_ratio"] = ratio
        yield sample


def add_noise(
    data,
    noise_lmdb_file,
    noise_prob: float = 0.0,
    noise_db_low: int = -5,
    noise_db_high: int = 25,
    single_channel: bool = True,
):
    """Add noise to mixture (Only for 1 channel mixture)

    Args:
        data: Iterable[{key, wav_mix, wav_spk1, wav_spk2, ..., spk1, spk2, ...}]
        noise_lmdb_file: noise LMDB data source.
        noise_db_low (int, optional): SNR lower bound. Defaults to -5.
        noise_db_high (int, optional): SNR upper bound. Defaults to 25.
        single_channel (bool, optional): Whether to force the noise file to be single channel.  # noqa
                                         Defaults to True.

    Returns:
        Iterable[{key, wav_mix, wav_spk1, wav_spk2, ..., spk1, spk2, ..., noise, snr}]  # noqa
    """
    noise_source = LmdbData(noise_lmdb_file)
    for sample in data:
        if noise_prob > random.random():
            assert "sample_rate" in sample.keys()
            if sample["wav_mix"].dim() != 2 or sample["wav_mix"].size(0) != 1:
                raise ValueError(
                    "Noise augmentation currently requires wav_mix [1, T]")
            tgt_fs = sample["sample_rate"]
            speech = sample["wav_mix"].numpy()  # [1, nsamples]
            nsamples = speech.shape[1]
            power = (speech**2).mean()
            noise_key, noise_data = noise_source.random_one()
            if noise_key.startswith(
                    "speech"):  # using interference speech as additive noise
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
                            f"Something wrong: {noise_lmdb_file}")

            if single_channel:
                num_ch = noise.shape[1]
                chs = [np.random.randint(num_ch)]
                noise = noise[:, chs]
            # noise: (Nmic, Time)
            noise = noise.T
            if tgt_fs and fs != tgt_fs:
                logging.warning(
                    f"Resampling noise to match the sampling rate ({fs} -> {tgt_fs} Hz)"  # noqa
                )
                noise = librosa.resample(noise,
                                         orig_sr=fs,
                                         target_sr=tgt_fs,
                                         res_type="kaiser_fast")
                if noise.shape[1] < nsamples:
                    noise = np.pad(
                        noise,
                        [(0, 0), (0, nsamples - noise.shape[1])],
                        mode="wrap",
                    )
                else:
                    noise = noise[:, :nsamples]
            noise_power = (noise**2).mean()
            scale = (10**(-noise_db / 20) * np.sqrt(power) /
                     np.sqrt(max(noise_power, 1e-10)))
            scaled_noise = scale * noise
            speech = speech + scaled_noise
            dtype = sample["wav_mix"].dtype
            sample["wav_mix"] = torch.as_tensor(speech, dtype=dtype)
            sample["noise"] = torch.as_tensor(scaled_noise, dtype=dtype)
            sample["snr"] = noise_db
        yield sample


def add_reverb(data, reverb_prob=0, reverb_conf=None, rng=random):
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
    if reverb_prob == 0:
        yield from data
        return

    if reverb_conf is None:
        # set the default simulation configuration
        reverb_conf = {
            "min_max_room": [[3, 3, 2.5], [10, 6, 4]],
            "rt60": [0.1, 0.7],
            "mic_dist": [0.2, 5.0],
        }

    for sample in data:
        apply = rng.random() < reverb_prob
        if not apply:
            yield sample
            continue

        assert "num_speaker" in sample.keys()
        assert "sample_rate" in sample.keys()

        rir_conf = dict(reverb_conf)
        rir_conf["num_src"] = sample["num_speaker"]
        rir_conf["sr"] = sample["sample_rate"]

        rirs, _ = RIR_sim(rir_conf)
        rirs = rirs[0]  # [num_speaker, rir_len]

        # Render every source in the same simulated single-channel room.
        for i in range(sample["num_speaker"]):
            wav_key = f"wav_spk{i+1}"
            wav = sample[wav_key]
            if wav.dim() != 2 or wav.size(0) != 1:
                raise ValueError(
                    "Online reverb currently requires [1, T] sources")
            audio = wav.numpy()
            rir = rirs[i:i + 1]
            out = signal.convolve(audio, rir, mode="full")[:, :audio.shape[1]]
            sample[wav_key] = torch.as_tensor(out, dtype=wav.dtype)

        yield sample
