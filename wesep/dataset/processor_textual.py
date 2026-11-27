# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import logging
import math
import random
import re

import numpy as np

from wesep.modules.textual.tokenizer import build_text_tokenizer
from wesep.utils.file_utils import load_json

_TEXTUAL_RESOURCE_CACHE = {}
_TEXTUAL_TOKENIZER_CACHE = {}


class EmptyTextWindowError(ValueError):
    """The current audio window contains no eligible annotated words."""


def _select_word_indices(item, sample, spk_slot, processing, generator):
    """Filter word times against the audio window, then apply word selection."""
    words = item.get("words")
    if (not isinstance(words, list) or not words
            or any(not isinstance(word, str) or not word.strip()
                   for word in words)):
        raise ValueError("Textual cues require a non-empty list of words")
    indices = list(range(len(words)))
    times = item.get("word_times")
    if times is not None:
        if not isinstance(times, list) or len(times) != len(words):
            raise ValueError("word_times must have one interval per word")
        ratio = sample.get(f"chunk_ratio_{spk_slot}",
                           sample.get("chunk_ratio"))
        if ratio is None:
            start, end = 0.0, sample["wav_mix"].shape[-1] / sample[
                "sample_rate"]
        else:
            start = round(ratio["start_ratio"] * ratio["orig_len"])
            end = round(ratio["end_ratio"] * ratio["orig_len"])
            start, end = start / sample["sample_rate"], end / sample[
                "sample_rate"]
        boundary = processing.get("boundary", "contained")
        if boundary not in ("contained", "overlap"):
            raise ValueError(f"Unsupported word boundary policy: {boundary}")
        indices = []
        for index, interval in enumerate(times):
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError("Each word interval must be [start, end]")
            left, right = map(float, interval)
            if not math.isfinite(left + right) or left < 0 or right <= left:
                raise ValueError(f"Invalid word interval: {interval}")
            keep = ((left >= start and right <= end) if boundary == "contained"
                    else (left < end and right > start))
            if keep:
                indices.append(index)
    if not indices:
        raise EmptyTextWindowError(
            "No labeled words inside the current audio window")

    selection = processing["selection"]
    mode = selection.get("type")
    if mode == "all":
        return indices
    if mode not in ("prefix", "random_span"):
        raise ValueError(f"Unsupported word selection: {mode}")
    bounds = selection.get("word_range")
    if (not isinstance(bounds, list) or len(bounds) != 2
            or any(type(value) is not int for value in bounds)
            or bounds[0] <= 0 or bounds[1] < bounds[0]):
        raise ValueError(
            "word_range must contain two positive ordered integers")
    count = min(generator.randint(*bounds), len(indices))
    first = 0 if mode == "prefix" else generator.randint(
        0,
        len(indices) - count)
    return indices[first:first + count]


def _read_word_text(item, indices, encoding):
    """Encode selected original words with the shared CPU text tokenizer."""
    cache_key = json.dumps(encoding, sort_keys=True)
    if cache_key not in _TEXTUAL_TOKENIZER_CACHE:
        _TEXTUAL_TOKENIZER_CACHE[cache_key] = build_text_tokenizer(encoding)
    tokenizer = _TEXTUAL_TOKENIZER_CACHE[cache_key]
    words = [item["words"][index] for index in indices]
    return tokenizer.encode_word_groups(words)


def _read_dae_keyword_phoneme(item, indices, encoding):
    """Select already encoded word groups without constructing a tokenizer."""
    labels = item.get("phn_label")
    if not isinstance(labels, list) or len(labels) != len(item["words"]):
        raise ValueError("phn_label must have one phoneme group per word")
    return [labels[index] for index in indices]


_TEXTUAL_READERS = {
    ("json", "word_text"): _read_word_text,
    ("json", "dae_keyword_phoneme"): _read_dae_keyword_phoneme,
}


def sample_textual_cue(data,
                       resource_path,
                       key_field,
                       cue_format,
                       processing,
                       scope="speaker",
                       required=True):
    """Attach one numeric phoneme sequence per target from either input path."""
    if scope != "speaker":
        raise ValueError("textual cue currently supports scope='speaker'")
    if cue_format not in _TEXTUAL_READERS:
        raise ValueError(f"Unsupported textual cue format: {cue_format}")
    if processing["encoding"].get("type") != "dae_phoneme":
        raise ValueError(
            "These textual paths currently require dae_phoneme encoding")
    if resource_path not in _TEXTUAL_RESOURCE_CACHE:
        _TEXTUAL_RESOURCE_CACHE[resource_path] = load_json(resource_path)
    resource = _TEXTUAL_RESOURCE_CACHE[resource_path]
    reader = _TEXTUAL_READERS[cue_format]

    for sample in data:
        slots = [key for key in sample if re.fullmatch(r"spk\d+", key)]
        slots.sort(key=lambda key: int(key[3:]))
        if not slots and required:
            raise KeyError("sample has no speaker slots")
        skip_sample = False
        for slot in slots:
            if key_field == "spk_id":
                key = sample[slot]
            elif key_field == "mix_spk_id":
                mix_key = sample.get(f"source_key_{slot}", sample.get("key"))
                if mix_key is None:
                    raise KeyError("sample missing key for textual cue lookup")
                key = f"{mix_key}::{sample[slot]}"
            else:
                raise ValueError(f"Unsupported textual key field: {key_field}")
            if key not in resource:
                if required:
                    raise KeyError(f"Textual cue not found: {key}")
                continue
            try:
                # A local seed fixes word count and position separately per target.
                generator = random
                seed = processing["selection"].get("seed")
                if seed is not None:
                    seed = f"{seed}:{key}".encode()
                    generator = random.Random(
                        int.from_bytes(hashlib.sha256(seed).digest(), "big"))
                item = resource[key]
                indices = _select_word_indices(item, sample, slot, processing,
                                               generator)
                groups = reader(item, indices, processing["encoding"])
                if any(not isinstance(group, list) or not group
                       for group in groups):
                    raise ValueError(
                        "Selected words require non-empty phoneme groups")
                ids = [value for group in groups for value in group]
                if any(
                        type(value) is not int or not 1 <= value <= 70
                        for value in ids):
                    raise ValueError("DAE word IDs must be 1..70; "
                                     "PAD/BOS/EOS belong outside word groups")
                sample[f"textual_{slot}"] = np.asarray(ids, dtype=np.int64)
            except EmptyTextWindowError as error:
                # Timed silence windows have no usable text cue for this target.
                if processing.get("skip_empty", False):
                    logging.warning("Skipping mix %s: no timed words for %s",
                                    sample.get("key"), slot)
                    skip_sample = True
                    break
                logging.warning("Failed to read textual cue %s: %s", key,
                                error)
                if required:
                    raise
            except Exception as error:
                logging.warning("Failed to read textual cue %s: %s", key,
                                error)
                if required:
                    raise
        if not skip_sample:
            yield sample
