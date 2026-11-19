# Copyright (c) 2025 Ke Zhang (kylezhang1118@gmail.com)
#
# SPDX-License-Identifier: Apache-2.0

import logging
import re

import av
import numpy as np
import torch

from wesep.utils.file_utils import load_json


def _audio_seconds(sample, spk_slot):
    ratio = sample.get(f"chunk_ratio_{spk_slot}",
                       sample.get("chunk_ratio", None))
    if ratio is not None:
        return ratio["orig_len"] / sample["sample_rate"]
    return sample["wav_mix"].shape[-1] / sample["sample_rate"]


def _align_last_dim_to_audio_duration(x, duration_sec, sample, spk_slot):
    """Align a precomputed temporal cue to the full audio duration."""
    if x.shape[-1] <= 0:
        raise ValueError("temporal cue must have non-zero last dimension")
    if duration_sec is None:
        return x

    duration_sec = float(duration_sec)
    if duration_sec <= 0:
        raise ValueError(f"duration_sec must be positive, got {duration_sec}")

    cur_len = x.shape[-1]
    target_len = int(
        round(_audio_seconds(sample, spk_slot) * cur_len / duration_sec))
    if target_len <= 0:
        raise ValueError(
            f"target temporal length must be positive, got {target_len}")
    if cur_len == target_len:
        return x
    if cur_len > target_len:
        return x[..., :target_len]

    pad_len = target_len - cur_len
    pad = x[..., -1:].expand(*x.shape[:-1], pad_len)
    return torch.cat([x, pad], dim=-1)


def _chunk_time_window(sample, spk_slot):
    """Recover the audio chunk's absolute time window from sample positions."""
    ratio = sample.get(f"chunk_ratio_{spk_slot}",
                       sample.get("chunk_ratio", None))
    sample_rate = sample["sample_rate"]
    if ratio is None:
        return 0.0, sample["wav_mix"].shape[-1] / sample_rate

    start = round(ratio["start_ratio"] * ratio["orig_len"])
    end = round(ratio["end_ratio"] * ratio["orig_len"])
    if start < 0 or end <= start:
        raise ValueError(f"Invalid video time window: {start}:{end}")
    return start / sample_rate, end / sample_rate


def _read_raw_video_item(item, sample, spk_slot):
    """Decode the video window as [H, W, C, T], padding with its last frame."""
    start_sec, end_sec = _chunk_time_window(sample, spk_slot)
    frames = []
    with av.open(item["path"]) as container:
        stream = container.streams.video[0]
        if not stream.average_rate or stream.average_rate <= 0:
            raise ValueError(f"Invalid video FPS: {item['path']}")
        fps = float(stream.average_rate)
        time_base = stream.time_base
        origin_pts = stream.start_time or 0
        first = int(start_sec * fps + 0.5)
        count = max(1, int((end_sec - start_sec) * fps + 0.5))
        stop = first + count

        # Seek to an earlier keyframe, then locate frames by their actual PTS.
        seek_pts = origin_pts + int(first / stream.average_rate / time_base)
        container.seek(seek_pts, stream=stream, backward=True)
        last_frame = None
        last_index = -1
        for frame in container.decode(video=0):
            frame_sec = (frame.pts - origin_pts) * time_base
            index = int(frame_sec * stream.average_rate + 0.5)
            if index >= stop:
                break
            last_frame = frame
            last_index = index
            if index >= first:
                frames.append(frame.to_ndarray(format="rgb24"))

        # Preserve the final frame when a padded audio chunk starts past EOF.
        if not frames:
            if last_frame is None or last_index >= first:
                raise RuntimeError(
                    f"No video frames for requested window: {item['path']}")
            frames.append(last_frame.to_ndarray(format="rgb24"))

    frames = frames[:count]
    frames.extend([frames[-1]] * (count - len(frames)))
    video = torch.from_numpy(np.stack(frames))
    return video.permute(1, 2, 3, 0)


def _read_muse_frontend_item(item, sample, spk_slot):
    """Read one precomputed visual feature and align its final time axis."""
    feat = np.load(item["path"])
    feat = torch.from_numpy(np.asarray(feat)).float()
    if feat.ndim < 2:
        raise ValueError(
            f"muse_frontend visual cue must have at least 2 dims, got {tuple(feat.shape)}"
        )
    feat = _align_last_dim_to_audio_duration(
        feat,
        item.get("duration_sec", None),
        sample,
        spk_slot,
    )
    ratio = sample.get(f"chunk_ratio_{spk_slot}",
                       sample.get("chunk_ratio", None))
    if ratio is not None:
        num_frames = feat.shape[-1]
        start_f = int(ratio["start_ratio"] * num_frames)
        end_f = int(ratio["end_ratio"] * num_frames)
        start_f = max(0, min(start_f, num_frames))
        crop_end = max(start_f + 1, min(end_f, num_frames))
        feat = feat[..., start_f:crop_end]
        if end_f > num_frames:
            pad = feat[..., -1:].expand(*feat.shape[:-1], end_f - num_frames)
            feat = torch.cat([feat, pad], dim=-1)
    return feat


# module-level cache (per worker process)
_SPK_RESOURCE_CACHE = {}


def sample_visual_cue(
    data,
    resource_path,
    key_field,
    cue_format,
    scope="speaker",
    required=True,
):
    """Attach `visual_spk{i}` tensors from MP4 or precomputed npy items.

    Args:
        data: samples containing audio timing metadata and `spk{i}` fields.
        resource_path: JSON mapping lookup keys to visual item lists.
        key_field: `spk_id` or `mix_spk_id`.
        cue_format: (`mp4`, `raw_video`) or (`npy`, `muse_frontend`).

    Returns:
        Samples with available `visual_spk{i}` tensors attached.
    """
    if scope != "speaker":
        raise ValueError("visual cue currently supports scope='speaker'")

    # Select the format decoder and load its index once per worker.
    if cue_format == ("mp4", "raw_video"):
        item_reader = _read_raw_video_item
    elif cue_format == ("npy", "muse_frontend"):
        item_reader = _read_muse_frontend_item
    else:
        raise ValueError(f"Unsupported visual cue format: {cue_format}")
    if resource_path not in _SPK_RESOURCE_CACHE:
        _SPK_RESOURCE_CACHE[resource_path] = load_json(resource_path)
    spk_resource = _SPK_RESOURCE_CACHE[resource_path]

    for sample in data:
        # Resolve the ordered target speaker slots in the current sample.
        spk_slots = [key for key in sample if re.fullmatch(r"spk\d+", key)]
        spk_slots.sort(key=lambda key: int(key[3:]))
        if not spk_slots:
            if required:
                raise KeyError("sample has no speaker slots (spk1, spk2, ...)")
            yield sample
            continue

        for slot in spk_slots:
            # Build the offline or original online utterance lookup key.
            if key_field == "spk_id":
                lookup_key = sample[slot]
            elif key_field == "mix_spk_id":
                mix_key = sample.get(f"source_key_{slot}",
                                     sample.get("key", None))
                if mix_key is None:
                    raise KeyError("sample missing 'key' for mix_spk_id cue")
                lookup_key = f"{mix_key}::{sample[slot]}"
            else:
                raise ValueError(
                    f"Unsupported key_field for visual cue: {key_field}")

            if lookup_key not in spk_resource:
                if required:
                    raise KeyError(f"fixed visual cue not found: {lookup_key}")
                continue

            items = spk_resource[lookup_key]
            if not items:
                if required:
                    raise RuntimeError(f"empty fixed visual cue: {lookup_key}")
                continue

            # Decode and time-align the first fixed visual item.
            try:
                visual = item_reader(items[0], sample, slot)
            except Exception as e:
                logging.warning(
                    f"Failed to read visual cue: {items[0]}, err={e}")
                if required:
                    raise
                continue

            sample[f"visual_{slot}"] = visual
        yield sample
