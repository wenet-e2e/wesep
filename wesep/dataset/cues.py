# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
#
# SPDX-License-Identifier: Apache-2.0

from wesep.dataset import (
    processor_speaker,
    processor_spatial,
    processor_textual,
    processor_visual,
)
from wesep.utils.file_utils import load_yaml

DEFAULT_CUE_TYPE_FORMAT = {
    "audio": ("wav", "waveform"),
    "visual": ("mp4", "raw_video"),
    "spatial": ("npy", "fields"),
}

CUE_FORMATS = {
    ("audio", "wav", "waveform"),
    ("audio", "npy", "spk_embedding"),
    ("visual", "mp4", "raw_video"),
    ("visual", "npy", "muse_frontend"),
    ("spatial", "npy", "fields"),
    ("textual", "json", "word_text"),
    ("textual", "json", "dae_keyword_phoneme"),
}


def build_cue_layer(dataset, cues_yaml, state, configs):
    """Build the model-requested cue pipelines declared in cues.yaml."""
    # Load the data-side cue descriptions once while building the pipeline.
    cues_conf = load_yaml(cues_yaml)
    if "cues" in cues_conf:
        cues_conf = cues_conf["cues"]

    # Attach only cues enabled by the current model configuration.
    requested_cues = configs.get("cues", {})
    for cue_name, cue_conf in cues_conf.items():
        if not requested_cues.get(cue_name, {}).get("use", False):
            continue
        if cue_name not in CUE_BUILDERS:
            raise ValueError(f"Unknown cue: {cue_name}")
        dataset = CUE_BUILDERS[cue_name](
            dataset,
            cue_name,
            cue_conf,
            state,
            configs,
        )

    return dataset


CUE_BUILDERS = {}


def register_cue(name):

    def deco(fn):
        CUE_BUILDERS[name] = fn
        return fn

    return deco


def _cue_common(cue_name, cue_conf, supported_policies):
    """Resolve and validate the shared configuration for one cue."""
    # Resolve the declared representation and validate the official formats.
    default = DEFAULT_CUE_TYPE_FORMAT.get(cue_name, (None, None))
    cue_type = cue_conf.get("type", default[0])
    cue_format = cue_conf.get("format", default[1])
    if cue_type is None or cue_format is None:
        raise ValueError(
            f"{cue_name} cue must explicitly declare 'type' and 'format'")
    format_key = (cue_name, cue_type, cue_format)
    if format_key not in CUE_FORMATS:
        supported = [
            f"{item[1]}:{item[2]}" for item in CUE_FORMATS
            if item[0] == cue_name
        ]
        raise ValueError(f"Unsupported {cue_name} cue format: "
                         f"type={cue_type}, format={cue_format}. "
                         f"Supported formats: {supported}")

    # Resolve the speaker-level lookup policy and its resource index.
    scope = cue_conf.get("scope", "speaker")
    guaranteed = cue_conf.get("guaranteed", True)
    if scope != "speaker":
        raise ValueError(
            f"{cue_name} cue currently supports scope='speaker', got: {scope}")

    policy = cue_conf.get("policy", {})
    policy_type = policy.get("type")
    key_field = policy.get("key")
    resource_path = policy.get("resource")
    if policy_type is None or key_field is None or resource_path is None:
        raise ValueError(f"Invalid {cue_name} cue policy config: {policy}")
    if policy_type not in supported_policies:
        raise ValueError(
            f"Unsupported {cue_name} cue policy for "
            f"type={cue_type}, format={cue_format}: {policy_type}")

    return {
        "type": cue_type,
        "format": cue_format,
        "scope": scope,
        "guaranteed": guaranteed,
        "policy_type": policy_type,
        "key_field": key_field,
        "resource_path": resource_path,
    }


@register_cue("audio")
def build_audio_cue(dataset, cue_name, cue_conf, state, configs):
    """Attach the configured speaker/audio cue processor."""
    info = _cue_common(cue_name, cue_conf, ("random", "fixed"))

    # Select the reader for the declared audio cue representation.
    cue_format = (info["type"], info["format"])
    if cue_format == ("wav", "waveform"):
        target_sr = configs.get("resample_rate", None)
    else:
        target_sr = None

    # Attach one cue tensor to every target speaker slot.
    dataset = dataset.apply(
        processor_speaker.sample_speaker_cue,
        info["resource_path"],
        key_field=info["key_field"],
        policy_type=info["policy_type"],
        cue_format=cue_format,
        scope=info["scope"],
        required=info["guaranteed"],
        target_sr=target_sr,
    )

    # Training-only enrollment augmentation stays with audio waveform cues.
    audio_conf = configs.get("cue_processing", {}).get("audio", {})
    if state == "train" and cue_format == ("wav", "waveform"):
        reverb_prob = audio_conf.get("reverb_enroll_prob", 0)
        if reverb_prob > 0:
            dataset = dataset.apply(processor_speaker.add_reverb_on_enroll,
                                    reverb_prob)

        noise_prob = audio_conf.get("noise_enroll_prob", 0)
        noise_lmdb_file = configs.get("noise_lmdb_file", None)
        if noise_prob > 0:
            assert noise_lmdb_file is not None
            dataset = dataset.apply(
                processor_speaker.add_noise_on_enroll,
                noise_lmdb_file,
                noise_prob,
            )

    return dataset


@register_cue("visual")
def build_visual_cue(dataset, cue_name, cue_conf, state, configs):
    """Attach the configured visual cue processor."""
    info = _cue_common(cue_name, cue_conf, ("fixed", ))

    cue_format = (info["type"], info["format"])
    return dataset.apply(
        processor_visual.sample_visual_cue,
        info["resource_path"],
        key_field=info["key_field"],
        cue_format=cue_format,
        scope=info["scope"],
        required=info["guaranteed"],
    )


@register_cue("spatial")
def build_spatial_cue(dataset, cue_name, cue_conf, state, configs):
    """Attach the configured spatial cue processor."""
    info = _cue_common(cue_name, cue_conf, ("fixed", ))

    fields = cue_conf.get("fields")
    if not fields:
        raise ValueError("Spatial cue requires non-empty 'fields'.")

    return dataset.apply(
        processor_spatial.sample_fixed_spatial_cue,
        info["resource_path"],
        fields,
        key_field=info["key_field"],
        field_defaults=cue_conf.get("field_defaults", {}),
        scope=info["scope"],
        required=info["guaranteed"],
    )


@register_cue("textual")
def build_textual_cue(dataset, cue_name, cue_conf, state, configs):
    """Build numeric textual cues using training-config processing rules."""
    info = _cue_common(cue_name, cue_conf, ("fixed", ))
    cue_format = (info["type"], info["format"])
    processing = dict(configs.get("cue_processing", {}).get("textual", {}))
    # The configured input must match the representation in the cue index.
    formats = {
        "text": ("json", "word_text"),
        "dae_phoneme": ("json", "dae_keyword_phoneme")
    }
    input_type = processing.get("input")
    if input_type not in formats:
        raise ValueError("textual.input must be text or dae_phoneme")
    if cue_format != formats[input_type]:
        raise ValueError(
            f"textual.input={input_type} requires {formats[input_type]}, "
            f"but cues.yaml declares {cue_format}; select matching cue lists")
    # Omitted stages retain every available word in the selected cue record.
    selections = processing.get("selection", {})
    if not isinstance(selections, dict):
        raise ValueError(
            "textual.selection must map train/val/test to policies")
    unknown = set(selections) - {"train", "val", "test"}
    if unknown:
        raise ValueError(
            f"Unknown textual selection stages: {sorted(unknown)}")
    selection = selections.get(state, {"type": "all"})
    if not isinstance(selection, dict):
        raise ValueError(f"Invalid textual selection for state={state}")
    seed = selection.get("seed")
    if seed is not None and type(seed) is not int:
        raise ValueError("textual selection seed must be an integer")
    processing["selection"] = selection
    if not isinstance(processing.get("encoding"), dict):
        raise ValueError("cue_processing.textual.encoding must be configured")
    if cue_conf.get("collate", {}).get("fill_value", 0) != 0:
        raise ValueError("DAE phoneme cues require fill_value=0")
    if cue_conf.get("collate", {}).get("align", "max") != "max":
        raise ValueError(
            "Textual cues must align to the longest phoneme sequence")
    return dataset.apply(
        processor_textual.sample_textual_cue,
        info["resource_path"],
        key_field=info["key_field"],
        cue_format=cue_format,
        processing=processing,
        scope=info["scope"],
        required=info["guaranteed"],
    )
