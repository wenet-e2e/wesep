"""Direct input readers used by the local extraction CLI."""

from pathlib import Path

import numpy as np
import torch
import torchaudio

AUDIO_SUFFIXES = {".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"}
VIDEO_SUFFIXES = {".avi", ".mov", ".mp4"}


class DirectInputReader:
    """Convert one direct audio or cue input into a model-ready tensor."""

    def __init__(self, sample_rate, model_config, config_path, model_dir):
        self.sample_rate = sample_rate
        self.model_config = model_config
        self.config_path = Path(config_path).resolve()
        self.model_dir = Path(model_dir).resolve()
        self.textual_tokenizer = None

    def _get_textual_tokenizer(self):
        """Build the configured DAE tokenizer only when raw text is used."""
        if self.textual_tokenizer is not None:
            return self.textual_tokenizer

        feature = self.model_config.get("textual",
                                        {}).get("features",
                                                {}).get("dae_kce", {})
        tokenizer_conf = feature.get("tokenizer", {})
        if "phoneme_map" not in tokenizer_conf:
            raise ValueError("Raw textual inference requires "
                             "textual.features.dae_kce.tokenizer.phoneme_map.")

        # Resolve recipe paths first, then paths relative to the model package.
        def resolve_path(value, required):
            if value is None:
                return None
            path = Path(value).expanduser()
            candidates = ([path] if path.is_absolute() else [
                path,
                self.config_path.parent / path,
                self.model_dir / path,
            ])
            for candidate in candidates:
                if candidate.is_file():
                    return str(candidate.resolve())
            if required:
                raise FileNotFoundError(
                    f"Textual tokenizer resource not found: {value}")
            return None

        from wesep.modules.textual.tokenizer import DAEPhonemeTokenizer
        self.textual_tokenizer = DAEPhonemeTokenizer(
            resolve_path(tokenizer_conf["phoneme_map"], required=True),
            resolve_path(tokenizer_conf.get("lexicon"), required=False),
        )
        return self.textual_tokenizer

    def _read_textual(self, value):
        """Read phoneme IDs from NPY, or tokenize a text file/string."""
        path = Path(value)
        suffix = path.suffix.lower()

        if suffix == ".npy":
            if not path.is_file():
                raise FileNotFoundError(f"Textual NPY not found: {path}")
            array = np.load(path, allow_pickle=False)
            if array.dtype == np.dtype("O"):
                raise ValueError(
                    f"Textual NPY must contain numeric phoneme IDs: {path}")
            tensor = torch.from_numpy(np.asarray(array)).long()
            if tensor.ndim == 2 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            if tensor.ndim != 1:
                raise ValueError(
                    "Textual phoneme IDs must have shape [L], got "
                    f"{tuple(tensor.shape)}")
            return tensor

        if suffix == ".txt":
            if not path.is_file():
                raise FileNotFoundError(
                    f"Textual input file not found: {path}")
            text = path.read_text(encoding="utf-8")
        else:
            text = str(value)

        ids = self._get_textual_tokenizer().encode(text)
        return torch.tensor(ids, dtype=torch.long)

    def read(self, value, key, input_type=None):
        """Decode one input according to its model key and representation."""
        if key == "textual_aux":
            return self._read_textual(value)

        path = Path(value)
        suffix = path.suffix.lower()
        if key == "wav_mix" and suffix not in AUDIO_SUFFIXES:
            raise ValueError(f"Mixture must be an audio file: {path}")

        # Audio inputs use channel-first float waveforms at the model rate.
        if suffix in AUDIO_SUFFIXES:
            waveform, sample_rate = torchaudio.load(str(path))
            if sample_rate != self.sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, sample_rate, self.sample_rate)
            return waveform.float()

        # NPY inputs already follow the selected frontend representation.
        if suffix == ".npy":
            array = np.load(path, allow_pickle=False)
            if array.dtype == np.dtype("O"):
                raise ValueError(
                    f"CLI NPY inputs must contain a numeric array: {path}")
            tensor = torch.from_numpy(np.asarray(array)).float()
            if (key == "audio_aux" and input_type == "embedding"
                    and tensor.ndim == 2 and tensor.shape[0] == 1):
                tensor = tensor.squeeze(0)
            return tensor

        # Video remains uint8 until its frontend runs on the model device.
        if suffix in VIDEO_SUFFIXES:
            from torchvision.io import read_video
            video, _, _ = read_video(str(path), pts_unit="sec")
            if video.numel() == 0:
                raise RuntimeError(f"Empty video: {path}")
            return video.permute(1, 2, 3, 0)  # [H, W, C, T]

        raise ValueError(f"Unsupported file type for {key!r}: {path}")
