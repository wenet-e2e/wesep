#!/usr/bin/env python3
"""Build model-independent word-text cues from LibriSpeech transcripts."""

import argparse
import json
from pathlib import Path

from wesep.utils.file_utils import write_json_records


def load_transcripts(root):
    """Read the full transcription associated with each LibriSpeech utterance."""
    transcripts = {}
    for path in sorted(Path(root).rglob("*.trans.txt")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                fields = line.strip().split(maxsplit=1)
                if len(fields) != 2:
                    continue
                key, text = fields
                if key in transcripts and transcripts[key] != text:
                    raise ValueError(f"Conflicting transcript: {key}")
                transcripts[key] = text
    if not transcripts:
        raise ValueError(f"No LibriSpeech transcripts found under {root}")
    return transcripts


def main():
    """Write one complete text record per mixture target without tokenization."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--librispeech-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    transcripts = load_transcripts(args.librispeech_root)
    cues = {}
    with open(args.samples, encoding="utf-8") as stream:
        for line in stream:
            sample = json.loads(line)
            utterances = sample["key"].split("_")
            if len(utterances) != len(sample["spk"]):
                raise ValueError(f"Source count mismatch: {sample['key']}")
            for utterance, speaker in zip(utterances, sample["spk"]):
                text = transcripts[utterance]
                cues[f"{sample['key']}::{speaker}"] = {
                    "text": text,
                    "words": text.split(),
                    "word_times": None
                }
    if not cues:
        raise ValueError("No textual cues generated")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_records(output, cues)
    print(f"Saved {len(cues)} word-text cues to {output}")


if __name__ == "__main__":
    main()
