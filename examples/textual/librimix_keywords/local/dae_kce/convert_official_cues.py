#!/usr/bin/env python3
"""Convert official selected keywords to the common per-target cue records."""

import argparse
import json
from pathlib import Path

from wesep.utils.file_utils import write_json_records


def main():
    """Preserve official word choices and, for offline input, their phonemes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--official-cues", required=True)
    parser.add_argument("--textual-input",
                        choices=("text", "dae_phoneme"),
                        required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.samples, encoding="utf-8") as stream:
        samples = {
            item["key"]: item["spk"]
            for item in map(json.loads, stream) if item
        }
    expected = {
        f"{key}::{speaker}"
        for key, speakers in samples.items() for speaker in speakers
    }
    cues = {}
    with open(args.official_cues, encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            source, mix_key = item["key"].split("/", maxsplit=1)
            if mix_key not in samples:
                continue
            slot = int(source.removeprefix("s")) - 1
            if slot < 0 or slot >= len(samples[mix_key]):
                raise ValueError(f"Invalid source slot: {item['key']}")
            key = f"{mix_key}::{samples[mix_key][slot]}"
            if key in cues:
                raise ValueError(f"Duplicate official target: {key}")
            words = item["normalized_text"].split()
            groups = item["phn_label"]
            indices = item["kw_candidate"]
            if len(words) != len(groups) or not indices:
                raise ValueError(f"Invalid official word groups: {key}")
            if any(
                    type(index) is not int or not 0 <= index < len(words)
                    for index in indices):
                raise ValueError(f"Invalid official word indices: {key}")
            selected = [words[index] for index in indices]
            text = " ".join(selected)
            if text != item["keywords_text"]:
                raise ValueError(f"Official keyword text mismatch: {key}")
            record = {"text": text, "words": selected, "word_times": None}
            if args.textual_input == "dae_phoneme":
                record["phn_label"] = [groups[index] for index in indices]
            cues[key] = record
    if not expected or set(cues) != expected:
        raise ValueError(f"Official cue coverage: {len(cues)}/{len(expected)}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_records(output, cues)
    print(f"Saved {len(cues)} official keyword cues to {output}")


if __name__ == "__main__":
    main()
