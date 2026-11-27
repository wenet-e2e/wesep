#!/usr/bin/env python3
"""Precompute the same per-word DAE phonemes used by online cue workers."""

import argparse
import json
from pathlib import Path

import yaml

from wesep.modules.textual.tokenizer import build_text_tokenizer
from wesep.utils.file_utils import write_json_records


def main():
    """Preserve complete words and timings while adding phoneme groups."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.input).resolve() == Path(args.output).resolve():
        raise ValueError("Input and output indexes must be different files")
    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    encoding = config["dataset_args"]["cue_processing"]["textual"]["encoding"]
    tokenizer = build_text_tokenizer(encoding)
    with open(args.input, encoding="utf-8") as stream:
        cues = json.load(stream)
    for item in cues.values():
        item["phn_label"] = tokenizer.encode_word_groups(item["words"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_records(output, cues)
    print(f"Saved {len(cues)} precomputed phoneme cues to {output}")


if __name__ == "__main__":
    main()
