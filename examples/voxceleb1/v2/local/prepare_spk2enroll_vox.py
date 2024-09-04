import json
from collections import defaultdict
from pathlib import Path


def get_spk2utt_from_wavscp(wav_scp_path):
    spk2utt = defaultdict(list)
    with open(wav_scp_path, "r") as readin:
        for line in readin:
            speaker_id = line.split("/")[0]
            uid, audio_path = line.strip().split()
            spk2utt[speaker_id].append((uid, str(audio_path)))

    return spk2utt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "wav_scp_path",
        type=str,
        help="Paths to Librispeech subsets",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="spk2utt_tse.json",
        help="Path to the output spk2utt json file",
    )
    args = parser.parse_args()

    spk2utt = get_spk2utt_from_wavscp(args.wav_scp_path)

    outfile = Path(args.outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w", encoding="utf-8") as f:
        json.dump(spk2utt, f)
