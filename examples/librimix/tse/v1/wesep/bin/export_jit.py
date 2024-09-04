from __future__ import print_function

import argparse
import os

import torch
import yaml

from wesep.models import get_model
from wesep.utils.checkpoint import load_pretrained_model


def get_args():
    parser = argparse.ArgumentParser(description="export your script model")
    parser.add_argument("--config", required=True, help="config file")
    parser.add_argument("--checkpoint", required=True, help="checkpoint model")
    parser.add_argument("--output_model", required=True, help="output file")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    with open(args.config, "r") as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    print(configs)

    model = get_model(configs["model"]["tse_model"])(
        **configs["model_args"]["tse_model"]
    )
    print(model)

    load_pretrained_model(model, args.checkpoint)
    model.eval()

    speaker_feat_dim = configs["dataset_args"]["fbank_args"].get(
        "num_mel_bins", 80
    )

    speaker_dummy_input = torch.ones(2, 300, speaker_feat_dim)
    mix_dummy_input = torch.ones(2, 81280)
    script_model = torch.jit.script(
        model, (mix_dummy_input, speaker_dummy_input)
    )
    script_model.save(args.output_model)
    print("Export model successfully, see {}".format(args.output_model))


if __name__ == "__main__":
    main()
