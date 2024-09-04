# Copyright (c) 2020 Mobvoi Inc (Di Wu)
#               2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import glob
import os.path
import re

import torch


def get_args():
    parser = argparse.ArgumentParser(description="average model")
    parser.add_argument("--dst_model", required=True, help="averaged model")
    parser.add_argument("--src_path",
                        required=True,
                        help="src model path for average")
    parser.add_argument("--num",
                        default=5,
                        type=int,
                        help="nums for averaged model")
    parser.add_argument(
        "--min_epoch",
        default=0,
        type=int,
        help="min epoch used for averaging model",
    )
    parser.add_argument(
        "--max_epoch",
        default=65536,  # Big enough
        type=int,
        help="max epoch used for averaging model",
    )
    parser.add_argument(
        "--mode",
        default="final",
        type=str,
        help="use last epochs for average or best epochs",
    )
    parser.add_argument(
        "--epochs",
        default="1,2,3,4,5",
        type=str,
        help="use last epochs for average or best epochs",
    )
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    if args.mode == "final":
        path_list = glob.glob("{}/*[!avg][!final][!latest].pt".format(
            args.src_path))
        path_list = sorted(
            path_list,
            key=lambda p: int(re.findall(r"(?<=checkpoint_)\d*(?=.pt)", p)[0]),
        )
        path_list = path_list[-args.num:]
    else:
        epoch_indexes = [x for x in args.epochs.split(",")]
        path_list = [
            os.path.join(args.src_path, "checkpoint_" + x + ".pt")
            for x in epoch_indexes
        ]
    print(path_list)
    avg = None
    num = args.num
    assert num == len(path_list)
    for path in path_list:
        print("Processing {}".format(path))
        states = torch.load(path, map_location=torch.device("cpu"))
        states = states["models"][0] if "models" in states else states
        if avg is None:
            avg = states
        else:
            for k in avg.keys():
                avg[k] += states[k]
    # average
    for k in avg.keys():
        if avg[k] is not None:
            # pytorch 1.6 use true_divide instead of /=
            avg[k] = torch.true_divide(avg[k], num)
    avg = {"models": [avg]}
    print("Saving to {}".format(args.dst_model))
    torch.save(avg, args.dst_model)


if __name__ == "__main__":
    main()
