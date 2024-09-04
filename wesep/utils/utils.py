# Copyright (c) 2022 Hongji Wang (jijijiang77@gmail.com)
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
import difflib
import logging
import os
import random
import shutil
import sys
from distutils.util import strtobool
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml


def str2bool(value: str) -> bool:
    return bool(strtobool(value))


def get_logger(outdir, fname):
    formatter = logging.Formatter(
        "[ %(levelname)s : %(asctime)s ] - %(message)s")
    logging.basicConfig(
        level=logging.DEBUG,
        format="[ %(levelname)s : %(asctime)s ] - %(message)s",
    )
    logger = logging.getLogger("Pyobj, f")
    # Dump log to file
    fh = logging.FileHandler(os.path.join(outdir, fname))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def setup_logger(rank, exp_dir, device_ids, MAX_NUM_LOG_FILES: int = 100):
    model_dir = os.path.join(exp_dir, "models")
    file_name = "train.log"
    if rank == 0:
        os.makedirs(model_dir, exist_ok=True)
        for i in range(MAX_NUM_LOG_FILES - 1, -1, -1):
            if i == 0:
                p = Path(os.path.join(exp_dir, file_name))
                pn = p.parent / (p.stem + ".1" + p.suffix)
            else:
                _p = Path(os.path.join(exp_dir, file_name))
                p = _p.parent / (_p.stem + f".{i}" + _p.suffix)
                pn = _p.parent / (_p.stem + f".{i + 1}" + _p.suffix)

            if p.exists():
                if i == MAX_NUM_LOG_FILES - 1:
                    p.unlink()
                else:
                    shutil.move(p, pn)
    dist.barrier(device_ids=[device_ids])  # let the rank 0 mkdir first
    return get_logger(exp_dir, file_name)


def parse_config_or_kwargs(config_file, **kwargs):
    """parse_config_or_kwargs

    :param config_file: Config file that has parameters, yaml format
    :param **kwargs: Other alternative parameters or overwrites for conf
    """
    with open(config_file) as con_read:
        yaml_config = yaml.load(con_read, Loader=yaml.FullLoader)
    # values from conf file are all possible params
    help_str = "Valid Parameters are:\n"
    help_str += "\n".join(list(yaml_config.keys()))
    # passed kwargs will override yaml conf
    # for key in kwargs.keys():
    #    assert key in yaml_config, "Parameter {} invalid!\n".format(key)
    # add the path of config file to dict
    if "config" not in kwargs:
        kwargs["config"] = config_file
    return dict(yaml_config, **kwargs)


def validate_path(dir_name):
    """Create the directory if it doesn't exist
    :param dir_name
    :return: None
    """
    dir_name = os.path.dirname(dir_name)  # get the path
    if not os.path.exists(dir_name) and (dir_name != ""):
        os.makedirs(dir_name)


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def generate_enahnced_scp(directory: str, extension: str = "wav"):
    source_dir = Path(directory)
    spk_scp = source_dir.joinpath("spk1.scp")
    audio_list = []

    for file_path in source_dir.rglob(f"*.{extension}"):
        audio_list.append(file_path)

    with open(spk_scp, "w") as f:
        for audio in audio_list:
            path = str(audio.resolve())
            ori_filename = audio.stem
            spk1_id = ori_filename.split("-")[1]
            # spk2_id = ori_filename.split("_")[1].split("-")[0]
            curr_spk = ori_filename.split("T")[1]
            prefix = "s1" if curr_spk == spk1_id else "s2"
            f_dash_index = ori_filename.find("-")
            l_dash_index = ori_filename.rfind("-")
            filename = ori_filename[f_dash_index + 1:l_dash_index]
            final_filename = prefix + "/" + filename + ".wav"
            line = final_filename + " " + path
            f.write(line + "\n")


def get_commandline_args():
    # ported from
    # https://github.com/espnet/espnet/blob/master/espnet/utils/cli_utils.py
    extra_chars = [
        " ",
        ";",
        "&",
        "(",
        ")",
        "|",
        "^",
        "<",
        ">",
        "?",
        "*",
        "[",
        "]",
        "$",
        "`",
        '"',
        "\\",
        "!",
        "{",
        "}",
    ]

    # Escape the extra characters for shell
    argv = [(arg.replace("'", "'\\''") if all(
        char not in arg
        for char in extra_chars) else "'" + arg.replace("'", "'\\''") + "'")
            for arg in sys.argv]

    return sys.executable + " " + " ".join(argv)


# ported from
# https://github.com/espnet/espnet/blob/master/espnet2/utils/config_argparse.py
class ArgumentParser(argparse.ArgumentParser):
    """Simple implementation of ArgumentParser supporting config file

    This class is originated from https://github.com/bw2/ConfigArgParse,
    but this class is lack of some features that it has.

    - Not supporting multiple config files
    - Automatically adding "--config" as an option.
    - Not supporting any formats other than yaml
    - Not checking argument type

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("--config", help="Give config file in yaml format")

    def parse_known_args(self, args=None, namespace=None):
        # Once parsing for setting from "--config"
        _args, _ = super().parse_known_args(args, namespace)
        if _args.config is not None:
            if not Path(_args.config).exists():
                self.error(f"No such file: {_args.config}")

            with open(_args.config, "r", encoding="utf-8") as f:
                d = yaml.safe_load(f)
            if not isinstance(d, dict):
                self.error("Config file has non dict value: {_args.config}")

            for key in d:
                for action in self._actions:
                    if key == action.dest:
                        break
                else:
                    self.error(
                        f"unrecognized arguments: {key} (from {_args.config})")

            # NOTE(kamo): Ignore "--config" from a config file
            # NOTE(kamo): Unlike "configargparse", this module doesn't
            #             check type. i.e. We can set any type value
            #             regardless of argument type.
            self.set_defaults(**d)
        return super().parse_known_args(args, namespace)


def get_layer(l_name, library=torch.nn):
    """Return layer object handler from library e.g. from torch.nn

    E.g. if l_name=="elu", returns torch.nn.ELU.

    Args:
        l_name (string): Case insensitive name for layer in library
                        (e.g. .'elu').
        library (module): Name of library/module where to search for
                          object handler with l_name e.g. "torch.nn".

    Returns:
        layer_handler (object): handler for the requested layer
                                e.g. (torch.nn.ELU)

    """

    all_torch_layers = [x for x in dir(torch.nn)]
    match = [x for x in all_torch_layers if l_name.lower() == x.lower()]
    if len(match) == 0:
        close_matches = difflib.get_close_matches(
            l_name, [x.lower() for x in all_torch_layers])
        raise NotImplementedError(
            "Layer with name {} not found in {}.\n Closest matches: {}".format(
                l_name, str(library), close_matches))
    elif len(match) > 1:
        close_matches = difflib.get_close_matches(
            l_name, [x.lower() for x in all_torch_layers])
        raise NotImplementedError(
            "Multiple matchs for layer with name {} not found in {}.\n "
            "All matches: {}".format(l_name, str(library), close_matches))
    else:
        # valid
        layer_handler = getattr(library, match[0])
        return layer_handler


# def spk2id(utt_spk_list):
#     _, spk_list = zip(*utt_spk_list)
#     spk_list = sorted(list(set(spk_list)))  # remove overlap and sort

#     spk2id_dict = {}
#     spk_list.sort()
#     for i, spk in enumerate(spk_list):
#         spk2id_dict[spk] = i
#     return spk2id_dict
