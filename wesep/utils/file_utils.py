import collections
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import kaldiio
import numpy as np
import soundfile


def read_lists(list_file):
    """list_file: only 1 column"""
    lists = []
    with open(list_file, "r", encoding="utf8") as fin:
        for line in fin:
            lists.append(line.strip())
    return lists


def read_vec_scp_file(scp_file):
    """
    Read the pre-extracted kaldi-format speaker embeddings.
    :param scp_file: path to xvector.scp
    :return: dict {wav_name: embedding}
    """
    samples_dict = {}
    for key, vec in kaldiio.load_scp_sequential(scp_file):
        if len(vec.shape) == 1:
            vec = np.expand_dims(vec, 0)
        samples_dict[key] = vec

    return samples_dict


def norm_embeddings(embeddings, kaldi_style=True):
    """
    Norm embeddings to unit length
    :param embeddings: input embeddings
    :param kaldi_style: if true, the norm should be embedding dimension
    :return:
    """
    scale = math.sqrt(embeddings.shape[-1]) if kaldi_style else 1.0
    if len(embeddings.shape) == 2:
        return (scale * embeddings.transpose() /
                np.linalg.norm(embeddings, axis=1)).transpose()
    elif len(embeddings.shape) == 1:
        return scale * embeddings / np.linalg.norm(embeddings)


def read_label_file(label_file):
    """
    Read the utt2spk file
    :param label_file: the path to utt2spk
    :return: dict {wav_name: spk_id}
    """
    labels_dict = {}
    with open(label_file, "r") as fin:
        for line in fin:
            tokens = line.strip().split()
            labels_dict[tokens[0]] = tokens[1]
    return labels_dict


def load_speaker_embeddings(scp_file, utt2spk_file):
    """
    :param scp_file:
    :param utt2spk_file:
    :return: {spk1: [emb1, emb2 ...], spk2: [emb1, emb2...]}
    """
    samples_dict = read_vec_scp_file(scp_file)
    labels_dict = read_label_file(utt2spk_file)
    spk2embeds = {}
    for key, vec in samples_dict.items():
        if len(vec.shape) == 1:
            vec = np.expand_dims(vec, 0)
        label = labels_dict[key]
        if label in spk2embeds.keys():
            spk2embeds[label].append(vec)
        else:
            spk2embeds[label] = [vec]
    return spk2embeds


# ported from
# https://github.com/espnet/espnet/blob/master/espnet2/fileio/read_text.py
def read_2columns_text(path: Union[Path, str]) -> Dict[str, str]:
    """Read a text file having 2 columns as dict object.

    Examples:
        wav.scp:
            key1 /some/path/a.wav
            key2 /some/path/b.wav

        >>> read_2columns_text('wav.scp')
        {'key1': '/some/path/a.wav', 'key2': '/some/path/b.wav'}

    """

    data = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for linenum, line in enumerate(f, 1):
            sps = line.rstrip().split(maxsplit=1)
            if len(sps) == 1:
                k, v = sps[0], ""
            else:
                k, v = sps

            if k in data:
                raise RuntimeError(f"{k} is duplicated ({path}:{linenum})")
            data[k] = v
    return data


# ported from
# https://github.com/espnet/espnet/blob/master/espnet2/fileio/read_text.py
def read_multi_columns_text(
    path: Union[Path, str],
    return_unsplit: bool = False
) -> Tuple[Dict[str, List[str]], Optional[Dict[str, str]]]:
    """Read a text file having 2 or more columns as dict object.

    Examples:
        wav.scp:
            key1 /some/path/a1.wav /some/path/a2.wav
            key2 /some/path/b1.wav /some/path/b2.wav  /some/path/b3.wav
            key3 /some/path/c1.wav
            ...

        >>> read_multi_columns_text('wav.scp')
        {'key1': ['/some/path/a1.wav', '/some/path/a2.wav'],
         'key2': ['/some/path/b1.wav', '/some/path/b2.wav',
                  '/some/path/b3.wav'],
         'key3': ['/some/path/c1.wav']}

    """

    data = {}

    if return_unsplit:
        unsplit_data = {}
    else:
        unsplit_data = None

    with Path(path).open("r", encoding="utf-8") as f:
        for linenum, line in enumerate(f, 1):
            sps = line.rstrip().split(maxsplit=1)
            if len(sps) == 1:
                k, v = sps[0], ""
            else:
                k, v = sps

            if k in data:
                raise RuntimeError(f"{k} is duplicated ({path}:{linenum})")

            data[k] = v.split() if v != "" else [""]
            if return_unsplit:
                unsplit_data[k] = v

    return data, unsplit_data


# ported from
# https://github.com/espnet/espnet/blob/master/espnet2/fileio/sound_scp.py
def soundfile_read(
    wavs: Union[str, List[str]],
    dtype=None,
    always_2d: bool = False,
    concat_axis: int = 1,
    start: int = 0,
    end: int = None,
    return_subtype: bool = False,
) -> Tuple[np.array, int]:
    if isinstance(wavs, str):
        wavs = [wavs]

    arrays = []
    subtypes = []
    prev_rate = None
    prev_wav = None
    for wav in wavs:
        with soundfile.SoundFile(wav) as f:
            f.seek(start)
            if end is not None:
                frames = end - start
            else:
                frames = -1
            if dtype == "float16":
                array = f.read(
                    frames,
                    dtype="float32",
                    always_2d=always_2d,
                ).astype(dtype)
            else:
                array = f.read(frames, dtype=dtype, always_2d=always_2d)
            rate = f.samplerate
            subtype = f.subtype
            subtypes.append(subtype)

        if len(wavs) > 1 and array.ndim == 1 and concat_axis == 1:
            # array: (Time, Channel)
            array = array[:, None]

        if prev_wav is not None:
            if prev_rate != rate:
                raise RuntimeError(
                    f"{prev_wav} and {wav} have mismatched sampling rate: "
                    f"{prev_rate} != {rate}")

            dim1 = arrays[0].shape[1 - concat_axis]
            dim2 = array.shape[1 - concat_axis]
            if dim1 != dim2:
                raise RuntimeError(
                    "Shapes must match with "
                    f"{1 - concat_axis} axis, but gut {dim1} and {dim2}")

        prev_rate = rate
        prev_wav = wav
        arrays.append(array)

    if len(arrays) == 1:
        array = arrays[0]
    else:
        array = np.concatenate(arrays, axis=concat_axis)

    if return_subtype:
        return array, rate, subtypes
    else:
        return array, rate


# ported from
# https://github.com/espnet/espnet/blob/master/espnet2/fileio/sound_scp.py
class SoundScpReader(collections.abc.Mapping):
    """Reader class for 'wav.scp'.

    Examples:
        wav.scp is a text file that looks like the following:

        key1 /some/path/a.wav
        key2 /some/path/b.wav
        key3 /some/path/c.wav
        key4 /some/path/d.wav
        ...

        >>> reader = SoundScpReader('wav.scp')
        >>> rate, array = reader['key1']

        If multi_columns=True is given and
        multiple files are given in one line
        with space delimiter, and  the output array are concatenated
        along channel direction

        key1 /some/path/a.wav /some/path/a2.wav
        key2 /some/path/b.wav /some/path/b2.wav
        ...

        >>> reader = SoundScpReader('wav.scp', multi_columns=True)
        >>> rate, array = reader['key1']

        In the above case, a.wav and a2.wav are concatenated.

        Note that even if multi_columns=True is given,
        SoundScpReader still supports a normal wav.scp,
        i.e., a wav file is given per line,
        but this option is disable by default
        because dict[str, list[str]] object is needed to be kept,
        but it increases the required amount of memory.
    """

    def __init__(
        self,
        fname,
        dtype=None,
        always_2d: bool = False,
        multi_columns: bool = False,
        concat_axis=1,
    ):
        self.fname = fname
        self.dtype = dtype
        self.always_2d = always_2d

        if multi_columns:
            self.data, _ = read_multi_columns_text(fname)
        else:
            self.data = read_2columns_text(fname)
        self.multi_columns = multi_columns
        self.concat_axis = concat_axis

    def __getitem__(self, key) -> Tuple[int, np.ndarray]:
        wavs = self.data[key]

        array, rate = soundfile_read(
            wavs,
            dtype=self.dtype,
            always_2d=self.always_2d,
            concat_axis=self.concat_axis,
        )
        # Returned as scipy.io.wavread's order
        return rate, array

    def get_path(self, key):
        return self.data[key]

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()
