# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2023 Shuai Wang (wsstriving@gmail.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random

import torch
import torch.distributed as dist
import torch.nn.functional as tf
from torch.utils.data import IterableDataset

import wesep.dataset.processor as processor
from wesep.utils.file_utils import read_lists


class Processor(IterableDataset):
    def __init__(self, source, f, *args, **kw):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    def __iter__(self):
        """Return an iterator over the source dataset processed by the
        given processor.
        """
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)

    def apply(self, f):
        assert callable(f)
        return Processor(self, f, *self.args, **self.kw)


class DistributedSampler:
    def __init__(self, shuffle=True, partition=True):
        self.epoch = -1
        self.update()
        self.shuffle = shuffle
        self.partition = partition

    def update(self):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        return dict(
            rank=self.rank,
            world_size=self.world_size,
            worker_id=self.worker_id,
            num_workers=self.num_workers,
        )

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample(self, data):
        """Sample data according to rank/world_size/num_workers

        Args:
            data(List): input data list

        Returns:
            List: data list after sample
        """
        data = list(range(len(data)))
        if len(data) <= self.num_workers:
            if self.shuffle:
                random.Random(self.epoch).shuffle(data)
        else:
            if self.partition:
                if self.shuffle:
                    random.Random(self.epoch).shuffle(data)
                data = data[self.rank :: self.world_size]
            data = data[self.worker_id :: self.num_workers]
        return data


class DataList(IterableDataset):
    def __init__(
        self, lists, shuffle=True, partition=True, repeat_dataset=False
    ):
        self.lists = lists
        self.repeat_dataset = repeat_dataset
        self.sampler = DistributedSampler(shuffle, partition)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        sampler_info = self.sampler.update()
        indexes = self.sampler.sample(self.lists)
        if not self.repeat_dataset:
            for index in indexes:
                data = dict(src=self.lists[index])
                data.update(sampler_info)
                yield data
        else:
            indexes_len = len(indexes)
            counter = 0
            while True:
                index = indexes[counter % indexes_len]
                counter += 1
                data = dict(src=self.lists[index])
                data.update(sampler_info)
                yield data


def tse_collate_fn_2spk(batch, mode="min"):
    # Warning: hard-coded for 2 speakers, will be deprecated in the future, use tse_collate_fn instead
    new_batch = {}

    wav_mix = []
    wav_targets = []
    spk_embeds = []
    spk = []
    key = []
    spk_label = []
    length_spk_embeds = []
    for s in batch:
        wav_mix.append(s["wav_mix"])
        wav_targets.append(s["wav_spk1"])
        spk.append(s["spk1"])
        key.append(s["key"])
        spk_embeds.append(torch.from_numpy(s["embed_spk1"].copy()))
        length_spk_embeds.append(spk_embeds[-1].shape[1])
        if "spk1_label" in s.keys():
            spk_label.append(s["spk1_label"])

        wav_mix.append(s["wav_mix"])
        wav_targets.append(s["wav_spk2"])
        spk.append(s["spk2"])
        key.append(s["key"])
        spk_embeds.append(torch.from_numpy(s["embed_spk2"].copy()))
        length_spk_embeds.append(spk_embeds[-1].shape[1])
        if "spk2_label" in s.keys():
            spk_label.append(s["spk2_label"])

    if not (len(set(length_spk_embeds)) == 1):
        if mode == "max":
            max_len = max(length_spk_embeds)
            for i in range(len(length_spk_embeds)):
                if len(spk_embeds[i].shape) == 2:
                    spk_embeds[i] = tf.pad(
                        spk_embeds[i],
                        (0, max_len - length_spk_embeds[i]),
                        "constant",
                        0,
                    )
                elif len(spk_embeds[i].shape) == 3:
                    spk_embeds[i] = tf.pad(
                        spk_embeds[i],
                        (0, 0, 0, max_len - length_spk_embeds[i]),
                        "constant",
                        0,
                    )
        if mode == "min":
            min_len = min(length_spk_embeds)
            for i in range(len(length_spk_embeds)):
                if len(spk_embeds[i].shape) == 2:
                    spk_embeds[i] = spk_embeds[i][:, :min_len]
                elif len(spk_embeds[i].shape) == 3:
                    spk_embeds[i] = spk_embeds[i][:, :min_len, :]

    new_batch["wav_mix"] = torch.concat(wav_mix)
    new_batch["wav_targets"] = torch.concat(wav_targets)
    new_batch["spk_embeds"] = torch.concat(spk_embeds)
    new_batch["length_spk_embeds"] = length_spk_embeds
    new_batch["spk"] = spk
    new_batch["key"] = key
    new_batch["spk_label"] = torch.as_tensor(spk_label)
    return new_batch


def tse_collate_fn(batch, mode="min"):
    # This is a more generalizable implementation for target speaker extraction
    # Support arbitrary number of speakers
    new_batch = {}
    wav_mix = []
    wav_targets = []
    spk_embeds = []
    spk = []
    key = []
    spk_label = []
    length_spk_embeds = []
    for s in batch:
        for i in range(s["num_speaker"]):
            wav_mix.append(s["wav_mix"])
            wav_targets.append(s["wav_spk{}".format(i + 1)])
            spk.append(s["spk{}".format(i + 1)])
            key.append(s["key"])
            spk_embeds.append(
                torch.from_numpy(s["embed_spk{}".format(i + 1)].copy())
            )
            length_spk_embeds.append(spk_embeds[-1].shape[1])
            if "spk{}_label".format(i + 1) in s.keys():
                spk_label.append(s["spk{}_label".format(i + 1)])

    if not (len(set(length_spk_embeds)) == 1):
        if mode == "max":
            max_len = max(length_spk_embeds)
            for i in range(len(length_spk_embeds)):
                if len(spk_embeds[i].shape) == 2:
                    spk_embeds[i] = tf.pad(
                        spk_embeds[i],
                        (0, max_len - length_spk_embeds[i]),
                        "constant",
                        0,
                    )
                elif len(spk_embeds[i].shape) == 3:
                    spk_embeds[i] = tf.pad(
                        spk_embeds[i],
                        (0, 0, 0, max_len - length_spk_embeds[i]),
                        "constant",
                        0,
                    )
        if mode == "min":
            min_len = min(length_spk_embeds)
            for i in range(len(length_spk_embeds)):
                if len(spk_embeds[i].shape) == 2:
                    spk_embeds[i] = spk_embeds[i][:, :min_len]
                elif len(spk_embeds[i].shape) == 3:
                    spk_embeds[i] = spk_embeds[i][:, :min_len, :]

    new_batch["wav_mix"] = torch.concat(wav_mix)
    new_batch["wav_targets"] = torch.concat(wav_targets)
    new_batch["spk_embeds"] = torch.concat(spk_embeds)
    new_batch["length_spk_embeds"] = (
        length_spk_embeds  # Not used, but maybe needed when using the enrollment utterance
    )
    new_batch["spk"] = spk
    new_batch["key"] = key
    new_batch["spk_label"] = torch.as_tensor(spk_label)
    return new_batch


def Dataset(
    data_type,
    data_list_file,
    configs,
    spk2embed_dict=None,
    spk1_embed=None,
    spk2_embed=None,
    state="train",
    joint_training=False,
    dict_spk=None,
    whole_utt=False,
    repeat_dataset=False,
    noise_prob=0,
    reverb_prob=0,
    noise_enroll_prob=0,
    reverb_enroll_prob=0,
    specaug_enroll_prob=0,
    noise_lmdb_file=None,
    online_mix=False,
):
    """Construct dataset from arguments
    We have two shuffle stage in the Dataset. The first is global
    shuffle at shards tar/raw/feat file level. The second is local shuffle
    at training samples level.

    Args:
        :param spk2_embed:
        :param online_mix:
        :param spk1_embed:
        :param data_type(str): shard/raw/feat
        :param data_list_file: data list file
        :param configs: dataset configs
        :param noise_prob:probility to add noise on mixture
        :param reverb_prob:probility to add reverb on mixture
        :param noise_enroll_prob:probility to add noise on enrollment speech
        :param reverb_enroll_prob:probility to add reverb on enrollment speech
        :param specaug_enroll_prob: probility to apply SpecAug on fbank of enrollment speech
        :param noise_lmdb_file: noise data source lmdb file
        :param whole_utt: use whole utt or random chunk
        :param repeat_dataset:
    """
    assert data_type in ["shard", "raw"]
    lists = read_lists(data_list_file)
    shuffle = configs.get("shuffle", False)
    # Global shuffle
    dataset = DataList(lists, shuffle=shuffle, repeat_dataset=repeat_dataset)
    if data_type == "shard":
        dataset = Processor(dataset, processor.url_opener)
        if not online_mix:
            dataset = Processor(dataset, processor.tar_file_and_group)
        else:
            dataset = Processor(
                dataset, processor.tar_file_and_group_single_spk
            )
    else:
        dataset = Processor(dataset, processor.parse_raw)

    if configs.get("filter_len", False) and state == "train":
        # Filter the data with unwanted length
        filter_conf = configs.get("filter_args", {})
        dataset = Processor(dataset, processor.filter_len, **filter_conf)
    # Local shuffle
    if shuffle and not online_mix:
        dataset = Processor(
            dataset, processor.shuffle, **configs["shuffle_args"]
        )

    # resample
    resample_rate = configs.get("resample_rate", 16000)
    dataset = Processor(dataset, processor.resample, resample_rate)

    if not whole_utt:
        # random chunk
        chunk_len = configs.get("chunk_len", resample_rate * 3)
        dataset = Processor(dataset, processor.random_chunk, chunk_len)

    if online_mix:
        dataset = Processor(
            dataset,
            processor.mix_speakers,
            configs.get("num_speakers", 2),
            configs.get("online_buffer_size", 1000),
        )
        if reverb_prob > 0:
            dataset = Processor(dataset, processor.add_reverb, reverb_prob)
        dataset = Processor(
            dataset,
            processor.snr_mixer,
            configs.get("use_random_snr", False),
        )
    if noise_prob > 0:
        assert noise_lmdb_file is not None
        dataset = Processor(
            dataset, processor.add_noise, noise_lmdb_file, noise_prob
        )
    speaker_feat = configs.get("speaker_feat", False)
    if state == "train":
        if not joint_training:
            dataset = Processor(
                dataset, processor.sample_spk_embedding, spk2embed_dict
            )
        else:
            dataset = Processor(
                dataset, processor.sample_enrollment, spk2embed_dict, dict_spk
            )
            if reverb_enroll_prob > 0:
                dataset = Processor(
                    dataset, processor.add_reverb_on_enroll, reverb_enroll_prob
                )
            if noise_enroll_prob > 0:
                assert noise_lmdb_file is not None
                dataset = Processor(
                    dataset,
                    processor.add_noise_on_enroll,
                    noise_lmdb_file,
                    noise_enroll_prob,
                )
            if speaker_feat:
                dataset = Processor(
                    dataset, processor.compute_fbank, **configs["fbank_args"]
                )
                dataset = Processor(dataset, processor.apply_cmvn)
                if specaug_enroll_prob > 0:
                    dataset = Processor(
                        dataset, processor.spec_aug, prob=specaug_enroll_prob
                    )
    else:
        if not joint_training:
            dataset = Processor(
                dataset,
                processor.sample_fix_spk_embedding,
                spk2embed_dict,
                spk1_embed,
                spk2_embed,
            )
        else:
            dataset = Processor(
                dataset,
                processor.sample_fix_spk_enrollment,
                spk2embed_dict,
                spk1_embed,
                spk2_embed,
                dict_spk,
            )
            if speaker_feat:
                dataset = Processor(
                    dataset, processor.compute_fbank, **configs["fbank_args"]
                )
                dataset = Processor(dataset, processor.apply_cmvn)

    return dataset
