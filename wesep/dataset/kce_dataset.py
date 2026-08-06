# ref: wenet dataset.py
import json
import copy
import random
import torch
from torch.utils.data import IterableDataset

from wesep.utils.kce_utils import read_list
from wesep.dataset.dataset import Processor, DistributedSampler
from wesep.dataset.kce_processor import (
    process_raw,
    process_interference,
    process_speech_feats,
    process_fix_keyword,
    process_text_feats,
    process_sampled_keyword_from_label,
    process_sot_label,
    process_list_data,
    process_save_meta,
    make_length,
    filter_max_length,
    make_batch,
    process_speaker_label,
    fetch_tensor,
    fetch_inference_data,
)


class KceDataList(IterableDataset):
    """KCE-specific DataList with support for speech/noise/RIR interference,
    predefined keyword sampling, and multi-match positive sampling.

    Extends wesep's simple DataList pattern with KWS+ASR data pipeline features.
    """
    def __init__(
        self,
        lists: list[str],   # jsonl format
        speech_interference: bool = False,
        noise_interference_list=None,
        rirs_list=None,
        shuffle=True,
        partition=True,
        predefined_keyword_config: dict = None,
    ):
        self.lists = lists

        self.uid2meta = {meta['key']: meta for meta in [json.loads(line) for line in lists]}

        self.sampler = DistributedSampler(shuffle, partition)
        self.speech_interference = speech_interference
        if noise_interference_list is not None:
            self.noise_interference_list = noise_interference_list
            self.num_noise_interference = len(self.noise_interference_list)
            self.noise_interference = True
        else:
            self.noise_interference = False
        if rirs_list is not None:
            self.rirs_list = rirs_list
            self.reverb = True
        else:
            self.reverb = False

        if predefined_keyword_config:
            self.pick_predefined_prob = predefined_keyword_config.get('pick_predefined_prob', 0)
            self.kwd2utts_jsonl = predefined_keyword_config.get('kwd2uttids_jsonl', None)
            assert self.pick_predefined_prob == 0 or self.kwd2utts_jsonl is not None
        else:
            self.pick_predefined_prob = 0
            self.kwd2utts_jsonl = None

        if self.kwd2utts_jsonl:
            with open(self.kwd2utts_jsonl, 'r') as rf:
                self.kwd2utts_list = [json.loads(line) for line in rf.readlines()]

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def padding(self, org_list, target_len):
        org_len = len(org_list)
        assert org_len < target_len
        num_repeat = target_len // org_len
        new_list = copy.deepcopy(org_list)
        for x in range(num_repeat):
            new_list += org_list
        return new_list

    def speech_interference_candidate(self, lists, indexes, num_candidate=5):
        idx = random.choices(indexes, k=num_candidate)
        candidate = []
        for i in idx:
            candidate.append(lists[i])
        if num_candidate == 1:
            candidate = candidate[0]
        return candidate

    def __iter__(self):
        sampler_info = self.sampler.update()
        world_indexes = list(range(len(self.lists)))
        rank_indexes = self.sampler.sample(world_indexes)
        if self.speech_interference:
            self_corrupt_lists = copy.deepcopy(self.lists)
            self_corrupt_indexes = list(range(len(self_corrupt_lists)))
            self_corrupt_indexes = self.sampler.sample(self_corrupt_indexes)
        if self.noise_interference:
            none_target_corrupt_lists = self.noise_interference_list
            none_target_corrupt_indexes = list(range(len(none_target_corrupt_lists)))
            none_target_corrupt_indexes = self.sampler.sample(none_target_corrupt_indexes)
        if self.reverb:
            rirs_lists = self.rirs_list
            rirs_indexes = list(range(len(rirs_lists)))
            rirs_indexes = self.sampler.sample(rirs_indexes)

        if self.kwd2utts_jsonl:
            # with kwd2utts_json file, use sampler for data sharding
            world_kwd2utts_indexes = list(range(len(self.kwd2utts_list)))
            rank_kwd2utts_indexes = self.sampler.sample(world_kwd2utts_indexes)

        for i, index in enumerate(rank_indexes):
            # random: zero-match (negative) & one-match
            if random.random() > self.pick_predefined_prob:
                sample = dict(src=self.lists[index], epoch=self.sampler.epoch, match_case='zero_or_one_match')
                sample.update(sampler_info)
                if self.speech_interference:
                    speech_interference_candidates = self.speech_interference_candidate(
                        self_corrupt_lists, self_corrupt_indexes, num_candidate=150
                    )
                    sample.update(speech_interference_candidates=copy.deepcopy(speech_interference_candidates))

                    one_neg_candidate = self.speech_interference_candidate(
                        self_corrupt_lists, self_corrupt_indexes, num_candidate=150
                    )
                    sample.update(neg_candidate=one_neg_candidate)
                    # sample keys: src, sample_info, speech_interference_candidates, neg_candidate
            # multi-match
            else:
                sample = dict(epoch=self.sampler.epoch, match_case='multiple_match')
                sample.update(sampler_info)

                kwd2utts_index = random.choice(rank_kwd2utts_indexes)
                """
                {
                    "keyword": "something",
                    "keyword_phn_length": 6,
                    "uids: [
                        {uid1, keyword_start_position, start_time}, {uid2, keyword_start_position, start_time}]" : List[Dict]
                    "keyword_phn_label": [55, 8, 44, 59, 35, 46], "uids": ["1988-147956-0012","2428-83705-0033"]
                }
                """
                kwd2utts_sample: dict = self.kwd2utts_list[kwd2utts_index]
                uids_info = random.choices(kwd2utts_sample['uids'], k=2)

                # set the utterance whose keyword appears first as the source utterance.
                source, interference = uids_info
                if source['start_time'] > interference['start_time']:
                    source, interference = interference, source

                # uid -> speech label
                sample.update(src=json.dumps(self.uid2meta[source['uid']]))
                sample.update(
                    speech_interference_candidates=[
                        json.dumps(self.uid2meta[interference['uid']])
                    ]
                )

                sample.update(is_multiple_match=True)

                sample.update(predefined_keyword_info={
                    'keyword': [kwd2utts_sample['phn_label']],  # we need an additional pair of brackets
                    'keyword_start_position': source['keyword_start_position']
                })

            if self.noise_interference:
                none_target_corrupt_candidates = self.speech_interference_candidate(
                    none_target_corrupt_lists, none_target_corrupt_indexes, num_candidates=5
                )
                sample.update(noise_interference_candidates=none_target_corrupt_candidates)

            if self.reverb:
                rirs_src = self.speech_interference_candidate(rirs_lists, rirs_indexes, num_candidate=1)
                sample.update(rirs=rirs_src)
            # sample keys: src, sample_info, speech_interference_candidates, multiple_match, predefined_keyword_info
            yield sample


def build_kce_dataset(data_conf: dict, data_list, tag_of_dataset='train'):
    """Build KCE data pipeline using wesep's Processor pattern.

    Args:
        data_conf: data configuration dict from YAML
        data_list: list of JSONL strings
        tag_of_dataset: 'train', 'valid', or 'test'

    Returns:
        Processor-wrapped IterableDataset pipeline
    """
    assert tag_of_dataset in ('train', 'valid', 'test')

    # check speech config
    speech_config: dict = data_conf.get('speech_config', None)
    if not speech_config:
        raise NotImplementedError(
            "speech_config should be specific, there are no any default config for " +
            "speech feats"
        )
    input_data_config = {'lists': data_list}
    shuffle = data_conf.get('shuffle', True)
    input_data_config.update({'shuffle': shuffle})

    addition_noise_config = {}
    if speech_config.get('speech_interference', {}).get('use', False):
        addition_noise_config.update(speech_interference=speech_config['speech_interference'])
        input_data_config.update({'speech_interference': True})

    if speech_config.get('noise_interference') is not None:
        addition_noise_config.update(noise_interference=speech_config['noise_interference'])
        if speech_config['noise_interference'].get('use', False):
            noise_interference_list = addition_noise_config['noise_interference']['corrupt_list']
            noise_interference_list = read_list(noise_interference_list)
            input_data_config.update({'noise_interference_list': noise_interference_list})

    if speech_config.get('rirs_list', False):
        rirs_list = speech_config['rirs_list']
        rirs_list = read_list(rirs_list)
        input_data_config.update({'rirs_list': rirs_list})

    sv_config = data_conf.get('sv_config', None)

    predefined_keyword_config = {}
    if "predefined_keyword_config" in data_conf:
        if tag_of_dataset == "train":
            predefined_keyword_config = data_conf['predefined_keyword_config'].get('train', None)
        elif tag_of_dataset == "valid":
            predefined_keyword_config = data_conf['predefined_keyword_config'].get('valid', None)
        else:
            # test
            predefined_keyword_config = {}

    input_data_config.update(predefined_keyword_config=predefined_keyword_config)

    # jsonl
    dataset = KceDataList(**input_data_config)

    # START DATA PROCESS!!!
    dataset = Processor(dataset, process_raw)

    if len(addition_noise_config) > 0:
        dataset = Processor(dataset, process_interference, addition_noise_config)

    # prepare speech feats
    dataset = Processor(dataset, process_speech_feats, speech_config)

    # prepare text feats, such as label, keyword etc.
    dataset = Processor(dataset, process_text_feats)

    # process keyword setting randomly
    keyword_selection_conf = data_conf['keyword_selection_conf']
    keyword_selection_strategy = keyword_selection_conf.pop('selection_strategy')
    keyword_selection_conf: dict = keyword_selection_conf['config']
    assert keyword_selection_strategy in ['random_sample', 'fixed_selection_from_jsonl']
    if keyword_selection_strategy == 'random_sample':
        keyword_selection_conf.update({'neg_len': 70})
        dataset = Processor(dataset, process_sampled_keyword_from_label, **keyword_selection_conf)
    elif keyword_selection_strategy == 'fixed_selection_from_jsonl':
        dataset = Processor(dataset, process_fix_keyword, **keyword_selection_conf)
    else:
        raise NotImplementedError("Not supported keyword selection strategy.")

    sot_label_config = data_conf.get('sot_config', None)
    if sot_label_config:
        dataset = Processor(dataset, process_sot_label, **sot_label_config)

    # save metadata of random online simulation
    meta_save_config: dict = data_conf.get('meta_save_config', {})
    if meta_save_config.get('use', False):
        dataset = Processor(dataset, process_save_meta, meta_save_config)

    # process list data
    dataset = Processor(dataset, process_list_data)

    # process length information
    dataset = Processor(dataset, make_length)

    # filter overly long mixed-speech samples to prevent OOM
    max_frames = speech_config.get('max_frames', None)
    if max_frames is not None:
        dataset = Processor(dataset, filter_max_length, max_frames)

    # read speaker label
    if sv_config:
        dataset = Processor(dataset, process_speaker_label, **sv_config)

    # make batch
    dataset = Processor(dataset, make_batch, data_conf.get('batch_size', 256))

    # if fetch_key from config is not None, use that fetch_key
    # fetch tensor
    fetch_key: list[str] = data_conf.get('fetch_key', 'speech,label,keyword').split(",")
    fetch_meta_key: list[str] = data_conf.get('fetch_meta_key', None)
    if fetch_meta_key:
        fetch_meta_key = fetch_meta_key.split(",")

    if tag_of_dataset in ('train', 'valid'):
        dataset = Processor(dataset, fetch_tensor, fetch_key)
    elif tag_of_dataset in ('test'):
        dataset = Processor(dataset, fetch_inference_data, fetch_key, fetch_meta_key)

    return dataset
