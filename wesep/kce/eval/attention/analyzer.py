import logging
from typing import Dict, List

import numpy as np
import torch

from .metrics import (
    compute_topk_ratio_normalized,
    compute_normalized_entropy,
    compute_spearman_and_highatt_ratio,
    compute_local_dtw,
    compute_local_cross_correlation,
    compute_sharpness,
    find_max_path,
)
from .visualization import draw_attention_heatmap
from wesep.utils.kce_utils import log_json_info


_KW_BORDER_TRIM = (2, -2)
_FRAME_BORDER_TRIM = (1, -1)


class AttentionAnalyzer:
    DEFAULT_CONFIG = {
        "nth_layers": [9],
        "high_att_threshold": 0.6,
        "min_high_att_frames": 4,
        "window_size_ratio": 3.5,
        "step_ratio": 0.5,
        "repeat_factor": 1,
        "top_percent": 0.05,
    }

    def __init__(self, config: dict = None):
        self.config = dict(self.DEFAULT_CONFIG)
        if config:
            self.config.update(config)

    def analyze_sample(self, att_scores, speech_len, keyword_len, uid="", match_case="", bidx=None) -> Dict[str, float]:
        results = {k: [] for k in [
            "sharpness", "top_5p", "normalized_entropy",
            "spearman_corr", "high_att_ratio",
            "local_dtw", "local_cross_correlation",
        ]}

        prefix = f"[{bidx}]" if bidx is not None else ""

        for nth_layer in self.config["nth_layers"]:
            att = att_scores[nth_layer - 1][0][:speech_len, :keyword_len].detach().cpu()
            correct_att = att[_FRAME_BORDER_TRIM[0]:_FRAME_BORDER_TRIM[1]]

            results["sharpness"].append(compute_sharpness(correct_att))
            results["top_5p"].append(compute_topk_ratio_normalized(correct_att, self.config["top_percent"]))
            results["normalized_entropy"].append(compute_normalized_entropy(correct_att))
            results["local_dtw"].append(compute_local_dtw(
                correct_att, window_size_ratio=self.config["window_size_ratio"],
                repeat_factor=self.config["repeat_factor"],
            ))
            results["local_cross_correlation"].append(compute_local_cross_correlation(
                correct_att, window_size_ratio=self.config["window_size_ratio"],
                step_ratio=self.config["step_ratio"],
            ))

            search_matrix = att.transpose(0, 1).detach().cpu()
            search_matrix = search_matrix[_KW_BORDER_TRIM[0]:_KW_BORDER_TRIM[1], :]
            this_kw_len = int(keyword_len) - 4

            max_sum, (start, end) = find_max_path(search_matrix)
            normed_max = float(max_sum / this_kw_len)

            log_json_info({
                "uid": uid, "match_case": match_case,
                "max_sum": max_sum, "keyword_len": this_kw_len,
                "normed_max_sum": normed_max,
                "start_frame_index": start, "end_frame_index": end,
            }, prefix=prefix)

            spearman_corr, high_att_ratio = compute_spearman_and_highatt_ratio(
                correct_att, high_att_threshold=self.config["high_att_threshold"],
                min_high_att_frames=self.config["min_high_att_frames"],
            )
            results["spearman_corr"].append(spearman_corr)
            results["high_att_ratio"].append(high_att_ratio)

        return {k: np.mean(v) for k, v in results.items()}

    def analyze_batch(self, stats_dict, match_cases, ret_cache, meta_data_dict, enable=True):
        if not enable:
            return

        att_scores = stats_dict["attention_maps"]
        speech_lens = stats_dict["speech_len"]
        keyword_lens = stats_dict["keyword_len"]
        uids = meta_data_dict["key"]

        for bidx, (uid, case) in enumerate(zip(uids, match_cases)):
            sample = self.analyze_sample(
                att_scores[bidx], speech_lens[bidx], keyword_lens[bidx],
                uid=uid, match_case=case, bidx=bidx,
            )
            for metric, value in sample.items():
                ret_cache[case][metric].append(value)

    def visualize_batch(self, stats_dict, match_cases, save_dir, phoneme_map_path,
                        batch_id=0, enable=True):
        if not enable:
            return

        draw_attention_heatmap(
            attention_scores=stats_dict["attention_maps"],
            speech_lengths=stats_dict["speech_len"],
            keyword_lengths=stats_dict["keyword_len"],
            keyword_labels=stats_dict["keyword_label"],
            match_cases=match_cases,
            save_dir=save_dir,
            phoneme_map_path=phoneme_map_path,
            batch_id=batch_id,
            nth_layers=self.config["nth_layers"],
        )
