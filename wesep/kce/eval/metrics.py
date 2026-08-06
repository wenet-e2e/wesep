from typing import Dict, List
from collections import defaultdict

import numpy as np
from editdistance import distance as editdistance

from wesep.utils.kce_utils import ctc_map, is_sublist


class KWSEvaluator:
    ALL_CASES = ["one_match", "multiple_match", "zero_match"]
    RESULT_KEYS = ["tp", "fp", "tn", "fn", "total_positive", "total_negative"]

    @classmethod
    def result_template(cls) -> Dict[str, int]:
        return {k: 0 for k in cls.RESULT_KEYS}

    @classmethod
    def build_flat_cache(cls) -> Dict[str, Dict[str, int]]:
        return {case: cls.result_template() for case in cls.ALL_CASES}

    def evaluate_batch(self, meta_data_dict, ctc_alignments, ground_truths, kws_targets, cache):
        for idx in range(len(ground_truths)):
            match_case = meta_data_dict["match_case"][idx]
            keyword_phn = meta_data_dict["keyword"][idx]
            hyp = ctc_map(ctc_alignments[idx])
            pred = int(is_sublist(hyp, keyword_phn))
            target = kws_targets[idx][0]

            cache[match_case]["total_positive"] += int(target == 1)
            cache[match_case]["total_negative"] += int(target == 0)

            if target == 0:
                if pred == 0:
                    cache[match_case]["tn"] += 1
                else:
                    cache[match_case]["fp"] += 1
            elif target == 1:
                if pred == 1:
                    cache[match_case]["tp"] += 1
                else:
                    cache[match_case]["fn"] += 1
            else:
                raise ValueError(f"Invalid kws_target: {target}")


class ASREvaluator:
    @classmethod
    def build_flat_cache(cls) -> Dict[str, defaultdict]:
        return {"one_match": defaultdict(list), "zero_match": defaultdict(list), "multiple_match": defaultdict(list)}

    def evaluate_batch(self, meta_data_dict, ctc_hyps, gts, cache):
        squeezed_hyps = [ctc_map(h) for h in ctc_hyps]
        for idx, (gt, hyp) in enumerate(zip(gts, squeezed_hyps)):
            per = editdistance(gt, hyp) / len(gt) if len(gt) > 0 else 0.0
            case = meta_data_dict["match_case"][idx]
            cache[case]["per"].append(per)


class MetricsComputer:
    @staticmethod
    def precision_recall_f1(pos_scores, neg_scores, threshold, reverse=False):
        pos_arr = np.array(pos_scores)
        neg_arr = np.array(neg_scores)
        if reverse:
            pos_preds = pos_arr <= threshold
            neg_preds = neg_arr <= threshold
        else:
            pos_preds = pos_arr >= threshold
            neg_preds = neg_arr >= threshold

        tp = np.sum(pos_preds)
        fp = np.sum(neg_preds)
        fn = len(pos_scores) - tp
        tn = len(neg_scores) - fp

        eps = 1e-8
        precision = tp / (tp + fp + eps)
        pos_recall = tp / (tp + fn + eps)
        neg_recall = tn / (tn + fp + eps)
        f1 = 2 * precision * pos_recall / (precision + pos_recall + eps)
        pos_acc = tp / len(pos_scores)
        neg_acc = tn / len(neg_scores)
        return precision, pos_recall, neg_recall, f1, pos_acc, neg_acc

    @staticmethod
    def compute_eer(y_true, y_score):
        from sklearn.metrics import roc_curve

        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
        fnr = 1 - tpr
        idx = np.nanargmin(np.abs(fnr - fpr))
        return (fpr[idx] + fnr[idx]) / 2, thresholds[idx]
