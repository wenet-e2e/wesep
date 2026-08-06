import os
from typing import Dict, List

import numpy as np

from wesep.kce.eval.metrics import MetricsComputer
from wesep.kce.eval.attention.visualization import (
    plot_pr_curve,
    plot_f1_curve,
    plot_recall_curve,
    plot_accuracy_curve,
    plot_metric_distribution,
)


class ThresholdSweeper:
    def __init__(self, num_thresholds=200, save_dir=None):
        self.num_thresholds = num_thresholds
        self.save_dir = save_dir

    def sweep(self, pos_scores, neg_scores, reverse=False):
        if not pos_scores or not neg_scores:
            return None

        scores = np.array(pos_scores + neg_scores)
        thresholds = np.linspace(scores.min(), scores.max(), self.num_thresholds)

        results = {k: [] for k in [
            "thresholds", "precisions", "pos_recalls", "neg_recalls",
            "f1s", "recall_sums", "pos_accuracies", "neg_accuracies",
        ]}

        best = {"recall_sum": 0}
        for thresh in thresholds:
            precision, r_pos, r_neg, f1, acc_pos, acc_neg = MetricsComputer.precision_recall_f1(
                pos_scores, neg_scores, thresh, reverse
            )
            results["thresholds"].append(thresh)
            results["precisions"].append(precision)
            results["pos_recalls"].append(r_pos)
            results["neg_recalls"].append(r_neg)
            results["f1s"].append(f1)
            results["recall_sums"].append(r_pos + r_neg)
            results["pos_accuracies"].append(acc_pos)
            results["neg_accuracies"].append(acc_neg)

            if (r_pos + r_neg) > best["recall_sum"]:
                best.update(
                    recall_sum=r_pos + r_neg, threshold=thresh,
                    precision=precision, pos_recall=r_pos, neg_recall=r_neg,
                    f1=f1, pos_accuracy=acc_pos, neg_accuracy=acc_neg,
                )

        return {"scan_results": results, "best_result": best}

    def plot_and_save(self, sweep_results, metric_name):
        if sweep_results is None or self.save_dir is None:
            return

        r = sweep_results["scan_results"]
        best = sweep_results["best_result"]

        plot_pr_curve(r["pos_recalls"], r["precisions"], metric_name, self.save_dir)
        plot_f1_curve(r["thresholds"], r["f1s"], metric_name, self.save_dir)
        plot_recall_curve(r["thresholds"], r["pos_recalls"], r["neg_recalls"], metric_name, self.save_dir)
        plot_accuracy_curve(r["thresholds"], r["pos_accuracies"], r["neg_accuracies"], metric_name, self.save_dir)

        path = os.path.join(self.save_dir, f"{metric_name}_best_result.txt")
        with open(path, "w") as f:
            for k, v in best.items():
                f.write(f"{k}: {v:.4f}\n" if isinstance(v, float) else f"{k}: {v}\n")

    def analyze_metric_for_match_cases(self, case_cache_dict, metric, reverse=False, save_plots=True):
        pos = case_cache_dict["one_match"].get(metric, [])
        neg = case_cache_dict["zero_match"].get(metric, [])

        print(f"\nMetric: {metric}")
        print(f"  pos: {len(pos)} samples  neg: {len(neg)} samples")
        if pos and neg:
            print(f"  pos range: [{min(pos):.4f}, {max(pos):.4f}]  "
                  f"neg range: [{min(neg):.4f}, {max(neg):.4f}]")

        sw = self.sweep(pos, neg, reverse)
        if sw is None:
            print(f"  Warning: no valid threshold for {metric}")
            return None

        best = sw["best_result"]
        print(f"  best thr={best['threshold']:.4f}  P={best['precision']:.4f}  "
              f"R+={best['pos_recall']:.4f}  R-={best['neg_recall']:.4f}  F1={best['f1']:.4f}")

        if save_plots and self.save_dir:
            self.plot_and_save(sw, metric)
            plot_metric_distribution(case_cache_dict, metric, self.save_dir)

        return best
