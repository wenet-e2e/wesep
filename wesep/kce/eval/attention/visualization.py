import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def draw_attention_heatmap(attention_scores, speech_lengths, keyword_lengths,
                           keyword_labels, match_cases, save_dir,
                           phoneme_map_path, batch_id=0, nth_layers=None, dpi=1000):
    if nth_layers is None:
        nth_layers = [9]

    idx2p = [line.split()[0] for line in open(phoneme_map_path)]

    for bidx in range(len(attention_scores)):
        ind = keyword_labels[bidx][:keyword_lengths[bidx]].detach().cpu().flip(0).tolist()
        ind = [idx2p[i] if i < len(idx2p) else str(i) for i in ind]

        for nth_layer in nth_layers:
            att = attention_scores[bidx][nth_layer - 1][0][
                :speech_lengths[bidx], :keyword_lengths[bidx]
            ]
            att = att.transpose(0, 1).flip(0).detach().cpu()
            att = att[2:-2, :]
            lab = ind[2:-2]

            df = pd.DataFrame(att, index=lab)
            plt.figure(figsize=(10, 5))
            sns.heatmap(df, annot=False, cmap="copper", yticklabels=lab)
            path = os.path.join(save_dir, f"batch_{batch_id}_sample_{bidx}.match_case_{match_cases[bidx]}.png")
            plt.savefig(path, dpi=dpi)
            plt.close()


def plot_metric_distribution(case_cache_dict, metric, save_dir, match_case_types=None):
    if match_case_types is None:
        match_case_types = ["one_match", "multiple_match", "zero_match"]

    plt.figure(figsize=(12, 6))
    for case in match_case_types:
        if case_cache_dict[case].get(metric):
            plt.hist(case_cache_dict[case][metric], bins=50, alpha=0.5, label=case)
    plt.xlabel(metric)
    plt.ylabel("Count")
    plt.title(f"Distribution of {metric}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"{metric}_hist.png"))
    plt.close()

    data, labels = [], []
    for case in match_case_types:
        if case_cache_dict[case].get(metric):
            data.append(case_cache_dict[case][metric])
            labels.append(case)

    if data:
        plt.figure(figsize=(8, 6))
        plt.boxplot(data, labels=labels)
        plt.ylabel(metric)
        plt.title(f"Boxplot of {metric}")
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"{metric}_boxplot.png"))
        plt.close()

    print(f"\nMetric: {metric}")
    for case, vals in [(c, case_cache_dict[c].get(metric, [])) for c in match_case_types]:
        if not vals:
            print(f"  {case}: no data")
            continue
        arr = np.array(vals)
        print(f"  {case}: mean={arr.mean():.4f}  std={arr.std():.4f}  min={arr.min():.4f}  max={arr.max():.4f}")


def _save_plot(fig, save_dir, name):
    fig.savefig(os.path.join(save_dir, name))
    plt.close(fig)


def plot_pr_curve(recalls, precisions, metric, save_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recalls, precisions, marker="o")
    ax.set_xlabel("Positive Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR Curve — {metric}")
    ax.grid(True)
    _save_plot(fig, save_dir, f"{metric}_pr_curve.png")


def plot_f1_curve(thresholds, f1_scores, metric, save_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thresholds, f1_scores, marker="x")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1")
    ax.set_title(f"F1 vs Threshold — {metric}")
    ax.grid(True)
    _save_plot(fig, save_dir, f"{metric}_f1_curve.png")


def plot_recall_curve(thresholds, pos_recalls, neg_recalls, metric, save_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thresholds, pos_recalls, label="Positive Recall")
    ax.plot(thresholds, neg_recalls, label="Negative Recall")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Recall")
    ax.set_title(f"Recall vs Threshold — {metric}")
    ax.legend()
    ax.grid(True)
    _save_plot(fig, save_dir, f"{metric}_recall_curve.png")


def plot_accuracy_curve(thresholds, pos_acc, neg_acc, metric, save_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thresholds, pos_acc, label="Positive Accuracy")
    ax.plot(thresholds, neg_acc, label="Negative Accuracy")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy vs Threshold — {metric}")
    ax.legend()
    ax.grid(True)
    _save_plot(fig, save_dir, f"{metric}_accuracy_curve.png")
