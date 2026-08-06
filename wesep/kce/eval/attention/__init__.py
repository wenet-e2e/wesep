"""
注意力分析模块初始化
"""
from .metrics import (
    compute_topk_ratio_normalized,
    compute_normalized_entropy,
    compute_spearman_and_highatt_ratio,
    compute_local_dtw,
    compute_local_cross_correlation,
    compute_sharpness,
    find_max_path
)
from .visualization import (
    draw_attention_heatmap,
    plot_metric_distribution,
    plot_pr_curve,
    plot_f1_curve,
    plot_recall_curve,
    plot_accuracy_curve
)
from .analyzer import AttentionAnalyzer

__all__ = [
    # Metrics
    'compute_topk_ratio_normalized',
    'compute_normalized_entropy',
    'compute_spearman_and_highatt_ratio',
    'compute_local_dtw',
    'compute_local_cross_correlation',
    'compute_sharpness',
    'find_max_path',
    # Visualization
    'draw_attention_heatmap',
    'plot_metric_distribution',
    'plot_pr_curve',
    'plot_f1_curve',
    'plot_recall_curve',
    'plot_accuracy_curve',
    # Analyzer
    'AttentionAnalyzer',
]
