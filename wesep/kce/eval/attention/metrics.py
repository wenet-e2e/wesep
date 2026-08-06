import numpy as np
import torch
from scipy.stats import spearmanr
from fastdtw import fastdtw
from scipy.signal import correlate


def compute_topk_ratio_normalized(att_map, top_percent=0.05):
    att_flat = att_map.flatten()
    total = att_flat.numel()
    k = max(1, int(total * top_percent))
    topk = torch.topk(att_flat, k).values
    return (topk.sum() / att_flat.sum()).item()


def compute_normalized_entropy(att_map):
    T, K = att_map.shape
    eps = 1e-12
    entropy = -torch.sum(att_map * torch.log(att_map + eps), dim=1)
    max_entropy = torch.log(torch.tensor(K, dtype=att_map.dtype, device=att_map.device))
    return (entropy / max_entropy).mean().item()


def compute_spearman_and_highatt_ratio(att_map, high_att_threshold=0.1, min_high_att_frames=5):
    att = att_map.detach().cpu()
    max_indices = torch.argmax(att, dim=-1)
    max_values = torch.max(att, dim=-1).values

    mask = max_values > high_att_threshold
    frame_idx = torch.arange(att.shape[0])[mask]
    kw_idx = max_indices[mask]

    if len(frame_idx) >= min_high_att_frames:
        corr, _ = spearmanr(frame_idx.numpy(), kw_idx.numpy())
    else:
        corr = 0.0

    ratio = len(frame_idx) / att.shape[0]
    return corr, ratio


def compute_local_dtw(att_map, window_size_ratio=3.5, repeat_factor=1):
    att_map = att_map.detach().cpu()
    T, K = att_map.shape
    max_indices = torch.argmax(att_map, dim=-1).numpy()
    kw_seq = np.repeat(np.arange(K), repeats=repeat_factor)

    win = min(int(K * window_size_ratio), T)
    step = max(1, int(win / 2))
    best = float("inf")

    for start in range(0, T - win + 1, step):
        dist, _ = fastdtw(max_indices[start:start + win], kw_seq, dist=lambda x, y: abs(x - y))
        if dist < best:
            best = dist
    return best / K


def compute_local_cross_correlation(att_map, window_size_ratio=3.5, step_ratio=0.5):
    att_map = att_map.detach().cpu()
    T, K = att_map.shape
    max_indices = torch.argmax(att_map, dim=-1).numpy()

    win = max(int(K * window_size_ratio), K)
    step = max(1, int(win * step_ratio))
    best = 0.0

    for start in range(0, T - win + 1, step):
        frames = np.arange(win)
        kw_idx = max_indices[start:start + win]
        if np.std(kw_idx) < 1e-6:
            continue
        frames_n = (frames - frames.mean()) / (frames.std() + 1e-8)
        kw_idx_n = (kw_idx - kw_idx.mean()) / (kw_idx.std() + 1e-8)
        corr = np.abs(correlate(frames_n, kw_idx_n, mode="valid")).max() / win
        if corr > best:
            best = corr
    return best


def compute_sharpness(att_map):
    return (torch.max(att_map) / torch.mean(att_map)).item()


def find_max_path(matrix):
    K, T = matrix.shape
    dp = [[0.0] * T for _ in range(K)]
    prev = [[None] * T for _ in range(K)]

    for t in range(T):
        dp[0][t] = matrix[0][t]

    for k in range(1, K):
        for t in range(1, T):
            if dp[k - 1][t - 1] > dp[k][t - 1]:
                dp[k][t] = dp[k - 1][t - 1] + matrix[k][t]
                prev[k][t] = (k - 1, t - 1)
            else:
                dp[k][t] = dp[k][t - 1] + matrix[k][t]
                prev[k][t] = (k, t - 1)

    max_sum = max(dp[K - 1])
    k, t = K - 1, np.argmax(dp[-1])

    while k == K - 1:
        k, t = prev[k][t]
    end = t + 1

    while k > 0 and t > 0:
        k, t = prev[k][t]
    start = t

    return float(max_sum), (start, end)
