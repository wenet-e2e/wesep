import numpy as np
from joblib import Parallel, delayed
from pesq import pesq
from pystoi.stoi import stoi


def cal_SISNR(est, ref, eps=1e-8):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        est: separated signal, numpy.ndarray, [T]
        ref: reference signal, numpy.ndarray, [T]
    Returns:
        SISNR
    """
    assert len(est) == len(ref)
    est_zm = est - np.mean(est)
    ref_zm = ref - np.mean(ref)

    t = np.sum(est_zm * ref_zm) * ref_zm / (np.linalg.norm(ref_zm) ** 2 + eps)
    return 20 * np.log10(
        eps + np.linalg.norm(t) / (np.linalg.norm(est_zm - t) + eps)
    )


def cal_SISNRi(est, ref, mix, eps=1e-8):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        est: separated signal, numpy.ndarray, [T]
        ref: reference signal, numpy.ndarray, [T]
    Returns:
        SISNR
    """
    assert len(est) == len(ref) == len(mix)
    sisnr1 = cal_SISNR(est, ref)
    sisnr2 = cal_SISNR(mix, ref)

    return sisnr1, sisnr1 - sisnr2


def cal_PESQ(est, ref):
    assert len(est) == len(ref)
    mode = "wb"
    p = pesq(16000, ref, est, mode)
    return p


def cal_PESQ_norm(est, ref):
    assert len(est) == len(ref)
    mode = "wb"
    try:
        # normalize PESQ to (0, 1)
        p = (pesq(16000, ref, est, mode) + 0.5) / 5
    except:
        # error can happen due to silent estimated signal
        p = None
    return p


def cal_PESQi(est, ref, mix):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        est: separated signal, numpy.ndarray, [T]
        ref: reference signal, numpy.ndarray, [T]
    Returns:
        SISNR
    """
    assert len(est) == len(ref) == len(mix)
    pesq1 = cal_PESQ(est, ref)
    pesq2 = cal_PESQ(mix, ref)

    return pesq1, pesq1 - pesq2


def cal_STOI(est, ref):
    assert len(est) == len(ref)
    p = stoi(ref, est, 16000)
    return p


def cal_STOIi(est, ref, mix):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        est: separated signal, numpy.ndarray, [T]
        ref: reference signal, numpy.ndarray, [T]
    Returns:
        SISNR
    """
    assert len(est) == len(ref) == len(mix)
    stoi1 = cal_STOI(est, ref)
    stoi2 = cal_STOI(mix, ref)

    return stoi1, stoi1 - stoi2


def batch_evaluation(metric, est, ref, lengths=None, parallel=False, n_jobs=8):
    """Calculate specified evaluation metrics in batches

    Args:
        metric (Callable): the function to calculate metric
        est (np.ndarray): separated signal, numpy.ndarray, [B, T]
        ref (np.ndarray): reference signal, numpy.ndarray, [B, T]
        lengths (np.ndarray, optional): specify the length of each signal. Defaults to None.
        parallel (bool, optional): whether to calculate metric in parallel. Default to False.
        n_jobs (int, optional): number of jobs, used when `parallel` is True. Defaults to 8.

    Returns:
        scores (np.ndarray): batched metrics, [B]
    """
    assert callable(metric)
    if lengths is not None:
        assert ((0 < lengths) & (lengths <= 1)).all()
        lengths = (lengths * est.size(1)).round().int().cpu()
        est = [p[:length].cpu() for p, length in zip(est, lengths)]
        ref = [t[:length].cpu() for t, length in zip(ref, lengths)]

    if parallel:
        while True:
            try:
                scores = Parallel(n_jobs=n_jobs, timeout=30)(
                    delayed(metric)(p, t) for p, t in zip(est, ref)
                )
                break
            except Exception as e:
                print(e)
                print("Evaluation timeout...... (will try again)")
    else:
        scores = []
        for p, t in zip(est, ref):
            score = metric(p, t)
            scores.append(score)

    if None in scores:
        return None

    return np.array(scores)
