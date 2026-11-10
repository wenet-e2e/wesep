# Metric implementations adapted from ESPnet enhancement scoring.
import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, Union

import soundfile
from mir_eval.separation import bss_eval_sources
from pystoi import stoi

from wesep.utils.score import cal_SISNR
from wesep.utils.utils import ArgumentParser, get_commandline_args, str2bool


def read_audio(path, sample_rate, channel, dtype):
    audio, rate = soundfile.read(path, dtype=dtype, always_2d=True)
    if not 0 <= channel < audio.shape[1]:
        raise ValueError(f"Channel {channel} is unavailable in {path}")
    audio = audio[:, channel]
    if rate != sample_rate:
        import torch
        from torchaudio.functional import resample

        audio = resample(torch.from_numpy(audio), rate, sample_rate).numpy()
    return audio


def metric_value(name, calculate, errors):
    try:
        value = float(calculate())
        if not math.isfinite(value):
            raise ValueError("non-finite result")
        return value
    except Exception as error:
        errors[name] = str(error)
        logging.warning("%s: %s", name, error)
        return None


def scoring(
    output_dir: str,
    dtype: str,
    log_level: Union[int, str],
    pairs: str,
    fs: str,
    target_channel: int,
    estimate_channel: int,
    use_dnsmos: bool,
    dnsmos_args: Dict,
    use_pesq: bool,
):
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    sample_rate = 16000 if fs == "16k" else 8000
    if use_dnsmos and sample_rate != 16000:
        raise ValueError("DNSMOS scoring requires --fs 16k")
    if use_dnsmos:
        if dnsmos_args["mode"] == "local":
            from wesep.utils.dnsmos import DNSMOS_local

            if not Path(dnsmos_args["primary_model"]).exists():
                raise ValueError(
                    f"The primary model {dnsmos_args['primary_model']} doesn't exist."
                    " You can download the model from https://github.com/microsoft/"
                    "DNS-Challenge/tree/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx")
            if not Path(dnsmos_args["p808_model"]).exists():
                raise ValueError(
                    f"The P808 model {dnsmos_args['p808_model']} doesn't exist."
                    " You can download the model from https://github.com/microsoft/"
                    "DNS-Challenge/tree/master/DNSMOS/DNSMOS/model_v8.onnx")
            dnsmos = DNSMOS_local(
                dnsmos_args["primary_model"],
                dnsmos_args["p808_model"],
                use_gpu=dnsmos_args["use_gpu"],
                convert_to_torch=dnsmos_args["convert_to_torch"],
                gpu_device=dnsmos_args["gpu_device"] - 1,
            )
            logging.warning("Using local DNSMOS models for evaluation")

        elif dnsmos_args["mode"] == "web":
            from wesep.utils.dnsmos import DNSMOS_web

            if not dnsmos_args["auth_key"]:
                raise ValueError(
                    "Please specify the authentication key for access to the Web-API. "
                    "You can apply for the AUTH_KEY at https://github.com/microsoft/"
                    "DNS-Challenge/blob/master/DNSMOS/README.md#to-use-the-web-api"
                )
            dnsmos = DNSMOS_web(dnsmos_args["auth_key"])
            logging.warning("Using the DNSMOS Web-API for evaluation")
    else:
        dnsmos = None

    if use_pesq:
        try:
            from pesq import PesqError, pesq

            logging.warning("Using the PESQ package for evaluation")
        except ImportError:
            raise ImportError(
                "Please install pesq and retry: pip install pesq") from None
    else:
        pesq = None

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(pairs, encoding="utf-8") as source, \
            (output / "scores.jsonl").open("w", encoding="utf-8") as writer:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            location = f"{pairs}:{line_number}"
            try:
                pair = json.loads(line)
                target_path, estimate_path = pair["target"], pair["estimate"]
                target = read_audio(target_path, sample_rate, target_channel, dtype)
                estimate = read_audio(estimate_path, sample_rate, estimate_channel, dtype)
                if target.size == 0 or target.shape != estimate.shape:
                    raise ValueError(
                        f"Audio lengths differ or are empty: {target.shape}, {estimate.shape}")
            except Exception as error:
                raise ValueError(f"{location}: {error}") from error
            logging.info("Scoring %s (%s)", location, pair.get("id", "no id"))
            errors = {}
            metrics = {
                "STOI": metric_value(
                    "STOI", lambda t=target, e=estimate: stoi(t, e, sample_rate) * 100, errors),
                "ESTOI": metric_value(
                    "ESTOI", lambda t=target, e=estimate: stoi(t, e, sample_rate, extended=True) * 100,
                    errors),
                "SI_SNR": metric_value(
                    "SI_SNR", lambda e=estimate, t=target: cal_SISNR(e, t), errors),
            }
            try:
                sdr, sir, sar, _ = bss_eval_sources(
                    target[None], estimate[None], compute_permutation=False)
                for name, values in [("SDR", sdr), ("SIR", sir), ("SAR", sar)]:
                    metrics[name] = metric_value(name, lambda v=values[0]: v, errors)
            except Exception as error:
                for name in ["SDR", "SIR", "SAR"]:
                    metrics[name] = None
                    errors[name] = str(error)
                logging.warning("BSS metrics: %s", error)
            if pesq:
                mode = "wb" if sample_rate == 16000 else "nb"
                name = f"PESQ_{mode.upper()}"

                def calculate_pesq(t=target, e=estimate, m=mode):
                    value = pesq(sample_rate, t, e, mode=m,
                                 on_error=PesqError.RETURN_VALUES)
                    if value < 0:
                        raise ValueError(f"PESQ error code {value}")
                    return value

                metrics[name] = metric_value(name, calculate_pesq, errors)
            if dnsmos:
                try:
                    result = dnsmos(estimate, sample_rate)
                    for name in ["BAK", "SIG", "OVRL", "P808_MOS"]:
                        metrics[name] = metric_value(name, lambda n=name, r=result: r[n], errors)
                except Exception as error:
                    for name in ["BAK", "SIG", "OVRL", "P808_MOS"]:
                        metrics[name] = None
                        errors[name] = str(error)
                    logging.warning("DNSMOS: %s", error)
            record = {"target": target_path, "estimate": estimate_path, "metrics": metrics}
            if "id" in pair:
                record["id"] = pair["id"]
            if errors:
                record["errors"] = errors
            writer.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            count += 1
    if not count:
        raise ValueError(f"No audio pairs found in {pairs}")
    logging.info("Scored %s audio pairs", count)


def merge_scores(inputs, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scores_path = output / "scores.jsonl"
    if any(Path(path).resolve() == scores_path.resolve() for path in inputs):
        raise ValueError("Merge inputs must differ from the merged scores.jsonl")
    totals, counts = {}, {}
    pair_count = 0
    with scores_path.open("w", encoding="utf-8") as writer:
        for path in inputs:
            with open(path, encoding="utf-8") as source:
                for line in source:
                    record = json.loads(line)
                    writer.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    pair_count += 1
                    for name, value in record["metrics"].items():
                        totals.setdefault(name, 0.0)
                        counts.setdefault(name, 0)
                        if value is not None and math.isfinite(value):
                            totals[name] += value
                            counts[name] += 1
    if not pair_count:
        raise ValueError("No scores to merge")
    summary = {name: {"mean": totals[name] / counts[name] if counts[name] else None,
                      "count": counts[name]} for name in totals}
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump({"num_pairs": pair_count, "metrics": summary}, stream, indent=2,
                  allow_nan=False)
        stream.write("\n")
    # Keep mean files available to the existing experiment-results viewer.
    for stale in output.glob("result_*.txt"):
        stale.unlink()
    with (output / "RESULTS.md").open("w", encoding="utf-8") as stream:
        stream.write(f"# Scoring results\n\nAudio pairs: {pair_count}\n\n")
        stream.write("| Metric | Mean | Valid pairs |\n| --- | --- | --- |\n")
        for name, values in summary.items():
            mean = values["mean"]
            text = f"{mean:.2f}" if mean is not None else "N/A"
            (output / f"result_{name.lower()}.txt").write_text(text + "\n")
            stream.write(f"| {name} | {text} | {values['count']} |\n")


def get_parser():
    parser = ArgumentParser(
        description="Score target/estimate audio pairs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use '_' instead of '-' as separator.
    # '-' is confusing if written in yaml.

    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float64"],
        help="Data type",
    )

    parser.add_argument("--pairs", help="JSONL with target, estimate and optional id")
    parser.add_argument("--merge_inputs", nargs="+", help="Worker scores.jsonl files")
    parser.add_argument("--fs", default="16k", choices=("8k", "16k"))
    parser.add_argument("--target_channel", type=int, default=0)
    parser.add_argument("--estimate_channel", type=int, default=0)

    group = parser.add_argument_group("DNSMOS related")
    group.add_argument("--use_dnsmos", type=str2bool, default=False)
    group.add_argument(
        "--dnsmos_mode",
        type=str,
        choices=("local", "web"),
        default="local",
        help="Use local DNSMOS model or web API for DNSMOS calculation",
    )
    group.add_argument(
        "--dnsmos_auth_key",
        type=str,
        default="",
        help="Required if dnsmsos_mode='web'",
    )
    group.add_argument(
        "--dnsmos_use_gpu",
        type=str2bool,
        default=False,
        help="used when dnsmsos_mode='local'",
    )
    group.add_argument(
        "--dnsmos_convert_to_torch",
        type=str2bool,
        default=False,
        help="used when dnsmsos_mode='local'",
    )
    group.add_argument("--dnsmos_primary_model",
                       type=str,
                       default="./DNSMOS/sig_bak_ovr.onnx",
                       help="Path to the primary DNSMOS model. "
                       "Required if dnsmsos_mode='local'")
    group.add_argument(
        "--dnsmos_p808_model",
        type=str,
        default="./DNSMOS/model_v8.onnx",
        help="Path to the p808 model. Required if dnsmsos_mode='local'",
    )
    group.add_argument("--dnsmos_gpu_device",
                       type=int,
                       default=1,
                       help="gpu device to use for DNSMOS evaluation. "
                       "Used when dnsmsos_mode='local'")

    group = parser.add_argument_group("PESQ related")
    group.add_argument(
        "--use_pesq",
        type=str2bool,
        default=False,
        help="Bebore setting this to True, please make sure that you or "
        "your institution have the license "
        "(check https://www.itu.int/rec/T-REC-P.862-200511-I!Amd2/en) to report PESQ",
    )
    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    kwargs = vars(parser.parse_args(cmd))
    kwargs.pop("config", None)
    merge_inputs = kwargs.pop("merge_inputs")
    if merge_inputs:
        merge_scores(merge_inputs, kwargs["output_dir"])
        return
    if not kwargs["pairs"]:
        parser.error("--pairs is required for scoring")
    dnsmos_args = {
        "mode": kwargs.pop("dnsmos_mode"),
        "auth_key": kwargs.pop("dnsmos_auth_key"),
        "primary_model": kwargs.pop("dnsmos_primary_model"),
        "p808_model": kwargs.pop("dnsmos_p808_model"),
        "use_gpu": kwargs.pop("dnsmos_use_gpu"),
        "convert_to_torch": kwargs.pop("dnsmos_convert_to_torch"),
        "gpu_device": kwargs.pop("dnsmos_gpu_device"),
    }
    kwargs["dnsmos_args"] = dnsmos_args
    scoring(**kwargs)


if __name__ == "__main__":
    main()
