#!/bin/bash
set -euo pipefail

WESEP_ROOT=${WESEP_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
export PYTHONPATH="${WESEP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
pairs=
output_dir=
score_nj=16
n_gpu=1
target_channel=0
estimate_channel=0
use_pesq=false
use_dnsmos=false
dnsmos_use_gpu=true
fs=16k

. "${WESEP_ROOT}/tools/parse_options.sh" || exit 1
if [ ! -s "${pairs}" ] || [ -z "${output_dir}" ]; then
    echo "Usage: $0 --pairs audio_pairs.jsonl --output_dir scoring [options]" >&2
    exit 1
fi
if ! [[ "${score_nj}" =~ ^[1-9][0-9]*$ && "${n_gpu}" =~ ^[1-9][0-9]*$ ]]; then
    echo "score_nj and n_gpu must be positive integers" >&2
    exit 1
fi
if [ "${fs}" != 16k ] && ${use_dnsmos}; then
    echo "Warning: disabling DNSMOS because it requires --fs 16k."
    use_dnsmos=false
fi

_logdir="${output_dir}/logdir"
mkdir -p "${_logdir}"
_logdir=$(cd "${_logdir}" && pwd)
echo "Preparing audio pairs: ${pairs}"
# Give each worker complete pairs, preserving input order and repeated IDs.
python - "${pairs}" "${_logdir}/pairs.jsonl" <<'PYTHON'
import json
import os
import sys

if os.path.realpath(sys.argv[1]) == os.path.realpath(sys.argv[2]):
    raise ValueError("Input pairs must differ from the working pairs file")

with open(sys.argv[1], encoding="utf-8") as source, \
        open(sys.argv[2], "w", encoding="utf-8") as output:
    for number, line in enumerate(source, 1):
        if not line.strip():
            continue
        try:
            pair = json.loads(line)
            for name in ("target", "estimate"):
                pair[name] = os.path.abspath(pair[name])
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"{sys.argv[1]}:{number}: {error}") from error
        output.write(json.dumps(pair, ensure_ascii=False) + "\n")
PYTHON
num_pairs=$(wc -l < "${_logdir}/pairs.jsonl")
if [ "${num_pairs}" -eq 0 ]; then
    echo "Error: no audio pairs in ${pairs}" >&2
    exit 1
fi
_nj=${score_nj}
if ${use_dnsmos} && ${dnsmos_use_gpu} && [ "${_nj}" -gt "${n_gpu}" ]; then
    _nj=${n_gpu}
fi
if [ "${_nj}" -gt "${num_pairs}" ]; then
    _nj=${num_pairs}
fi
split_pairs=()
score_files=()
for n in $(seq "${_nj}"); do
    split_pairs+=("${_logdir}/pairs.${n}.jsonl")
    score_files+=("${_logdir}/output.${n}/scores.jsonl")
    mkdir -p "${_logdir}/output.${n}"
    : > "${_logdir}/output.${n}/scores.jsonl"
    : > "${_logdir}/tse_scoring.${n}.log"
done
LC_ALL=C perl "${WESEP_ROOT}/tools/split_scp.pl" "${_logdir}/pairs.jsonl" "${split_pairs[@]}"

metrics="STOI ESTOI SI_SNR SDR SIR SAR"
if ${use_pesq}; then metrics+=" PESQ"; fi
dnsmos_device=disabled
if ${use_dnsmos}; then
    metrics+=" DNSMOS"
    dnsmos_device=CPU
    if ${dnsmos_use_gpu}; then dnsmos_device="GPU (requested)"; fi
fi
printf 'Audio pairs: %s\nScoring workers: %s\nMetrics: %s\n' "${num_pairs}" "${_nj}" "${metrics}"
printf 'DNSMOS device: %s\nOutput: %s\nWorker logs: %s/tse_scoring.*.log\n' \
    "${dnsmos_device}" "${output_dir}" "${_logdir}"

scoring_opts=()
if ${use_dnsmos}; then
    primary_model_path="${PWD}/DNSMOS/sig_bak_ovr.onnx"
    p808_model_path="${PWD}/DNSMOS/model_v8.onnx"
    mkdir -p DNSMOS
    for model in sig_bak_ovr.onnx model_v8.onnx; do
        if [ ! -s "DNSMOS/${model}" ]; then
            wget -O "DNSMOS/${model}" \
                "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/${model}"
        fi
    done
    scoring_opts+=(--dnsmos_mode local
        --dnsmos_primary_model "${primary_model_path}"
        --dnsmos_p808_model "${p808_model_path}"
        --dnsmos_use_gpu "${dnsmos_use_gpu}")
fi

echo "Initializing scoring workers; waiting for the first result..."
# Monitor flushed result rows while run.pl owns the scoring workers.
status=0
python - "${_logdir}" "${num_pairs}" "${_nj}" \
    "${WESEP_ROOT}/tools/run.pl" \
    python -u -m wesep.bin.score \
    --pairs "${_logdir}/pairs.JOB.jsonl" \
    --output_dir "${_logdir}/output.JOB" \
    --fs "${fs}" \
    --target_channel "${target_channel}" \
    --estimate_channel "${estimate_channel}" \
    --use_pesq "${use_pesq}" \
    --use_dnsmos "${use_dnsmos}" \
    --dnsmos_gpu_device JOB \
    "${scoring_opts[@]}" <<'PYTHON' &
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from tqdm import tqdm

logdir, total, jobs, runner = sys.argv[1:5]
total, jobs = int(total), int(jobs)
files = [Path(logdir) / f"output.{i}/scores.jsonl" for i in range(1, jobs + 1)]
started = time.monotonic()
previous = 0
pending_notices = {Path(logdir) / f"tse_scoring.{i}.log" for i in range(1, jobs + 1)}
seen_notices = set()
interactive = sys.stdout.isatty()


def interrupted(signum, frame):
    raise KeyboardInterrupt


signal.signal(signal.SIGTERM, interrupted)
process = subprocess.Popen(
    [runner, f"JOB=1:{jobs}", "tse_scoring.JOB.log", *sys.argv[5:]],
    cwd=logdir, start_new_session=True, env=dict(os.environ, LC_ALL="C"))
try:
    with tqdm(total=total, desc="Scoring", unit="pair", file=sys.stdout,
              disable=not interactive) as bar:
        while True:
            code = process.poll()
            completed = 0
            for path in files:
                with path.open(encoding="utf-8") as stream:
                    completed += sum(line.endswith("\n") for line in stream)
            bar.update(completed - previous)
            previous = completed
            # Surface provider fallback warnings from worker logs in the terminal.
            for path in tuple(pending_notices):
                if not path.exists():
                    continue
                text = path.read_text(errors="replace")
                for line in text.splitlines():
                    if "DNSMOS_CPU_FALLBACK:" in line:
                        notice = line.split("DNSMOS_CPU_FALLBACK:", 1)[1].strip()
                        if notice not in seen_notices:
                            bar.write(f"WARNING: {notice}", file=sys.stdout)
                            seen_notices.add(notice)
                        pending_notices.discard(path)
                        break
                if "Starting pair scoring:" in text:
                    pending_notices.discard(path)
            if not interactive:
                elapsed = time.monotonic() - started
                eta = f"{elapsed * (total - completed) / completed:.0f}s" if completed else "pending"
                print(f"Scoring: {completed}/{total} ({completed / total:.1%}); "
                      f"elapsed {elapsed:.0f}s; ETA {eta}", flush=True)
            if code is not None:
                break
            time.sleep(5)
    if code:
        raise SystemExit(code)
except KeyboardInterrupt:
    print("\nScoring interrupted; stopping this run's workers.", file=sys.stderr)
    raise SystemExit(130)
finally:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
PYTHON
monitor_pid=$!
stop_scoring() {
    trap - INT TERM
    kill -TERM "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
    exit "$1"
}
trap 'stop_scoring 130' INT
trap 'stop_scoring 143' TERM
wait "${monitor_pid}" || status=$?
trap - INT TERM
if [ "${status}" -ne 0 ]; then
    echo "Scoring failed (exit ${status}). Worker logs: ${_logdir}/tse_scoring.*.log" >&2
    for n in $(seq "${_nj}"); do
        log="${_logdir}/tse_scoring.${n}.log"
        if [ -f "${log}" ]; then
            echo "--- ${log} ---" >&2
            tail -n 12 "${log}" >&2
        fi
    done
    exit "${status}"
fi

echo "Scoring workers finished. Merging results..."

python -m wesep.bin.score --output_dir "${output_dir}" \
    --merge_inputs "${score_files[@]}"
echo "Scores: ${output_dir}/scores.jsonl"
echo "Summary: ${output_dir}/RESULTS.md"
