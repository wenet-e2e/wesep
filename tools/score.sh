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
done
perl "${WESEP_ROOT}/tools/split_scp.pl" "${_logdir}/pairs.jsonl" "${split_pairs[@]}"

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

# Each process writes its own result file; merge only after all jobs succeed.
(
    cd "${_logdir}"
    "${WESEP_ROOT}/tools/run.pl" JOB=1:"${_nj}" tse_scoring.JOB.log \
        python -m wesep.bin.score \
        --pairs "${_logdir}/pairs.JOB.jsonl" \
        --output_dir "${_logdir}/output.JOB" \
        --fs "${fs}" \
        --target_channel "${target_channel}" \
        --estimate_channel "${estimate_channel}" \
        --use_pesq "${use_pesq}" \
        --use_dnsmos "${use_dnsmos}" \
        --dnsmos_gpu_device JOB \
        "${scoring_opts[@]}"
)

python -m wesep.bin.score --output_dir "${output_dir}" \
    --merge_inputs "${score_files[@]}"
echo "Scores: ${output_dir}/scores.jsonl"
echo "Summary: ${output_dir}/RESULTS.md"
