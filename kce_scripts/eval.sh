#!/bin/bash
# KCE Evaluation Script under wesep
# Usage: bash kce_scripts/eval.sh

set -euo pipefail

# Paths (absolute to avoid working-directory confusion)
KCE_ROOT="/home/zhanghaoyi/workspace/DAE-TSE/kce"
WESEP_ROOT="/home/zhanghaoyi/workspace/wesep"

# Model checkpoint
checkpoint_path="${KCE_ROOT}/exp/ls860_ctc1.0_sv0_bs48_epoch150_lr1e-3_sInterference1_nInterference0_4gpu/kwatt_asr_149.pt"

# Test data list
test_list="${KCE_ROOT}/data/dump/datalist/test-clean.jsonl"

# Decoding config
decoding_config="${WESEP_ROOT}/kce_configs/eval_fix_keyword.yaml"

# Inference result tag
inference_dirname_tag="wesep_verify/test-clean"

# GPU settings
use_gpu_id=${1:-0}
rank=0
world_size=1

# Run from KCE_ROOT so that relative paths in data.yaml resolve correctly
cd "${KCE_ROOT}"

# Activate wesep conda env (has compatible PyTorch CUDA version)
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate wesep 2>/dev/null || true

export PYTHONPATH="${WESEP_ROOT}:${KCE_ROOT}:${PYTHONPATH:-}"
export PYTHONIOENCODING=UTF-8

echo "============================================"
echo "KCE Evaluation (wesep package)"
echo "============================================"
echo "Checkpoint: ${checkpoint_path}"
echo "Test list:  ${test_list}"
echo "Config:     ${decoding_config}"
echo "GPU:        ${use_gpu_id}"
echo "Tag:        ${inference_dirname_tag}"
echo "============================================"

python -m wesep.bin.eval_kce \
    --test_list "${test_list}" \
    --checkpoint_path "${checkpoint_path}" \
    --inference_dirname_tag "${inference_dirname_tag}" \
    --decoding_config "${decoding_config}" \
    --use_gpu_id "${use_gpu_id}" \
    --rank "${rank}" \
    --world_size "${world_size}"

echo ""
echo "Evaluation completed!"
echo "Results in: $(dirname ${checkpoint_path})/decode/${inference_dirname_tag}/log/"
