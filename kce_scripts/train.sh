#!/bin/bash
# KCE Training Script under wesep
# Usage: bash kce_scripts/train.sh --gpu 0,1 --config kce_configs/train.yaml

set -euo pipefail

WESEP_ROOT="/home/zhanghaoyi/workspace/wesep"
KCE_ROOT="/home/zhanghaoyi/workspace/DAE-TSE/kce"

# Default config
config="${WESEP_ROOT}/kce_configs/train.yaml"

# Parse custom args
gpu="0"
world_size=1
step=1
port="1234"

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu) gpu="$2"; shift 2 ;;
        --config) config="$2"; shift 2 ;;
        --step) step="$2"; shift 2 ;;
        --port) port="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Count GPUs
IFS=',' read -ra GPU_ARRAY <<< "$gpu"
num_gpu=${#GPU_ARRAY[@]}
world_size=$num_gpu

# Run from KCE_ROOT so that relative paths in data.yaml resolve correctly
cd "${KCE_ROOT}"

# Activate wesep conda env (has compatible PyTorch CUDA version)
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate wesep 2>/dev/null || true

export PYTHONPATH="${WESEP_ROOT}:${KCE_ROOT}:${PYTHONPATH:-}"
export PYTHONIOENCODING=UTF-8

echo "============================================"
echo "KCE Training (wesep package)"
echo "============================================"
echo "Config:      ${config}"
echo "GPUs:        ${gpu}"
echo "World size:  ${world_size}"
echo "Step:        ${step}"
echo "============================================"

if [ $num_gpu -eq 1 ]; then
    # Single GPU
    CUDA_VISIBLE_DEVICES=$gpu python -m wesep.bin.train_kce \
        --config "${config}" \
        --world_size 1 \
        --rank 0 \
        --gpu 0 \
        --step "${step}" \
        --port "${port}"
else
    # Multi-GPU with DDP
    for ((i=0; i<num_gpu; i++)); do
        rank=$i
        gpu_id=${GPU_ARRAY[$i]}
        CUDA_VISIBLE_DEVICES=$gpu_id python -m wesep.bin.train_kce \
            --config "${config}" \
            --world_size "${world_size}" \
            --rank "${rank}" \
            --gpu 0 \
            --step "${step}" \
            --port "${port}" &
        sleep 2
    done
    wait
fi

echo ""
echo "Training completed!"
