#!/usr/bin/env bash

# Copyright 2026 Ke Zhang (kylezhang1118@gmail.com)

set -euo pipefail
. ./path.sh || exit 1

# Run no stage by default; select the stage range explicitly.
stage=-1
stop_stage=-1

# Stage 1: Libri2Mix audio and keyword cue indexes
librimix_root=       # Existing Libri2Mix root containing wav16k/min
librispeech_root=    # LibriSpeech root containing *.trans.txt files
data=data
noise_type=clean
test_cue=4 # 1 | 2 | 3 | 4 | full
textual_input= # Empty: read dataset_args.cue_processing.textual.input from conf.
# text: store text; data workers perform phonemization during training.
# dae_phoneme: precompute per-word phoneme IDs during Stage 1.
# Optional Stage 1 override; keep it consistent with the training conf.

# Training data format
data_type=raw # shard/raw

# Training
gpus=
config=confs/tse_bsrnn_dae.yaml # Select text or dae_phoneme in this config.
exp_dir=
checkpoint=
num_avg=

# Inference and scoring
fs=
save_results=true
use_pesq=true
use_dnsmos=true
dnsmos_use_gpu=true

. "${WESEP_ROOT}/tools/parse_options.sh" || exit 1
. "${WESEP_ROOT}/tools/resolve_config.sh" || exit 1

# Resolve values with explicit CLI/submit overrides first, then config, then
# recipe fallback defaults.
gpus=$(resolve_yaml_value "${gpus}" "${config}" "gpus" "[0]")
# Read the nested data-processing choice without loading a model.
if [ -z "${textual_input}" ]; then
  textual_input=$(python -c 'import sys, yaml
with open(sys.argv[1], encoding="utf-8") as stream:
    print(yaml.safe_load(stream)["dataset_args"]["cue_processing"]["textual"]["input"])
' "${config}")
fi

gpus=${gpus// /}
num_gpus=$(echo "${gpus}" | awk -F ',' '{print NF}')
exp_dir=$(resolve_yaml_value "${exp_dir}" "${config}" "exp_dir" "exp/TSE_BSRNN_TEXTUAL_KEYWORDS")
num_avg=$(resolve_yaml_value "${num_avg}" "${config}" "num_avg" "5")
fs=$(resolve_fs "${fs}" "${config}" "16k")

# Select an existing test index without rebuilding audio or cue records.
case "${test_cue}" in
  1|2|3|4) test_cues_name="cues_kw${test_cue}.yaml" ;;
  full) test_cues_name=cues_full.yaml ;;
  *) echo "--test_cue must be 1, 2, 3, 4, or full." >&2; exit 1 ;;
esac

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Stage 0: Download official DAE-KCE resources"
  bash local/dae_kce/download_resources.sh
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Stage 1: Build WeSep audio and textual cue indexes"
  bash local/prepare_data.sh \
    --mix-data-path "${librimix_root}/wav16k/min" \
    --librispeech-root "${librispeech_root}" \
    --data "${data}" \
    --noise-type "${noise_type}" \
    --textual-input "${textual_input}" \
    --config "${config}"
fi

data=${data}/${noise_type}

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && [ "${data_type}" = "shard" ]; then
  echo "Stage 2: Create shards"
  for split in train-100 dev test; do
    python "${WESEP_ROOT}/tools/make_shards_from_samples.py" \
      --samples "${data}/${split}/samples.jsonl" \
      --num_utts_per_shard 1000 \
      --num_threads 16 \
      --prefix shards \
      --shuffle \
      "${data}/${split}/shards" \
      "${data}/${split}/shard.list"
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Stage 3: Train"
  if [ -z "${checkpoint}" ] && [ -f "${exp_dir}/models/latest_checkpoint.pt" ]; then
    checkpoint=${exp_dir}/models/latest_checkpoint.pt
  fi
  export OMP_NUM_THREADS=8
  torchrun --standalone --nnodes=1 --nproc_per_node="${num_gpus}" \
    "${WESEP_ROOT}/wesep/bin/train.py" \
    --config "${config}" \
    --exp_dir "${exp_dir}" \
    --gpus "${gpus}" \
    --num_avg "${num_avg}" \
    --data_type "${data_type}" \
    --train_data "${data}/train-100/${data_type}.list" \
    --train_cues "${data}/train-100/cues.yaml" \
    --train_samples "${data}/train-100/samples.jsonl" \
    --val_data "${data}/dev/${data_type}.list" \
    --val_cues "${data}/dev/cues.yaml" \
    --val_samples "${data}/dev/samples.jsonl" \
    ${checkpoint:+--checkpoint "${checkpoint}"}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Stage 4: Average checkpoints"
  # This stage runs only when explicitly selected; edit the epochs below.
  python "${WESEP_ROOT}/wesep/bin/average_model.py" \
    --dst_model "${exp_dir}/models/avg_best_model.pt" \
    --src_path "${exp_dir}/models" \
    --mode best \
    --epochs "138,141"
fi

if [ -z "${checkpoint}" ] && [ -f "${exp_dir}/models/avg_best_model.pt" ]; then
  checkpoint=${exp_dir}/models/avg_best_model.pt
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Stage 5: Infer"
  infer_gpu="${gpus#[}"
  infer_gpu="${infer_gpu%]}"
  infer_gpu="${infer_gpu%%,*}"
  python "${WESEP_ROOT}/wesep/bin/infer.py" \
    --config "${config}" \
    --fs "${fs}" \
    --gpus "${infer_gpu}" \
    --exp_dir "${exp_dir}" \
    --data_type "${data_type}" \
    --test_data "${data}/test/${data_type}.list" \
    --test_cues "${data}/test/${test_cues_name}" \
    --test_samples "${data}/test/samples.jsonl" \
    --save_wav "${save_results}" \
    ${checkpoint:+--checkpoint "${checkpoint}"}
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Stage 6: Score"
  "${WESEP_ROOT}/tools/score.sh" \
    --pairs "${exp_dir}/audio/audio_pairs.jsonl" \
    --output_dir "${exp_dir}/scoring" \
    --fs "${fs}" \
    --use_pesq "${use_pesq}" \
    --use_dnsmos "${use_dnsmos}" \
    --dnsmos_use_gpu "${dnsmos_use_gpu}" \
    --n_gpu "${num_gpus}"
fi
