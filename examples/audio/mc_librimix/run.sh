#!/usr/bin/env bash

# Copyright 2023 Shuai Wang (wangshuai@cuhk.edu.cn)
#           2026 Ke Zhang (kylezhang1118@gmail.com)

set -euo pipefail
. ./path.sh || exit 1

# Run no stage by default; select the stage range explicitly.
stage=-1
stop_stage=-1

# Before Stage 1, run Stage 0 in examples/spatial/mc_librimix to
# generate the multichannel waveforms, and run Stage 1 in
# examples/audio/librimix to prepare the mono speaker cue indexes. This recipe
# then creates its own WeSep samples.jsonl, raw.list, and cues.yaml files.
mc_librimix_root=
audio_cue_root=../librimix/data/clean
data=data
data_type=raw # shard/raw

# Training
gpus=
config=confs/tse_bsrnn_spk.yaml
exp_dir=exp/TSE_BSRNN_SPK_MULTICHANNEL
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
gpus=${gpus// /}
num_gpus=$(echo "${gpus}" | awk -F ',' '{print NF}')
exp_dir=$(resolve_yaml_value "${exp_dir}" "${config}" "exp_dir" "exp/TSE_BSRNN_SPK_MULTICHANNEL")
num_avg=$(resolve_yaml_value "${num_avg}" "${config}" "num_avg" "10")
fs=$(resolve_fs "${fs}" "${config}" "16k")

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Stage 1: Build multichannel samples and speaker cue indexes"
  local/prepare_data.sh \
    --dataset-root "${mc_librimix_root}" \
    --audio-cue-root "${audio_cue_root}" \
    --data "${data}"
fi

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
  python "${WESEP_ROOT}/wesep/bin/infer.py" \
    --config "${config}" \
    --fs "${fs}" \
    --gpus 0 \
    --exp_dir "${exp_dir}" \
    --data_type "${data_type}" \
    --test_data "${data}/test/${data_type}.list" \
    --test_cues "${data}/test/cues.yaml" \
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
