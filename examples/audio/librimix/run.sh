#!/bin/bash

# Copyright 2023 Shuai Wang (wangshuai@cuhk.edu.cn)
#           2026 Ke Zhang (kylezhang1118@gmail.com)
. ./path.sh || exit 1

# General configuration
stage=-1
stop_stage=-1

# Data preparation related
data=data
fs=16k
min_max=min
noise_type="clean"
data_type="raw" # shard/raw
Libri2Mix_dir=/YourPATH/librimix/Libri2Mix
mix_data_path="${Libri2Mix_dir}/wav${fs}/${min_max}"

# Training
gpus=
config=confs/tse_bsrnn_spk.yaml
exp_dir=exp/TSE_BSRNN_SPK
checkpoint=
num_avg=

# Inferencing and scoring related
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
exp_dir=$(resolve_yaml_value "${exp_dir}" "${config}" "exp_dir" "exp/TSE_BSRNN_SPK")
num_avg=$(resolve_yaml_value "${num_avg}" "${config}" "num_avg" "10")
fs=$(resolve_fs "${fs}" "${config}" "16k")

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare datasets ..."
  ./local/prepare_data.sh --mix_data_path ${mix_data_path} \
    --data ${data} \
    --noise_type ${noise_type} \
    --stage 1 \
    --stop-stage 4
fi

data=${data}/${noise_type}

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && [ "${data_type}" = "shard" ]; then
  echo "Making shards from samples.jsonl ..."
  for dset in train-100 dev test; do
  #  for dset in train-360; do
    python "${WESEP_ROOT}/tools/make_shards_from_samples.py" \
      --samples ${data}/${dset}/samples.jsonl \
      --num_utts_per_shard 1000 \
      --num_threads 16 \
      --prefix shards \
      --shuffle \
      ${data}/${dset}/shards \
      ${data}/${dset}/shard.list
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Start training ..."
  if [ -z "${checkpoint}" ] && [ -f "${exp_dir}/models/latest_checkpoint.pt" ]; then
    checkpoint="${exp_dir}/models/latest_checkpoint.pt"
  fi
  train_script="${WESEP_ROOT}/wesep/bin/train.py"
  export OMP_NUM_THREADS=8
  torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    ${train_script} --config $config \
    --exp_dir ${exp_dir} \
    --gpus "${gpus}" \
    --num_avg ${num_avg} \
    --data_type "${data_type}" \
    --train_data ${data}/train-100/${data_type}.list \
    --train_cues ${data}/train-100/cues.yaml \
    --train_samples ${data}/train-100/samples.jsonl \
    --val_data ${data}/dev/${data_type}.list \
    --val_cues ${data}/dev/cues.yaml \
    --val_samples ${data}/dev/samples.jsonl \
    ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Do model average ..."
  avg_model=$exp_dir/models/avg_best_model.pt
  # This stage runs only when explicitly selected; edit the epochs below.
  python "${WESEP_ROOT}/wesep/bin/average_model.py" \
    --dst_model $avg_model \
    --src_path $exp_dir/models \
    --mode best \
    --epochs "138,141"
fi
if [ -z "${checkpoint}" ] && [ -f "${exp_dir}/models/avg_best_model.pt" ]; then
  checkpoint="${exp_dir}/models/avg_best_model.pt"
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Start inferencing ..."
  python "${WESEP_ROOT}/wesep/bin/infer.py" --config $config \
    --fs ${fs} \
    --gpus 0 \
    --exp_dir ${exp_dir} \
    --data_type "${data_type}" \
    --test_data ${data}/test/${data_type}.list \
    --test_cues ${data}/test/cues.yaml \
    --test_samples ${data}/test/samples.jsonl \
    --save_wav ${save_results} \
    ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Start scoring ..."
  "${WESEP_ROOT}/tools/score.sh" --pairs "${exp_dir}/audio/audio_pairs.jsonl" \
    --output_dir "${exp_dir}/scoring" \
    --fs ${fs} \
    --use_pesq "${use_pesq}" \
    --use_dnsmos "${use_dnsmos}" \
    --dnsmos_use_gpu "${dnsmos_use_gpu}" \
    --n_gpu "${num_gpus}"
fi
