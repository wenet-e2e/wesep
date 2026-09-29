#!/usr/bin/env bash

# Copyright 2026 Ke Zhang (kylezhang1118@gmail.com)

set -euo pipefail
. ./path.sh || exit 1

# Run no stage by default; select the stage range explicitly.
stage=-1
stop_stage=-1

# Stage 0: VoxCeleb2Mix generation
voxceleb2_root=/YourPATH/voxceleb2/mp4             # Existing VoxCeleb2 orig/{train,test} MP4 root
voxceleb2mix_root=          # Generated min-length VoxCeleb2Mix root
# Use the official MuSE CSV for directly comparable train/val/test mixtures.
# https://raw.githubusercontent.com/zexupan/MuSE/master/data/voxceleb2-800/mixture_data_list_2mix.csv
# Leave it empty to generate WeSep's deterministic MuSE-style manifest.
mixture_manifest=
visual_frontend=raw_video   # raw_video keeps MP4 cues; muse precomputes features
visual_frontend_checkpoint= # Checkpoint required by the selected frontend
                            # this checkpoint is from Pre-trained Weights of
			                #  the repo 'https://github.com/smeetrs/deep_avsr'
                            # You also download it via the huggingface:
                            # ```
                            # pip install huggingface_hub
                            # mkdir ./pretrain_networks
                            # hf download shuanguanma/DeepAVSR_Weights visual_frontend.pt \
			                #    --local-dir./pretrain_networks/
			                # ```

visual_device=cuda

# Stage 1: WeSep lists from the generated dataset root
data=data

# Training data format
data_type=raw # shard/raw

# Training
gpus=
# The model visual input must match the representation in visual_cue.json.
config=confs/tse_bsrnn_visual.yaml # Use *_muse_frontend.yaml for muse cues.
exp_dir=exp/TSE_BSRNN_VIS
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
exp_dir=$(resolve_yaml_value "${exp_dir}" "${config}" "exp_dir" "exp/TSE_BSRNN_VIS")
num_avg=$(resolve_yaml_value "${num_avg}" "${config}" "num_avg" "10")
fs=$(resolve_fs "${fs}" "${config}" "16k")

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Stage 0: Generate fixed min-length VoxCeleb2Mix data"
  simulation_args=(
    --voxceleb2-root "${voxceleb2_root}"
    --output-root "${voxceleb2mix_root}"
  )
  if [ -n "${mixture_manifest}" ]; then
    simulation_args+=(--mixture-manifest "${mixture_manifest}")
  fi
  simulation_args+=(
    --visual-frontend "${visual_frontend}"
    --visual-frontend-checkpoint "${visual_frontend_checkpoint}"
    --visual-device "${visual_device}"
  )
  local/simulate_data.sh "${simulation_args[@]}"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Stage 1: Build WeSep lists from the visual dataset root"
  local/prepare_data.sh \
    --dataset-root "${voxceleb2mix_root}" \
    --data "${data}"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && [ "${data_type}" = "shard" ]; then
  echo "Stage 2: Create shards"
  for dset in train val test; do
    python "${WESEP_ROOT}/tools/make_shards_from_samples.py" \
      --samples "${data}/${dset}/samples.jsonl" \
      --num_utts_per_shard 1000 \
      --num_threads 16 \
      --prefix shards \
      --shuffle \
      "${data}/${dset}/shards" \
      "${data}/${dset}/shard.list"
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Stage 3: Train"
  if [ -z "${checkpoint}" ] && [ -f "${exp_dir}/models/latest_checkpoint.pt" ]; then
    checkpoint="${exp_dir}/models/latest_checkpoint.pt"
  fi
  export OMP_NUM_THREADS=8
  torchrun --standalone --nnodes=1 --nproc_per_node="${num_gpus}" \
    "${WESEP_ROOT}/wesep/bin/train.py" \
    --config "${config}" \
    --exp_dir "${exp_dir}" \
    --gpus "${gpus}" \
    --num_avg "${num_avg}" \
    --data_type "${data_type}" \
    --train_data "${data}/train/${data_type}.list" \
    --train_cues "${data}/train/cues.yaml" \
    --train_samples "${data}/train/samples.jsonl" \
    --val_data "${data}/val/${data_type}.list" \
    --val_cues "${data}/val/cues.yaml" \
    --val_samples "${data}/val/samples.jsonl" \
    ${checkpoint:+--checkpoint "${checkpoint}"}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Stage 4: Average checkpoints"
  avg_model=${exp_dir}/models/avg_best_model.pt
  # This stage runs only when explicitly selected; edit the epochs below.
  python "${WESEP_ROOT}/wesep/bin/average_model.py" \
    --dst_model "${avg_model}" \
    --src_path "${exp_dir}/models" \
    --mode best \
    --epochs "138,141"
fi
if [ -z "${checkpoint}" ] && [ -f "${exp_dir}/models/avg_best_model.pt" ]; then
  checkpoint="${exp_dir}/models/avg_best_model.pt"
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
  python "${WESEP_ROOT}/tools/build_tse_reference_scp.py" \
    --samples "${data}/test/samples.jsonl" \
    --output "${data}/test/single.wav.scp" \
    --inference-scp "${exp_dir}/audio/spk1.scp" \
    --check-source-files
  "${WESEP_ROOT}/tools/score.sh" --dset "${data}/test" \
    --exp_dir "${exp_dir}" \
    --fs ${fs} \
    --use_pesq "${use_pesq}" \
    --use_dnsmos "${use_dnsmos}" \
    --dnsmos_use_gpu "${dnsmos_use_gpu}" \
    --n_gpu "${num_gpus}"
fi
