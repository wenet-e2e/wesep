#!/bin/bash

# Copyright 2023 Shuai Wang (wangshuai@cuhk.edu.cn)

. ./path.sh || exit 1

stage=-1
stop_stage=-1

HOST_NODE_ADDR="localhost:29402"
num_nodes=1
job_id=2025

data=data
fs=16k
min_max=min
noise_type="clean"
data_type="shard"  # shard/raw
Vox1_dir=/YourPATH/voxceleb/VoxCeleb1/wav
Libri2Mix_dir=/YourPATH/librimix/Libri2Mix          #For validate and test the TSE model.
mix_data_path="${Libri2Mix_dir}/wav${fs}/${min_max}"

gpus="[0,1,2,3]"
num_avg=5 #10
checkpoint=
config=confs/bsrnn_online.yaml #_PretrainedResNet34.yaml
exp_dir=exp/BSRNN_Online/no_spk_transform_multiply #_PretrainedResNet34
save_results=true


. tools/parse_options.sh || exit 1


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare datasets ..."
  ./local/prepare_data.sh --single_data_path ${Vox1_dir} \
    --mix_data_path ${mix_data_path} \
    --data ${data} \
    --noise_type ${noise_type} \
    --stage 1 \
    --stop-stage 4
fi

data=${data}/${noise_type}

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Covert train and test data to ${data_type}..."
  for dset in train-vox1; do
    python tools/make_shard_online.py --num_utts_per_shard 1000 \
        --num_threads 16 \
        --prefix shards \
        --shuffle \
        ${data}/$dset/wav.scp ${data}/$dset/utt2spk \
        ${data}/$dset/shards_online ${data}/$dset/shard_online.list
  done
  for dset in dev test; do
    python tools/make_shard_list_premix.py --num_utts_per_shard 48 \
      --num_threads 16 \
      --prefix shards \
      --shuffle \
      ${data}/$dset/wav.scp ${data}/$dset/utt2spk \
      ${data}/$dset/shards ${data}/$dset/shard.list
  done
fi

if [ -z "${checkpoint}" ] && [ -f "${exp_dir}/models/latest_checkpoint.pt" ]; then
  checkpoint="${exp_dir}/models/latest_checkpoint.pt"
fi
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  #  rm -r $exp_dir
  echo "Start training ..."
  export OMP_NUM_THREADS=8
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  #torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus \
           --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint=$HOST_NODE_ADDR \
    wesep/bin/train.py --config $config \
    --exp_dir ${exp_dir} \
    --gpus $gpus \
    --num_avg ${num_avg} \
    --data_type "${data_type}" \
    --train_data ${data}/train-vox1/${data_type}_online.list \
    --train_utt2spk ${data}/train-vox1/utt2spk \
    --train_spk2utt ${data}/train-vox1/spk2enroll.json \
    --val_data ${data}/dev/${data_type}.list \
    --val_spk2utt ${data}/dev/single.wav.scp \
    --val_spk1_enroll ${data}/dev/spk1.enroll \
    --val_spk2_enroll ${data}/dev/spk2.enroll \
    ${checkpoint:+--checkpoint $checkpoint}
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Do model average ..."
  avg_model=$exp_dir/models/avg_best_model.pt
  python wesep/bin/average_model.py \
    --dst_model $avg_model \
    --src_path $exp_dir/models \
    --num ${num_avg} \
    --mode best \
    --epochs "121,129,130,141,144" # no_spk_transform_multiply, 19200:bs(4)*ngpus(4)*iters(1200), SI-SNR: 13.80
    #--epochs "125,129,130,142,144" # no_spk_transform_multiply_PretrainedResNet34JointTrain, 19200:bs(4)*ngpus(4)*iters(1200), SI-SNR: 13.49
    #--epochs "129,140,141,146,149"  # no_spk_transform_multiply_PretrainedResNet34, 19200:bs(4)*ngpus(4)*iters(1200), SI-SNR: 14.67
    #--epochs "88,101,102,104,106"  # no_spk_transform_multiply.bak, 每个epoch跑vox1-dev全量数据:~bs(4)*ngpus(8)*iters(2300), SI-SNR: 15.20
fi

if [ -z "${checkpoint}" ] && [ -f "${exp_dir}/models/avg_best_model.pt" ]; then
  checkpoint="${exp_dir}/models/avg_best_model.pt"
fi
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  python wesep/bin/infer.py --config $config \
      --gpus 0 \
      --exp_dir ${exp_dir} \
      --data_type "${data_type}" \
      --test_data ${data}/test/${data_type}.list \
      --test_spk1_enroll ${data}/test/spk1.enroll \
      --test_spk2_enroll ${data}/test/spk2.enroll \
      --test_spk2utt ${data}/test/single.wav.scp \
      --save_wav ${save_results} \
      ${checkpoint:+--checkpoint $checkpoint}
fi

#./run.sh --stage 4 --stop-stage 4 --exp_dir exp/BSRNN/train_clean_460/multiply_no_spk_transform/
#./run.sh --stage 5 --stop-stage 5 --config exp/BSRNN/train_clean_460/multiply_no_spk_transform/config.yaml  --exp_dir exp/BSRNN/train_clean_460/multiply_no_spk_transform/ --checkpoint exp/BSRNN/train_clean_460/multiply_no_spk_transform/models/avg_best_model.pt