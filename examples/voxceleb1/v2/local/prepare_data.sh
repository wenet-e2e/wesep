#!/bin/bash
# Copyright (c) 2023 Shuai Wang (wsstriving@gmail.com)

stage=-1
stop_stage=-1

single_data_path='./voxceleb/VoxCeleb1/wav/'
mix_data_path='./Libri2Mix/wav16k/min/'

data=data
noise_type=clean
num_spk=2

. tools/parse_options.sh || exit 1

data=$(realpath ${data})

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare the meta files for the vox1 single speaker datasets"

  for dataset in train-vox1; do
    echo "Preparing files for" $dataset

    # Prepare the meta data for the online mix data
    mkdir -p "${data}"/$noise_type/${dataset}
    find ${single_data_path} -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' | sort >"${data}"/$noise_type/${dataset}/wav.scp
    awk '{print $1}' "${data}"/$noise_type/${dataset}/wav.scp | awk -F "/" '{print $0,$1}' >"${data}"/$noise_type/${dataset}/utt2spk

    python local/prepare_spk2enroll_vox.py \
      "${data}/$noise_type/${dataset}/wav.scp" \
      --outfile "${data}"/$noise_type/${dataset}/spk2enroll.json 
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Prepare the meta files for the val and test datasets"

  for dataset in dev test; do
    echo "Preparing files for" $dataset

    # Prepare the meta data for the mixed data
    dataset_path=$mix_data_path/$dataset/mix_${noise_type}
    mkdir -p "${data}"/$noise_type/${dataset}
    find ${dataset_path}/ -type f -name "*.wav" | awk -F/ '{print $NF}' |
      awk -v path="${dataset_path}" '{print $1 , path "/" $1 , path "/../s1/" $1 , path "/../s2/" $1}' |
      sed 's#.wav##' | sort -k1,1 >"${data}"/$noise_type/${dataset}/wav.scp
    awk '{print $1}' "${data}"/$noise_type/${dataset}/wav.scp |
      awk -F[_-] '{print $0, $1,$4}' >"${data}"/$noise_type/${dataset}/utt2spk

    # Prepare the meta data for single speakers
    dataset_path=$mix_data_path/$dataset/s1
    find ${dataset_path}/ -type f -name "*.wav" | awk -F/ '{print "s1/" $NF, $0}' | sort -k1,1 >"${data}"/$noise_type/${dataset}/single.wav.scp
    awk '{print $1}' "${data}"/$noise_type/${dataset}/single.wav.scp | grep 's1' |
      awk -F[-_/] '{print $0, $2}' >"${data}"/$noise_type/${dataset}/single.utt2spk

    dataset_path=$mix_data_path/$dataset/s2
    find ${dataset_path}/ -type f -name "*.wav" | awk -F/ '{print "s2/" $NF, $0}' | sort -k1,1 >>"${data}"/$noise_type/${dataset}/single.wav.scp

    awk '{print $1}' "${data}"/$noise_type/${dataset}/single.wav.scp | grep 's2' |
      awk -F[-_/] '{print $0, $5}' >>"${data}"/$noise_type/${dataset}/single.utt2spk
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "stage 3: Prepare LibriMix target-speaker enroll signal"

  for dset in dev test; do
    python local/prepare_spk2enroll_librispeech.py \
      "${mix_data_path}/${dset}" \
      --is_librimix True \
      --outfile "${data}"/$noise_type/${dset}/spk2enroll.json \
      --audio_format wav
  done

  for dset in dev test; do
    if [ $num_spk -eq 2 ]; then
      url="https://raw.githubusercontent.com/BUTSpeechFIT/speakerbeam/main/egs/libri2mix/data/wav8k/min/${dset}/map_mixture2enrollment"
    else
      url="https://raw.githubusercontent.com/BUTSpeechFIT/speakerbeam/main/egs/libri3mix/data/wav8k/min/${dset}/map_mixture2enrollment"
    fi

    output_file="${data}/${noise_type}/${dset}/mixture2enrollment"
    wget -O "$output_file" "$url"
  done

  for dset in dev test; do
    python local/prepare_librimix_enroll.py \
      "${data}"/$noise_type/${dset}/wav.scp \
      "${data}"/$noise_type/${dset}/spk2enroll.json \
      --mix2enroll "${data}/${noise_type}/${dset}/mixture2enrollment" \
      --num_spk ${num_spk} \
      --train False \
      --output_dir "${data}"/${noise_type}/${dset} \
      --outfile_prefix "spk"
  done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Download the pre-trained speaker encoders (Resnet34 & Ecapa-TDNN512) from wespeaker..."
  mkdir wespeaker_models
  wget https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet34.zip
  unzip voxceleb_resnet34.zip -d wespeaker_models
  wget https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_ECAPA512.zip
  unzip voxceleb_ECAPA512.zip -d wespeaker_models
fi

# if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
#   echo "Prepare the speaker embeddings using wespeaker pretrained models"
#   for dataset in dev test train-100; do
#     mkdir -p "${data}"/$noise_type/${dataset}
#     echo "Preparing files for" $dataset
#     wespeaker --task embedding_kaldi \
#               --wav_scp "${data}"/$noise_type/${dataset}/single.wav.scp \
#               --output_file "${data}"/$noise_type/${dataset}/embed \
#               -p wespeaker_models/voxceleb_resnet34 \
#               -g 0 # GPU idx
#   done
# fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then 
  if [ ! -d "${data}/raw_data/musan" ]; then
    mkdir -p ${data}/raw_data/musan
    # 
    echo "Downloading musan.tar.gz ..."
    echo "This may take a long time. Thus we recommand you to download all archives above in your own way first."
    wget --no-check-certificate https://openslr.elda.org/resources/17/musan.tar.gz -P ${data}/raw_data
    md5=$(md5sum ${data}/raw_data/musan.tar.gz | awk '{print $1}')
    [ $md5 != "0c472d4fc0c5141eca47ad1ffeb2a7df" ] && echo "Wrong md5sum of musan.tar.gz" && exit 1

    echo "Decompress all archives ..." 
    tar -xzvf ${data}/raw_data/musan.tar.gz -C ${data}/raw_data

    rm -rf ${data}/raw_data/musan.tar.gz 
  fi

  echo "Prepare wav.scp for musan ..."
  mkdir -p ${data}/musan 
  find ${data}/raw_data/musan -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' >${data}/musan/wav.scp

  # Convert all musan data to LMDB
  echo "conver musan data to LMDB ..."
  python tools/make_lmdb.py ${data}/musan/wav.scp ${data}/musan/lmdb
fi