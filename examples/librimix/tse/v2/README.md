## Tutorial on LibriMix

If you meet any problems when going through this tutorial, please feel free to ask in github issues. Thanks for any kind of feedback.


### First Experiment

We provide a recipe `examples/librimix/tse/v2/run.sh` on LibriMix data.

The recipe is simple and we suggest you run each stage one by one manually and check the result to understand the whole process.

```bash
cd examples/librimix/tse/v2
bash run.sh --stage 1 --stop_stage 1
bash run.sh --stage 2 --stop_stage 2
bash run.sh --stage 3 --stop_stage 3
bash run.sh --stage 4 --stop_stage 4
bash run.sh --stage 5 --stop_stage 5
bash run.sh --stage 6 --stop_stage 6
```

You could also just run the whole script
```bash
bash run.sh --stage 1 --stop_stage 6
```

------

### Stage 1: Prepare Training Data

Prior to executing this phase, we assume that you have locally stored or can access the LibriMix dataset and you should assign the data path to `Libri2Mix_dir`. 

As the LibriMix dataset is available in multiple versions, each determined by factors like the number of sources in the mixtures and the sampling rate, you can choose the desired version by adjusting the following variables in `run.sh`:

+ `fs`: the sample rate of the dataset, valid options are `16k` and `8k`.
+ `min_max`: the mode of mixtures, valiad options are `min` and `max`.
+ `noise_type`: the type of mixture, valiad options are `clean` and `both`.

In our recipe, we opt for the Libri2Mix data with a sampling rate of 16kHz, in 'min' mode, and without noise, thus configuring as follows:

``` bash
fs=16k
min_max=min
noise_type="clean"
Libri2Mix_dir=/path/to/Libri2Mix
```

After configuring the desired dataset version, running the script for the first phase will generate the prepared data files. By default, these files are stored in the `data` directory in the current location. 

```bash
data=data # you can change this to any directory
```

In this stage, `local/prepare_data.sh`accomplishes three tasks (Main differences with v1 version):

1. Organizes the original Libri2Mix dataset into three directoies `dev`, `test` and `train_100`/`train_360`, each containing the following files:

    + `single.utt2spk`: each line records two space-separated columns: `clean_wav_id` and `speaker_id`

        ```text
        s1/103-1240-0003_1235-135887-0017.wav 103
        s1/103-1240-0004_4195-186237-0003.wav 103
        ...
        ```

    + `utt2spk`: each line records three space-separated columns: `mixture_wav_id`, `speaker1_id` and `speaker2_id`.

        ```
        103-1240-0003_1235-135887-0017 103 1235
        103-1240-0004_4195-186237-0003 103 4195
        ...
        ```

    + `single.wav.scp`: each line records two space-separated columns: `clean_wav_id` and `clean_wav_path`

        ```
        s1/103-1240-0003_1235-135887-0017.wav /Data/Libri2Mix/wav16k/min/train-100/s1/103-1240-0003_1235-135887-0017.wav
        s1/103-1240-0004_4195-186237-0003.wav /Data/Libri2Mix/wav16k/min/train-100/s1/103-1240-0004_4195-186237-0003.wav
        ...
        ```

    + `wav.scp`: each line records four space-separated columns: `mixture_wav_id`, `mixtrue_wav_path`, `clean_wav1_path` and `clean_wav2_path`.

        ```
        103-1240-0003_1235-135887-0017 /Data/Libri2Mix/wav16k/min/train-100/mix_clean/103-1240-0003_1235-135887-0017.wav /Data/Libri2Mix/wav16k/min/train-100/mix_clean/../s1/103-1240-0003_1235-135887-0017.wav /Data/Libri2Mix/wav16k/min/train-100/mix_clean/../s2/103-1240-0003_1235-135887-0017.wav
        103-1240-0004_4195-186237-0003 /Data/Libri2Mix/wav16k/min/train-100/mix_clean/103-1240-0004_4195-186237-0003.wav /Data/Libri2Mix/wav16k/min/train-100/mix_clean/../s1/103-1240-0004_4195-186237-0003.wav /Data/Libri2Mix/wav16k/min/train-100/mix_clean/../s2/103-1240-0004_4195-186237-0003.wav
        ...
        ```

2. Prepare LibriMix target-speaker enroll signal. This step will generate one `json` file in the `dev`, `test` and `train_100`/`train_360` directories, and additional three files in the `dev` and `test` directories respectively:

    + `spk2enroll.json`: A JSON file, where the format of the stored key-value pairs is `{spk_id: [[spk_id_with_prefix_or_suffix, wav_path], ...]}`.

        ```
        "652": [["652-129742-0010", "/Data/Libri2Mix/wav16k/min/dev/s1/652-129742-0010_3081-166546-0071.wav"], 
        ..., 
        ["652-129742-0000", "/Data/Libri2Mix/wav16k/min/dev/s1/652-129742-0000_1993-147966-0004.wav"]],
        ...
        ```

    + `mixture2enrollment`: each line records three space-separated columns: `mixture_wav_id`, `clean_wav_id` and `enrollment_wav_id`.

        ```
        4077-13754-0001_5142-33396-0065 4077-13754-0001 s1/4077-13754-0004_5142-36377-0020
        4077-13754-0001_5142-33396-0065 5142-33396-0065 s1/5142-36377-0003_1320-122612-0014
        ...
        ```

    + `spk1.enroll`: each line records two space-separated columns: `mixture_wav_id` and `enrollment_wav_id`.

        ```
        1272-128104-0000_2035-147961-0014 s1/1272-135031-0015_2277-149896-0006.wav
        1272-128104-0003_2035-147961-0016 s1/1272-135031-0013_1988-147956-0016.wav
        ...
        ```

    + `spk2.enroll`: each line records two space-separated columns: `mixture_wav_id` and `enrollment_wav_id`.

        ```
        1272-128104-0000_2035-147961-0014 s1/2035-152373-0009_3000-15664-0016.wav
        1272-128104-0003_2035-147961-0016 s2/6313-66129-0013_2035-152373-0012.wav
        ...
        ```

At the end of this stage, the directory structure of `data` should look like this:

```
data/
|__ clean/ # the noise_type you chose
    |__ dev/
    |   |__ mixture2enrollment
    |   |__ single.utt2spk
    |   |__ single.wav.scp
    |   |__ spk1.enroll
    |   |__ spk2.enroll
    |   |__ spk2enroll.json
    |   |__ utt2spk
    |   |__ wav.scp
    |
    |__ test/ # the same as dev/
    |
    |__ train_100/
        |__ single.utt2spk
        |__ single.wav.scp
        |__ spk2enroll.json
        |__ utt2spk
        |__ wav.scp
```

3. Download the speaker encoders (Resnet34 & Ecapa-TDNN512) from wespeaker for training the TSE model with pretrained speaker encoder. The models will be unzipped into `wespeaker_models/`.
Find more speaker models in https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md.

4. Prepare the speaker embeddings using wespeaker pretrained models. (Not needed, and comment off in v2 version by default.)

This step will generate two files in the `dev`, `test`, and `train_100` directories respectively:

    + `embed.ark`: Kaldi ark file that stores the speaker embeddings.

    + `embed.scp`: each line records two space-separated columns: `clean_wav_id` and `spk_embed_path`

        ```
        s1/103-1240-0003_1235-135887-0017.wav workspace/wesep/examples/librimix/tse/v1/data/clean/train-100/embed.ark:1450569
        s1/103-1240-0004_4195-186237-0003.wav workspace/wesep/examples/librimix/tse/v1/data/clean/train-100/embed.ark:10622715
        ...
        ```

------

### Stage 2: Convert Data Format

This stage involves transforming the data into `shard` format, which is better suited for large datasets. Its core idea is to make the audio and labels of multiple small data(such as 1000 pieces), into compressed packets (tar) and read them based on the IterableDataset of Pytorch. For a detailed explanation of the  `shard` format, please refer to the [documentation](https://github.com/wenet-e2e/wenet/blob/main/docs/UIO.md) available in Wenet.

This stage will generate a subdirectory and a file in the `dev`, `test`, and `train_100` directories respectively:

+ `shards/`: this directory stores the compressed packets (tar) files.

    ```bash
    ls shards
    shards_000000000.tar  shards_000000001.tar  shards_000000002.tar ...
    ```

+ `shard.list`: each line records the path to the corresponding tar file.

    ```
    data/clean/dev/shards/shards_000000000.tar
    data/clean/dev/shards/shards_000000001.tar
    data/clean/dev/shards/shards_000000002.tar
    ...
    ```

At the end of this stage, the directory structure of `data` should look like this:

```
data/
|__ clean/ # the noise_type you chose
    |__ dev/
    |   |__ single.utt2spk, single.wav.scp, ... # files generated by Stage 1
    |   |__ shard.list
    |   |__ shards/
    |       |__ shards_000000000.tar
    |       |__ shards_000000001.tar
    |       |__ shards_000000002.tar
    |   
    |__ test/ # the same as dev/
    |
    |__ train_100/
        |__ single.utt2spk, single.wav.scp, ... # files generated by Stage 1
        |__ shard.list
        |__ shards/
        	|__ shards_000000000.tar
        	|__ ...
        	|__ shards_000000013.tar
```

------

### Stage 3: Neural Networking Training

You can configure network training related parameters through the configuration file. We provide some ready-to-use configuration files in the recipe. If you wish to write your own configuration files or understand the meaning of certain parameters in the configuration files, you can refer to the following information:

+ **overall training process related**

    ```yaml
    seed: 42
    exp_dir: exp/BSRNN
    enable_amp: false
    gpus: '0,1'
    log_batch_interval: 100
    save_epoch_interval: 1
    ```

    Explanations for some of the parameters mentioned above:

    + `seed`: specify a random seed.
    + `exp_dir`: specify the experiment directory.
    + `enable_amp`: whether enable automatic mixed precision.
    + `gpus`: specify the visible GPUs during training.
    + `log_batch_interval`: specify after how many batch iterations to record in the log.
    + `save_epoch_interval`: specify after how many batch epoches to save a checkpoint.

+ **dataset and dataloader realted**

    ```yaml
    dataset_args:
      resample_rate: 16000
      sample_num_per_epoch: 0
      shuffle: true
      shuffle_args:
        shuffle_size: 2500
      whole_utt: false
      chunk_len: 48000
      online_mix: false
      speaker_feat: &speaker_feat true
      fbank_args:
        num_mel_bins: 80
        frame_shift: 10
        frame_length: 25
        dither: 1.0
      
      "Usually you don't need to manually write the data part of the configuration into the config file, it will be automatically generated."
      data_type: "shard"
      train_data: "data/clean/train_100/shard.list"
      train_utt2spk: "data/clean/train_100/single.utt2spk"
      train_spk2utt: "data/clean/train_100/spk2enroll.json"
      val_data: "data/clean/dev/shard.list"
      val_utt2spk: "data/clean/dev/single.utt2spk"
      val_spk1_enroll: "data/clean/dev/spk1.enroll"
      val_spk2_enroll: "data/clean/dev/spk2.enroll"
      val_spk2utt: "data/clean/dev/single.wav.scp"
    
    
    dataloader_args:
      batch_size: 12  # A800
      drop_last: true
      num_workers: 6
      pin_memory: false
      prefetch_factor: 6
    ```

    Explanations for some of the parameters mentioned above:

    + `resample_rate`: All audio in the dataset will be resampled to this specified sample rate. Defaults to `16000`.
    + `sample_num_per_epoch`: Specifies how many samples from the full training set will be iterated over in each epoch during training. The default is `0`, which means iterating over the entire training set.
    + `shuffle`: Whether to perform *global* shuffle, i.e., shuffling at shards tar/raw/feat file level. Defaults to `true`.
    + `shuffle_size`: Parameters related to *local* shuffle. Local shuffle maintains a buffer, and shuffling is only performed when the number of data items in the buffer reaches the s`shuffle_size`. Defaults to `2500`.
    + `whole_utt`: Whether the network input and training target are the entire audio segment. Defaults to `false`.
    + `chunk_len`: This parameter only takes effect when `whole_utt` is set to `false`. It indicates the length of the segment to be extracted from the complete audio as the network input and training target. Defaults to `48000`.
    + `online_mix`: Whether dynamic mixing speakers when loading data, `shuffle` will not take effect if this parameter is set to `true`. Defaults to `false`.
    + `speaker_feat`: Whether transform the enrollment from waveform to fbank. Recommended setting to `true`. Defaults to `false`.
    + `num_mel_bins`: The parameter of fbank. The feature dimension of the fbank. Defaults to `80`.
    + `frame_shift`: The parameter of fbank. The time of frame shift in `ms`. Defaults to `10`.
    + `frame_length`: The parameter of fbank. The frame length in `ms`. Defaults to `25`.
    + `dither`: The parameter of fbank. Whether add noise to fbank feature. Defaults to `1.0`.
    + `data_type`: Specify the type of dataset, with valid options being `shard` and `raw`. Defaults to `shard`.
    + `train_data`: File containing paths to the training set files.
    + `train_utt2spk`: Each line of the file specified by this parameter consists of `clean_wav_id` and `speaker_id`, separated by a space(e.g. `single.utt2spk`  generated in Stage 1).
    + `train_spk2utt`: The file specified by this parameter is only used when the `joint_training` parameter is set to `true`. Each line of the file contains `speaker_id` and `enrollment_wav_id`.
    + `val_data`: File containing paths to the validation set files.
    + `val_utt2spk`: Similiar to `train_utt2spk`.
    + `val_spk1_enroll`: Each line of the file specified by this parameter consists of `mixtrue_wav_id` and `speaker1_enrollment_wav_id`, separated by a space.
    + `val_spk2_enroll`: Each line of the file specified by this parameter consists of `mixtrue_wav_id` and `speaker2_enrollment_wav_id`, separated by a space.
    + `val_spk2utt`: Each line of the file specified by this parameter consists of `clean_wav_id` and `clean_wav_path`, separated by a space(e.g. `single.wav.scp`  generated in Stage 1).
        + We have denoted this parameter as `val_spk2utt`, but it is actually assigned the `single.wav.scp` file as its value. This might be perplexing for users familiar with file formats in Kaldi or ESPnet, where the `spk2utt` file typically consists of lines containing `spk_id` and `wav_id`, whereas the `wav.scp` file's lines contain `wav_id` and `wav_path`.
        + Nevertheless, upon closer examination of its role in subsequent procedures, it becomes evident that it is indeed employed to create a dictionary mapping speaker IDs to audio samples. 
    + `batch_size`: how many samples per batch to load. Please note that the batch size mentioned here refers to the **batch size per GPU**. So, if you are training on two GPUs within a single node and set the batch size to 16, it is equivalent to setting the batch size to 32 in a single-GPU, single-node scenario.
    + `drop_last`: set to `true` to drop the last incomplete batch, if the dataset size is not divisible by the batch size. If `false` and the size of dataset is not divisible by the batch size, then the last batch will be smaller.
    + `num_workers`: how many subprocesses to use for data loading. `0` means that the data will be loaded in the main process.
    + `pin_memory`: If `true`, the data loader will copy Tensors into device/CUDA pinned memory before returning them.
    + `prefetch_factor`: number of batches loaded in advance by each worker.

+ **loss function related**

    ```yaml
    loss: SISDR
    loss_args: { }

    ### For joint training the speaker encoder with CE loss. Put SISDR in the first position for validation set.
    loss: [SISDR, CE]               
    loss_args:
      loss_posi: [[0],[1]]
      loss_weight: [[1.0],[1.0]]
    ```

    Explanations for some of the parameters mentioned above:

    + `loss`: the loss function used for training.
    + `loss_args`: the required arguments for the loss function.
    + `loss_posi`: Select which outputs from the TSE model the loss function works on.
    + `loss_weight`: The weight of loss calculated from corresponding loss function.

    In addition to some common loss functions, we also support the use of GAN loss. You can enable this feature by setting `use_gan_loss` to `true` in  `run.sh`. Once enabled, the TSE model serves as the generator, and another convolutional neural network acts as the discriminator, engaging in adversarial training. The final loss of the TSE model is a combination of the losses specified in the configuration file and the GAN loss. By default, the weight for the former is set to` 0.95`, while the latter is set to `0.05`.

    Due to the compatibility with GAN loss, the parameters mentioned below often differentiate between `tse_model` and `discriminator` under a single parameter. In such cases, we no longer provide separate explanations for each parameter.

+ **neural network structure related**

    ```yaml
    model:
      tse_model: BSRNN
    model_args:
      tse_model:
        sr: 16000
        win: 512
        stride: 128
        feature_dim: 128
        num_repeat: 6
        spk_emb_dim: 256
        spk_fuse_type: 'multiply'
        use_spk_transform: False
        multi_fuse: False
        joint_training: True       ### You should always set this para to `True` when using v2 version.
        spk_model: ResNet34 
          spk_model_init: None
          spk_args: None
        spk_emb_dim: 256
        spk_model_freeze: False   
        spk_feat: *speaker_feat   
        feat_type: "consistent"
        multi_task: False
        spksInTrain: 251
    
    model_init:
      tse_model: exp/BSRNN/no_spk_transform-multiply_fuse/models/latest_checkpoint.pt
      discriminator: null
    ```

    Explanations for some of the parameters mentioned above:

    + `model`: specify the neural network used for training.
    + `model_args`: specify model-specific parameters.
    + `spk_fuse_type`: specify the fusion method of the speaker embedding. Support `concat`, `additive`, `multiply` and `FiLM`. 
    + `multi_fuse`: whether fuse the speaker embedding multiple times.
    + `joint_training`: specify whether the speaker encoder for extracting speaker embeddings is jointly trained with the TSE model. Always set this to `true`. Do NOT use it to control if training with pretrained speaker encoders. Defaluts to `false`.
    + `spk_model`: specify the speaker model. Supports most speaker models in wespeaker: https://github.com/wenet-e2e/wespeaker/tree/master.
    + `spk_model_init`: the path of the pre-trained speaker model. Find more pretrained models in https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md. Set `None` for training the speaker model with TSE model from scratch.
    + `spk_args`: specify speaker model-specific parameters.
    + `spk_emb_dim`: the feature dimension of speaker embedding extracted from the speaker encoder.
    + `spk_model_freeze`: whether freeze the weights in speaker encoder. Set `True` when using pretrained speaker encoder. 
    + `spk_feat`: Use the defined parameters in `dataset_args` to determine whether to perform feature extraction of enrollment within the model.
    + `feat_type`: specify the type of enrollment's feature, when `spk_feat` is `False`.
    + `multi_task`: whether use such as `CE` loss function for jointly training the speaker encoder. This parameter needs to be coordinated with the `loss`.
    + `spksInTrain`: specify the speaker number in the training dataset. wsj0-2mix: 101, Libri2mix-100: 251, Libri2mix-360:921.
    + `model_init`: whether to initialize the model with an existing checkpoint. Use `null` for no initialization. If you want to initialize, provide the checkpoint path. Defaults to `null`.
    

+ **model optimization related**

    ```yaml
    num_epochs: 150
    clip_grad: 5.0
    
    optimizer:
      tse_model: Adam
    optimizer_args:
      tse_model:
        lr: 0.001
        weight_decay: 0.0001
    
    scheduler:
      tse_model: ExponentialDecrease
    scheduler_args:
      tse_model:
        final_lr: 2.5e-05
        initial_lr: 0.001
        warm_from_zero: false
        warm_up_epoch: 0
    ```

    Explanations for some of the parameters mentioned above:

    + `num_epochs`: total number of training epochs.
    + `clip_grad`: set the threshold for gradient clipping.
    + `optimizer`: set the optimizer.
    + `optimizer_args`: the required arguments for optimizer. Not used in currently version. The learning rate and scheduler are determined by `scheduler_args`.
    + `scheduler`: set the scheduler.
    + `scheduler_args`: the required arguments for scheduler.

+ **others**

    ```yaml
    num_avg: 2
    ```

    Explanations for some of the parameters mentioned above:

    + `num_avg`: numbers for averaged model.

To avoid frequent changes to the configuration file, we support **overwriting values in the configuration file** directly within `run.sh`. For example, running the following command in `run.sh` will overwrite the visible GPU from `'0,1'` to ``'0'`` in the above configuration file:

```bash
  torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    ${train_script} --config confs/config.yaml \
    --gpus "[0]" \
```

At the end of this stage, an experiment directory will be created in the current directory, containing the following files:

```
${exp_dir}/
|__ train.log
|__ config.yaml
|__ models/
	|__ checkpoint_1.pt
	|__ ...
	|__ checkpoint_150.pt
	|__ final_checkpoint.pt -> checkpoint_150.pt
	|__ latest_checkpoint.pt -> checkpoint_150.pt
```

------

### Stage 4: Apply Model Average

In this stage, we perform model averaging, and you need to specify the following parameters in `run.sh`:

+ `dst_model`: the path to save the averaged model.
+ `src_path`: source models path for average.
+ `num`: number of source models for the averaged model.
+ `mode`: the mode for model averaging. Validate options are `final` and `best`.
    + `final`: filters and sorts the latest PyTorch model files in the source directory. Averages the states of the last `num` models based on a numerical sorting of their filenames.
    + `best`: directly uses user-specified epochs to select specific model checkpoint files. Averages the states of these selected models.
+ `epochs`: this parameter only takes effect when `mode`  is set to `best` and is used to specify the epoch index of the checkpoint that will be used as source models.

------

### Stage 5: Extract Speech Using the Trained Model

After training is complete, you can execute stage 5 to extract the target speaker's speech using the trained model. In this stage, it mainly calls `wesep/bin/infer.py`, and you need to provide the following parameters for this script:

+ `config`: the configuration file used in Stage 3.
+ `fs`: the sample rate of the audio data.
+ `gpus`: the index of the visible GPU.
+ `exp_dir`: the experiment directory.
+ `data_type`: the type of dataset, with valid options being `shard` and `raw`. Defaults to `shard`.
+ `test_data`: similiar to `train_data`.
+ `test_spk1_enroll`: similiar to `dev_spk1_enroll`.
+ `test_spk2_enroll`: similiar to `dev_spk2_enroll`.
+ `test_spk2utt`: similiar to `dev_spk2utt`.
+ `save_wav`: control if save the extracted speech in `exp_dir/audio`.
+ `checkpoint`: the path to the checkpoint used for extracting the target speaker's speech.

At the end of this stage, the structure of  the experiment directory should look like this:

```
${exp_dir}/
|__ train.log
|__ config.yaml
|__ models/
|__ infer.log
|__ audio/
	|__ spk1.scp # each line records two space-separated columns: `target_wav_id` and `target_wav_path`
	|__ Utt1001-4992-41806-0008_6930-75918-0015-T4992.wav
	|__ ...
	|__ Utt999-61-70968-0003_2830-3980-0008-T61.wav
```

------

### Stage 6: Scoring

In this stage, we evaluate the quality of the generated speech using common objective metrics. The default metrics include **STOI**, **SDR**, **SAR**, **SIR**, and **SI_SNR**. In addition to these metrics, you can also include **PESQ** and **DNS_MOS** by setting the values of `use_pesq` and `use_dnsmos` to `true`. Please be aware that DNS_MOS is exclusively supported for audio samples with a **16 kHz** sampling rate. For audio with different sampling rates, refrain from employing DNS_MOS for assessment.

At the end of this stage, a markdown file `RESULTS.md` will be created under `exp` directory, the directory structure of `exp` should look like this:

```
exp/BSRNN/
|__ ${exp_dir}
|	|__ train.log, ... # files and directories generated in Stage 5
|	|__ scoring/
|
|__ RESULTS.md
```