# WeSep 2024

This branch contains the WeSep toolkit accompanying the Interspeech 2024 paper,
with target speaker extraction models, training and inference code, and LibriMix
and VoxCeleb recipes. Model architectures are in [`wesep/models`](wesep/models),
and experiment scripts are in [`examples`](examples).

For the current toolkit, see the
[`master` branch](https://github.com/wenet-e2e/wesep/tree/master).

## Download

Clone the 2024 version into its own directory:

```sh
git clone --branch wesep_2024 --single-branch \
  https://github.com/wenet-e2e/wesep.git wesep_2024
cd wesep_2024
```

For an existing clone, create a local branch tracking the 2024 version:

```sh
git fetch origin
git switch --track origin/wesep_2024
```

To download a ZIP, open the
[`wesep_2024` branch](https://github.com/wenet-e2e/wesep/tree/wesep_2024),
select **Code > Download ZIP**, and extract the archive. Open a terminal in the
extracted directory to follow the installation steps below.

## Installation

Create a dedicated environment with Python 3.9, PyTorch 1.12.1, and
TorchAudio 0.12.1. Run these commands from the repository root:

```sh
conda create -n wesep_2024 python=3.9
conda activate wesep_2024
conda install pytorch=1.12.1 torchaudio=0.12.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

For development in a Git checkout, enable the formatting hooks:

```sh
pre-commit install
```

## Recipes

Start with the [LibriMix v2 tutorial](examples/librimix/tse/v2/README.md) for
step-by-step data preparation, training, inference, and scoring. Set the dataset
paths and GPU selection in the recipe before running its stages.

- [LibriMix v2](examples/librimix/tse/v2/README.md)
- [LibriMix v1](examples/librimix/tse/v1/README.md)
- [VoxCeleb online training](examples/voxceleb1/v2/run_online.sh)

Run each recipe from its own directory, as shown in its instructions.

## The Target Speaker Extraction Task

> Target speaker extraction (TSE) focuses on isolating the speech of a specific target speaker from overlapped multi-talker speech, which is a typical setup in the cocktail party problem.
WeSep is featured with flexible target speaker modeling, scalable data management, effective on-the-fly data simulation, structured recipes and deployment support.

<img src="resources/tse.png" width="600px">

## Features (To Do List)

- [x] On the fly data simulation
  - [x] Dynamic Mixture simulation
  - [x] Dynamic Reverb simulation
  - [x] Dynamic Noise simulation
- [x] Support time- and frequency- domain models
    - Time-domain
        - [x] conv-tasnet based models
            - [x] Spex+
    - Frequency domain
        - [x] pBSRNN
        - [x] pDPCCN
        - [x] tf-gridnet (Extremely slow, need double check)
- [ ] Training Criteria
    - [x] SISNR loss
    - [x] GAN loss  (Need further investigation)
- [ ] Datasets
  - [x] Libri2Mix (Illustration for pre-mixed speech)
  - [x] VoxCeleb (Illustration for online training)
  - [ ] WSJ0-2Mix
- [ ] Speaker Embedding
  - [x] Wespeaker Intergration
  - [x] Joint Learned Speaker Embedding
  - [x] Different fusion methods
- [ ] Pretrained models
- [ ] CLI Usage
- [x] Runtime

## Data Pipe Design

Following Wenet and Wespeaker, WeSep organizes the data processing modules as a pipeline of a set of different processors. The following figure shows such a pipeline with essential processors.

<img src="resources/datapipe.png" width="800px">

## Discussion

For Chinese users, you can scan the QR code on the left to join our group directly. If it has expired, please scan the personal Wechat QR code on the right.

|<img src='resources/Wechat_group.jpg' style=" width: 200px; height: 300px;">|<img src='resources/Wechat.jpg' style=" width: 200px; height: 300px;">|
| ---- | ---- |



## Citations
If you find WeSep useful, please cite it as

```bibtex
@inproceedings{wang24fa_interspeech,
  title     = {WeSep: A Scalable and Flexible Toolkit Towards Generalizable Target Speaker Extraction},
  author    = {Shuai Wang and Ke Zhang and Shaoxiong Lin and Junjie Li and Xuefei Wang and Meng Ge and Jianwei Yu and Yanmin Qian and Haizhou Li},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {4273--4277},
  doi       = {10.21437/Interspeech.2024-1840},
}
```
