# VoxCeleb2Mix Visual-Cue Recipe

> Last updated: 2026-09-21

This Linux recipe creates min-length audio-visual mixtures from VoxCeleb2 and
trains BSRNN with raw MP4 visual cues. This is the maintained visual recipe for
the v0.1 research preview.

Stage 0 expects the original video hierarchy:

```text
<VoxCeleb2>/orig/{train,test}/<speaker>/<video>/<clip>.mp4
```

For mixtures directly comparable with MuSE sample selection and gains, pass
the official manifest linked in `run.sh`. Omitting it creates a deterministic
custom manifest.

Raw-video route:

```bash
./run.sh --stage 0 --stop_stage 1 \
  --voxceleb2_root /path/to/VoxCeleb2/orig \
  --voxceleb2mix_root /path/to/VoxCeleb2Mix \
  --mixture_manifest /path/to/mixture_data_list_2mix.csv \
  --visual_frontend raw_video \
  --config confs/tse_bsrnn_visual.yaml
```

## Experimental compatibility path

The code also retains a precomputed MuSE-frontend path for internal comparison:

```bash
./run.sh --stage 0 --stop_stage 1 \
  --voxceleb2_root /path/to/VoxCeleb2/orig \
  --voxceleb2mix_root /path/to/VoxCeleb2Mix \
  --mixture_manifest /path/to/mixture_data_list_2mix.csv \
  --visual_frontend muse \
  --visual_frontend_checkpoint /path/to/visual_frontend.pt \
  --config confs/tse_bsrnn_visual_muse_frontend.yaml
```

Stage 0 writes `visual_cue.json`, which records whether Stage 1 should expose
MP4/raw-video or NPY/MuSE cues. Stage 1 reads only the generated dataset root
and creates explicit `samples.jsonl`, `raw.list`, and `cues.yaml` files for
`train`, `val`, and `test`. The selected cue representation and model config
must match.

`/path/to/visual_frontend.pt ` is from Pre-trained Weights of the repo 'https://github.com/smeetrs/deep_avsr'
You also download it via the huggingface:
```
pip install huggingface_hub
mkdir ./pretrain_networks
hf download shuanguanma/DeepAVSR_Weights visual_frontend.pt --local-dir./pretrain_networks/

```

Raw MP4 decoding uses more host memory and data-loader time. Adjust batch size,
worker count, and prefetching according to the available host and GPU memory.

The MuSE-frontend configuration is not part of the official model list or the
validated v0.1 release scope.

## Stages

Stages 0–6 generate VoxCeleb2Mix, build WeSep lists, optionally create shards,
train, average checkpoints, infer, and score. Raw-video data preparation and
training are the validated v0.1 path; inference and scoring remain under
release-wide validation.
