from __future__ import print_function

import json
import os
import time
from contextlib import ExitStack
from pathlib import Path

import fire
import soundfile
import torch
from torch.utils.data import DataLoader

from wesep.dataset.dataset import Dataset
from wesep.dataset.collate import (
    BASE_COLLECT_KEYS,
    build_collect_keys,
    tse_collate_fn,
)
from wesep.models import get_model
from wesep.utils.checkpoint import load_pretrained_model
from wesep.utils.score import cal_SISNRi
from wesep.utils.file_utils import load_yaml
from wesep.utils.utils import (
    get_logger,
    parse_config_or_kwargs,
    set_seed,
)


def infer(config="confs/conf.yaml", **kwargs):
    start = time.time()
    total_SISNR = 0
    total_SISNRi = 0
    total_cnt = 0
    accept_cnt = 0

    configs = parse_config_or_kwargs(config, **kwargs)
    sign_save_wav = configs.get(
        "save_wav", True)  # Control if save the extracted speech as .wav

    rank = 0
    set_seed(configs["seed"] + rank)
    gpu = configs.get("gpus", 0)
    if isinstance(gpu, (list, tuple)):
        if not gpu:
            raise ValueError("gpus must contain at least one device")
        gpu = gpu[0]
    gpu = int(gpu)
    device = (torch.device("cuda:{}".format(gpu))
              if gpu >= 0 else torch.device("cpu"))

    sample_rate = configs.get("fs", None)
    if sample_rate is None or sample_rate == "16k":
        sample_rate = 16000
    else:
        sample_rate = 8000

    model = get_model(configs["model"]["tse_model"])(
        configs["model_args"]["tse_model"],
        defer_pretrained=True,
    )
    model_path = os.path.join(configs["checkpoint"])
    load_pretrained_model(model, model_path)

    logger = get_logger(configs["exp_dir"], "infer.log")
    logger.info("Load checkpoint from {}".format(model_path))
    save_audio_dir = os.path.join(configs["exp_dir"], "audio")
    if sign_save_wav:
        if not os.path.exists(save_audio_dir):
            try:
                os.makedirs(save_audio_dir)
                print(f"Directory {save_audio_dir} created successfully.")
            except OSError as e:
                print(f"Error creating directory {save_audio_dir}: {e}")
        else:
            print(f"Directory {save_audio_dir} already exists.")
    else:
        print("Do NOT save the results in wav.")

    model = model.to(device)
    model.eval()

    configs["dataset_args"]["whole_utt"] = True
    test_dataset = Dataset(
        configs["data_type"],
        configs["test_data"],
        configs["dataset_args"],
        state="test",
        repeat_dataset=configs.get("repeat_dataset", False),
        cues_yaml=configs.get("test_cues", None),
        expand_targets=True,
    )
    test_collect_keys = build_collect_keys(
        load_yaml(configs["test_cues"]),
        configs["dataset_args"],
        BASE_COLLECT_KEYS,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=lambda batch: tse_collate_fn(batch, test_collect_keys))

    # Resolve targets by sample key and speaker ID, not output filenames.
    sample_index = None
    if sign_save_wav and configs.get("test_samples"):
        sample_index = {}
        with open(configs["test_samples"], encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                sample = json.loads(line)
                if sample["key"] in sample_index:
                    raise ValueError(f"Duplicate sample key: {sample['key']}")
                if len(set(sample["spk"])) != len(sample["spk"]):
                    raise ValueError(f"Duplicate speaker in {sample['key']}")
                sample_index[sample["key"]] = sample
    elif sign_save_wav:
        logger.info("No test_samples provided; saving audio without pairs.")

    with ExitStack() as stack, torch.no_grad():
        if sample_index is not None:
            pairs_path = Path(
                configs.get("pairs_file",
                            os.path.join(save_audio_dir, "audio_pairs.jsonl")))
            pairs_path.parent.mkdir(parents=True, exist_ok=True)
            # Flush each pair as soon as its output audio has been saved.
            pairs_stream = stack.enter_context(
                pairs_path.open("w", encoding="utf-8", buffering=1))
            logger.info("Audio pairs: %s", pairs_path.resolve())
        for i, batch in enumerate(test_dataloader):
            spk = batch["spk"]
            key = batch["key"]

            # Move tensor fields without changing bool, integer, or float dtype.
            for batch_key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[batch_key] = value.to(device)

            model_outputs = model(batch)
            speech = model_outputs["speech"]
            if speech.ndim != 3 or speech.size(1) != 1:
                raise RuntimeError(
                    "Current TSE inference expects speech shaped [B, 1, T], "
                    f"got {tuple(speech.shape)}")
            outputs = speech[:, 0].cpu()

            ref = batch["wav_target"][:, 0].cpu().numpy()
            mix = batch["wav_mix"][:, 0].cpu().numpy()

            min_len = min(ref.shape[-1], outputs.shape[-1], mix.shape[-1])
            ref = ref[..., :min_len]
            outputs = outputs[..., :min_len]
            ests = outputs.numpy()
            mix = mix[..., :min_len]

            # Save and score every speaker-expanded TSE output in the batch.
            for idx in range(len(ests)):
                if sign_save_wav:
                    if sample_index is not None:
                        sample = sample_index[key[idx]]
                        if spk[idx] not in sample["spk"]:
                            raise ValueError(
                                f"Unknown target speaker {spk[idx]} in {key[idx]}")
                        target_path = Path(sample["src"][spk[idx]][0]).resolve()
                    file_path = os.path.join(
                        save_audio_dir,
                        f"Utt{total_cnt + 1}-{key[idx]}-T{spk[idx]}.wav",
                    )
                    output_to_save = outputs[idx]
                    peak = output_to_save.abs().amax()
                    if peak.item() > 1.0:
                        output_to_save = (output_to_save /
                                          peak.clamp_min(1e-8) * 0.9)
                    soundfile.write(
                        file_path,
                        output_to_save.numpy(),
                        sample_rate,
                    )
                    if sample_index is not None:
                        pair = {
                            "id": f"{key[idx]}_{spk[idx]}",
                            "target": str(target_path),
                            "estimate": os.path.abspath(file_path),
                        }
                        pairs_stream.write(
                            json.dumps(pair, ensure_ascii=False) + "\n")

                sisnr, delta = cal_SISNRi(ests[idx], ref[idx], mix[idx])
                logger.info(
                    "Num={} | Utt={} | Target speaker={} | SI-SNR={:.2f} | SI-SNRi={:.2f}"
                    .format(total_cnt + 1, key[idx], spk[idx], sisnr, delta))
                total_SISNR += sisnr
                total_SISNRi += delta
                total_cnt += 1
                if delta > 1:
                    accept_cnt += 1

        end = time.time()
    logger.info("Extracted targets: %s", total_cnt)

    logger.info("Time Elapsed: {:.1f}s".format(end - start))
    logger.info("Average SI-SNR: {:.2f}".format(total_SISNR / total_cnt))
    logger.info("Average SI-SNRi: {:.2f}".format(total_SISNRi / total_cnt))
    logger.info(
        "Acceptance rate of Utterances with SI-SDRi > 1 dB: {:.2f}".format(
            accept_cnt / total_cnt * 100))


if __name__ == "__main__":
    fire.Fire(infer)
