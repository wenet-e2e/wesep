from __future__ import print_function

import os
import time

import fire
import soundfile
import torch
from torch.utils.data import DataLoader

from wesep.dataset.dataset import Dataset, tse_collate_fn_2spk
from wesep.models import get_model
from wesep.utils.checkpoint import load_pretrained_model
from wesep.utils.file_utils import read_label_file, read_vec_scp_file
from wesep.utils.score import cal_SISNRi
from wesep.utils.utils import (
    generate_enahnced_scp,
    get_logger,
    parse_config_or_kwargs,
    set_seed,
)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"


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
    gpu = configs["gpus"]
    device = (torch.device("cuda:{}".format(gpu))
              if gpu >= 0 else torch.device("cpu"))

    sample_rate = configs.get("fs", None)
    if sample_rate is None or sample_rate == "16k":
        sample_rate = 16000
    else:
        sample_rate = 8000

    model = get_model(
        configs["model"]["tse_model"])(**configs["model_args"]["tse_model"])
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

    test_spk_embeds = configs.get("test_spk_embeds", None)
    test_spk1_embed_scp = configs["test_spk1_enroll"]
    test_spk2_embed_scp = configs["test_spk2_enroll"]
    joint_training = configs["model_args"]["tse_model"].get(
        "joint_training", None)
    if not joint_training and test_spk_embeds:
        test_spk2embed_dict = read_vec_scp_file(test_spk_embeds)
    else:
        test_spk2embed_dict = read_label_file(configs["test_spk2utt"])

    test_spk1_embed = read_label_file(test_spk1_embed_scp)
    test_spk2_embed = read_label_file(test_spk2_embed_scp)

    lines = len(test_spk2embed_dict)

    test_dataset = Dataset(
        configs["data_type"],
        configs["test_data"],
        configs["dataset_args"],
        test_spk2embed_dict,
        test_spk1_embed,
        test_spk2_embed,
        state="test",
        joint_training=joint_training,
        whole_utt=configs.get("whole_utt", True),
        repeat_dataset=configs.get("repeat_dataset", False),
    )
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 collate_fn=tse_collate_fn_2spk)
    test_iter = lines // 2
    logger.info("test number: {}".format(test_iter))

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            features = batch["wav_mix"]
            targets = batch["wav_targets"]
            enroll = batch["spk_embeds"]
            spk = batch["spk"]
            key = batch["key"]

            features = features.float().to(device)  # (B,T,F)
            targets = targets.float().to(device)
            enroll = enroll.float().to(device)

            outputs = model(features, enroll)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            if torch.min(outputs.max(dim=1).values) > 0:
                outputs = ((outputs /
                            abs(outputs).max(dim=1, keepdim=True)[0] *
                            0.9).cpu().numpy())
            else:
                outputs = outputs.cpu().numpy()

            if sign_save_wav:
                file1 = os.path.join(
                    save_audio_dir,
                    f"Utt{total_cnt + 1}-{key[0]}-T{spk[0]}.wav",
                )
                soundfile.write(file1, outputs[0], sample_rate)
                file2 = os.path.join(
                    save_audio_dir,
                    f"Utt{total_cnt + 1}-{key[1]}-T{spk[1]}.wav",
                )
                soundfile.write(file2, outputs[1], sample_rate)

            ref = targets.cpu().numpy()
            ests = outputs
            mix = features.cpu().numpy()

            if ests[0].size != ref[0].size:
                end = min(ests[0].size, ref[0].size, mix[0].size)
                ests_1 = ests[0][:end]
                ref_1 = ref[0][:end]
                mix_1 = mix[0][:end]
                SISNR1, delta1 = cal_SISNRi(ests_1, ref_1, mix_1)
            else:
                SISNR1, delta1 = cal_SISNRi(ests[0], ref[0], mix[0])

            logger.info(
                "Num={} | Utt={} | Target speaker={} | SI-SNR={:.2f} | SI-SNRi={:.2f}"
                .format(total_cnt + 1, key[0], spk[0], SISNR1, delta1))
            total_SISNR += SISNR1
            total_SISNRi += delta1
            total_cnt += 1
            if delta1 > 1:
                accept_cnt += 1

            if ests[1].size != ref[1].size:
                end = min(ests[1].size, ref[1].size, mix[1].size)
                ests_2 = ests[1][:end]
                ref_2 = ref[1][:end]
                mix_2 = mix[1][:end]
                SISNR2, delta2 = cal_SISNRi(ests_2, ref_2, mix_2)
            else:
                SISNR2, delta2 = cal_SISNRi(ests[1], ref[1], mix[1])
            logger.info(
                "Num={} | Utt={} | Target speaker={} | SI-SNR={:.2f} | SI-SNRi={:.2f}"
                .format(total_cnt + 1, key[1], spk[1], SISNR2, delta2))
            total_SISNR += SISNR2
            total_SISNRi += delta2
            total_cnt += 1
            if delta2 > 1:
                accept_cnt += 1

            # if (i + 1) == test_iter:
            #     break
        end = time.time()
    # generate the scp file of the enhanced speech for scoring
    if sign_save_wav:
        generate_enahnced_scp(os.path.abspath(save_audio_dir), extension="wav")

    logger.info("Time Elapsed: {:.1f}s".format(end - start))
    logger.info("Average SI-SNR: {:.2f}".format(total_SISNR / total_cnt))
    logger.info("Average SI-SNRi: {:.2f}".format(total_SISNRi / total_cnt))
    logger.info(
        "Acceptance rate of Utterances with SI-SDRi > 1 dB: {:.2f}".format(
            accept_cnt / total_cnt * 100))


if __name__ == "__main__":
    fire.Fire(infer)
