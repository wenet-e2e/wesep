# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import re
from pprint import pformat

import fire
import matplotlib.pyplot as plt
import tableprint as tp
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader

import wesep.utils.schedulers as schedulers
from wesep.dataset.dataset import Dataset, tse_collate_fn, tse_collate_fn_2spk
from wesep.models import get_model
from wesep.utils.checkpoint import (
    load_checkpoint,
    load_pretrained_model,
    save_checkpoint,
)
from wesep.utils.executor_gan import ExecutorGAN
from wesep.utils.file_utils import (
    load_speaker_embeddings,
    read_label_file,
    read_vec_scp_file,
)
from wesep.utils.losses import parse_loss
from wesep.utils.utils import parse_config_or_kwargs, set_seed, setup_logger

MAX_NUM_log_files = 100  # The maximum number of log-files to be kept
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def train(config="conf/config.yaml", **kwargs):
    """Trains a model on the given features and spk labels.

    :config: A training configuration. Note that all parameters in the
             config can also be manually adjusted with --ARG VALUE
    :returns: None
    """
    configs = parse_config_or_kwargs(config, **kwargs)
    checkpoint = configs.get("checkpoint", None)
    if checkpoint is not None:
        checkpoint = os.path.realpath(checkpoint)
    find_unused_parameters = configs.get("find_unused_parameters", False)
    gan_loss_weight = configs.get("gan_loss_weight", 0.05)

    # dist configs
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    gpu = int(configs["gpus"][rank])
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend="nccl")

    # Log rotation
    model_dir = os.path.join(configs["exp_dir"], "models")
    logger = setup_logger(rank, configs["exp_dir"], gpu, MAX_NUM_log_files)

    print("-------------------", dist.get_rank(), world_size)
    if world_size > 1:
        logger.info("training on multiple gpus, this gpu {}".format(gpu))

    if rank == 0:
        logger.info("exp_dir is: {}".format(configs["exp_dir"]))
        logger.info("<== Passed Arguments ==>")
        # Print arguments into logs
        for line in pformat(configs).split("\n"):
            logger.info(line)

    # seed
    set_seed(configs["seed"] + rank)

    # support multiple losses, e.g., criterion = [SISNR, CE]
    criterion = configs.get("loss", None)
    if criterion:
        criterion = parse_loss(criterion)
    else:
        criterion = [
            parse_loss("SISNR"),
        ]
    # loss_posi is used to store the indices when the model has multiple outputs
    # loss_posi[i][j] stores the index of the output used for i-th criterion,
    # that is, output[loss_posi[i][j]] is used for the i-th criterion.
    loss_posi = configs["loss_args"].get(
        "loss_posi",
        [
            [
                0,
            ]
        ],
    )
    # loss_weight[i][j] stores the loss weight of output[loss_posi[i][j]] for the i-th criterion.
    loss_weight = configs["loss_args"].get(
        "loss_weight",
        [
            [
                1.0,
            ]
        ],
    )
    loss_args = (loss_posi, loss_weight)

    # embeds
    tr_spk_embeds = configs["train_spk_embeds"]
    tr_single_utt2spk = configs["train_utt2spk"]
    joint_training = configs["model_args"]["tse_model"].get(
        "joint_training", False
    )
    multi_task = configs["model_args"]["tse_model"].get("multi_task", False)

    # dict_spk: {spk_id: int_label}
    dict_spk = {}
    if not joint_training:
        tr_spk2embed_dict = load_speaker_embeddings(
            tr_spk_embeds, tr_single_utt2spk
        )
        multi_task = False
    else:
        with open(configs["train_spk2utt"], "r") as f:
            tr_spk2embed_dict = json.load(f)
            # tr_spk2embed_dict: {spk_id: [[spk_id, wav_path], ...]}
            if multi_task:
                for i, j in enumerate(
                    tr_spk2embed_dict.keys()
                ):  # Generate the dictionary for speakers in training set
                    dict_spk[j] = i

    with open(tr_single_utt2spk, "r") as f:
        tr_lines = f.readlines()

    val_spk_embeds = configs["val_spk_embeds"]
    val_spk1_enroll = configs["val_spk1_enroll"]
    val_spk2_enroll = configs["val_spk2_enroll"]

    if not joint_training:
        val_spk2embed_dict = read_vec_scp_file(val_spk_embeds)
    else:
        val_spk2embed_dict = read_label_file(configs["val_spk2utt"])

    val_spk1_embed = read_label_file(val_spk1_enroll)
    val_spk2_embed = read_label_file(val_spk2_enroll)

    with open(val_spk_embeds, "r") as f:
        val_lines = f.readlines()

    # dataset and dataloader
    train_dataset = Dataset(
        configs["data_type"],
        configs["train_data"],
        configs["dataset_args"],
        tr_spk2embed_dict,
        None,
        None,
        state="train",
        joint_training=joint_training,
        dict_spk=dict_spk,
        whole_utt=configs.get("whole_utt", False),
        repeat_dataset=configs.get("repeat_dataset", True),
        reverb=configs["dataset_args"].get("reverb", False),
        noise=configs["dataset_args"].get("noise", False),
        noise_lmdb_file=configs["dataset_args"].get("noise_lmdb_file", None),
        online_mix=configs["dataset_args"].get("online_mix", False),
    )
    val_dataset = Dataset(
        configs["data_type"],
        configs["val_data"],
        configs["dataset_args"],
        val_spk2embed_dict,
        val_spk1_embed,
        val_spk2_embed,
        state="val",
        joint_training=joint_training,
        whole_utt=configs.get("whole_utt", False),
        repeat_dataset=True,
        reverb=False,
        online_mix=False,
    )
    train_dataloader = DataLoader(
        train_dataset, **configs["dataloader_args"], collate_fn=tse_collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset,
        **configs["dataloader_args"],
        collate_fn=tse_collate_fn_2spk,
    )
    batch_size = configs["dataloader_args"]["batch_size"]
    if configs["dataset_args"].get("sample_num_per_epoch", 0) > 0:
        sample_num_per_epoch = configs["dataset_args"]["sample_num_per_epoch"]
    else:
        sample_num_per_epoch = len(tr_lines) // 2
    epoch_iter = sample_num_per_epoch // world_size // batch_size
    val_iter = len(val_lines) // 2 // world_size // batch_size
    if rank == 0:
        logger.info("<== Dataloaders ==>")
        logger.info("train dataloaders created")
        logger.info("epoch iteration number: {}".format(epoch_iter))
        logger.info("val iteration number: {}".format(val_iter))

    # model
    model_list = []
    scheduler_list = []
    optimizer_list = []

    logger.info("<== Model ==>")
    model = get_model(configs["model"]["tse_model"])(
        **configs["model_args"]["tse_model"]
    )
    num_params = sum(param.numel() for param in model.parameters())

    if rank == 0:
        logger.info("tse_model size: {}".format(num_params))
        # print model
        for line in pformat(model).split("\n"):
            logger.info(line)

    # ddp_model
    model.cuda()
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model, find_unused_parameters=find_unused_parameters
    )
    device = torch.device("cuda")

    if rank == 0:
        logger.info("<== TSE Model Loss ==>")
        logger.info("loss criterion is: " + str(configs["loss"]))

    configs["optimizer_args"]["tse_model"]["lr"] = configs["scheduler_args"][
        "tse_model"
    ]["initial_lr"]
    optimizer = getattr(torch.optim, configs["optimizer"]["tse_model"])(
        ddp_model.parameters(), **configs["optimizer_args"]["tse_model"]
    )
    if rank == 0:
        logger.info("<== TSE Model Optimizer ==>")
        logger.info("optimizer is: " + configs["optimizer"]["tse_model"])

    # scheduler
    configs["scheduler_args"]["tse_model"]["num_epochs"] = configs[
        "num_epochs"
    ]
    configs["scheduler_args"]["tse_model"]["epoch_iter"] = epoch_iter
    configs["scheduler_args"]["scale_ratio"] = 1.0

    scheduler = getattr(schedulers, configs["scheduler"]["tse_model"])(
        optimizer, **configs["scheduler_args"]["tse_model"]
    )
    if rank == 0:
        logger.info("<== TSE Model Scheduler ==>")
        logger.info("scheduler is: " + configs["scheduler"]["tse_model"])

    if configs["model_init"]["tse_model"] is not None:
        logger.info(
            "Load initial model from {}".format(
                configs["model_init"]["tse_model"]
            )
        )
        load_pretrained_model(ddp_model, configs["model_init"]["tse_model"])
    elif checkpoint is None:
        logger.info("Train model from scratch ...")

    for c in criterion:
        c = c.to(device)

    # append to list
    model_list.append(ddp_model)
    optimizer_list.append(optimizer)
    scheduler_list.append(scheduler)
    scaler = torch.cuda.amp.GradScaler(enabled=configs["enable_amp"])

    # discriminator
    discriminator = get_model(configs["model"]["discriminator"])(
        **configs["model_args"]["discriminator"]
    )
    num_params = sum(param.numel() for param in discriminator.parameters())
    # optimizer
    configs["optimizer_args"]["discriminator"]["lr"] = configs[
        "scheduler_args"
    ]["discriminator"]["initial_lr"]
    # scheduler
    configs["scheduler_args"]["discriminator"]["num_epochs"] = configs[
        "num_epochs"
    ]
    configs["scheduler_args"]["discriminator"]["epoch_iter"] = epoch_iter
    configs["scheduler_args"]["discriminator"]["scale_ratio"] = 1.0
    # ddp model
    discriminator.cuda()
    ddp_discriminator = torch.nn.parallel.DistributedDataParallel(
        discriminator, find_unused_parameters=find_unused_parameters
    )
    optimizer_d = getattr(torch.optim, configs["optimizer"]["discriminator"])(
        ddp_discriminator.parameters(),
        **configs["optimizer_args"]["discriminator"],
    )
    scheduler_d = getattr(schedulers, configs["scheduler"]["discriminator"])(
        optimizer_d, **configs["scheduler_args"]["discriminator"]
    )

    # initialize discriminator
    if configs["model_init"]["discriminator"] is not None:
        logger.info(
            "Load initial discriminator from {}".format(
                configs["model_init"]["discriminator"]
            )
        )
        load_pretrained_model(
            ddp_discriminator,
            configs["model_init"]["discriminator"],
            type="discriminator",
        )
    elif checkpoint is None:
        logger.info("Train discriminator from scratch ...")

    # If specify checkpoint, load some info from checkpoint.
    if checkpoint is not None:
        load_checkpoint(
            model_list, optimizer_list, scheduler_list, scaler, checkpoint
        )
        start_epoch = (
            int(re.findall(r"(?<=checkpoint_)\d*(?=.pt)", checkpoint)[0]) + 1
        )
        logger.info("Load checkpoint: {}".format(checkpoint))
    else:
        start_epoch = 1

    model_list.append(ddp_discriminator)
    optimizer_list.append(optimizer_d)
    scheduler_list.append(scheduler_d)

    if rank == 0:
        logger.info("<== Discriminator Model ==>")
        logger.info("discriminator size: {}".format(num_params))
        for line in pformat(discriminator).split("\n"):
            logger.info(line)
        logger.info("<== Discriminator Optimizer ==>")
        logger.info("optimizer is: " + configs["optimizer"]["discriminator"])
        logger.info("<== Discriminator Scheduler ==>")
        logger.info("scheduler is: " + configs["scheduler"]["discriminator"])

        # save config.yaml
        saved_config_path = os.path.join(configs["exp_dir"], "config.yaml")
        with open(saved_config_path, "w") as fout:
            data = yaml.dump(configs)
            fout.write(data)

    logger.info("start_epoch: {}".format(start_epoch))

    # training
    dist.barrier(device_ids=[gpu])  # synchronize here
    if rank == 0:
        logger.info("<========== Training process ==========>")
        header = [
            "Train/Val",
            "Epoch",
            "iter",
            "SE_Loss",
            "G_Loss",
            "D_Loss",
            "LR",
        ]
        for line in tp.header(header, width=10, style="grid").split("\n"):
            logger.info(line)
    dist.barrier(device_ids=[gpu])  # synchronize here

    executor = ExecutorGAN()
    executor.step = 0

    train_losses = []
    val_losses = []
    train_d_losses = []
    val_d_losses = []
    for epoch in range(start_epoch, configs["num_epochs"] + 1):
        train_dataset.set_epoch(epoch)

        train_loss, train_d_loss = executor.train(
            train_dataloader,
            model_list,
            epoch_iter,
            optimizer_list,
            criterion,
            scheduler_list,
            scaler=scaler,
            epoch=epoch,
            logger=logger,
            enable_amp=configs["enable_amp"],
            clip_grad=configs["clip_grad"],
            log_batch_interval=configs["log_batch_interval"],
            device=device,
            se_loss_weight=loss_args,
            gan_loss_weight=gan_loss_weight,
            multi_task=multi_task,
        )

        val_loss, val_d_loss = executor.cv(
            val_dataloader,
            model_list,
            val_iter,
            criterion,
            epoch=epoch,
            logger=logger,
            enable_amp=configs["enable_amp"],
            log_batch_interval=configs["log_batch_interval"],
            device=device,
        )

        if rank == 0:
            logger.info(
                "Epoch {} Train info train_loss {}, train_d_loss {}".format(
                    epoch, train_loss, train_d_loss
                )
            )
            logger.info(
                "Epoch {} Val info val_loss {}, val_d_loss {}".format(
                    epoch, val_loss, val_d_loss
                )
            )
            train_losses.append(train_loss)
            train_d_losses.append(train_d_loss)
            val_losses.append(val_loss)
            val_d_losses.append(val_d_loss)

            best_loss = val_loss
            scheduler.best = best_loss
            # plot
            plt.figure()
            plt.title("Loss of Train and Validation")
            x = list(range(start_epoch, epoch + 1))
            plt.plot(
                x, train_losses, "b-", label="train_G_loss", linewidth=0.8
            )
            plt.plot(
                x, train_d_losses, "r-", label="train_D_loss", linewidth=0.8
            )
            plt.plot(x, val_losses, "c-", label="val_G_loss", linewidth=0.8)
            plt.plot(x, val_d_losses, "m-", label="val_D_loss", linewidth=0.8)
            plt.legend()
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.xticks(range(start_epoch, epoch + 1, 1))
            plt.savefig(
                f"{configs['exp_dir']}/{configs['model']['tse_model']}.png"
            )
            plt.close()

        if rank == 0:
            if (
                epoch % configs["save_epoch_interval"] == 0
                or epoch >= configs["num_epochs"] - configs["num_avg"]
            ):
                save_checkpoint(
                    model_list,
                    optimizer_list,
                    scheduler_list,
                    scaler,
                    os.path.join(model_dir, "checkpoint_{}.pt".format(epoch)),
                )
            try:
                os.symlink(
                    "checkpoint_{}.pt".format(epoch),
                    os.path.join(model_dir, "latest_checkpoint.pt"),
                )
            except FileExistsError:
                os.remove(os.path.join(model_dir, "latest_checkpoint.pt"))
                os.symlink(
                    "checkpoint_{}.pt".format(epoch),
                    os.path.join(model_dir, "latest_checkpoint.pt"),
                )

    if rank == 0:
        os.symlink(
            "checkpoint_{}.pt".format(configs["num_epochs"]),
            os.path.join(model_dir, "final_checkpoint.pt"),
        )
        logger.info(tp.bottom(len(header), width=10, style="grid"))


if __name__ == "__main__":
    fire.Fire(train)
