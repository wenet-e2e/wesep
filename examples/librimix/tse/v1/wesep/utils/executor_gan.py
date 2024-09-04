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

from contextlib import nullcontext

import tableprint as tp

# if your python version < 3.7 use the below one
import torch
import torch.nn.functional as F

from wesep.utils.funcs import clip_gradients
from wesep.utils.score import batch_evaluation, cal_PESQ_norm


class ExecutorGAN:
    def __init__(self):
        self.step = 0

    def train(
        self,
        dataloader,
        models,
        epoch_iter,
        optimizers,
        criterion,
        schedulers,
        scaler,
        epoch,
        enable_amp,
        logger,
        clip_grad=5.0,
        log_batch_interval=100,
        device=torch.device("cuda"),
        se_loss_weight=0.95,
        gan_loss_weight=0.05,
        multi_task=False,
    ):
        """Train one epoch"""
        assert (
            len(models) == len(optimizers) == len(schedulers) == 2
        ), "Currently only support one discriminator"
        model, discriminator = models
        optimizer, optimizer_dis = optimizers
        scheduler, scheduler_dis = schedulers

        model.train()
        discriminator.train()
        log_interval = log_batch_interval
        accum_grad = 1
        losses = []
        se_losses = []
        dis_losses = []

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_context = model.join
        else:
            model_context = nullcontext

        with model_context():
            for i, batch in enumerate(dataloader):
                features = batch["wav_mix"]
                targets = batch["wav_targets"]
                enroll = batch[
                    "spk_embeds"
                ]  # embeddings when when not joint training, enrollment wavforms when joint training
                spk_label = batch[
                    "spk_label"
                ]  # spk_lable is an empty list when not joint training and multi-task
                one_labels = torch.ones(features.size(0))

                cur_iter = (epoch - 1) * epoch_iter + i
                scheduler.step(cur_iter)
                scheduler_dis.step(cur_iter)

                features = features.float().to(device)
                targets = targets.float().to(device)
                enroll = enroll.float().to(device)
                spk_label = spk_label.to(device)
                one_labels = one_labels.float().to(device)

                # calculate discriminator loss
                with torch.cuda.amp.autocast(enabled=enable_amp):
                    outputs = model(features, enroll)
                    if not isinstance(outputs, (list, tuple)):
                        outputs = [outputs]
                    # outputs is a list of tensors, each tensor has shape (Batch, samples)
                    if multi_task:
                        # remove the predicted spk_label from the outputs list
                        enhanced_wavs = torch.stack(outputs[:-1], dim=0)
                    else:
                        # enhanced_wavs: [N, Batch, samples], N is the number of output of the model
                        enhanced_wavs = torch.stack(outputs, dim=0)
                    d_loss = self._calculate_discriminator_loss(
                        discriminator,
                        targets,
                        enhanced_wavs.detach(),
                        features.detach(),
                    )

                dis_losses.append(d_loss.item())
                total_dis_loss_avg = sum(dis_losses) / len(dis_losses)
                # updata discriminator
                optimizer_dis.zero_grad()
                # scaler does nothing here if enable_amp=False
                scaler.scale(d_loss).backward()
                scaler.unscale_(optimizer_dis)
                clip_gradients(discriminator, clip_grad)
                scaler.step(optimizer_dis)
                scaler.update()

                # calculate generator loss
                with torch.cuda.amp.autocast(enabled=enable_amp):
                    se_loss = 0
                    for ii in range(len(criterion)):
                        for ji in range(
                            len(se_loss_weight[0][ii])
                        ):  # se_loss_weight[0]: 2-D array,loss_posi; se_loss_weight[1]: 2-D array,loss_weight.
                            if multi_task and ii == (len(criterion) - 1):
                                se_loss += se_loss_weight[1][ii][ji] * (
                                    criterion[ii](
                                        outputs[se_loss_weight[0][ii][ji]],
                                        spk_label,
                                    ).mean()
                                    / accum_grad
                                )
                                continue
                            se_loss += se_loss_weight[1][ii][ji] * (
                                criterion[ii](
                                    outputs[se_loss_weight[0][ii][ji]], targets
                                ).mean()
                                / accum_grad
                            )
                    gan_loss = 0
                    len_output = (
                        len(outputs) - 1 if multi_task else len(outputs)
                    )
                    for j in range(len_output):
                        enhanced_fake_metric = discriminator(
                            targets, outputs[j]
                        )
                        gan_loss += F.mse_loss(
                            enhanced_fake_metric.flatten(),
                            one_labels,
                        )
                    g_loss = se_loss + gan_loss_weight * gan_loss

                losses.append(g_loss.item())
                se_losses.append(se_loss.item())
                total_loss_avg = sum(losses) / len(losses)
                total_se_loss_avg = sum(se_losses) / len(se_losses)

                # updata the generator
                optimizer.zero_grad()
                # scaler does nothing here if enable_amp=False
                scaler.scale(g_loss).backward()
                scaler.unscale_(optimizer)
                clip_gradients(model, clip_grad)
                scaler.step(optimizer)
                scaler.update()

                if (i + 1) % log_interval == 0:
                    logger.info(
                        tp.row(
                            (
                                "TRAIN",
                                epoch,
                                i + 1,
                                total_se_loss_avg,
                                total_loss_avg * accum_grad,
                                total_dis_loss_avg * accum_grad,
                                optimizer.param_groups[0]["lr"],
                            ),
                            width=10,
                            style="grid",
                        )
                    )
                if (i + 1) == epoch_iter:
                    break
            total_loss_avg = sum(losses) / len(losses)
            total_dis_loss_avg = sum(dis_losses) / len(dis_losses)
            return total_loss_avg, total_dis_loss_avg

    def cv(
        self,
        dataloader,
        models,
        val_iter,
        criterion,
        epoch,
        enable_amp,
        logger,
        log_batch_interval=100,
        device=torch.device("cuda"),
    ):
        """Cross validation on"""
        assert len(models) == 2, "Currently only support one discriminator"
        model, discriminator = models
        model.eval()
        discriminator.eval()
        log_interval = log_batch_interval
        losses = []
        se_losses = []
        dis_losses = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                features = batch["wav_mix"]
                targets = batch["wav_targets"]
                enroll = batch["spk_embeds"]
                one_labels = torch.ones(features.size(0))

                features = features.float().to(device)  # (B,T,F)
                targets = targets.float().to(device)
                enroll = enroll.float().to(device)
                one_labels = one_labels.float().to(device)

                with torch.cuda.amp.autocast(enabled=enable_amp):
                    outputs = model(features, enroll)
                    if not isinstance(outputs, (list, tuple)):
                        outputs = [outputs]
                    # calculate discriminator loss
                    d_loss = self._calculate_discriminator_loss(
                        discriminator,
                        targets,
                        outputs[0].unsqueeze(0),
                        features,
                    )

                dis_losses.append(d_loss.item())
                total_dis_loss_avg = sum(dis_losses) / len(dis_losses)

                # calculate generator loss
                with torch.cuda.amp.autocast(enabled=enable_amp):
                    se_loss = criterion[0](outputs[0], targets).mean()
                    enhanced_fake_metric = discriminator(targets, outputs[0])
                    gan_loss = F.mse_loss(
                        enhanced_fake_metric.flatten(),
                        one_labels,
                    )
                    g_loss = se_loss + gan_loss

                losses.append(g_loss.item())
                se_losses.append(se_loss.item())
                total_loss_avg = sum(losses) / len(losses)
                total_se_loss_avg = sum(se_losses) / len(se_losses)

                if (i + 1) % log_interval == 0:
                    logger.info(
                        tp.row(
                            (
                                "VAL",
                                epoch,
                                i + 1,
                                total_se_loss_avg,
                                total_loss_avg,
                                total_dis_loss_avg,
                                "-",
                            ),
                            width=10,
                            style="grid",
                        )
                    )
                if (i + 1) == val_iter:
                    break
        return total_loss_avg, total_dis_loss_avg

    def mse_loss(self, output, target):
        return F.mse_loss(output.flatten(), target)

    def _calculate_discriminator_loss(
        self,
        discriminator,
        clean_wavs,
        enhanced_wavs,
        noisy_wavs,
    ):
        """Calculate the discriminator loss

        Args:
            discriminator (torch.nn.Module): the discriminator model
            clean_wavs (torch.Tensor): the clean waveforms, [Batch, samples]
            enhanced_wavs (torch.Tensor): the predicted waveforms, [N, Batch, samples]
            noisy_wavs (torch.Tensor): the noisy waveforms, [Batch, samples]

        Returns:
            torch.Tensor: the discriminator loss
        """

        def calculate_mse_loss(output, target):
            if target is not None:
                target = torch.FloatTensor(target).to(device)
                return self.mse_loss(output, target)
            return 0

        device = clean_wavs.device
        one_labels = torch.ones(clean_wavs.size(0)).float().to(device)

        noisy_fake_metric = discriminator(clean_wavs, noisy_wavs)
        clean_fake_metric = discriminator(clean_wavs, clean_wavs)

        audio_ref = clean_wavs.detach().cpu().numpy()
        audio_noisy = noisy_wavs.detach().cpu().numpy()

        noisy_real_metric = batch_evaluation(
            cal_PESQ_norm, audio_noisy, audio_ref, parallel=False
        )

        loss_d_clean = self.mse_loss(clean_fake_metric, one_labels)
        loss_d_noisy = calculate_mse_loss(noisy_fake_metric, noisy_real_metric)
        d_loss = loss_d_clean + loss_d_noisy

        # unbind enhanced_wavs to get a list of tensors, each tensor has shape (Batch, samples)
        enhanced_wavs = torch.unbind(enhanced_wavs, dim=0)

        for enhanced_wav in enhanced_wavs:
            enhanced_fake_metric = discriminator(clean_wavs, enhanced_wav)
            audio_est = enhanced_wav.detach().cpu().numpy()

            enhanced_real_metric = batch_evaluation(
                cal_PESQ_norm, audio_est, audio_ref, parallel=False
            )

            loss_d_enhanced = calculate_mse_loss(
                enhanced_fake_metric, enhanced_real_metric
            )

            d_loss += loss_d_enhanced

        return d_loss
