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

from wesep.utils.funcs import clip_gradients,compute_fbank,apply_cmvn
import random 

class Executor:

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
            se_loss_weight=1.0,
            multi_task=False,
            SSA_enroll_prob=0,
            fbank_args=None,
            sample_rate=16000,
            speaker_feat=True
    ):
        """Train one epoch"""
        model = models[0]
        optimizer = optimizers[0]
        scheduler = schedulers[0]

        model.train()
        log_interval = log_batch_interval
        accum_grad = 1
        losses = []

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_context = model.join
        else:
            model_context = nullcontext

        with model_context():
            for i, batch in enumerate(dataloader):
                features = batch["wav_mix"]
                targets = batch["wav_targets"]
                # embeddings when not joint training, enrollment wavforms
                # when joint training
                enroll = batch["spk_embeds"]
                # spk_lable is an empty list when not joint training
                # and multi-task
                spk_label = batch["spk_label"]

                cur_iter = (epoch - 1) * epoch_iter + i
                scheduler.step(cur_iter)

                features = features.float().to(device)  # (B,T,F)
                targets = targets.float().to(device)
                enroll = enroll.float().to(device)
                spk_label = spk_label.to(device)

                with torch.cuda.amp.autocast(enabled=enable_amp):
                    if SSA_enroll_prob >0:
                        if SSA_enroll_prob>random.random():
                            with torch.no_grad():
                                outputs = model(features, enroll)
                                est_speech = outputs[0]
                                self_fbank = est_speech
                                if fbank_args!=None and speaker_feat==True:
                                    self_fbank = compute_fbank(est_speech,**fbank_args,sample_rate=sample_rate)
                                    self_fbank = apply_cmvn(self_fbank)
                            outputs = model(features, self_fbank)
                        else:
                            outputs = model(features, enroll)
                    else: 
                        outputs = model(features, enroll)
                    if not isinstance(outputs, (list, tuple)):
                        outputs = [outputs]
                    loss = 0
                    for ii in range(len(criterion)):
                        # se_loss_weight: ([position in outputs[0], [1]],
                        #                 [weights:[1.0], [0.5]])
                        for ji in range(len(se_loss_weight[0][ii])):
                            if (multi_task and criterion[ii].__class__.__name__
                                    == "CrossEntropyLoss"):
                                loss += se_loss_weight[1][ii][ji] * (
                                    criterion[ii](
                                        outputs[se_loss_weight[0][ii][ji]],
                                        spk_label,
                                    ).mean() / accum_grad)
                                continue
                            loss += se_loss_weight[1][ii][ji] * (criterion[ii](
                                outputs[se_loss_weight[0][ii][ji]],
                                targets).mean() / accum_grad)

                losses.append(loss.item())
                total_loss_avg = sum(losses) / len(losses)

                # updata the model
                optimizer.zero_grad()
                # scaler does nothing here if enable_amp=False
                scaler.scale(loss).backward()
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
                                total_loss_avg * accum_grad,
                                optimizer.param_groups[0]["lr"],
                            ),
                            width=10,
                            style="grid",
                        ))
                if (i + 1) == epoch_iter:
                    break
            total_loss_avg = sum(losses) / len(losses)
            return total_loss_avg, 0

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
        model = models[0]

        model.eval()
        log_interval = log_batch_interval
        losses = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                features = batch["wav_mix"]
                targets = batch["wav_targets"]
                enroll = batch["spk_embeds"]

                features = features.float().to(device)  # (B,T,F)
                targets = targets.float().to(device)
                enroll = enroll.float().to(device)

                with torch.cuda.amp.autocast(enabled=enable_amp):
                    outputs = model(features, enroll)
                    if not isinstance(outputs, (list, tuple)):
                        outputs = [outputs]
                    # By default, the first loss is used as the indicator
                    # of the validation set.
                    loss = criterion[0](outputs[0], targets).mean()

                losses.append(loss.item())
                total_loss_avg = sum(losses) / len(losses)

                if (i + 1) % log_interval == 0:
                    logger.info(
                        tp.row(
                            ("VAL", epoch, i + 1, total_loss_avg, "-"),
                            width=10,
                            style="grid",
                        ))
                if (i + 1) == val_iter:
                    break
        return total_loss_avg, 0
