import auraloss
import torch.nn as nn
import torchmetrics.audio as audio_metrics
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio

"""Get a loss function with its name from the configuration file."""
valid_losses = {}

torch_losses = {
    "L1": nn.L1Loss(),
    "L2": nn.MSELoss(),
    "CE": nn.CrossEntropyLoss(),
}

torchmetrics_losses = {
    # Not tested
    "PIT": audio_metrics.PermutationInvariantTraining(
        scale_invariant_signal_noise_ratio
    ),
}

auraloss_losses = {
    "STFT": auraloss.freq.STFTLoss(),
    "MultiResolutionSTFT": auraloss.freq.MultiResolutionSTFTLoss(),
    "SISDR": auraloss.time.SISDRLoss(),
    "SISNR": auraloss.time.SISDRLoss(),
    "SNR": auraloss.time.SNRLoss(),
}

valid_losses.update(torch_losses)
valid_losses.update(auraloss_losses)
valid_losses.update(torchmetrics_losses)


def parse_loss(loss):
    loss_functions = []
    if not isinstance(loss, list):
        loss = [loss]
    for i in range(len(loss)):
        loss_name = loss[i]
        loss_functions.append(valid_losses.get(loss_name))
    return loss_functions
