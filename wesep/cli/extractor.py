import os
import sys

from silero_vad import load_silero_vad, get_speech_timestamps
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import yaml
import soundfile

from wesep.cli.hub import Hub
from wesep.cli.utils import get_args
from wesep.models import get_model
from wesep.utils.checkpoint import load_pretrained_model
from wesep.utils.utils import set_seed


class Extractor:

    def __init__(self, model_dir: str):
        set_seed()

        config_path = os.path.join(model_dir, "config.yaml")
        model_path = os.path.join(model_dir, "avg_model.pt")
        with open(config_path, "r") as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
            if 'spk_model_init' in configs['model_args']['tse_model']:
                configs['model_args']['tse_model']['spk_model_init'] = False
        self.model = get_model(configs["model"]["tse_model"])(
            **configs["model_args"]["tse_model"]
        )
        load_pretrained_model(self.model, model_path)
        self.model.eval()
        self.vad = load_silero_vad()
        self.table = {}
        self.resample_rate = configs["dataset_args"].get("resample_rate", 16000)
        self.apply_vad = False
        self.device = torch.device("cpu")
        self.wavform_norm = True
        self.output_norm = True

        self.speaker_feat = configs["model_args"]["tse_model"].get("spk_feat", False)
        self.joint_training = configs["model_args"]["tse_model"].get(
            "joint_training", False
        )

    def set_wavform_norm(self, wavform_norm: bool):
        self.wavform_norm = wavform_norm

    def set_resample_rate(self, resample_rate: int):
        self.resample_rate = resample_rate

    def set_vad(self, apply_vad: bool):
        self.apply_vad = apply_vad

    def set_device(self, device: str):
        self.device = torch.device(device)
        self.model = self.model.to(self.device)

    def set_output_norm(self, output_norm: bool):
        self.output_norm = output_norm

    def compute_fbank(
        self,
        wavform,
        sample_rate=16000,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        cmn=True,
    ):
        feat = kaldi.fbank(
            wavform,
            num_mel_bins=num_mel_bins,
            frame_length=frame_length,
            frame_shift=frame_shift,
            sample_frequency=sample_rate,
        )
        if cmn:
            feat = feat - torch.mean(feat, 0)
        return feat

    def extract_speech(self, audio_path: str, audio_path_2: str):
        pcm_mix, sample_rate_mix = torchaudio.load(
            audio_path, normalize=self.wavform_norm
        )
        pcm_enroll, sample_rate_enroll = torchaudio.load(
            audio_path_2, normalize=self.wavform_norm
        )
        return self.extract_speech_from_pcm(pcm_mix,
                                            sample_rate_mix,
                                            pcm_enroll,
                                            sample_rate_enroll)

    def extract_speech_from_pcm(self,
                                pcm_mix: torch.Tensor,
                                sample_rate_mix: int,
                                pcm_enroll: torch.Tensor,
                                sample_rate_enroll: int):
        if self.apply_vad:
            # TODO(Binbin Zhang): Refine the segments logic, here we just
            # suppose there is only silence at the start/end of the speech
            # Only do vad on the enrollment
            vad_sample_rate = 16000
            wav = pcm_enroll
            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sample_rate_enroll != vad_sample_rate:
                transform = torchaudio.transforms.Resample(
                    orig_freq=sample_rate_enroll, new_freq=vad_sample_rate
                )
                wav = transform(wav)

            segments = get_speech_timestamps(wav, self.vad, return_seconds=True)
            pcmTotal = torch.Tensor()
            if len(segments) > 0:  # remove all the silence
                for segment in segments:
                    start = int(segment["start"] * sample_rate_enroll)
                    end = int(segment["end"] * sample_rate_enroll)
                    pcmTemp = pcm_enroll[0, start:end]
                    pcmTotal = torch.cat([pcmTotal, pcmTemp], 0)
                pcm_enroll = pcmTotal.unsqueeze(0)
            else:  # all silence, nospeech
                return None

        pcm_mix = pcm_mix.to(torch.float)
        if sample_rate_mix != self.resample_rate:
            pcm_mix = torchaudio.transforms.Resample(
                orig_freq=sample_rate_mix, new_freq=self.resample_rate
            )(pcm_mix)
        pcm_enroll = pcm_enroll.to(torch.float)
        if sample_rate_enroll != self.resample_rate:
            pcm_enroll = torchaudio.transforms.Resample(
                orig_freq=sample_rate_enroll, new_freq=self.resample_rate
            )(pcm_enroll)

        if self.joint_training:
            if self.speaker_feat:
                feats = self.compute_fbank(
                    pcm_enroll, sample_rate=self.resample_rate, cmn=True
                )
                feats = feats.unsqueeze(0)
                feats = feats.to(self.device)
            else:
                feats = pcm_enroll

            with torch.no_grad():
                outputs = self.model(pcm_mix, feats)
                outputs = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            target_speech = outputs.to(torch.device("cpu"))
            if self.output_norm:
                target_speech = target_speech / abs(target_speech).max(dim=1, keepdim=True).values * 0.9
            return target_speech
        else:
            return None


def load_model(language: str) -> Extractor:
    model_path = Hub.get_model(language)
    return Extractor(model_path)


def load_model_local(model_dir: str) -> Extractor:
    return Extractor(model_dir)


def main():
    args = get_args()
    if args.pretrain == "":
        if args.bsrnn:
            model = load_model("bsrnn")
        else:
            model = load_model(args.language)
    else:
        model = load_model_local(args.pretrain)
    model.set_resample_rate(args.resample_rate)
    model.set_vad(args.vad)
    model.set_device(args.device)
    model.set_output_norm(args.output_norm)
    if args.task == "extraction":
        speech = model.extract_speech(args.audio_file, args.audio_file2)
        if speech is not None:
            soundfile.write(args.output_file, speech[0], args.resample_rate)
            print("Succeed, see {}".format(args.output_file))
        else:
            print("Fails to extract the target speech")
    else:
        print("Unsupported task {}".format(args.task))
        sys.exit(-1)


if __name__ == "__main__":
    main()
