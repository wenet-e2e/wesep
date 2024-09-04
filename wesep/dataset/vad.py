import numpy as np
import soundfile as sf


class VoiceActivityDetection:

    def __init__(self, wave):
        self.wave = wave

    def segmentation(self, overlap, slice_len):
        frequency = 16000
        signal = self.wave
        self.seg_len = len(signal) / frequency
        self.slice_len = slice_len
        overlap = 2

        slices = np.arange(0, self.seg_len, slice_len - overlap, dtype=np.intc)
        # print(slices)
        audio_slices = []
        for start, end in zip(slices[:-1], slices[1:]):
            start_audio = start * frequency
            end_audio = (end + overlap) * frequency
            audio_slice = signal[int(start_audio):int(end_audio)]
            # print(len(audio_slice))
            audio_slices.append(audio_slice)

            # wavfile.write('slices{}.wav'.format(start), 16000, audio_slice)
        # print(len(audio_slices))
        return audio_slices

    def calc_energy(self, audio):
        # for a in enumerate(audio):
        #     if (a == 0.0):
        #         a = 0.00001

        # print(np.sum(np.sum(audio**2)))

        energy = audio / np.sum(np.sum(audio**2) + 1e-8) * 1e2
        # print(len(audio))
        return energy

    def select(self):
        audio_slices = self.segmentation(overlap=1, slice_len=4)
        energies = []
        for audio in audio_slices:
            chunk_len = len(audio) / 10
            chunk_slice = np.arange(0,
                                    len(audio) + chunk_len,
                                    chunk_len,
                                    dtype=np.intc)

            for start, end in zip(chunk_slice[:-1], chunk_slice[1:]):

                energy = self.calc_energy(audio[start:end])
                # print(energy)
                for i, _ in enumerate(energy):
                    if (energy[i]) == 0:
                        energy[i] = 0.00001
                        # print(energy[i])
                energies.append(sum(energy))

        # print(energies)

        threshold = np.quantile(energies, 0.25)
        print(threshold)

        if threshold < 0.0001:
            threshold = 0.0001

        fin_audios = []
        i = 0
        for audio in audio_slices:
            chunk_len = len(audio) / 10
            chunk_slice = np.arange(0,
                                    len(audio) + chunk_len,
                                    chunk_len,
                                    dtype=np.intc)
            count = 0
            for start, end in zip(chunk_slice[:-1], chunk_slice[1:]):
                energy = self.calc_energy(audio[start:end])
                # if 50% enenrgy > threshold
                # print(energy)
                print(sum(i >= threshold for i in energy))
                if sum(i >= threshold for i in energy) >= chunk_len // 2:
                    count += 1
                # save seg
            # print(count)
            if count >= 5:
                sf.write("output{}.wav".format(i), audio, 16000)
                if len(audio) < self.slice_len * 16000:
                    # print(self.slice_len*16000-len(audio))
                    audio = np.concatenate(
                        [audio,
                         np.zeros(self.slice_len * 16000 - len(audio))])
                fin_audios.append(audio)

            i += 1

        if len(fin_audios) == 0:
            fin_audios.append(np.zeros(self.slice_len * 16000))
        return fin_audios
