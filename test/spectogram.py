from diffwave.inference import predict as diffwave_predict
import torchaudio
import torchaudio.transforms as TT
import torch
import matplotlib.pyplot as plt

import scipy.io.wavfile as scipy_wav

## Carregar o áudio
audio_path = './data/audio_original.wav'  # Substitua pelo seu caminho de áudio
sample_rate, waveform = scipy_wav.read(audio_path)

waveform = torch.tensor(waveform, dtype=torch.float32).reshape(1, -1)

audio = torch.clamp(waveform[0], -1.0, 1.0)
hop_samples=256
n_fft=1024
n_mels=80

mel_args = {
    'sample_rate': sample_rate,
    'win_length': hop_samples * 4,
    'hop_length': hop_samples,
    'n_fft': n_fft,
    'f_min': 20.0,
    'f_max': sample_rate / 2.0,
    'n_mels': n_mels,
    'power': 1.0,
    'normalized': True,
}
mel_spec_transform = TT.MelSpectrogram(**mel_args)

spectrogram = mel_spec_transform(audio)
spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)

plt.imshow(spectrogram)
plt.tight_layout()
plt.savefig("./result/spec.png")