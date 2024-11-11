from model_lib.inference import predict as diffwave_predict
import torchaudio
import torchaudio.transforms as TT
import torch
import matplotlib.pyplot as plt
import os

import scipy.io.wavfile as scipy_wav

input_dir = "./data"
output_dir = "./result"

filename = 'audio_original.wav'
# filename = 'LJ025-0076.wav'
# filename = 'reference_0.wav'

sample_rate, waveform = scipy_wav.read(os.path.join(input_dir, filename))

waveform = torch.tensor(waveform, dtype=torch.float32).reshape(1, -1)
# Verifique a forma do waveform
print("Forma do waveform:", waveform.shape)  # Espera-se [C, T]
print("sample_rate:", sample_rate)

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

# plt.imshow(spectrogram)
# plt.tight_layout()
# plt.savefig("./result/spec.png")

# spectrogram = spectrogram.reshape(spectrogram.shape[0], 1, spectrogram.shape[1])
# print(spectrogram.shape)

# # Redimensionar para [N, C, W]
# # Aqui N=1 (1 amostra), C=1 (1 canal), W=number of frames
# # O mel_spectrogram deve já estar em [C, n_mels, W]
# if spectrogram.dim() == 3:  # Se for 3D, adiciona a dimensão do batch
#     spectrogram = spectrogram.unsqueeze(0)  # Adiciona dimensão para N
# else:
#     raise ValueError("O espectrograma não tem a dimensão esperada.")

model_dir = './model/diffwave-ljspeech-22kHz-1000578.pt'
#spectrogram = # get your hands on a spectrogram in [N,C,W] format
audio, sample_rate = diffwave_predict(spectrogram,model_dir, fast_sampling=False)
print(audio)
# Suponha que 'audio' seja o tensor que você obteve como saída
# Se o seu tensor for 1D, adicione uma dimensão para transformá-lo em 2D
if audio is None:
    raise ValueError("O tensor de áudio não foi gerado corretamente.")

# Se o seu tensor for 1D, adicione uma dimensão para transformá-lo em 2D
if audio.dim() == 1:  # Se for apenas [T]
    audio = audio.unsqueeze(0)  # Adiciona a dimensão do canal

# Verifique a forma do tensor após a transformação
print("Forma do áudio:", audio.shape)  # Deve ser [C, T]

# Mover o tensor para a CPU antes de salvar
audio = audio.cpu()

# Agora você pode salvar o áudio
torchaudio.save(os.path.join(output_dir, filename), audio, sample_rate=sample_rate)
