
from diffwave.inference import predict as diffwave_predict
import torchaudio
import torchaudio.transforms as T
import torch
from matplotlib import pyplot as plt

import torchaudio
torchaudio.set_audio_backend("soundfile")

## Carregar o áudio
audio_path = 'LJ037-0171.wav'  # Substitua pelo seu caminho de áudio
waveform, sample_rate = torchaudio.load(audio_path)

# Verifique a forma do waveform
print("Forma do waveform:", waveform.shape)  # Espera-se [C, T]

# Criar o espectrograma Mel
mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=80, f_max=8000)
mel_spectrogram = mel_transform(waveform)
print(mel_spectrogram)
plt.imshow(mel_spectrogram)
# Converter para escala logarítmica
log_mel_spectrogram = 10 * torch.log10(mel_spectrogram.clamp(min=1e-10))  # Evitar log(0)

# Verifique a forma do mel_spectrogram
print("Forma do mel_spectrogram:", mel_spectrogram.shape)  # Espera-se [C, n_mels, W]

# Redimensionar para [N, C, W]
# Aqui N=1 (1 amostra), C=1 (1 canal), W=number of frames
# O mel_spectrogram deve já estar em [C, n_mels, W]
if log_mel_spectrogram.dim() == 3:  # Se for 3D, adiciona a dimensão do batch
    spectrogram = log_mel_spectrogram.unsqueeze(0)  # Adiciona dimensão para N
else:
    raise ValueError("O espectrograma não tem a dimensão esperada.")




model_dir = '/home/pedro.rocha/diffwave-ljspeech-22kHz-1000578.pt'
#spectrogram = # get your hands on a spectrogram in [N,C,W] format
audio, sample_rate = diffwave_predict(mel_spectrogram,model_dir=model_dir, fast_sampling=True)
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
output_path = 'output_audio.wav'
torchaudio.save(output_path, audio, sample_rate=sample_rate)
print(2)

# audio is a GPU tensor in [N,T] format.

