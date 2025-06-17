import os
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import hilbert

def find_cpa(signal):
    # Calcula a envoltória do sinal (Hilbert transform)
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    # Encontra o índice do pico (CPA)
    cpa_index = np.argmax(envelope)
    return cpa_index

def fragment_at_cpa(input_folder, output_folder, window_seconds=15):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            file_path = os.path.join(input_folder, filename)
            sample_rate, data = wav.read(file_path)

            # Encontra o CPA
            cpa_index = find_cpa(data)
            cpa_time = cpa_index / sample_rate

            # Define a janela ao redor do CPA (ex: 15 segundos)
            window_samples = window_seconds * sample_rate
            start = max(0, cpa_index - window_samples // 2)
            end = min(len(data), cpa_index + window_samples // 2)

            # Extrai o segmento
            segment = data[start:end]

            # Salva o arquivo
            output_filename = f"{os.path.splitext(filename)[0]}_CPA.wav"
            output_path = os.path.join(output_folder, output_filename)
            wav.write(output_path, sample_rate, segment.astype(np.int16))

            print(f"Arquivo {filename} fragmentado no CPA (tempo: {cpa_time:.2f}s).")

# Uso
input_folder = "/data/4classes_full/A"
output_folder = "/home/leticia.luz/Documents/generative_simulator/test/4classes_CPA_15s/A"
fragment_at_cpa(input_folder, output_folder, window_seconds=15)