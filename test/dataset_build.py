from glob import glob
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample
import numpy as np
import os

df = pd.read_csv("/home/leticia.luz/Documents/generative_simulator/generative_simulator/data/data.csv")

# class_id = "pilot_vessel"
class_id = "bulk_carrier"
filenames = glob(f"/data/{class_id}/*.wav", recursive=True)

# Certifique-se de que a pasta de saida existe
output_dir = f"./data/Slices/{class_id}"
os.makedirs(output_dir, exist_ok=True)

# Nova frequência de amostragem
target_sample_rate = 16000  # 16 kHz

# Loop pelos arquivos de áudio
for filename in filenames:
    try:
        # Extrai o ID do nome do arquivo
        id = int(filename[-8:-4])
        
        # Busca o tempo de CPA no DataFrame
        cpa_time = df.loc[df['IARA ID'] == id, 'CPA time'].values[0]
        
        # Lê o arquivo de áudio com scipy
        original_sample_rate, audio_data = wavfile.read(filename)
        
        # Calcula o número de amostras correspondentes aos 5 segundos na taxa original
        segment_duration_samples = int(5 * original_sample_rate)
        
        # Calcula o índice de início e fim dos 5 segundos centrados em cpa_time
        center_sample = int(cpa_time * original_sample_rate)
        start_sample = max(0, center_sample - segment_duration_samples // 2)
        end_sample = start_sample + segment_duration_samples
        
        # Ajusta o final se ultrapassar o tamanho do áudio
        if end_sample > len(audio_data):
            end_sample = len(audio_data)
            start_sample = max(0, end_sample - segment_duration_samples)
        
        # Extrai o trecho do áudio
        audio_segment = audio_data[start_sample:end_sample]
        
        # Redimensiona o áudio para 16 kHz
        num_samples = int(len(audio_segment) * target_sample_rate / original_sample_rate)
        audio_segment_resampled = resample(audio_segment, num_samples)
        
        # Cria o caminho para salvar o arquivo na pasta "./test"
        new_filename = os.path.join('./trn_data', os.path.basename(filename))
        
        # Salva o segmento redimensionado em 16 kHz em um novo arquivo .wav
        wavfile.write(new_filename, target_sample_rate, audio_segment_resampled.astype(np.int16))
        
        print(f"{filename}: Trecho de 5 segundos extraído e salvo em {new_filename} com 16 kHz")

    except Exception as e:
        print(f"Erro ao processar {filename}: {e}")
        continue
