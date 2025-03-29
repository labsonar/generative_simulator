import os
import numpy as np
import scipy.io.wavfile as wav

def fragment_wav_files(input_folder, output_folder, segment_length=15, overlap=5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            file_path = os.path.join(input_folder, filename)
            sample_rate, data = wav.read(file_path)

            segment_samples = segment_length * sample_rate
            overlap_samples = overlap * sample_rate
            step_size = segment_samples - overlap_samples

            num_segments = (len(data) - overlap_samples) // step_size

            for i in range(int(num_segments)):
                start_sample = i * step_size
                end_sample = start_sample + segment_samples
                segment = data[start_sample:end_sample]

                output_filename = f"{os.path.splitext(filename)[0]}_seg{i+1}.wav"
                output_path = os.path.join(output_folder, output_filename)

                wav.write(output_path, sample_rate, segment.astype(np.int16))

            print(f"Arquivo {filename} fragmentado em {int(num_segments)} segmentos.")

input_folder = "/data/4classes_full/D"
output_folder = "/data/4classes_15s/D"
segment_length = 15
overlap = 10

fragment_wav_files(input_folder, output_folder, segment_length, overlap)