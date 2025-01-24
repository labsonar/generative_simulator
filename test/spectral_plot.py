from glob import glob
from argparse import ArgumentParser
import numpy as np
import lps_sp.acoustical.analysis as lps
import scipy.io.wavfile as scipy_wav
import matplotlib.pyplot as plt
import lps_sp.signal as signal
import ml.visualization as vis
import os

print("oi")
def main(args):
	analysis = lps.SpectralAnalysis.SPECTROGRAM
	normalization = signal.Normalization.NORM_L2

	output_dir = f"./data/plots"
	os.makedirs(output_dir, exist_ok=True)

	specs = []
	labels = []

	class_idx = 0
	print("PPPPPPPPPPPPPP")
	print(args.dirs)
	for directory in args.dirs:


		filenames = glob(f"{directory}/*.wav")
		print('CHEGOU AQUI ')
		if not filenames:
			print(f"No .wav files found in {directory}")
			continue

		for filename in filenames:
			fs, data = scipy_wav.read(filename)
			print('lendo')
			S, f, t = analysis.apply(data=data, fs=fs)
			mean = np.mean(S, axis=1)
			norm_mean = normalization.apply(mean)

			specs.append(norm_mean)
			labels.append(class_idx)

			# Plota o espectrograma médio
			plt.plot(f, norm_mean)
			print('PLOTANDOOOOOOOooooooo')

		# Salva o gráfico de espectrograma para o diretório atual
		plt.title(f"Spectrogram Mean - {os.path.basename(directory)}")
		plt.savefig(os.path.join(output_dir, f"spectrogram_{class_idx}.png"))
		plt.close()

		class_idx += 1

	# Exporta o t-SNE com todas as classes
	vis.export_tsne(np.array(specs), np.array(labels), os.path.join(output_dir, "tsne.png"))


if __name__ == '__main__':
	parser = ArgumentParser(description='Analyze and plot spectrograms from multiple directories of wav files.')
	parser.add_argument('dirs', nargs='+', help='folders with wav files to analyze (each folder is a class)')
	main(parser.parse_args())
