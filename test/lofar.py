import os
import librosa
import IPython
import numpy as np
import matplotlib as plt

if not os.path.exists('./data/audio_example_1.wav'):
  !wget https://github.com/natmourajr/data/raw/main/audio_example_1.wav

samples , sampling_rate = librosa.load("./data/audio_example_1.wav")
print("Sampling rate is %i Hz"%(sampling_rate))
print("Number of Samples in this file is %i"%(samples.shape[0]))
print("Record time is %1.3f seconds"%(float(samples.shape[0]/sampling_rate)))

IPython.display.Audio(samples, rate=sampling_rate)
# Sampling rate is 22050 Hz
# Number of Samples in this file is 1323000
# Record time is 60.000 seconds
n_bins_left = 400
lofar_signal, freq, time = lofar(data=samples,fs=sampling_rate, n_pts_fft=1024,
                                 n_overlap=0.0, spectrum_bins_left=n_bins_left)
plt.imshow(lofar_signal,cmap="jet",
           extent=[1, n_bins_left, lofar_signal.shape[0],1],
           aspect="auto")
plt.xticks(np.linspace(0,n_bins_left,9),rotation=45)
cbar = plt.colorbar()

cbar.ax.set_ylabel('dB',fontweight='bold') ;



def lofar(data, fs, n_pts_fft=1024, n_overlap=0,
          spectrum_bins_left=None, tpsw_args):

    if not isinstance(data, np.ndarray):
        raise NotImplementedError

    freq, time, power = spectrogram(data,
                                    window=('hann'),
                                    nperseg=n_pts_fft,
                                    noverlap=n_overlap,
                                    nfft=n_pts_fft,
                                    fs=fs,
                                    detrend=False,
                                    axis=0,
                                    scaling='spectrum',
                                    mode='magnitude')
    # For stereo, without further changes, the genreated spectrogram has shape (freq, channel, time)
    if power.ndim == 3:  # temporary fix for stereo audio.
        power = power.mean(axis=1)
        power = power.squeeze()

    power = np.absolute(power)
    power = power / tpsw(power)  # , tpsw_args)
    power = np.log10(power)
    power[power < -0.2] = 0

    if spectrum_bins_left is None:
        spectrum_bins_left = power.shape[0]*0.8
    power = power[:spectrum_bins_left, :]
    freq = freq[:spectrum_bins_left]

    return np.transpose(power), freq, time

