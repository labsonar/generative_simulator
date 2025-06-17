import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, convolve
import librosa
from datetime import datetime

class LOFAR_Processor:
    def __init__(self, debug_mode=False, save_debug_plots=False):
        self.debug_mode = debug_mode
        self.save_debug_plots = save_debug_plots



############################################## Caminho outputs

        self.debug_output_dir = "./lofar_output/D"
        os.makedirs(self.debug_output_dir, exist_ok=True)
        
    def _debug_plot(self, data, title, filename, xlabel=None, ylabel=None, figsize=(12, 6)):
        if self.debug_mode:
            plt.figure(figsize=figsize)
            if data.ndim == 1:
                plt.plot(data)
            else:
                plt.imshow(data, aspect='auto', origin='lower', cmap='jet')
                plt.colorbar()
            plt.title(title)
            if xlabel: plt.xlabel(xlabel)
            if ylabel: plt.ylabel(ylabel)
            if self.save_debug_plots:
                plt.savefig(f"{self.debug_output_dir}/{filename}.png")
            plt.show()
    
    def tpsw(self, signal, npts=None, n=None, p=None, a=None, debug_name=""):
        """Two-Pass Split Window com debug integrado"""
        if self.debug_mode:
            print(f"\n[TPSW {debug_name}] Input shape: {signal.shape}")
            self._debug_plot(signal, f"Sinal Original - {debug_name}", 
                           f"1_tpsw_input_{debug_name}", "Amostras", "Amplitude")

        x = np.copy(signal)
        if x.ndim == 1:
            x = x[:, np.newaxis]
        if npts is None:
            npts = x.shape[0]
        if n is None:
            n = int(round(npts*.04/2.0+1))
        if p is None:
            p = int(round(n / 8.0 + 1))
        if a is None:
            a = 2.0

        # Janela de suavização
        h = np.concatenate((np.ones((n-p+1)), np.zeros(2*p-1), np.ones((n-p+1))), axis=None) if p > 0 else np.ones((1, 2*n+1))
        h /= np.linalg.norm(h, 1)

        def apply_on_spectre(xs):
            return convolve(h, xs, mode='full')

        # Primeira passagem
        mx = np.apply_along_axis(apply_on_spectre, arr=x, axis=0)
        ix = int(np.floor((h.shape[0] + 1)/2.0) )
        mx = mx[ix-1:npts+ix-1]
        
        # Correção de bordas
        ixp = ix - p
        mult = 2 * ixp / np.concatenate([np.ones(p-1)*ixp, range(ixp, 2*ixp+1)], axis=0)[:, np.newaxis]
        mx[:ix, :] = mx[:ix, :] * np.matmul(mult, np.ones((1, x.shape[1])))
        mx[npts-ix:npts, :] = mx[npts-ix:npts, :] * np.matmul(np.flipud(mult), np.ones((1, x.shape[1])))

        # Segunda passagem
        indl = (x - a*mx) > 0
        x = np.where(indl, mx, x)
        mx = np.apply_along_axis(apply_on_spectre, arr=x, axis=0)
        mx = mx[ix-1:npts+ix-1, :]
        
        # Correção final
        mx[:ix, :] = mx[:ix, :] * np.matmul(mult, np.ones((1, x.shape[1])))
        mx[npts-ix:npts, :] = mx[npts-ix:npts, :] * np.matmul(np.flipud(mult), np.ones((1, x.shape[1])))

        if signal.ndim == 1:
            mx = mx[:, 0]

        if self.debug_mode:
            self._debug_plot(mx, f"Resultado TPSW - {debug_name}", 
                           f"2_tpsw_output_{debug_name}", "Amostras", "Amplitude")
            print(f"[TPSW {debug_name}] Output range: {np.min(mx):.2f} to {np.max(mx):.2f}")

        return mx

    def lofar(self, data, fs, n_pts_fft=1024, n_overlap=0, spectrum_bins_left=None):
        """Processamento LOFAR completo com debug integrado"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.debug_mode:
            print("\n" + "="*50)
            print(f"LOFAR Processing - Debug Session {timestamp}")
            print("="*50)
            print(f"[Input] Samples: {len(data)}, SR: {fs}Hz, Duration: {len(data)/fs:.2f}s")
            self._debug_plot(data, "Sinal de Entrada", "0_input_waveform", "Tempo (s)", "Amplitude")

        # Espectrograma
        freq, time, power = spectrogram(data,
                                      window='hann',
                                      nperseg=n_pts_fft,
                                      noverlap=n_overlap,
                                      nfft=n_pts_fft,
                                      fs=fs,
                                      detrend=False,
                                      axis=0,
                                      scaling='spectrum',
                                      mode='magnitude')

        if self.debug_mode:
            print(f"\n[Espectrograma] Shape: {power.shape}, Freq: {freq[0]:.1f}-{freq[-1]:.1f}Hz")
            self._debug_plot(10*np.log10(power+1e-10), "Espectrograma Bruto (dB)", 
                           "1_raw_spectrogram", "Tempo (s)", "Frequência (Hz)")

        # Processamento TPSW
        power = np.absolute(power)
        power_tpsw = self.tpsw(power, debug_name="Espectrograma")
        
        # Pós-processamento
        power_log = np.log10(power_tpsw + 1e-10)  # Evita log(0)
        power_log[power_log < -0.2] = 0

        if spectrum_bins_left is None:
            spectrum_bins_left = int(power.shape[0]*0.8)
        power_final = power_log[:spectrum_bins_left, :]
        freq_final = freq[:spectrum_bins_left]

        if self.debug_mode:
            print("\n[Resultado Final]")
            print(f"Shape: {power_final.shape}, Freq: {freq_final[0]:.1f}-{freq_final[-1]:.1f}Hz")
            self._debug_plot(np.transpose(power_final), "Espectrograma LOFAR Final", 
                           "2_final_lofar", "Tempo (s)", "Frequência (Hz)")

        return np.transpose(power_final), freq_final, time

    def process_audio_folder(self, folder_path, output_dir=None, file_extensions=['.wav', '.mp3', '.flac']):
        """
        Processa todos os arquivos de áudio em uma pasta
        
        Args:
            folder_path (str): Caminho para a pasta com os arquivos de áudio
            output_dir (str): Pasta para salvar os resultados (None para usar pasta padrão)
            file_extensions (list): Extensões de arquivo a serem processadas
        """
        if output_dir is None:
            output_dir = os.path.join(self.debug_output_dir, "batch_processing")
            os.makedirs(output_dir, exist_ok=True)
        
        # Lista todos os arquivos na pasta
        audio_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in file_extensions):
                    audio_files.append(os.path.join(root, file))
        
        if not audio_files:
            print(f"Nenhum arquivo de áudio encontrado em {folder_path} com extensões {file_extensions}")
            return
        
        print(f"\nIniciando processamento de {len(audio_files)} arquivos de áudio...")
        
        results = []
        for i, audio_file in enumerate(audio_files, 1):
            try:
                print(f"\n[{i}/{len(audio_files)}] Processando: {os.path.basename(audio_file)}")
                
                # Carrega o áudio
                samples, sr = librosa.load(audio_file, sr=None)
                
                # Processa com LOFAR
                lofar_result, freq, time = self.lofar(
                    data=samples,
                    fs=sr,
                    n_pts_fft=1024,
                    n_overlap=128,
                    spectrum_bins_left=400
                )
                
                # Salva os resultados
                base_name = os.path.splitext(os.path.basename(audio_file))[0]
                np.save(os.path.join(output_dir, f"{base_name}_lofar.npy"), lofar_result)
                
                # Gera plot final e salva
                plt.figure(figsize=(12, 6))
                plt.imshow(lofar_result, aspect='auto', origin='lower', cmap='jet')
                plt.colorbar()
                plt.title(f"LOFAR Result - {base_name}")
                plt.xlabel("Tempo (s)")
                plt.ylabel("Frequência (Hz)")
                plt.savefig(os.path.join(output_dir, f"{base_name}_lofar_plot.png"))
                plt.close()
                
                results.append({
                    'filename': audio_file,
                    'result': lofar_result,
                    'frequencies': freq,
                    'times': time
                })
                
            except Exception as e:
                print(f"Erro ao processar {audio_file}: {str(e)}")
                continue
        
        print("\nProcessamento concluído!")
        return results

# Exemplo de uso
if __name__ == "__main__":
    # Configuração
    processor = LOFAR_Processor(debug_mode=True, save_debug_plots=True)
    
    # Opção 1: Processar um arquivo específico (como antes)
    # samples, sr = librosa.load('./audio_example.wav', sr=None)
    # lofar_result, freq, time = processor.lofar(data=samples, fs=sr)
    

############################################### Caminho inputs

    # Opção 2: Processar todos os áudios de uma pasta
    audio_folder = "/home/leticia.luz/Documents/generative_simulator/test/4classes_CPA_15s/D"  # Substitua pelo caminho da sua pasta
    processor.process_audio_folder(audio_folder)



    #TODO 
    # Perguntar pro natanael o funcionamento da funcao tpsw
    # Perguntar o sentido de os graficos serem apresentados para a classe inteira
    # Perguntar se faz sentido os gr[aficos de LOFAR apresentarem coloracao completamente verde
