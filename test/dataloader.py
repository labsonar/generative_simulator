import os
import numpy as np
import pandas as pd
import scipy.io as scipy
import torch
from torch.utils.data import Dataset, DataLoader
from fast_ml.model_development import train_valid_test_split

# Junta os arquivos em um único DataFrame
def union_df1(path_A, path_B, path_C, path_D):
    dic = {"Audio": [], "Class": []}
    paths = [path_A, path_B, path_C, path_D]
    for path_class in paths:
        for arquivo in os.listdir(path_class):
            caminho_arquivo = os.path.join(path_class, arquivo)
            if os.path.isfile(caminho_arquivo):
                dic["Audio"].append(caminho_arquivo)
                dic["Class"].append(path_class[-1])  # Pega 'A', 'B', 'C' ou 'D'
    df = pd.DataFrame(dic)
    return df

# Divide em treino, validação e teste
def dataset_split(df):
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(
        df, target="Class", train_size=0.6, valid_size=0.2, test_size=0.2
    )
    train = pd.concat([X_train, y_train], axis=1)
    valid = pd.concat([X_valid, y_valid], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    return train, valid, test

# Lê o áudio e retorna os tensores
def getitem(data, idx):
    sample = data.iloc[idx]
    wav_file = sample['Audio']
    classification = sample['Class']
    _, audio_data = scipy.wavfile.read(wav_file)
    audio_data = audio_data / np.max(np.abs(audio_data), axis=0)
    audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
    classification_tensor = torch.tensor(classification, dtype=torch.long)
    return audio_tensor, classification_tensor

# Dataset personalizado
class AudioDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return getitem(self.data, idx)

# Cria os dataloaders
def get_dataloaders(train, valid, test, batch_size=1):
    return {
        "train": DataLoader(AudioDataset(train), batch_size=batch_size, shuffle=True),
        "valid": DataLoader(AudioDataset(valid), batch_size=batch_size, shuffle=False),
        "test": DataLoader(AudioDataset(test), batch_size=batch_size, shuffle=False),
    }

# --- Execução principal ---
if __name__ == "__main__":
    df = union_df1("/data/4classes_15s/A", "/data/4classes_15s/B", "/data/4classes_15s/C", "/data/4classes_15s/D")

    # Mapeia classes para números
    df["Class"] = df["Class"].map({'A': 0, 'B': 1, 'C': 2, 'D': 3})

    train_df, valid_df, test_df = dataset_split(df)
    test_df = test_df[:38]  # Filtro extra, se necessário

    dataloaders = get_dataloaders(train_df, valid_df, test_df)

    # Exemplo de iteração
    for batch in dataloaders["test"]:
        audio, label = batch
        print("Shape do áudio:", audio.shape)
        print("Classe:", label)
        break