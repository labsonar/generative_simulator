import torch
import torch.utils.data as torch_data
import scipy.io as scipy
import numpy as np
import pandas as pd
import os
import tqdm
from fast_ml.model_development import train_valid_test_split


def union_df1(path_A, path_B, path_C, path_D):
    dic = {"Audio": [], "Class": []}
    paths = [path_A, path_B, path_C, path_D]
    for path_class in paths:
        for arquivo in os.listdir(path_class):
            caminho_arquivo = os.path.join(path_class, arquivo)
            if os.path.isfile(caminho_arquivo):
                dic["Audio"].append(caminho_arquivo)
                dic["Class"].append(path_class[-1])
        df = pd.DataFrame(dic)

    return df


def dataset_split(df):
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(df1,target = "Class", train_size=0.6, valid_size=0.2, test_size=0.2)
    train = pd.concat([X_train, y_train], axis=1)
    valid =  pd.concat([X_valid, y_valid], axis=1)
    test = pd.concat([X_test, y_test], axis=1)


    return train, valid, test



def getitem(data,idx):
    
        sample = data.iloc[idx]
        wav_file = sample['Audio']
        classification = sample['Class']

        _, audio_data = scipy.wavfile.read(wav_file)

        audio_data = audio_data / np.max(np.abs(audio_data), axis=0)

        audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
        classification_tensor = torch.tensor(classification, dtype=torch.long)

        return audio_tensor, classification_tensor

class Audio(torch_data.Dataset):


    def __init__(self,data):
        self.data = data


    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return getitem(self.data,idx)
    

if __name__ == "__main__":
    df1 = union_df1("/content/sample_data/A" ,"/content/sample_data/B","/content/sample_data/C","/content/sample_data/D")
    train, valid, test = dataset_split(df1)
    print(train)
    print(valid)
    print(test)
