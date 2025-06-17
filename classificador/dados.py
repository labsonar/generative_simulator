import os
import pandas as pd
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
 

if __name__ == '__main__':
    df1 = union_df1("/data/4classes/A" ,"/data/4classes/B","/data/4classes/C","/data/4classes/D")
    train, valid, test = dataset_split(df1)
    print(train)
    print(valid)
    print(test)
    