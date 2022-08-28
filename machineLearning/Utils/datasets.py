import numpy as np
import pandas as pd
import random
import os

def load_data(file_path):
    if file_path.endswith("csv"):
        data = pd.read_csv(file_path)
    elif file_path.endswith("xlsx"):
        data = pd.read_excel(file_path)
    elif file_path.endswith("feather"):
        data = pd.read_feather(file_path)
    elif file_path.endswith("json"):
        data = pd.read_json(file_path)
    elif file_path.endswith("parquet"):
        data = pd.read_parquet(file_path)
    elif file_path.endswith("hdf"):
        data = pd.read_hdf(file_path)
    elif file_path.endswith("txt"):
        data = pd.read_table(file_path, delimiter=",")
    else:
        print("Your file format is not supported !")
        raise TypeError;

    return data

def train_test_split(df_X, df_y, ratio=0.8, seed=42):
    X_train = df_X.sample(frac=ratio, replace=False, random_state=seed)
    X_test = df_X.drop(X_train.index, axis=0)
    y_train = df_y.sample(frac=ratio, replace=False, random_state=seed)
    y_test = df_y.drop(y_train.index, axis=0)
    print("training samples are {:}, testing samples are {:}.".format(len(X_train), len(X_test)))

    return X_train, X_test, y_train, y_test

def show_data(df, rows=5):
    print("----------------dataset is like this:-----------------")
    print(df.sample(n=rows))
    print("The shape is: ", df.shape)
    print("-----------------------------------------------------")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def Normalize(x):
    avg = np.average(x, axis=0)
    std = np.std(x, axis=0, ddof=1)
    x = (x - avg) / std

    return x, avg, std

def MinMaxScale(x):
    min = np.min(x, axis=0)
    max = np.max(x, axis=0)
    x = (x - min) / (max - min)

    return x, min, max
