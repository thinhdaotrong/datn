from data.common import TangQuangHuy
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_data_2(path_csv, percent):
    data = pd.read_csv(path_csv)
    Gender_map = {'Female': 0, 'Male': 1}
    data['Gender'] = data['Gender'].map(Gender_map)
    Dataset_map = {1: -1, 2: 1}
    data['Dataset'] = data['Dataset'].map(Dataset_map)
    X = data.iloc[:, 0:-1]
    X = X.to_numpy()
    print(X)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    print(X)
    Y = data.iloc[:, 10]
    Y = Y.to_numpy()
    X_train, X_test, y_train, y_test = tts(
        X, Y, test_size=percent, random_state=1)
    # print(X_train)

    # print(np.sum((y_train == 1).astype("uint8")))
    # print(np.sum((y_train == -1).astype("uint8")))
    # print(y_train.shape)

    return X_train, X_test, y_train, y_test
