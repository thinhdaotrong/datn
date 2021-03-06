import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize, MinMaxScaler


def load_data_1(path_csv, percent):
    data = pd.read_csv(path_csv)
    diag_map = {'Abnormal': 1, 'Normal': -1}
    data['Label class'] = data['Label class'].map(diag_map)
    X = data.values[:, 0:-1]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    Y = data.values[:, 6]
    X_train, X_test, y_train, y_test = tts(
        X, Y, test_size=percent, random_state=42)

    # print(np.sum((y_train == 1).astype("uint8")))
    # print(np.sum((y_train == -1).astype("uint8")))
    # print(y_train.shape)

    return X_train, X_test, y_train, y_test
