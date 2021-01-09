import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier

def TangQuangHuy(x_origin, y_origin, percent):
    x_positive = []
    x_negative = []
    for index in range(y_origin.shape[0]):
        label = y_origin[index]
        x = x_origin[index, :]
        if label == 1.0:
            x_positive.append(x)
        else:
            x_negative.append(x)
    x_positive = np.asarray(x_positive)
    x_negative = np.asarray(x_negative)
    index_positive = int(x_positive.shape[0] * percent)
    index_negative = int(x_negative.shape[0] * percent)
    x_positive_train = x_positive[index_positive:, :]
    x_positive_test = x_positive[0:index_positive, :]

    y_positive_train = np.asarray(
        [1.0 for i in range((x_positive_train.shape[0]))])
    y_positive_test = np.asarray(
        [1.0 for i in range((x_positive_test.shape[0]))])
    x_negative_train = x_negative[index_negative:, :]
    x_negative_test = x_negative[0:index_negative, :]
    y_negative_train = np.asarray(
        [-1.0 for i in range((x_negative_train.shape[0]))])
    y_negative_test = np.asarray(
        [-1.0 for i in range((x_negative_test.shape[0]))])
    x_train = np.concatenate((x_positive_train, x_negative_train), axis=0)
    y_train = np.concatenate((y_positive_train, y_negative_train))
    x_test = np.concatenate((x_positive_test, x_negative_test), axis=0)
    y_test = np.concatenate((y_positive_test, y_negative_test))
    index_shuffle = np.random.permutation(x_train.shape[0])
    x_train_new = x_train[index_shuffle, :]
    y_train_new = y_train[index_shuffle]
    return x_train_new, x_test, y_train_new, y_test


def load_data(path_csv, percent):
    data = pd.read_csv(path_csv)
    diag_map = {0: -1, 1: 1}
    data['Label'] = data['Label'].map(diag_map)
    # X = data.values[:, 2:-1]

    # X = data[['CommonNeighbor', 'AdamicAdar', 'JaccardCoefficient', 'PreferentialAttachment', 'ResourceAllocation', 'ShortestPath', 'CommonCountry']]
    # X = data[['CommonNeighbor', 'AdamicAdar', 'JaccardCoefficient', 'PreferentialAttachment', 'ResourceAllocation', 'ShortestPath', 'CommonCountry']]
    X = data[['CommonNeighbor', 'AdamicAdar', 'JaccardCoefficient']]
    X = X.to_numpy()
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    Y = data['Label']
    Y = Y.to_numpy()

    print(np.sum((Y == 1).astype("uint8")))
    print(np.sum((Y == -1).astype("uint8")))

    # X_train, X_test, y_train, y_test = TangQuangHuy(X, Y, percent)
    X1_train, X1_test, y1_train, y1_test = tts(
        X, Y, test_size=0.85, random_state=1)

    X_train, X_test, y_train, y_test = tts(
        X1_train, y1_train, test_size=percent, random_state=1)

    print(np.sum((y1_train == 1).astype("uint8")))
    print(np.sum((y1_train == -1).astype("uint8")))

    return X_train, X_test, y_train, y_test
