import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier

def RemoveNegative(x_origin, y_origin, remove_length):
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

    # x_positive = x_positive[:-remove_length, :]
    x_negative = x_negative[:-remove_length, :]

    y_positive = np.asarray(
        [1.0 for i in range((x_positive.shape[0]))])
    y_negative = np.asarray(
        [-1.0 for i in range((x_negative.shape[0]))])

    x_data = np.concatenate((x_positive, x_negative), axis=0)
    y_data = np.concatenate((y_positive, y_negative), axis=0)

    # index_shuffle = np.random.permutation(x_data.shape[0])
    # x_data_new = x_data[index_shuffle, :]
    # y_data_new = y_data[index_shuffle]

    return x_data, y_data

def RemovePositive(x_origin, y_origin, remove_length):
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

    x_positive = x_positive[:-remove_length, :]
    # x_negative = x_negative[:-remove_length, :]

    y_positive = np.asarray(
        [1.0 for i in range((x_positive.shape[0]))])
    y_negative = np.asarray(
        [-1.0 for i in range((x_negative.shape[0]))])

    x_data = np.concatenate((x_positive, x_negative), axis=0)
    y_data = np.concatenate((y_positive, y_negative), axis=0)

    # index_shuffle = np.random.permutation(x_data.shape[0])
    # x_data_new = x_data[index_shuffle, :]
    # y_data_new = y_data[index_shuffle]

    return x_data, y_data



def load_data_7(path_csv, percent):
    data = pd.read_csv(path_csv)
    diag_map = {0: -1, 1: 1}
    data['Label'] = data['Label'].map(diag_map)
    # X = data.values[:, 2:-1]

    # X = data[['CommonNeighbor', 'AdamicAdar', 'JaccardCoefficient', 'PreferentialAttachment', 'ResourceAllocation', 'ShortestPath', 'CommonCountry']]
    # X = data[['PreferentialAttachment', 'ResourceAllocation', 'ShortestPath']]
    # X = data[['JaccardCoefficient', 'PreferentialAttachment', 'ResourceAllocation', 'ShortestPath']]
    # X = data[['JaccardCoefficient', 'PreferentialAttachment', 'ResourceAllocation', 'CommonCountry']]
    # X = data[['JaccardCoefficient', 'PreferentialAttachment', 'CommonCountry']]
    # X = data[['CommonNeighbor', 'AdamicAdar', 'ShortestPath']]
    # X = data[['CommonNeighbor', 'AdamicAdar', 'ResourceAllocation']]
    # X = data[['CommonNeighbor', 'AdamicAdar']]
    # X = data[['CommonNeighbor', 'AdamicAdar', 'PreferentialAttachment']]
    # X = data[['CommonNeighbor', 'AdamicAdar', 'CommonCountry']]
    # X = data[['CommonNeighbor', 'ShortestPath', 'CommonCountry']]
    # X = data[['CommonNeighbor', 'PreferentialAttachment']]
    # X = data[['CommonNeighbor', 'CommonCountry']]
    # X = data[['JaccardCoefficient', 'PreferentialAttachment', 'ResourceAllocation']]
    X = data[['CommonNeighbor', 'AdamicAdar', 'JaccardCoefficient']]
    X = X.to_numpy()
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    Y = data['Label']
    Y = Y.to_numpy()

    # X, Y = RemoveNegative(X, Y, 604)  
    # X, Y = RemovePositive(X, Y, 12)
    
    X, Y = RemovePositive(X, Y, 150)

    print(np.sum((Y == 1).astype("uint8")))
    print(np.sum((Y == -1).astype("uint8")))

    # X1_train, X1_test, y1_train, y1_test = tts(
    #     X, Y, test_size=0.96, random_state=1)

    # X1_train, y1_train = RemovePositive(X1_train, y1_train, 170)

    # X_train, X_test, y_train, y_test = tts(
    #     X1_train, y1_train, test_size=percent, random_state=1)

    X_train, X_test, y_train, y_test = tts(
        X, Y, test_size=percent, random_state=1)

    # print(np.sum((y1_train == 1).astype("uint8")))
    # print(np.sum((y1_train == -1).astype("uint8")))

    return X_train, X_test, y_train, y_test
