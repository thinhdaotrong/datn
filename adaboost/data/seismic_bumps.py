import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


def load_data_4(path_csv, percent):
    dataset = pd.read_csv(path_csv)
    Dataset_map = {1: 1, 0: -1}
    dataset['class'] = dataset['class'].map(Dataset_map)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 18].values
    labelencoder_X = LabelEncoder()
    X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
    onehotencoder_X = OneHotEncoder(handle_unknown='ignore')
    onehotencoder_X.fit_transform(X).toarray()
    X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
    onehotencoder_X = OneHotEncoder(handle_unknown='ignore')
    onehotencoder_X.fit_transform(X).toarray()
    X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
    onehotencoder_X = OneHotEncoder(handle_unknown='ignore')
    onehotencoder_X.fit_transform(X).toarray()
    X[:, 7] = labelencoder_X.fit_transform(X[:, 7])
    onehotencoder_X = OneHotEncoder(handle_unknown='ignore')
    onehotencoder_X.fit_transform(X).toarray()
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=percent, random_state=1)

    # sc_X = StandardScaler()
    # X_train = sc_X.fit_transform(X_train)
    # X_test = sc_X.transform(X_test)
    # pca = PCA(n_components=11)
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.transform(X_test)
    return X_train, X_test, y_train, y_test
