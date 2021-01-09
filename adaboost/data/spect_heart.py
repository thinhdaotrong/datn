import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, MinMaxScaler

def load_data_5(path_csv, percent):
    #read data from csv
    dataset = pd.read_csv(path_csv)
    Dataset_map = {1 : -1, 0: 1}
    dataset['OVERALL_DIAGNOSIS'] = dataset['OVERALL_DIAGNOSIS'].map(Dataset_map) 
    #tranfer to feature and label
    X = dataset.values[:, 1:23]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    y = dataset.values[:, 0]
    # Splitting the dataset into trainig and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = percent, random_state = 1)
    return X_train, X_test, y_train, y_test
