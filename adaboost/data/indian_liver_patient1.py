import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_data_2(path_csv, percent):
    #load data
    data = pd.read_csv(path_csv)
    Gender_map = {'Female': 0, 'Male': 1.0}
    #convert string to numberic
    data['Gender'] = data['Gender'].map(Gender_map)
    Dataset_map = {1 : -1, 2: 1}
    data['Dataset'] = data['Dataset'].map(Dataset_map)
    #Define X, y 
    y = data['Dataset']
    X = data.iloc[:, 1:10]
    #Scaler data
    X = X.to_numpy()
    y = y.to_numpy()
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X[:,2:10] = imputer.fit_transform(X[ :,2:10])
    X_train, X_test, y_train, y_test = tts(X, y, test_size = percent, random_state = 1)

    #Sclaler data
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    # Analysis the data
    pca = PCA(n_components = 6)

    X_train  = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    print(X_train.shape)
    #split data and label
    return X_train,X_test, y_train, y_test

# =============================================================================
# 
# data = pd.read_csv('E:/My project/soict/cvxopt_2/data/indian_liver_patient.csv')
# Gender_map = {'Female': 1.0, 'Male': 0.0}
# #convert string to numberic
# data['Gender'] = data['Gender'].map(Gender_map)
# Dataset_map = {1 : -1, 2: 1}
# data['Dataset'] = data['Dataset'].map(Dataset_map)
# #Define X, y 
# y = data['Dataset']
# X = data.iloc[:, 1:10]
# #Scaler data
# X_normalized = MinMaxScaler().fit_transform(X.values)
# X = pd.DataFrame(X_normalized)
# X = X.to_numpy()
# y = y.to_numpy()
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# X[:,2:10] = imputer.fit_transform(X[ :,2:10])
# =============================================================================
