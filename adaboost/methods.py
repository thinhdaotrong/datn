import numpy as np
from sklearn.metrics import classification_report

def intinitialization_weight_adjustment(N):
    '''
    N la so diem du lieu cua X
    '''
    return np.ones(N) / N


def intinitialization_instance_categorization(N):
    '''
    Input: N la so diem du lieu cua X
    Output: Vecto ban dau cua C trong bai bao 2016
    '''
    return np.ones(N)


def predict_svm(X, w, b):
    '''
    Input: tap data du lieu dau vao, X shaped (N, d)
        w, b la bo model phan lop, w shaped (d, ), b shaped ()
    Output: la gia tri predict cua lan SVM thu i
    '''
    return np.sign(X.dot(w) + b)


def find_true_false_index(y, pred):
    '''
    Tim gia tri dung sai cua moi lan phan loai
    Input: y la gia tri label cua data
        pred la gia tri sau khi phan lop
    Outpit index cua phan tu dung va sai
    '''
    true_index = np.where(y == pred)[0]
    false_index = np.where(y != pred)[0]
    return true_index, false_index


def confident(W, false_index):
    '''
    Input: 
        W: weight adjusntment, shaped (N, 1)
        false_index: wrong predict, length <= N
    Output:
        confident of model shaped ()
    '''
    eps = np.sum(W[false_index]) / np.sum(W)
    return 1 / 2 * np.log((1 - eps) / eps)


def update_weight_adjustment(W, alpha, true_index, false_index):
    '''
    Input:
        W: i-th weight adjustment
        alpha: ith_confident of Adaboost
        true_index, false_index: 
    Output:
        W (i+1)-th weight adjustment 
    '''
    W[true_index] = W[true_index] * np.exp(-1 * alpha)
    W[false_index] = W[false_index] * np.exp(alpha)
    return W / np.sum(W)


def update_instance_categorization(X, y, w, b):
    # Obtain categorization_weight
    C = np.ones(X.shape[0])
    A = 1 - y * (X.dot(w) + b)
    # BSV_weight
    num_of_BSV = np.where((A > 0) & (A < 2))[0].shape[0]
    pos_BSV = np.where((A > 0) & (A < 2) & (y == 1))[0]
    num_of_pos_BSV = pos_BSV.shape[0]
    neg_BSV = np.where((A > 0) & (A < 2) & (y == -1))[0]
    num_of_neg_BSV = neg_BSV.shape[0]
    if (num_of_pos_BSV != 0):
        C[pos_BSV] = num_of_BSV / (2 * (num_of_pos_BSV))
    if (num_of_neg_BSV != 0):
        C[neg_BSV] = num_of_BSV / (2 * (num_of_neg_BSV))
    # SV weight
    num_of_SV = np.where(A == 0)[0].shape[0]
    if (num_of_SV != 0):
        pos_SV = np.where((A == 0) & (y == 1))[0]
        num_of_pos_SV = pos_SV.shape[0]
        if (num_of_pos_SV != 0):
            C[pos_SV] = num_of_SV / (2 * num_of_pos_SV)
        neg_SV = np.where((A == 0) & (y == -1))[0]
        num_of_neg_SV = neg_SV.shape[0]
        if (num_of_neg_SV != 0):
            C[neg_SV] = num_of_SV / (2 * num_of_neg_SV)
    # positive noise
    positive_noise = np.where(((A <= 2) & (y == 1)))[0]
    num_of_positive_noise = positive_noise.shape[0]
    num_of_positive = np.where(y == 1)[0].shape[0]
    C[positive_noise] = np.exp(num_of_positive_noise / num_of_positive)

    return C


def get_eval(test_pred, y_test):
    acc = np.sum((test_pred == y_test).astype("uint8")) / test_pred.shape[0]
    result = classification_report(y_test, test_pred)
    print(result)
    list_new = [f for f in result.split("\n") if f != ""]
    new_arr = []
    for line in list_new:
        line_new = [f for f in line.split(" ") if f != ""]
        if "avg" in line_new:
            line_new = [line_new[0] + " " + line_new[1], line_new[2], line_new[3], line_new[4], line_new[5]]
        new_arr.append(line_new)
    # print(new_arr)
    dict_result = {new_arr[0][0]: {}, new_arr[0][1]: {}, new_arr[0][2]: {}, new_arr[0][3]: {}}
    # print(new_arr)
    for index, arr in enumerate(new_arr[1:]):
        count = 0
        if index == 2:
            continue
        # print(arr)
        for key in dict_result.keys():
            dict_result[key][arr[0]] = float(arr[count + 1])
            count += 1
    dict_result["accuracy"] = acc
    return dict_result