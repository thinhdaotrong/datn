import numpy as np
from data.Vertebral_column import load_data_1
from data.common import load_data
# import trainning_of_adaboost as toa
import svm
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
import trainning_of_adaboost as toa
from sklearn.ensemble import AdaBoostClassifier


# X_train, X_test, y_train, y_test = load_data_1(
    # "data/enable-data/Vertebral_column.csv", 0.3)

X_train, X_test, y_train, y_test = load_data(
    "data/enable-data/co_author_08_new.csv", 0.3)

w, b = svm.fit(X_train, y_train, C=800)
test_pred = np.sign(X_test.dot(w)+b)
print(classification_report(y_test, test_pred))
print(roc_auc_score(y_test, test_pred))

w, b, a = toa.fit(X_train, y_train, M=10, C=800, instance_categorization=False)
test_pred = toa.predict(X_test, w, b, a, M=10)
print(classification_report(y_test, test_pred))
print(roc_auc_score(y_test, test_pred))

w, b, a = toa.fit(X_train, y_train, M=10, C=800, instance_categorization=True)
test_pred = toa.predict(X_test, w, b, a, M=10)
print(classification_report(y_test, test_pred))
print(roc_auc_score(y_test, test_pred))


# test SVM
# model = SVC(kernel='linear', C=100)
# model.fit(X_train, y_train)
# test_svmpred = model.predict(X_test)
# test_accuracy = classification_report(y_test, test_svmpred)
# print(test_accuracy)


clf = AdaBoostClassifier(SVC(kernel='linear', C=800),
                          n_estimators=10, algorithm='SAMME')
clf.fit(X_train, y_train)
test_adapred = clf.predict(X_test)
print(classification_report(y_test, test_adapred))
print(roc_auc_score(y_test, test_adapred))
