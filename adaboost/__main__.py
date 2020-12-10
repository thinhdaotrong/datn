import numpy as np
from data.Vertebral_column import load_data_1
# import trainning_of_adaboost as toa
import svm
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import trainning_of_adaboost as toa

# test SVM


# model = SVC(kernel = 'linear', C = 100)
# model.fit(X_train,y_train)
# test_svmpred = model.predict(X_test)
# test_accuracy = classification_report(y_test,test_svmpred)
# print(test_accuracy)

X_train, X_test, y_train, y_test = load_data_1("data/enable-data/Vertebral_column.csv", 0.3)

w, b = svm.fit(X_train, y_train, C = 100)
test_pred = np.sign(X_test.dot(w)+b)
print(classification_report(y_test,test_pred))

w, b, a = toa.fit(X_train, y_train, M=50, C=100, instance_categorization=False)
test_pred = toa.predict(X_test, w, b, a, M=50)
print(classification_report(y_test,test_pred))

acc = np.sum((test_pred == y_test).astype("uint8")) / test_pred.shape[0]

result = classification_report(y_test, test_pred)
list_new = [f for f in result.split("\n") if f != ""]
new_arr = []
for line in list_new:
    line_new = [f for f in line.split(" ") if f != ""]
    if "avg" in line_new:
        line_new = [line_new[0] + " " + line_new[1], line_new[2], line_new[3], line_new[4], line_new[5]]
    new_arr.append(line_new)

dict_result = {new_arr[0][0] : {}, new_arr[0][1] : {}, new_arr[0][2] : {}, new_arr[0][3] : {}}

for arr in new_arr[1:]:
    count = 0
    for key in dict_result.keys():
        dict_result[key][arr[0]] = arr[count + 1]
        count += 1



# from sklearn.ensemble import AdaBoostClassifier
# clf = AdaBoostClassifier(SVC(kernel = 'linear', C = 100),n_estimators=50, algorithm='SAMME')
# clf.fit(X_train, y_train)
# test_adapred = clf.predict(X_test)
# print(classification_report(y_test,test_adapred))
