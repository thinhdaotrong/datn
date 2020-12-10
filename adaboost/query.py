from random import choice
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import os, socket
import uuid
from data.Vertebral_column import load_data_1
from data.indian_liver_patient import load_data_2
from data.churn import load_data_3
from sklearn.metrics import classification_report
import trainning_of_adaboost as toa
from methods import get_eval
from werkzeug.utils import secure_filename
import numpy as np
import svm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP


desktop_agents = [
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/602.2.14 (KHTML, like Gecko) Version/10.0.1 Safari/602.2.14',
    'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:50.0) Gecko/20100101 Firefox/50.0']


def random_headers():
    return {'User-Agent': choice(desktop_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'}


def create_response(data, error_code, error_message):
    response = {
        "data": data,
        "errorCode": error_code,
        "errorMessage": error_message,
    }
    return response


def jsonify_str(output_list):
    with app.app_context():
        with app.test_request_context():
            result = jsonify(output_list)
    return result


app = Flask(__name__)
CORS(app, support_credentials=True)

@app.route("/query", methods=['POST'])
@cross_origin(supports_credentials=True)
def query():
    m = request.args.get('m', default=50, type=int)
    c = request.args.get('c', default=100, type=int)
    instance_categorization = request.args.get('instance_categorization', default='true', type=str)
    instance_categorization = instance_categorization.lower()
    if instance_categorization == 'false':
        instance_categorization = False
    else:
        instance_categorization = True
    percent_test = request.args.get('percent_test', default=0.1, type=float)
    print(percent_test)
    if request.method == "POST":
        try:
            csv_data = request.files["file"]
            print(csv_data.filename)
            filename = csv_data.filename
            name_save_csv = os.path.join("csv", uuid.uuid4().hex + ".csv")
            csv_data.save(name_save_csv)
        except Exception as ex:
            print(ex)
            return jsonify_str(create_response("", "", ""))
    else:
        return jsonify_str(create_response("", "", ""))
    if filename == 'Vertebral_column.csv':
        X_train, X_test, y_train, y_test = load_data_1(name_save_csv, percent_test)
    if filename == 'indian_liver_patient.csv':
        X_train, X_test, y_train, y_test = load_data_2(name_save_csv, percent_test)
    if filename == 'churn.csv':
        X_train, X_test, y_train, y_test = load_data_3(name_save_csv, percent_test)
    os.remove(name_save_csv)
    w, b, a = toa.fit(X_train, y_train, M=m, C=c, instance_categorization=instance_categorization)
    test_pred = toa.predict(X_test, w, b, a, M=m)
    result = get_eval(test_pred, y_test)
    
    return jsonify_str(create_response(result, "", ""))

@app.route("/query1", methods=['POST'])
@cross_origin(supports_credentials=True)
def query1():
    m = request.args.get('m', default=50, type=int)
    c = request.args.get('c', default=100, type=int)
    instance_categorization = request.args.get('instance_categorization', default='true', type=str)
    instance_categorization = instance_categorization.lower()
    if instance_categorization == 'false':
        instance_categorization = False
    else:
        instance_categorization = True
    percent_test = request.args.get('percent_test', default=0.1, type=float)
    print(percent_test)
    if request.method == "POST":
        try:
            csv_data = request.files["file"]
            print(csv_data.filename)
            filename = csv_data.filename
            name_save_csv = os.path.join("csv", uuid.uuid4().hex + ".csv")
            csv_data.save(name_save_csv)
        except Exception as ex:
            print(ex)
            return jsonify_str(create_response("", "", ""))
    else:
        return jsonify_str(create_response("", "", ""))
    if filename == 'Vertebral_column.csv':
        X_train, X_test, y_train, y_test = load_data_1(name_save_csv, percent_test)
    if filename == 'indian_liver_patient.csv':
        X_train, X_test, y_train, y_test = load_data_2(name_save_csv, percent_test)
    if filename == 'churn.csv':
        X_train, X_test, y_train, y_test = load_data_3(name_save_csv, percent_test)
    os.remove(name_save_csv)
    w, b = svm.fit(X_train, y_train, C = 100)
    test_pred = np.sign(X_test.dot(w)+b)
    result = get_eval(test_pred, y_test)
    
    return jsonify_str(create_response(result, "", ""))



app.run("localhost", 1702, threaded=False, debug=True)
# app.run(get_ip(), 1702, threaded=False, debug=False)
