from flask import Flask, jsonify
from predictor.predict import load_data,process_data,normalize_data
import numpy as np
import requests

#GLOBAL VARIABLES

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World test 4!'

@app.route('/load')
def load():
    toPredict = np.loadtxt(".//test_data//predict_from_server.csv", delimiter=",")
    toPredict_list = toPredict.tolist()
    # df = load_data(data_path)
    # targets_ohe = process_data(df)
    # x_pred = normalize_data(df)

    payload={
        "instances":[{'dense_input':toPredict_list}]
    }
    #response = requests.post('http://localhost:8501/v1/models/mercury:predict', json=payload)
    response = requests.post('http://tf-serving-server:8501/v1/models/mercury:predict',json = payload)
    predictions = response.json()['predictions'][0]
    return jsonify(predictions)

if __name__ == '__main__':
    app.run('0.0.0.0')
