from flask import Flask, jsonify
from predictor.predict import load_data,process_data,normalize_data
import numpy as np
import requests

#GLOBAL VARIABLES
DEV_HOST = 'http://localhost:8501/v1/models/mercury:predict'
PROD_HOST = 'http://tf-serving-server:8501/v1/models/mercury:predict'

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World test 4!'

@app.route('/load')
def load():
    toPredict = np.loadtxt(".//test_data//predict_from_server.csv", delimiter=",")
    toPredict_list = toPredict.tolist()


    payload={
        "instances":[{'dense_input':toPredict_list}]
    }
    response = requests.post(PROD_HOST, json=payload)
    predictions = response.json()['predictions'][0]
    return jsonify(predictions)

if __name__ == '__main__':
    app.run('0.0.0.0')
