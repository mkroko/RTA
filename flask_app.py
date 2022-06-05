from flask import Flask
from flask import request, jsonify
import numpy as np
import pickle
from math import log10
import model_training

app = Flask(__name__)

@app.route('/')
def hello():
    return "hello"

@app.route('/predict', methods=['GET'])
def get_prediction():
    sepal_l = float(request.args.get('sl'))
    petal_l = float(request.args.get('pl'))

    features = [sepal_l,petal_l]

    with open('mod.pkl', "rb") as picklefile:
        model = pickle.load(picklefile)

    predicted_class = int(model.predict(features))

    return jsonify(features=features, predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(host="0.0.0.0")
