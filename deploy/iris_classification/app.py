import numpy as np
import joblib
import pickle
from flask import Flask, request, render_template, make_response, jsonify

app = Flask(__name__)
model = joblib.load('../../model/01_iris_classification_Model.pkl')

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/verificar', methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)

    if 0 in prediction:
        prediction_text = 'The Flower Species is Setosa'
    if 1 in prediction:
        prediction_text = 'The Flower Species is Versicolor'
    if 2 in prediction:
        prediction_text = 'The Flower Species is Virginica'
    return render_template("index.html", prediction_text = prediction_text) 

    # sepal_length = request.form['sepal_length']
    # sepal_width = request.form['sepal_width']
    # petal_length = request.form['petal_length']
    # petal_width = request.form['petal_width']
    # teste = np.array([sepal_length, sepal_width, petal_length, petal_width])
    # print(teste)
    # classe = model.predict(teste)[0]
    # print(f'Classe predita: {classe}')
    # return render_template('index.html', classe=str(classe))

if __name__ == "__main__":
    app.run(debug=True)