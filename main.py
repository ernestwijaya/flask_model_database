from flask import Flask, jsonify, request
import pandas as pd
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

loaded_model = joblib.load("models/model_diabetes.pkl")
loaded_scaler = joblib.load("models/standar_scaler.pkl")
columns = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age'
]

@app.route('/')
def index():
    return jsonify({
        "meta" : {
            "status" : "Success",
            "message" : "Welcome to Diabetes API"
        },
        "data" : None
    })

@app.route('/api/predict', methods=["POST"])
def predict():
    data = request.get_json()

    X_input = pd.DataFrame([data], columns=columns)

    X_input_scaled =  loaded_scaler.transform(X_input)

    prediction = loaded_model.predict(X_input_scaled)

    prediction_probability = loaded_model.predict_proba(X_input)

    print(loaded_model.classes)

    return jsonify({
        "meta": {
            "status": "Success",
            "message": "Prediction"
        },
        "data": "Positive" if prediction.tolist()[0] == 1 else "Negative"
    })

if __name__ == '__main__':
    app.run(debug=True)


