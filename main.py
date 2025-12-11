import joblib
import pandas as pd
import tensorflow as tf
import numpy as np
import io

from flask_cors import CORS
from flask import Flask, jsonify, request

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input



app = Flask(__name__)
CORS(app)

loaded_model = joblib.load("models/model_diabetes.pkl")
loaded_scaler = joblib.load("models/standar_scaler.pkl")
loaded_model_rps = tf.keras.models.load_model("models/RPS.keras")

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

rps_class_names = ['Paper', 'Rock', 'Scissors']

# @app.route('/')
# def index():
#     return jsonify({
#         "meta" : {
#             "status" : "Success",
#             "message" : "Welcome to Diabetes API"
#         },
#         "data" : None
#     })

@app.route('/api/predict', methods=["POST"])
def predict():
    
    data = request.get_json()

    X_input = pd.DataFrame([data], columns=columns)

    X_input_scaled =  loaded_scaler.transform(X_input)

    prediction = loaded_model.predict(X_input_scaled)

    return jsonify({
        "meta": {
            "status": "Success",
            "message": "Prediction"
        },
        "data": "Positive" if prediction.tolist()[0] == 1 else "Negative"
    })


@app.route("/api/predict-rps", methods=["POST"])
def predict_rps():
    #1. Validasi request file 
    if 'file' not in request.files: 
        return jsonify({
            "meta" : {
                "status" : 400,
                "message" : "No file in the request"
            }
        })
        
    #2. Membaca file gambar 
    file = request.files['file']
    img_bytes = io.BytesIO(file.read())
    
    #3. Preprocessing gambar 
    img = image.load_img(img_bytes, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_batch = np.expand_dims(img_array, axis=0)

    #4. Proses Prediksi (Inferensi)
    prediction = loaded_model_rps.predict(img_batch)
    predicted_class_index = np.argmax(prediction)
    
    #5. Mengembalikan respon
    return jsonify({
        "meta" : {
            "status" : 200,
            "message" : "Prediction successful"
        }, 
        "data" : {
            "prediction" : rps_class_names[predicted_class_index], 
            "probability" : f"{np.max(prediction) * 100:.2f}%"
        }
    })

if __name__ == '__main__':
    app.run(debug=True)
