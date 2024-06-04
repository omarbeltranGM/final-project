from flask import Flask, request, jsonify
import joblib
import numpy as np

# Cargar el modelo y el scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return "Bienvenido a la API de predicción de condiciones del motor."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
