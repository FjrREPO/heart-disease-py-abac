from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import logging
from datetime import datetime

logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
CORS(app)

try:
    model = tf.keras.models.load_model("./heart_disease_model.h5")
    dummy_input = np.random.random((1, 13, 1))
    model.predict(dummy_input)
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Failed to load model: {str(e)}")
    raise

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        logging.info(f"Received prediction request: {data}")
        
        values = np.array([[
            int(data['age']),
            int(data['sex']), 
            int(data['cp']),
            int(data['trestbps']),
            int(data['chol']),
            int(data['fbs']),
            int(data['restecg']), 
            int(data['thalach']),
            int(data['exang']),
            int(data['oldpeak']),
            int(data['slope']),
            int(data['ca']),
            int(data['thal'])
        ]]).reshape(1, 13, 1)
        
        prediction = model.predict(values)
        probability = float(prediction[0][0])
        
        pos_prob = round(probability, 4)
        neg_prob = round(1 - probability, 4)
        
        if pos_prob + neg_prob != 1:
            pos_prob = round(pos_prob, 4)
            neg_prob = 1 - pos_prob
        
        return jsonify({
            "success": True,
            "prediction": int(probability >= 0.5),
            "probability": {
                "negative": neg_prob,
                "positive": pos_prob
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

if __name__ == "__main__":
    app.run(debug=True)