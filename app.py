from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load model and gender encoder
model = joblib.load("model/calories_model.pkl")

gender_encoder = joblib.load("model/gender_encoder.pkl")


@app.route("/", methods=["GET"])
def home():
    return "🎯 Calories Predictor API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        # Encode gender
        gender = gender_encoder.transform([data["Gender"]])[0]

        # Prepare input features
        features = np.array([
            gender,
            data["Age"],
            data["Height"],
            data["Weight"],
            data["Duration"],
            data["Heart_Rate"],
            data["Body_Temp"]
        ]).reshape(1, -1)

        # Predict calories
        prediction = model.predict(features)[0]
        return jsonify({"predicted_calories": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    

@app.route("/health", methods=["GET"])
def health():
    return "OK", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render uses PORT env var
    app.run(host="0.0.0.0", port=port)
