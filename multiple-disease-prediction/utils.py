import numpy as np
import pickle
import tensorflow as tf

# Load models
diabetes_model = tf.keras.models.load_model("models/diabetes_model.h5")
heart_model = tf.keras.models.load_model("models/heart_model.h5")
kidney_model = tf.keras.models.load_model("models/kidney_model.h5")

# Load scalers
with open("models/diabetes_scaler.pkl", "rb") as f:
    diabetes_scaler = pickle.load(f)
with open("models/heart_scaler.pkl", "rb") as f:
    heart_scaler = pickle.load(f)
with open("models/kidney_scaler.pkl", "rb") as f:
    kidney_scaler = pickle.load(f)


def predict_diseases(data):
    """
    data: dict from HTML form { "age": "...", "bp": "...", "glucose": "...", ... }
    returns: dict { "diabetes": (prediction, prob), "heart": (prediction, prob), "kidney": (prediction, prob) }
    """

    results = {}

    try:
        # ---------------- Diabetes ----------------
        diabetes_features = [
            float(data.get("Pregnancies", 0)),
            float(data.get("Glucose", 0)),
            float(data.get("BloodPressure", 0)),
            float(data.get("SkinThickness", 0)),
            float(data.get("Insulin", 0)),
            float(data.get("BMI", 0)),
            float(data.get("DiabetesPedigreeFunction", 0)),
            float(data.get("Age", 0)),
        ]
        diabetes_input = diabetes_scaler.transform([diabetes_features])
        diabetes_prob = diabetes_model.predict(diabetes_input)[0][0]
        results["diabetes"] = ("Positive" if diabetes_prob > 0.5 else "Negative", float(diabetes_prob))

        # ---------------- Heart ----------------
        heart_features = [
            float(data.get("age", 0)),
            float(data.get("sex", 0)),
            float(data.get("cp", 0)),
            float(data.get("trestbps", 0)),
            float(data.get("chol", 0)),
            float(data.get("fbs", 0)),
            float(data.get("restecg", 0)),
            float(data.get("thalach", 0)),
            float(data.get("exang", 0)),
            float(data.get("oldpeak", 0)),
            float(data.get("slope", 0)),
            float(data.get("ca", 0)),
            float(data.get("thal", 0)),
        ]
        heart_input = heart_scaler.transform([heart_features])
        heart_prob = heart_model.predict(heart_input)[0][0]
        results["heart"] = ("Positive" if heart_prob > 0.5 else "Negative", float(heart_prob))

        # ---------------- Kidney ----------------
        kidney_features = [
            float(data.get("age", 0)),
            float(data.get("bp", 0)),
            float(data.get("sg", 0)),
            float(data.get("al", 0)),
            float(data.get("su", 0)),
            float(data.get("rbc", 0)),
            float(data.get("pc", 0)),
            float(data.get("pcc", 0)),
            float(data.get("ba", 0)),
            float(data.get("bgr", 0)),
            float(data.get("bu", 0)),
        ]
        kidney_input = kidney_scaler.transform([kidney_features])
        kidney_prob = kidney_model.predict(kidney_input)[0][0]
        results["kidney"] = ("Positive" if kidney_prob > 0.5 else "Negative", float(kidney_prob))

    except Exception as e:
        results["error"] = str(e)

    return results
