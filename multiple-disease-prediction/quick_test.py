import model_utils

model_utils.load_models()   # prints loaded models/scalers/encoders

sample = {
    "Pregnancies": "2", "Glucose": "120", "BloodPressure": "70", "SkinThickness": "20",
    "Insulin": "79", "BMI": "28.0", "DiabetesPedigreeFunction": "0.5", "Age": "33",
    "age": "33", "sex": "1", "cp": "3", "trestbps": "130", "chol": "250",
    "fbs": "0", "restecg": "1", "thalach": "150", "exang": "0",
    "oldpeak": "1.2", "slope": "2", "ca": "0", "thal": "2",
    "bp": "80", "sg": "1.02", "al": "1", "su": "0",
    "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent",
    "bgr": "120", "bu": "40"
}

print(model_utils.predict_all(sample))
