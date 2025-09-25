import os
import joblib
import pickle
import numpy as np
from typing import Dict, Any
from tensorflow.keras.models import load_model

# ---------- GLOBALS ----------
MODELS: Dict[str, Any] = {}
SCALERS: Dict[str, Any] = {}
ENCODERS: Dict[str, Any] = {}   # keyed by feature name, e.g. 'rbc' -> LabelEncoder()

# Feature lists (must match training)
FEATURES = {
    "diabetes": [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ],
    "heart": [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ],
    "kidney": [
        "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba",
        "bgr", "bu"
    ]
}

# Categorical features that were label-encoded for kidney
KIDNEY_CATEGORICAL = ["rbc", "pc", "pcc", "ba"]

DEFAULT_MODEL_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "models")


# ---------- LOAD HELPERS ----------
def _try_load_with_joblib_or_pickle(path):
    """
    Try joblib.load first; if fails, try pickle.load.
    """
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            raise


def load_models(models_dir: str = None) -> None:
    """
    Load models, scalers, and encoders into memory.
    """
    global MODELS, SCALERS, ENCODERS
    MODELS.clear()
    SCALERS.clear()
    ENCODERS.clear()

    if models_dir is None:
        models_dir = DEFAULT_MODEL_DIR

    for key in FEATURES.keys():
        model_path = os.path.join(models_dir, f"{key}_model.h5")
        scaler_path = os.path.join(models_dir, f"{key}_scaler.pkl")
        scaler_path_alt = os.path.join(models_dir, f"{key}_scaler.joblib")

        # load model
        if os.path.exists(model_path):
            try:
                MODELS[key] = load_model(model_path)
                print(f"[model_utils] ✅ Loaded model: {model_path}")
            except Exception as e:
                print(f"[model_utils] ❌ ERROR loading model {model_path}: {e}")
        else:
            print(f"[model_utils] ⚠️ Model not found: {model_path}")

        # load scaler (try pkl or joblib)
        scaler_obj = None
        for p in (scaler_path, scaler_path_alt):
            if os.path.exists(p):
                try:
                    scaler_obj = _try_load_with_joblib_or_pickle(p)
                    SCALERS[key] = scaler_obj
                    print(f"[model_utils] ✅ Loaded scaler: {p}")
                    break
                except Exception as e:
                    print(f"[model_utils] ❌ ERROR loading scaler {p}: {e}")

        # load encoders for kidney categorical columns (if present)
        if key == "kidney":
            for col in KIDNEY_CATEGORICAL:
                for candidate in (
                    os.path.join(models_dir, f"{col}_le.pkl"),
                    os.path.join(models_dir, f"{col}_le.joblib"),
                    os.path.join(models_dir, f"{key}_{col}_le.pkl"),
                    os.path.join(models_dir, f"{key}_{col}_le.joblib"),
                ):
                    if os.path.exists(candidate):
                        try:
                            enc = _try_load_with_joblib_or_pickle(candidate)
                            ENCODERS[col] = enc
                            print(f"[model_utils] ✅ Loaded encoder for '{col}' from {candidate}")
                            break
                        except Exception as e:
                            print(f"[model_utils] ❌ ERROR loading encoder {candidate}: {e}")
            if not ENCODERS:
                print("[model_utils] ℹ️ No label encoders found for kidney categorical columns. You must provide numeric encoded values or create & save encoders.")


# ---------- HELPERS ----------
def _convert_to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        raise ValueError(f"Unable to convert value to float: {x!r}")


def _prepare_feature_vector(input_dict: Dict[str, Any], feature_list: list) -> np.ndarray:
    """
    Build feature vector in exact order. Accepts:
      - numeric strings (converted to float)
      - for categorical features: if an encoder is present, accepts the raw string and will transform
    """
    vec, missing = [], []
    for f in feature_list:
        raw = input_dict.get(f)
        if raw is None or str(raw).strip() == "":
            missing.append(f)
            continue

        # if this feature has a saved encoder, use it (handles kidney categorical columns)
        if f in ENCODERS:
            enc = ENCODERS[f]
            # if user provided numeric already, try to convert
            try:
                val = _convert_to_float(raw)
            except Exception:
                # assume string label: transform with encoder
                try:
                    val = enc.transform([str(raw)])[0]
                except Exception as e:
                    raise ValueError(f"Error encoding feature '{f}' with value {raw!r}: {e}")
        else:
            # no encoder: convert to float
            try:
                val = _convert_to_float(raw)
            except Exception:
                raise ValueError(f"Feature '{f}' expected numeric value but got: {raw!r}")

        vec.append(val)

    if missing:
        raise ValueError(f"Missing required features: {missing}")
    return np.array(vec, dtype=np.float32).reshape(1, -1)


def _model_predict_proba(model, X: np.ndarray):
    preds = np.array(model.predict(X, verbose=0))

    # binary case
    if preds.ndim == 2 and preds.shape[1] == 1:
        prob = float(preds[0, 0])
        label = int(prob >= 0.5)
        confidence = prob if label == 1 else 1.0 - prob
        return label, confidence, preds.tolist()

    # multi-class
    try:
        probs = preds[0]
        if np.any(probs < 0) or not np.isclose(np.sum(probs), 1.0, atol=1e-3):
            exps = np.exp(probs - np.max(probs))
            probs = exps / np.sum(exps)
    except Exception:
        probs = preds.flatten()

    idx = int(np.argmax(probs))
    confidence = float(probs[idx])
    return idx, confidence, preds.tolist()


# ---------- PUBLIC API ----------
def predict_for(key: str, input_dict: Dict[str, Any]) -> Dict[str, Any]:
    if key not in FEATURES:
        raise ValueError(f"Unknown model key: {key}")
    if key not in MODELS:
        raise RuntimeError(f"Model '{key}' not loaded. Run load_models().")

    X = _prepare_feature_vector(input_dict, FEATURES[key])

    if key in SCALERS and SCALERS[key] is not None:
        X = SCALERS[key].transform(X)

    label, confidence, raw = _model_predict_proba(MODELS[key], X)
    return {"label": int(label), "confidence": float(confidence), "raw": raw}


def predict_all(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    results = {}
    for key in FEATURES.keys():
        try:
            if key in MODELS:
                results[key] = predict_for(key, input_dict)
            else:
                results[key] = {"error": "model_not_loaded"}
        except Exception as e:
            results[key] = {"error": str(e)}
    print(f"[model_utils] ✅ Predictions complete: {list(results.keys())}")
    return results


if __name__ == "__main__":
    load_models()
