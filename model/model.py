# models.py
"""
Utility to load different model types and produce 24-hour predictions for O3 and NO2.
Expected model file naming (place in models/ directory):
 - gru_o3.h5, gru_no2.h5          (Keras GRU)
 - sltm_o3.h5, sltm_no2.h5        (Keras SLTM - if you have it)
 - rf_o3.joblib, rf_no2.joblib    (sklearn RandomForest)
 - xgb_o3.joblib, xgb_no2.joblib   (xgboost joblib)
If models are missing, the code returns synthetic fallback predictions.
"""

import os
import numpy as np
import pandas as pd
import joblib
from typing import Tuple, Dict

# For Keras models (GRU/SLTM)
try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def _load_sklearn(path: str):
    return joblib.load(path)


def _load_keras(path: str):
    if load_model is None:
        raise RuntimeError("Keras not available in environment (tensorflow).")
    return load_model(path)


def load_model_for(model_name: str, pollutant: str):
    """
    model_name: one of "GRU", "SLTM", "RandomForest", "XGBoost" (case-insensitive)
    pollutant: "o3" or "no2"
    """
    model_key = model_name.strip().lower()
    pollutant = pollutant.strip().lower()

    # file name conventions
    if model_key in ("gru", "sltm"):
        filename = f"{model_key}_{pollutant}.h5"
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            return _load_keras(path)
        else:
            return None
    elif model_key in ("randomforest", "rf"):
        filename = f"rf_{pollutant}.joblib"
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            return _load_sklearn(path)
        else:
            return None
    elif model_key in ("xgboost", "xgb"):
        filename = f"xgb_{pollutant}.joblib"
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            return _load_sklearn(path)
        else:
            return None
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def make_dummy_prediction(seed: int = 0, base: float = 30.0, trend: float = 0.2) -> np.ndarray:
    """Return 24 hourly values as fallback."""
    rng = np.random.RandomState(seed)
    hours = np.arange(24)
    values = base + trend * hours + rng.normal(scale=2.5, size=24)
    return np.maximum(values, 0.0)


def prepare_features_for(site: str, now_ts=None) -> np.ndarray:
    """
    Build the feature array the models expect. This is a placeholder.
    Replace with your real feature engineering.
    For this template we return a dummy 24x1 array or a vector of len 24.
    """
    # Example: Use hour-of-day one-hot or scalar hour as a simple feature
    hours = np.arange(24)
    hour_features = (hours / 23.0).reshape(-1, 1)
    # Add site index as feature
    site_idx = abs(hash(site)) % 100 / 100.0
    site_col = np.full((24, 1), site_idx)
    X = np.hstack([hour_features, site_col])
    # If your model expects different shape, adapt here
    return X


def predict_24h(site: str, model_name: str) -> pd.DataFrame:
    """
    Return a DataFrame with index = datetimes for next 24 hours and columns ["o3", "no2"].
    """
    model_name = model_name.strip()
    # Load models for each pollutant
    m_o3 = load_model_for(model_name, "o3")
    m_no2 = load_model_for(model_name, "no2")

    # placeholder features
    X = prepare_features_for(site)

    # index: next 24 hours
    now = pd.Timestamp.now().floor("H")
    index = pd.date_range(now + pd.Timedelta(hours=1), periods=24, freq="H")

    # Predict
    if m_o3 is not None:
        try:
            # Keras models expect 3D input sometimes; adapt accordingly
            if hasattr(m_o3, "predict") and X.ndim == 2 and len(X.shape) == 2:
                # assume sklearn-like prediction
                y_o3 = m_o3.predict(X)
            else:
                y_o3 = m_o3.predict(X)
            y_o3 = np.ravel(y_o3)[:24]
        except Exception:
            y_o3 = make_dummy_prediction(seed=1, base=40.0, trend=0.5)
    else:
        y_o3 = make_dummy_prediction(seed=1, base=40.0, trend=0.5)

    if m_no2 is not None:
        try:
            if hasattr(m_no2, "predict") and X.ndim == 2:
                y_no2 = m_no2.predict(X)
            else:
                y_no2 = m_no2.predict(X)
            y_no2 = np.ravel(y_no2)[:24]
        except Exception:
            y_no2 = make_dummy_prediction(seed=2, base=20.0, trend=0.1)
    else:
        y_no2 = make_dummy_prediction(seed=2, base=20.0, trend=0.1)

    df = pd.DataFrame({"o3": y_o3, "no2": y_no2}, index=index)
    return df
