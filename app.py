# models.py
"""
Model loader + prediction helpers for pollution forecasting.
Expect model files in ./models/ with names:
    <site>_<model>_<pollutant>.<ext>
Examples:
    mukherjee_nagar_gru_o3.h5
    mukherjee_nagar_randomforest_no2.pkl

Model loading rules:
 - GRU, SLTM -> Keras .h5
 - RandomForest, XGBoost -> joblib/pickle .pkl

If model file missing, this returns a deterministic dummy fallback 24-hour forecast.
Replace prepare_features_for(...) with your real feature engineering.
"""

import os
import numpy as np
import pandas as pd
import joblib
from typing import Tuple
from datetime import datetime, timedelta

# Optional import for Keras models
try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

def safe_name(s: str) -> str:
    """normalise site or other names into filename-friendly strings"""
    return str(s).strip().lower().replace(" ", "_").replace("-", "_")

def model_filename(site: str, model_name: str, pollutant: str) -> str:
    site_f = safe_name(site)
    model_f = model_name.strip().lower()
    pollutant_f = pollutant.strip().lower()
    # choose extension depending on model
    if model_f in ("gru", "sltm"):
        ext = "h5"
    else:
        ext = "pkl"
    return f"{site_f}_{model_f}_{pollutant_f}.{ext}"

def load_model_file(site: str, model_name: str, pollutant: str):
    fname = model_filename(site, model_name, pollutant)
    path = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(path):
        return None, path
    model_key = model_name.strip().lower()
    try:
        if model_key in ("gru", "sltm"):
            if load_model is None:
                raise RuntimeError("TensorFlow/Keras not available to load .h5 models.")
            m = load_model(path, compile=False)
            return m, path
        else:
            m = joblib.load(path)
            return m, path
    except Exception as e:
        # on any load error return None
        print(f"Error loading {path}: {e}")
        return None, path

def make_dummy_prediction(site: str, pollutant: str, base: float = 50.0, seed: int = 0) -> np.ndarray:
    """Return deterministic 24-hour fallback values (no negative)."""
    rng = np.random.RandomState(abs(hash(site + pollutant)) % 2**32 + seed)
    hours = np.arange(24)
    trend = (np.sin(hours / 24 * 2 * np.pi) * 5.0)  # diurnal pattern
    vals = base + trend + rng.normal(scale=3.0, size=24)
    return np.clip(vals, 0.0, None)

def prepare_features_for(site: str):
    """
    Placeholder feature builder for each of next 24 hours.
    Real pipeline should use past observations + meteorology + time features.
    We'll return a 24 x F numpy array where F=2 (hour_norm, site_idx).
    """
    hours = np.arange(24)
    hour_norm = (hours / 23.0).reshape(-1, 1)
    site_idx = (abs(hash(site)) % 100) / 100.0
    site_col = np.full((24, 1), site_idx)
    X = np.hstack([hour_norm, site_col])  # shape (24,2)
    return X

def predict_with_model_obj(model_obj, X: np.ndarray) -> np.ndarray:
    """
    Attempt to get predictions from model_obj for 24 rows in X.
    Handles: sklearn-like models (predict), Keras models (predict).
    If model_obj is None, raise ValueError.
    """
    if model_obj is None:
        raise ValueError("model_obj is None")
    try:
        # Keras models sometimes expect 3D shape (samples, timesteps, features)
        import numpy as _np
        if hasattr(model_obj, "predict") and len(X.shape) == 2:
            # Try sklearn-style predict
            y = model_obj.predict(X)
            return np.ravel(y)[:24]
    except Exception:
        pass

    # fallback: try keras predict with reshape
    try:
        # reshape to (samples, timesteps=1, features)
        X3 = X.reshape((X.shape[0], 1, X.shape[1]))
        y = model_obj.predict(X3, verbose=0)
        return np.ravel(y)[:24]
    except Exception as e:
        raise RuntimeError(f"Model predict failed: {e}")

def predict_24h(site: str, model_name: str) -> Tuple[pd.DataFrame, dict]:
    """
    Return DataFrame indexed by next 24 hourly timestamps with columns ['o3','no2'].
    Also returns a dict of model_paths used for each pollutant.
    """
    pollutants = ["o3", "no2"]
    preds = {}
    model_paths = {}
    X = prepare_features_for(site)  # shape (24, F)
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    idx = [now + timedelta(hours=i+1) for i in range(24)]

    for p in pollutants:
        model_obj, path = load_model_file(site, model_name, p)
        model_paths[p] = path
        if model_obj is None:
            # fallback dummy — base different for pollutants
            base = 60.0 if p == "o3" else 30.0
            preds[p] = make_dummy_prediction(site, p, base=base, seed=1)
        else:
            try:
                y = predict_with_model_obj(model_obj, X)
                # ensure length 24
                if len(y) < 24:
                    y = np.pad(y, (0, 24 - len(y)), "edge")
                preds[p] = np.clip(np.array(y[:24], dtype=float), 0.0, None)
            except Exception as e:
                print(f"Prediction failed for {path}: {e}")
                base = 60.0 if p == "o3" else 30.0
                preds[p] = make_dummy_prediction(site, p, base=base, seed=2)

    df = pd.DataFrame(preds, index=pd.to_datetime(idx))
    df.index.name = "timestamp"
    return df, model_paths
