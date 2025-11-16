# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model

# ---------------- Step 1: Define options ----------------
SITE_NAMES = [
    "Mukherjee Nagar", "Uttam Nagar", "Lajpat Nagar", "Narela",
    "Patparganj", "Pooth Khurd", "Gokulpuri"
]
SITE_TO_NUM = {name: i+1 for i, name in enumerate(SITE_NAMES)}
MODEL_NAMES = ["LSTM", "GRU"]
METRIC_NAMES = ["O3", "NO2"]

# ---------------- Step 2: Sidebar selections ----------------
st.title("Delhi Air Pollution Forecaster")
site_choice = st.sidebar.selectbox("Select Site:", SITE_NAMES)
model_choice = st.sidebar.selectbox("Select Model:", MODEL_NAMES)
element_choice = st.sidebar.selectbox("Select Element:", METRIC_NAMES)

# ---------------- Step 3: Paths ----------------
site_num = SITE_TO_NUM[site_choice]

MODEL_DIR = "LSTM Models"
SCALER_DIR = "scaler"
DATA_DIR = "Data"

model_key = f"site_{site_num}_{element_choice}_model (1).h5"
scaler_key = f"site_{site_num}_scalers (1).pkl"
data_file = f"site_{site_num}_train_data.csv"

model_path = os.path.join(MODEL_DIR, model_key)
scaler_path = os.path.join(SCALER_DIR, scaler_key)
data_path = os.path.join(DATA_DIR, data_file)

# ---------------- Step 4: Load model and scaler ----------------
model = load_model(model_path, compile=False)

with open(scaler_path, 'rb') as f:
    scaler_obj = joblib.load(f)

# Extract scalers and feature order
scaler_X = scaler_obj['scaler_X']
scaler_y = scaler_obj[f'scaler_y_{element_choice}']
input_features = scaler_obj.get('feature_columns')  # exact feature order

if input_features is None:
    st.error("Scaler object does not contain 'feature_columns'. Please save feature columns during training.")
    st.stop()

# ---------------- Step 5: Load data ----------------
df = pd.read_csv(data_path)

# ---------------- Step 6: Prepare input sequence ----------------
def create_recent_sequence(df, input_features, time_steps=24):
    """
    Pulls the most recent 'time_steps' rows of features for prediction.
    Returns shape (1, time_steps, num_features)
    """
    # Ensure we have enough rows
    if len(df) < time_steps:
        raise ValueError(f"Data has only {len(df)} rows, but {time_steps} required for sequence.")
    
    return df[input_features].values[-time_steps:].reshape(1, time_steps, -1)

# Create input sequence
try:
    X_input = create_recent_sequence(df, input_features)
except Exception as e:
    st.error(f"Error creating input sequence: {e}")
    st.stop()

# Scale input
X_input_scaled = scaler_X.transform(X_input.reshape(-1, X_input.shape[-1])).reshape(X_input.shape)

# ---------------- Step 7: Make prediction ----------------
y_pred_scaled = model.predict(X_input_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# ---------------- Step 8: Display predictions ----------------
st.subheader(f"Next 24h {element_choice} prediction for {site_choice} ({model_choice})")

prediction_df = pd.DataFrame({
    "Hour": np.arange(1, 25),
    f"{element_choice}_prediction": y_pred.flatten()
})

st.line_chart(prediction_df.set_index("Hour"))
st.dataframe(prediction_df)
