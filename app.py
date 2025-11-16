# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model

# -------------------------------
# Step 1: Define options
# -------------------------------
SITE_NAMES = [
    "Mukherjee Nagar", "Uttam Nagar", "Lajpat Nagar", "Narela", 
    "Patparganj", "Pooth Khurd", "Gokulpuri"
]
SITE_TO_NUM = {name: i+1 for i, name in enumerate(SITE_NAMES)}
MODEL_NAMES = ["LSTM", "GRU"]
METRIC_NAMES = ["O3", "NO2"]

# -------------------------------
# Step 2: Sidebar UI
# -------------------------------
st.title("Delhi Air Pollution Forecaster")

site_choice = st.sidebar.selectbox("Select Site:", SITE_NAMES)
model_choice = st.sidebar.selectbox("Select Model:", MODEL_NAMES)
element_choice = st.sidebar.selectbox("Select Element:", METRIC_NAMES)

# -------------------------------
# Step 3: File paths
# -------------------------------
site_num = SITE_TO_NUM[site_choice]

DATA_DIR = "Data"
SCALER_DIR = "scaler"
LSTM_DIR = "LSTM Models"

data_file = f"site_{site_num}_train_data.csv"
data_path = os.path.join(DATA_DIR, data_file)

# -------------------------------
# Step 4: Load model & scaler
# -------------------------------
if model_choice == "LSTM":
    # LSTM from LSTM Models folder
    model_file = f"site_{site_num}_{element_choice}_model (1).h5"
    model_path = os.path.join(LSTM_DIR, model_file)
    model = load_model(model_path, compile=False)

    # scaler stored in scaler folder
    scaler_file = f"site_{site_num}_scalers (1).pkl"
    scaler_path = os.path.join(SCALER_DIR, scaler_file)
    with open(scaler_path, 'rb') as f:
        scaler_obj = joblib.load(f)

elif model_choice == "GRU":
    # GRU model itself is stored as pickle in scaler folder
    model_file = f"site_{site_num}_{element_choice}_gru_model.pkl"
    model_path = os.path.join(SCALER_DIR, model_file)
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
    
    # GRU scaler is also in scaler folder
    scaler_file = f"site_{site_num}_gru_scalers.pkl"
    scaler_path = os.path.join(SCALER_DIR, scaler_file)
    with open(scaler_path, 'rb') as f:
        scaler_obj = joblib.load(f)

# Extract scalers and feature columns
scaler_X = scaler_obj['scaler_X']
scaler_y = scaler_obj[f'scaler_y_{element_choice}']
feature_columns = scaler_obj['feature_columns']  # must be saved during training

# -------------------------------
# Step 5: Load data
# -------------------------------
df = pd.read_csv(data_path)

# -------------------------------
# Step 6: Prepare recent sequence
# -------------------------------
def create_recent_sequence(df, input_features, time_steps=24):
    """
    Pull the most recent data (last 'time_steps' rows) for the given features.
    Returns: shape (1, time_steps, num_features)
    """
    return df[input_features].values[-time_steps:].reshape(1, time_steps, -1)

input_features = feature_columns
X_input = create_recent_sequence(df, input_features)

# -------------------------------
# Step 7: Scale input
# -------------------------------
X_input_scaled = scaler_X.transform(X_input.reshape(-1, X_input.shape[-1])).reshape(X_input.shape)

# -------------------------------
# Step 8: Prediction
# -------------------------------
if model_choice == "LSTM":
    y_pred_scaled = model.predict(X_input_scaled)
else:  # GRU
    # GRU model might need flattened input
    y_pred_scaled = model.predict(X_input_scaled.reshape(X_input_scaled.shape[0], -1))

y_pred = scaler_y.inverse_transform(y_pred_scaled)

# -------------------------------
# Step 9: Display predictions
# -------------------------------
st.subheader(f"Next 24h {element_choice} prediction for {site_choice} ({model_choice})")

prediction_df = pd.DataFrame({
    "Hour": np.arange(1, 25),
    f"{element_choice}_prediction": y_pred.flatten()
})

st.line_chart(prediction_df.set_index("Hour"))
st.dataframe(prediction_df)
