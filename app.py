# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler, StandardScaler
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Define options
# -----------------------------
SITE_NAMES = [
    "Mukherjee Nagar", "Uttam Nagar", "Lajpat Nagar", "Narela",
    "Patparganj", "Pooth Khurd", "Gokulpuri"
]
SITE_TO_NUM = {name: i+1 for i, name in enumerate(SITE_NAMES)}
MODEL_NAMES = ["LSTM", "GRU"]
METRIC_NAMES = ["O3", "NO2"]

# -----------------------------
# Step 2: Streamlit sidebar
# -----------------------------
st.title("Delhi Air Pollution Forecaster")

site_choice = st.sidebar.selectbox("Select Site:", SITE_NAMES)
model_choice = st.sidebar.selectbox("Select Model:", MODEL_NAMES)
element_choice = st.sidebar.selectbox("Select Element:", METRIC_NAMES)

# -----------------------------
# Step 3: Paths for model, scaler, data
# -----------------------------
site_num = SITE_TO_NUM[site_choice]

MODEL_DIR = "LSTM Models"  # folder containing .h5 models
SCALER_DIR = "scaler"      # folder containing scaler .pkl files
DATA_DIR = "Data"          # folder containing CSV files

# Example filenames
model_key = f"site_{site_num}_{element_choice}_model (1).h5"
scaler_key = f"site_{site_num}_scalers (1).pkl"
data_key = f"site_{site_num}_train_data.csv"

model_path = os.path.join(MODEL_DIR, model_key)
scaler_path = os.path.join(SCALER_DIR, scaler_key)
data_path = os.path.join(DATA_DIR, data_key)

# -----------------------------
# Step 4: Load model safely
# -----------------------------
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.stop()

model = load_model(model_path, compile=False)  # avoid H5 metric deserialization issues

# -----------------------------
# Step 5: Load scaler
# -----------------------------
if not os.path.exists(scaler_path):
    st.error(f"Scaler file not found: {scaler_path}")
    st.stop()

with open(scaler_path, 'rb') as f:
    scaler_obj = joblib.load(f)

scaler_X = scaler_obj['scaler_X']
scaler_y = scaler_obj[f'scaler_y_{element_choice}']

# -----------------------------
# Step 6: Load site CSV data
# -----------------------------
if not os.path.exists(data_path):
    st.error(f"Data file not found: {data_path}")
    st.stop()

df = pd.read_csv(data_path)

# -----------------------------
# Step 7: Prepare recent sequence
# -----------------------------
def create_recent_sequence(df, input_features, time_steps=24):
    """
    Pull the most recent data (last 'time_steps' rows) for the given features.
    Returns: shape (1, time_steps, num_features)
    """
    return df[input_features].values[-time_steps:].reshape(1, time_steps, -1)

# Adjust input_features to match your CSV columns
input_features = ['O3', 'NO2', 'PM2.5', 'PM10']  

X_input = create_recent_sequence(df, input_features)

# Scale the input
X_input_scaled = scaler_X.transform(X_input.reshape(-1, X_input.shape[-1])).reshape(X_input.shape)

# -----------------------------
# Step 8: Predict
# -----------------------------
y_pred_scaled = model.predict(X_input_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# -----------------------------
# Step 9: Display predictions
# -----------------------------
st.subheader(f"Next 24h {element_choice} prediction for {site_choice} ({model_choice})")

prediction_df = pd.DataFrame({
    "Hour": np.arange(1, 25),
    f"{element_choice}_prediction": y_pred.flatten()
})

st.line_chart(prediction_df.set_index("Hour"))
st.dataframe(prediction_df)

