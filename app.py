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
MODEL_NAMES = ["LSTM"]  # Running only LSTM for now
METRIC_NAMES = ["O3", "NO2"]

# -------------------------------
# Step 2: Sidebar UI
# -------------------------------
st.title("Delhi Air Pollution Forecaster")

site_choice = st.sidebar.selectbox("Select Site:", SITE_NAMES)
model_choice = st.sidebar.selectbox("Select Model:", MODEL_NAMES)
element_choice = st.sidebar.selectbox("Select Element:", METRIC_NAMES)

# -------------------------------
# Step 3: Paths
# -------------------------------
site_num = SITE_TO_NUM[site_choice]

MODEL_DIR = "LSTM Models"
SCALER_DIR = "scaler"
DATA_DIR = "Data"

model_file = f"site_{site_num}_{element_choice}_model (1).h5"
scaler_file = f"site_{site_num}_scalers (1).pkl"
data_file = f"site_{site_num}_train_data.csv"

model_path = os.path.join(MODEL_DIR, model_file)
scaler_path = os.path.join(SCALER_DIR, scaler_file)
data_path = os.path.join(DATA_DIR, data_file)

# -------------------------------
# Step 4: Load model and scaler
# -------------------------------
model = load_model(model_path, compile=False)

with open(scaler_path, 'rb') as f:
    scaler_obj = joblib.load(f)

scaler_X = scaler_obj['scaler_X']
scaler_y = scaler_obj[f'scaler_y_{element_choice}']

# -------------------------------
# Step 5: Define features
# -------------------------------
# Must match what was used during training
feature_columns = [
    'O3_forecast', 'NO2_forecast', 'T_forecast', 'q_forecast',
    'u_forecast', 'v_forecast', 'w_forecast',
    'NO2_satellite', 'HCHO_satellite', 'ratio_satellite'
]

# -------------------------------
# Step 6: Load data
# -------------------------------
df = pd.read_csv(data_path)

# -------------------------------
# Step 7: Prepare recent sequence
# -------------------------------
def create_recent_sequence(df, feature_columns, time_steps=24):
    """
    Pull the most recent data (last 'time_steps' rows) for the given features.
    Returns: shape (1, time_steps, num_features)
    """
    return df[feature_columns].values[-time_steps:].reshape(1, time_steps, -1)

X_input = create_recent_sequence(df, feature_columns)

# -------------------------------
# Step 8: Scale and predict
# -------------------------------
# Flatten for scaler, then reshape back
X_input_scaled = scaler_X.transform(X_input.reshape(-1, X_input.shape[-1])).reshape(X_input.shape)

y_pred_scaled = model.predict(X_input_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# -------------------------------
# Step 9: Display results
# -------------------------------
st.subheader(f"Next 24h {element_choice} prediction for {site_choice} ({model_choice})")

prediction_df = pd.DataFrame({
    "Hour": np.arange(1, 25),
    f"{element_choice}_prediction": y_pred.flatten()
})

st.line_chart(prediction_df.set_index("Hour"))
st.dataframe(prediction_df)
