#Step 1: Define your options
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
# In your app.py
SITE_NAMES = [
    "Mukherjee Nagar", "Uttam Nagar", "Lajpat Nagar", "Narela", 
    "Patparganj", "Pooth Khurd", "Gokulpuri"
]
SITE_TO_NUM = {name: i+1 for i, name in enumerate(SITE_NAMES)}
MODEL_NAMES = ["LSTM", "GRU"]
METRIC_NAMES = ["O3", "NO2"]


#Step 2: Sidebar (or main UI) selections
st.title("Delhi Air Pollution Forecaster")

# Sidebar controls
site_choice = st.sidebar.selectbox("Select Site:", SITE_NAMES)
model_choice = st.sidebar.selectbox("Select Model:", MODEL_NAMES)
element_choice = st.sidebar.selectbox("Select Element:", METRIC_NAMES)


#Step 3: Dynamic file/model selection and loading
site_num = SITE_TO_NUM[site_choice]

# Set up paths for model & scaler
MODEL_DIR = "LSTM Models"
SCALER_DIR = "scaler"
# For LSTM, model file example: site_1_O3_model.h5, scaler: site_1_scalers (1).pkl

model_key = f"site_{site_num}_{element_choice}_model (1).h5"
scaler_key = f"site_{site_num}_scalers (1).pkl"

model_path = os.path.join(MODEL_DIR, model_key)
scaler_path = os.path.join("scaler", scaler_key)
data_path = os.path.join("Data", f"site_{site_num}_train_data.csv")

# Load model and scaler
model = load_model(model_path, compile=False)
with open(scaler_path, 'rb') as f:
    scaler_obj = joblib.load(f)

# Extract input and output scalers
scaler_X = scaler_obj['scaler_X']
scaler_y = scaler_obj[f'scaler_y_{element_choice}']  # dynamic for selected element

def create_recent_sequence(df, input_features, time_steps=24):
    """
    Pull the most recent data (last 'time_steps' rows) for the given features.
    Returns: shape (1, time_steps, num_features)
    """
    return df[input_features].values[-time_steps:].reshape(1, time_steps, -1)


input_features = ['O3', 'NO2', 'PM2.5', 'PM10']  # adjust to your CSV

X_input = create_recent_sequence(df, input_features)

# Flatten for scaler, then reshape back
X_input_scaled = scaler_X.transform(X_input.reshape(-1, X_input.shape[-1])).reshape(X_input.shape)
y_pred_scaled = model.predict(X_input_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

#Step 5: Displaying predictions (1 value per hour)
# Display predictions as dataframe or chart
st.subheader(f"Next 24h {element_choice} prediction for {site_choice} ({model_choice})")
prediction_df = pd.DataFrame({
    "Hour": np.arange(1, 25),
    f"{element_choice}_prediction": y_pred.flatten()
})
st.line_chart(prediction_df.set_index("Hour"))
st.dataframe(prediction_df)


# 6





