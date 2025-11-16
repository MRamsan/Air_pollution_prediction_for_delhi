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
MODEL_NAMES = ["LSTM", "GRU", "RandomForest", "XGBoost"]
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
scaler_path = os.path.join("scaler.pkl", scaler_key)
data_path = os.path.join("Data", f"site_{site_num}_train_data.csv")

# Load model and scaler
model = load_model(model_path)
with open(scaler_path, 'rb') as f:
    scaler_obj = joblib.load(f)
scaler_X = scaler_obj['scaler_X']
scaler_y = scaler_obj[f'scaler_y_{element_choice}']


#Step 4: Data Loading, Preprocessing, and Prediction


# For prediction: use your latest available inputs (or build as in your training phase)
def create_recent_sequence(df, input_features, time_steps=24):
    # Implement this to pull the most recent data (last 24 rows/features for a site)
    # The result: shape (1, time_steps, num_features)
    # Example:
    return df[input_features].values[-time_steps:].reshape(1, time_steps, -1)

df = pd.read_csv(data_path)
X_input = create_recent_sequence(df, input_features)

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



