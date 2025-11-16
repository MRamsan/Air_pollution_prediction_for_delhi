# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model

# -------------------------------
# Step 0: Setup directories
# -------------------------------
BASE_DIR = os.path.dirname(__file__)  # directory of app.py

MODEL_DIR = os.path.join(BASE_DIR, "LSTM Models")
SCALER_DIR = os.path.join(BASE_DIR, "scaler")
DATA_DIR = os.path.join(BASE_DIR, "Data")

# -------------------------------
# Step 1: Site & Element options
# -------------------------------
SITE_NAMES = [
    "Mukherjee Nagar", "Uttam Nagar", "Lajpat Nagar", "Narela",
    "Patparganj", "Pooth Khurd", "Gokulpuri"
]
SITE_TO_NUM = {name: i + 1 for i, name in enumerate(SITE_NAMES)}
ELEMENT_NAMES = ["O3", "NO2"]

# -------------------------------
# Step 2: Sidebar UI
# -------------------------------
st.title("Delhi Air Pollution Forecaster")

site_choice = st.sidebar.selectbox("Select Site:", SITE_NAMES)
element_choice = st.sidebar.selectbox("Select Element:", ELEMENT_NAMES)

site_num = SITE_TO_NUM[site_choice]

# -------------------------------
# Step 3: File paths
# -------------------------------
model_file = f"site_{site_num}_{element_choice}_model (1).h5"
scaler_file = f"site_{site_num}_scalers (1).pkl"
data_file = f"site_{site_num}_train_data.csv"

model_path = os.path.join(MODEL_DIR, model_file)
scaler_path = os.path.join(SCALER_DIR, scaler_file)
data_path = os.path.join(DATA_DIR, data_file)

# -------------------------------
# Step 4: Preprocessing function
# -------------------------------
def preprocess_lstm_df(df):
    df = df.copy()
    # Strip column names
    df.columns = [c.strip() for c in df.columns]
    # Strip object columns
    for c in df.select_dtypes(include='object').columns:
        df[c] = df[c].astype(str).str.strip()
    # Convert numeric columns
    numeric_cols = [
        'O3_forecast', 'NO2_forecast', 'T_forecast',
        'q_forecast', 'u_forecast', 'v_forecast', 'w_forecast',
        'O3_target', 'NO2_target'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    # Build timestamp features
    if {'year','month','day','hour'}.issubset(df.columns):
        df['timestamp'] = pd.to_datetime(df[['year','month','day','hour']], errors='coerce')
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['weekday'] = df['timestamp'].dt.weekday
    # Drop rows with NaNs in features or targets
    feature_cols = ['hour', 'day', 'month', 'weekday',
                    'O3_forecast', 'NO2_forecast', 'T_forecast',
                    'q_forecast', 'u_forecast', 'v_forecast', 'w_forecast']
    target_cols = [c for c in ['O3_target', 'NO2_target'] if c in df.columns]
    df = df.dropna(subset=feature_cols + target_cols)
    return df

# -------------------------------
# Step 5: Check files exist
# -------------------------------
for path in [model_path, scaler_path, data_path]:
    if not os.path.exists(path):
        st.error(f"Required file not found: {path}")
        st.stop()

# -------------------------------
# Step 6: Load model and scaler
# -------------------------------
model = load_model(model_path, compile=False)

with open(scaler_path, 'rb') as f:
    scaler_obj = joblib.load(f)

scaler_X = scaler_obj['scaler_X']
scaler_y = scaler_obj[f'scaler_y_{element_choice}']

# -------------------------------
# Step 7: Load and preprocess data
# -------------------------------
df1 = pd.read_csv(data_path)
df = preprocess_lstm_df(df1)

# -------------------------------
# Step 8: Prepare LSTM input
# -------------------------------
feature_columns = [
    'hour', 'day', 'month', 'weekday',
    'O3_forecast', 'NO2_forecast', 'T_forecast',
    'q_forecast', 'u_forecast', 'v_forecast', 'w_forecast'
]

def create_recent_sequence(df, feature_columns, time_steps=24):
    return df[feature_columns].values[-time_steps:].reshape(1, time_steps, len(feature_columns))

X_input = create_recent_sequence(df, feature_columns)

# Flatten for scaler
X_input_flat = X_input.reshape(-1, len(feature_columns))
X_input_scaled_flat = scaler_X.transform(X_input_flat)
X_input_scaled = X_input_scaled_flat.reshape(X_input.shape)

# -------------------------------
# Step 9: Predict
# -------------------------------
y_pred_scaled = model.predict(X_input_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# -------------------------------
# Step 10: Display results
# -------------------------------
st.subheader(f"Next 24h {element_choice} prediction for {site_choice}")

prediction_df = pd.DataFrame({
    "Hour": np.arange(1, 25),
    f"{element_choice}_prediction": y_pred.flatten()
})

st.line_chart(prediction_df.set_index("Hour"))
st.dataframe(prediction_df)
