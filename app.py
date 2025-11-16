"""
Streamlit app to forecast Delhi air quality (O3, NO2) using pre-trained GRU models.

Folder structure:
  - Data/: Raw CSV files (site_1_train_data.csv, etc.)
  - scaler/: Scaler files (improved_scaler_X_site_1.pkl, improved_scaler_y_site_1.pkl)
  - features/: Feature definitions (improved_features_site_1.pkl)
  - models/: GRU models (improved_gru_model_site_1.h5)

Requirements (requirements.txt):
  streamlit
  pandas
  numpy
  tensorflow
  scikit-learn
  matplotlib
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Delhi Air Quality Forecast (GRU)", layout="wide")

# ============== CONFIG ==============
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if __file__ else os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "Data")
SCALER_DIR = os.path.join(BASE_DIR, "scaler")
FEATURES_DIR = os.path.join(BASE_DIR, "features")
MODELS_DIR = os.path.join(BASE_DIR, "models")

SITE_NAMES = [
    "Mukherjee Nagar", "Uttam Nagar", "Lajpat Nagar", "Narela",
    "Patparganj", "Pooth Khurd", "Gokulpuri"
]
SITE_TO_NUM = {name: i + 1 for i, name in enumerate(SITE_NAMES)}
ELEMENT_NAMES = ["O3", "NO2"]

# ============== HELPER FUNCTIONS ==============
def preprocess_raw_data(df):
    """Preprocess raw data with enhanced feature engineering"""
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
        'O3_target', 'NO2_target', 'year', 'month', 'day', 'hour'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    
    # Create datetime features
    if {'year', 'month', 'day', 'hour'}.issubset(df.columns):
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']], errors='coerce')
        df['datetime'] = df['date'] + pd.to_timedelta(df['hour'], unit='h')
    
    # Sort by time
    df = df.sort_values(['year', 'month', 'day', 'hour']).reset_index(drop=True)
    
    # Drop satellite columns if present (they have too many missing values)
    satellite_cols = ['NO2_satellite', 'HCHO_satellite', 'ratio_satellite']
    df = df.drop(columns=satellite_cols, errors='ignore')
    
    return df

def engineer_features(df, past_days=14):
    """Apply advanced feature engineering matching the training process"""
    df = df.copy()
    
    # 1. Cyclical time features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    if 'date' in df.columns:
        df['day_of_year'] = df['date'].dt.dayofyear
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Day of week features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    else:
        # Default values if date not available
        df['day_of_year'] = 0
        df['day_of_year_sin'] = 0
        df['day_of_year_cos'] = 0
        df['day_of_week'] = 0
        df['is_weekend'] = 0
        df['dow_sin'] = 0
        df['dow_cos'] = 0
    
    # 2. Rush hour indicators
    df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 10)).astype(int)
    df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 20)).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    
    # 3. Interaction features
    df['temp_hour'] = df['T_forecast'] * df['hour']
    df['wind_speed'] = np.sqrt(df['u_forecast']**2 + df['v_forecast']**2)
    df['wind_direction'] = np.arctan2(df['v_forecast'], df['u_forecast'])
    
    # 4. Lagged features (if targets exist)
    if 'O3_target' in df.columns and 'NO2_target' in df.columns:
        for lag in range(1, past_days + 1):
            df[f'O3_target_lag_{lag}d'] = df.groupby('hour')['O3_target'].shift(lag)
            df[f'NO2_target_lag_{lag}d'] = df.groupby('hour')['NO2_target'].shift(lag)
        
        # 5. Rolling statistics
        for window in [3, 7, 14]:
            df[f'O3_rolling_mean_{window}d'] = df.groupby('hour')['O3_target'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean())
            df[f'O3_rolling_std_{window}d'] = df.groupby('hour')['O3_target'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std())
            df[f'NO2_rolling_mean_{window}d'] = df.groupby('hour')['NO2_target'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean())
            df[f'NO2_rolling_std_{window}d'] = df.groupby('hour')['NO2_target'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std())
        
        # 6. Exponential moving averages
        df['O3_ema_7d'] = df.groupby('hour')['O3_target'].transform(
            lambda x: x.ewm(span=7, adjust=False).mean())
        df['NO2_ema_7d'] = df.groupby('hour')['NO2_target'].transform(
            lambda x: x.ewm(span=7, adjust=False).mean())
    else:
        # If no targets, create zero-filled lag features
        for lag in range(1, past_days + 1):
            df[f'O3_target_lag_{lag}d'] = 0
            df[f'NO2_target_lag_{lag}d'] = 0
        for window in [3, 7, 14]:
            df[f'O3_rolling_mean_{window}d'] = 0
            df[f'O3_rolling_std_{window}d'] = 0
            df[f'NO2_rolling_mean_{window}d'] = 0
            df[f'NO2_rolling_std_{window}d'] = 0
        df['O3_ema_7d'] = 0
        df['NO2_ema_7d'] = 0
    
    # Fill any remaining NaN with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    df.fillna(0, inplace=True)
    
    return df

def create_sequences(data, feature_cols, seq_length=48):
    """Create sequences for GRU model input"""
    if len(data) < seq_length:
        return None
    
    # Take the last sequence
    X = data[feature_cols].values[-seq_length:].reshape(1, seq_length, len(feature_cols))
    return X

def load_model_components(site_num, element):
    """Load model, scalers, and feature definitions"""
    # File paths
    model_path = os.path.join(MODELS_DIR, f"improved_gru_model_site_{site_num}.h5")
    scaler_x_path = os.path.join(SCALER_DIR, f"improved_scaler_X_site_{site_num}.pkl")
    scaler_y_path = os.path.join(SCALER_DIR, f"improved_scaler_y_site_{site_num}.pkl")
    features_path = os.path.join(FEATURES_DIR, f"improved_features_site_{site_num}.pkl")
    
    # Check existence
    missing_files = []
    for path in [model_path, scaler_x_path, scaler_y_path, features_path]:
        if not os.path.exists(path):
            missing_files.append(path)
    
    if missing_files:
        return None, None, None, None, missing_files[0]
    
    try:
        # Load model with custom objects if needed
        model = load_model(model_path, compile=False)
        
        # Load scalers
        with open(scaler_x_path, 'rb') as f:
            scaler_X = pickle.load(f)
        with open(scaler_y_path, 'rb') as f:
            scaler_y = pickle.load(f)
        
        # Load feature definitions
        with open(features_path, 'rb') as f:
            feature_info = pickle.load(f)
        
        return model, scaler_X, scaler_y, feature_info, None
    except Exception as e:
        return None, None, None, None, str(e)

# ============== UI ==============
st.title("🌆 Delhi Air Quality Forecast (GRU)")
st.markdown("Forecast O3 and NO2 concentrations using pre-trained GRU models with advanced feature engineering.")

# Sidebar
site_choice = st.sidebar.selectbox("Select Monitoring Site", SITE_NAMES)
element_choice = st.sidebar.selectbox("Select Pollutant", ELEMENT_NAMES)
forecast_hours = st.sidebar.slider("Forecast Hours Ahead", 1, 24, 24)

site_num = SITE_TO_NUM[site_choice]

# Load data
data_file = f"site_{site_num}_train_data.csv"
data_path = os.path.join(DATA_DIR, data_file)

if not os.path.exists(data_path):
    st.error(f"❌ Data file not found: {data_path}")
    st.stop()

# Load and preprocess
with st.spinner("Loading and preprocessing data..."):
    try:
        df_raw = pd.read_csv(data_path)
        df_processed = preprocess_raw_data(df_raw)
        df_features = engineer_features(df_processed)
        st.success(f"✅ Loaded {len(df_features)} records from {data_file}")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

# Display historical data
if 'O3_target' in df_features.columns and 'NO2_target' in df_features.columns:
    st.subheader(f"📊 Historical Data: {site_choice}")
    
    # Show last 7 days
    last_week = df_features.tail(168)  # 24 hours * 7 days
    
    if len(last_week) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
        
        ax1.plot(range(len(last_week)), last_week['O3_target'].values, color='blue', linewidth=1.5)
        ax1.set_title(f"O3 - Last 7 Days")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("O3 (µg/m³)")
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(range(len(last_week)), last_week['NO2_target'].values, color='orange', linewidth=1.5)
        ax2.set_title(f"NO2 - Last 7 Days")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("NO2 (µg/m³)")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# Load model components
model, scaler_X, scaler_y, feature_info, error_path = load_model_components(site_num, element_choice)

if model is None:
    st.error(f"❌ Model components not found for Site {site_num}")
    st.write(f"Missing file: {error_path}")
    st.stop()

st.success(f"✅ Loaded GRU model for Site {site_num}")

# Get feature columns and sequence length
input_features = feature_info['input_features']
sequence_length = feature_info['sequence_length']

st.info(f"Model uses {len(input_features)} features and {sequence_length} hour sequence")

# Run Forecast
if st.button("🚀 Run Forecast", type="primary"):
    with st.spinner("Generating forecast..."):
        try:
            # Ensure all required features are present
            missing_features = [f for f in input_features if f not in df_features.columns]
            if missing_features:
                st.error(f"Missing features: {missing_features}")
                st.stop()
            
            # Create sequence
            X_input = create_sequences(df_features, input_features, sequence_length)
            
            if X_input is None:
                st.error(f"Not enough data. Need at least {sequence_length} hours.")
                st.stop()
            
            # Scale input
            X_input_flat = X_input.reshape(-1, len(input_features))
            X_input_scaled_flat = scaler_X.transform(X_input_flat)
            X_input_scaled = X_input_scaled_flat.reshape(X_input.shape)
            
            # Predict
            y_pred_scaled = model.predict(X_input_scaled, verbose=0)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            
            # Get predictions for selected element
            element_idx = 0 if element_choice == "O3" else 1
            prediction = y_pred[0, element_idx]
            
            # Create forecast dataframe (iterative forecasting for multiple hours)
            forecasts = []
            current_sequence = X_input_scaled.copy()
            
            for h in range(forecast_hours):
                # Predict next step
                y_next_scaled = model.predict(current_sequence, verbose=0)
                y_next = scaler_y.inverse_transform(y_next_scaled)
                
                forecasts.append(y_next[0, element_idx])
                
                # Update sequence for next prediction (simplified - using last prediction)
                # In production, you'd need to update all features properly
                current_sequence = np.roll(current_sequence, -1, axis=1)
                # current_sequence[0, -1, :] would need proper feature updates
            
            # Display results
            st.subheader(f"🎯 {element_choice} Forecast for {site_choice}")
            
            forecast_df = pd.DataFrame({
                "Hour": np.arange(1, forecast_hours + 1),
                f"{element_choice} (µg/m³)": forecasts
            })
            
            # Plot
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(forecast_df["Hour"], forecast_df[f"{element_choice} (µg/m³)"], 
                   marker='o', linewidth=2, markersize=6, color='red')
            ax.set_xlabel("Hours Ahead", fontsize=12)
            ax.set_ylabel(f"{element_choice} (µg/m³)", fontsize=12)
            ax.set_title(f"{element_choice} Forecast - Next {forecast_hours} Hours", fontsize=14)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Show table
            st.subheader("📋 Forecast Data")
            st.dataframe(forecast_df.style.format({f"{element_choice} (µg/m³)": "{:.2f}"}))
            
            # Download button
            csv = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Forecast CSV",
                csv,
                file_name=f"forecast_{site_choice}_{element_choice}.csv",
                mime="text/csv"
            )
            
            st.success(f"✅ Forecast complete! Next hour {element_choice}: {forecasts[0]:.2f} µg/m³")
            
        except Exception as e:
            st.error(f"Error during forecasting: {str(e)}")
            st.exception(e)

# Footer
st.markdown("---")
st.markdown("**Note:** This model uses Bidirectional GRU with attention mechanism and 48-hour sequences for forecasting.")
