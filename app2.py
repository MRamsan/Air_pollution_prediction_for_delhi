"""
Streamlit app to forecast Delhi air quality (O3, NO2) using pre-trained GRU models.
Includes an AI chatbot for recommendations and answering questions.

Folder structure:
  - Data/: Raw CSV files (site_1_train_data.csv, etc.)
  - scaler_gru/: Scaler files (site_1_scalers.pkl)
  - features_gru/: Feature definitions (site_1_features.pkl)
  - models_gru/: GRU models (site_1_gru_model.h5)

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
from datetime import datetime

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Delhi Air Quality Forecast", layout="wide", page_icon="🌆")

# ============== CONFIG ==============
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if __file__ else os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "Data")
SCALER_DIR = os.path.join(BASE_DIR, "scaler_gru")
FEATURES_DIR = os.path.join(BASE_DIR, "features_gru")
MODELS_DIR = os.path.join(BASE_DIR, "models_gru")

SITE_NAMES = [
    "Mukherjee Nagar", "Uttam Nagar", "Lajpat Nagar", "Narela",
    "Patparganj", "Pooth Khurd", "Gokulpuri"
]
SITE_TO_NUM = {name: i + 1 for i, name in enumerate(SITE_NAMES)}
ELEMENT_NAMES = ["O3", "NO2"]

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None
if 'current_site' not in st.session_state:
    st.session_state.current_site = None
if 'current_element' not in st.session_state:
    st.session_state.current_element = None

# ============== HELPER FUNCTIONS ==============
def preprocess_raw_data(df):
    """Preprocess raw data with enhanced feature engineering"""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    
    for c in df.select_dtypes(include='object').columns:
        df[c] = df[c].astype(str).str.strip()
    
    numeric_cols = [
        'O3_forecast', 'NO2_forecast', 'T_forecast',
        'q_forecast', 'u_forecast', 'v_forecast', 'w_forecast',
        'O3_target', 'NO2_target', 'year', 'month', 'day', 'hour'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    
    if {'year', 'month', 'day', 'hour'}.issubset(df.columns):
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']], errors='coerce')
        df['datetime'] = df['date'] + pd.to_timedelta(df['hour'], unit='h')
    
    df = df.sort_values(['year', 'month', 'day', 'hour']).reset_index(drop=True)
    satellite_cols = ['NO2_satellite', 'HCHO_satellite', 'ratio_satellite']
    df = df.drop(columns=satellite_cols, errors='ignore')
    
    return df

def engineer_features(df, past_days=14):
    """Apply advanced feature engineering matching the training process"""
    df = df.copy()
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    if 'date' in df.columns:
        df['day_of_year'] = df['date'].dt.dayofyear
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    else:
        for col in ['day_of_year', 'day_of_year_sin', 'day_of_year_cos', 
                    'day_of_week', 'is_weekend', 'dow_sin', 'dow_cos']:
            df[col] = 0
    
    df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 10)).astype(int)
    df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 20)).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    
    df['temp_hour'] = df['T_forecast'] * df['hour']
    df['wind_speed'] = np.sqrt(df['u_forecast']**2 + df['v_forecast']**2)
    df['wind_direction'] = np.arctan2(df['v_forecast'], df['u_forecast'])
    
    if 'O3_target' in df.columns and 'NO2_target' in df.columns:
        for lag in range(1, past_days + 1):
            df[f'O3_target_lag_{lag}d'] = df.groupby('hour')['O3_target'].shift(lag)
            df[f'NO2_target_lag_{lag}d'] = df.groupby('hour')['NO2_target'].shift(lag)
        
        for window in [3, 7, 14]:
            df[f'O3_rolling_mean_{window}d'] = df.groupby('hour')['O3_target'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean())
            df[f'O3_rolling_std_{window}d'] = df.groupby('hour')['O3_target'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std())
            df[f'NO2_rolling_mean_{window}d'] = df.groupby('hour')['NO2_target'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean())
            df[f'NO2_rolling_std_{window}d'] = df.groupby('hour')['NO2_target'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std())
        
        df['O3_ema_7d'] = df.groupby('hour')['O3_target'].transform(
            lambda x: x.ewm(span=7, adjust=False).mean())
        df['NO2_ema_7d'] = df.groupby('hour')['NO2_target'].transform(
            lambda x: x.ewm(span=7, adjust=False).mean())
    else:
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
    X = data[feature_cols].values[-seq_length:].reshape(1, seq_length, len(feature_cols))
    return X

def load_model_components(site_num, element):
    """Load model, scalers, and feature definitions"""
    model_path = os.path.join(MODELS_DIR, f"site_{site_num}_gru_model.h5")
    scaler_path = os.path.join(SCALER_DIR, f"site_{site_num}_scalers.pkl")
    features_path = os.path.join(FEATURES_DIR, f"site_{site_num}_features.pkl")
    
    missing_files = []
    if not os.path.exists(model_path):
        missing_files.append(f"Model: {model_path}")
    if not os.path.exists(scaler_path):
        missing_files.append(f"Scaler: {scaler_path}")
    if not os.path.exists(features_path):
        missing_files.append(f"Features: {features_path}")
    
    if missing_files:
        return None, None, None, None, "\n".join(missing_files)
    
    try:
        model = load_model(model_path, compile=False)
        
        with open(scaler_path, 'rb') as f:
            scaler_obj = pickle.load(f)
        
        if isinstance(scaler_obj, dict):
            scaler_X = scaler_obj.get('scaler_X', None)
            scaler_y = scaler_obj.get(f'scaler_y_{element}', None)
            
            if scaler_X is None or scaler_y is None:
                scaler_X = scaler_obj.get('X', scaler_obj.get('input', list(scaler_obj.values())[0]))
                scaler_y = scaler_obj.get(element, scaler_obj.get('y', scaler_obj.get('output', list(scaler_obj.values())[1] if len(scaler_obj) > 1 else scaler_X)))
        else:
            scaler_X = scaler_obj
            scaler_y = scaler_obj
        
        with open(features_path, 'rb') as f:
            feature_info = pickle.load(f)
        
        if not isinstance(feature_info, dict):
            feature_info = {
                'input_features': feature_info,
                'target_features': ['O3_target', 'NO2_target'],
                'sequence_length': 48
            }
        
        if 'sequence_length' not in feature_info:
            feature_info['sequence_length'] = 48
        
        return model, scaler_X, scaler_y, feature_info, None
        
    except Exception as e:
        import traceback
        return None, None, None, None, f"Error loading files: {str(e)}\n{traceback.format_exc()}"

# ============== CHATBOT FUNCTIONS ==============
def get_air_quality_category(value, pollutant):
    """Categorize air quality based on pollutant levels"""
    if pollutant == "O3":
        if value <= 50: return "Good", "🟢"
        elif value <= 100: return "Moderate", "🟡"
        elif value <= 150: return "Unhealthy for Sensitive Groups", "🟠"
        elif value <= 200: return "Unhealthy", "🔴"
        else: return "Very Unhealthy", "🟣"
    else:  # NO2
        if value <= 40: return "Good", "🟢"
        elif value <= 80: return "Moderate", "🟡"
        elif value <= 180: return "Unhealthy for Sensitive Groups", "🟠"
        elif value <= 280: return "Unhealthy", "🔴"
        else: return "Very Unhealthy", "🟣"

def generate_recommendations(forecast_data, site, pollutant):
    """Generate health recommendations based on forecast"""
    if forecast_data is None:
        return "Please run a forecast first to get personalized recommendations."
    
    avg_value = forecast_data[f"{pollutant} (µg/m³)"].mean()
    max_value = forecast_data[f"{pollutant} (µg/m³)"].max()
    min_value = forecast_data[f"{pollutant} (µg/m³)"].min()
    
    category, emoji = get_air_quality_category(avg_value, pollutant)
    
    recommendations = f"""
📍 **Location:** {site}
🔬 **Pollutant:** {pollutant}
{emoji} **Air Quality:** {category}

**24-Hour Forecast Summary:**
- Average: {avg_value:.2f} µg/m³
- Peak: {max_value:.2f} µg/m³ (Hour {forecast_data[f"{pollutant} (µg/m³)"].idxmax() + 1})
- Lowest: {min_value:.2f} µg/m³ (Hour {forecast_data[f"{pollutant} (µg/m³)"].idxmin() + 1})

**Health Recommendations:**
"""
    
    if category == "Good":
        recommendations += """
✅ Air quality is satisfactory
✅ Outdoor activities are safe for everyone
✅ Normal outdoor exercise is recommended
"""
    elif category == "Moderate":
        recommendations += """
⚠️ Acceptable air quality
⚠️ Sensitive individuals should limit prolonged outdoor exertion
✅ General public can enjoy outdoor activities
"""
    elif category == "Unhealthy for Sensitive Groups":
        recommendations += """
🚨 Sensitive groups (children, elderly, respiratory patients) should:
- Limit outdoor activities during peak hours
- Keep windows closed
- Use air purifiers indoors
✅ General public: Reduce prolonged outdoor exertion
"""
    elif category == "Unhealthy":
        recommendations += """
⛔ Everyone should:
- Avoid outdoor activities
- Stay indoors with air purifiers
- Wear N95 masks if going outside
- Keep medications handy (asthma patients)
"""
    else:  # Very Unhealthy
        recommendations += """
🚫 HEALTH ALERT - Take immediate action:
- Stay indoors at all times
- Seal windows and doors
- Use air purifiers continuously
- Avoid all physical exertion
- Seek medical attention if symptoms develop
"""
    
    # Add time-specific recommendations
    peak_hour = forecast_data[f"{pollutant} (µg/m³)"].idxmax() + 1
    recommendations += f"\n**Best Time for Outdoor Activities:** Hour {forecast_data[f'{pollutant} (µg/m³)'].idxmin() + 1} (lowest pollution)"
    recommendations += f"\n**Avoid:** Hour {peak_hour} (peak pollution)"
    
    return recommendations

def answer_question(question, forecast_data, site, pollutant):
    """Simple rule-based chatbot to answer questions"""
    question_lower = question.lower()
    
    # Greetings
    if any(word in question_lower for word in ["hello", "hi", "hey", "greetings"]):
        return "Hello! 👋 I'm your air quality assistant. I can help you understand the forecast and provide health recommendations. Ask me anything about the air quality predictions!"
    
    # What questions
    if "what is" in question_lower or "what are" in question_lower:
        if "o3" in question_lower or "ozone" in question_lower:
            return "**Ozone (O3)** is a gas composed of three oxygen atoms. Ground-level ozone is created by chemical reactions between nitrogen oxides and volatile organic compounds in the presence of sunlight. It can cause respiratory problems, especially during hot weather."
        elif "no2" in question_lower or "nitrogen dioxide" in question_lower:
            return "**Nitrogen Dioxide (NO2)** is a reddish-brown gas formed from vehicle emissions and industrial processes. It can irritate airways, aggravate respiratory diseases, and contribute to the formation of smog and acid rain."
    
    # When questions
    if forecast_data is not None:
        if "when" in question_lower and ("best" in question_lower or "safe" in question_lower):
            min_hour = forecast_data[f"{pollutant} (µg/m³)"].idxmin() + 1
            min_value = forecast_data[f"{pollutant} (µg/m³)"].min()
            return f"The best time for outdoor activities is around **Hour {min_hour}** when {pollutant} levels are lowest at **{min_value:.2f} µg/m³**."
        
        if "when" in question_lower and ("worst" in question_lower or "avoid" in question_lower or "peak" in question_lower):
            max_hour = forecast_data[f"{pollutant} (µg/m³)"].idxmax() + 1
            max_value = forecast_data[f"{pollutant} (µg/m³)"].max()
            return f"Avoid outdoor activities around **Hour {max_hour}** when {pollutant} levels peak at **{max_value:.2f} µg/m³**."
    
    # Health questions
    if "health" in question_lower or "safe" in question_lower or "risk" in question_lower:
        if forecast_data is not None:
            return generate_recommendations(forecast_data, site, pollutant)
        else:
            return "Please run a forecast first, then I can provide personalized health recommendations based on the predicted air quality levels."
    
    # Recommendation questions
    if "recommend" in question_lower or "should i" in question_lower or "can i" in question_lower:
        if forecast_data is not None:
            avg_value = forecast_data[f"{pollutant} (µg/m³)"].mean()
            category, emoji = get_air_quality_category(avg_value, pollutant)
            
            if "exercise" in question_lower or "run" in question_lower or "jog" in question_lower:
                if category in ["Good", "Moderate"]:
                    return f"{emoji} Yes, outdoor exercise is generally safe. Air quality is {category}. However, avoid peak pollution hours."
                else:
                    return f"{emoji} Not recommended. Air quality is {category}. Consider indoor exercise instead."
            
            if "mask" in question_lower:
                if category in ["Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy"]:
                    return f"✅ Yes, wearing an N95 or N99 mask is recommended. Air quality is {category}."
                else:
                    return f"Mask is optional. Air quality is {category}, which is generally safe."
            
            return generate_recommendations(forecast_data, site, pollutant)
        else:
            return "Please run a forecast first to get personalized recommendations."
    
    # Comparison questions
    if forecast_data is not None:
        if "compare" in question_lower or "difference" in question_lower:
            avg_value = forecast_data[f"{pollutant} (µg/m³)"].mean()
            return f"The average {pollutant} level for the next 24 hours at {site} is **{avg_value:.2f} µg/m³**. This can vary significantly throughout the day due to traffic patterns, weather, and industrial activity."
    
    # How questions
    if "how" in question_lower:
        if "improve" in question_lower or "reduce" in question_lower:
            return """**Ways to reduce air pollution exposure:**
1. Stay indoors during peak pollution hours (morning and evening rush)
2. Use air purifiers with HEPA filters
3. Keep windows closed when outdoor air quality is poor
4. Use public transport or carpool
5. Plant trees around your home
6. Avoid exercising near high-traffic areas"""
    
    # Default response
    return """I can help you with:
- Understanding O3 and NO2 pollutants
- Finding the best/worst times for outdoor activities
- Health recommendations based on forecasts
- Safety advice for exercise and outdoor activities
- Ways to reduce pollution exposure

Try asking: "When is the best time for outdoor activities?" or "What health precautions should I take?" """

# ============== MAIN UI ==============
st.title("🌆 Delhi Air Quality Forecast & AI Assistant")

# Create tabs
tab1, tab2 = st.tabs(["📊 Forecast", "💬 AI Assistant"])

# Sidebar
st.sidebar.header("Configuration")
site_choice = st.sidebar.selectbox("Select Monitoring Site", SITE_NAMES)
element_choice = st.sidebar.selectbox("Select Pollutant", ELEMENT_NAMES)
forecast_hours = st.sidebar.slider("Forecast Hours Ahead", 1, 24, 24)

site_num = SITE_TO_NUM[site_choice]

# Update session state
st.session_state.current_site = site_choice
st.session_state.current_element = element_choice

# TAB 1: FORECAST
with tab1:
    st.markdown("Forecast O3 and NO2 concentrations using pre-trained GRU models with advanced feature engineering.")
    
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
        
        last_week = df_features.tail(168)
        
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
    with st.spinner("Loading model..."):
        model, scaler_X, scaler_y, feature_info, error_path = load_model_components(site_num, element_choice)

    if model is None:
        st.error(f"❌ Model components not found for Site {site_num}")
        st.write(f"**Error details:** {error_path}")
        st.stop()

    st.success(f"✅ Loaded GRU model for Site {site_num}")

    input_features = feature_info['input_features']
    sequence_length = feature_info['sequence_length']

    st.info(f"Model uses {len(input_features)} features and {sequence_length} hour sequence")

    # Run Forecast
    if st.button("🚀 Run Forecast", type="primary"):
        with st.spinner("Generating forecast..."):
            try:
                missing_features = [f for f in input_features if f not in df_features.columns]
                if missing_features:
                    st.error(f"Missing features: {missing_features}")
                    st.stop()
                
                X_input = create_sequences(df_features, input_features, sequence_length)
                
                if X_input is None:
                    st.error(f"Not enough data. Need at least {sequence_length} hours.")
                    st.stop()
                
                X_input_flat = X_input.reshape(-1, len(input_features))
                X_input_scaled_flat = scaler_X.transform(X_input_flat)
                X_input_scaled = X_input_scaled_flat.reshape(X_input.shape)
                
                y_pred_scaled = model.predict(X_input_scaled, verbose=0)
                y_pred = scaler_y.inverse_transform(y_pred_scaled)
                
                element_idx = 0 if element_choice == "O3" else 1
                prediction = y_pred[0, element_idx]
                
                forecasts = []
                current_sequence = X_input_scaled.copy()
                
                for h in range(forecast_hours):
                    y_next_scaled = model.predict(current_sequence, verbose=0)
                    y_next = scaler_y.inverse_transform(y_next_scaled)
                    forecasts.append(y_next[0, element_idx])
                    current_sequence = np.roll(current_sequence, -1, axis=1)
                
                # Store in session state
                forecast_df = pd.DataFrame({
                    "Hour": np.arange(1, forecast_hours + 1),
                    f"{element_choice} (µg/m³)": forecasts
                })
                st.session_state.forecast_data = forecast_df
                
                # Display results
                st.subheader(f"🎯 {element_choice} Forecast for {site_choice}")
                
                # Get air quality category
                avg_value = np.mean(forecasts)
                category, emoji = get_air_quality_category(avg_value, element_choice)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Next Hour", f"{forecasts[0]:.2f} µg/m³")
                col2.metric("24h Average", f"{avg_value:.2f} µg/m³")
                col3.metric("Air Quality", f"{emoji} {category}")
                
                # Plot
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(forecast_df["Hour"], forecast_df[f"{element_choice} (µg/m³)"], 
                       marker='o', linewidth=2, markersize=6, color='red')
                ax.axhline(y=avg_value, color='green', linestyle='--', 
                          label=f'Average: {avg_value:.2f}', alpha=0.7)
                ax.set_xlabel("Hours Ahead", fontsize=12)
                ax.set_ylabel(f"{element_choice} (µg/m³)", fontsize=12)
                ax.set_title(f"{element_choice} Forecast - Next {forecast_hours} Hours", fontsize=14)
                ax.legend()
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
                
                st.success(f"✅ Forecast complete! You can now ask the AI Assistant for recommendations in the next tab.")
                
            except Exception as e:
                st.error(f"Error during forecasting: {str(e)}")
                st.exception(e)

# TAB 2: AI ASSISTANT
with tab2:
    st.markdown("### 💬 AI Air Quality Assistant")
    st.markdown("Ask me questions about air quality, health recommendations, and safety advice!")
    
    # Quick action buttons
    st.markdown("**Quick Questions:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🏃 Can I exercise outside?"):
            st.session_state.chat_history.append({
                "role": "user",
                "content": "Can I exercise outside?"
            })
    with col2:
        if st.button("⏰ Best time to go out?"):
            st.session_state.chat_history.append({
                "role": "user",
                "content": "When is the best time for outdoor activities?"
            })
    with col3:
        if st.button("💊 Health recommendations?"):
            st.session_state.chat_history.append({
                "role": "user",
                "content": "What health recommendations do you have?"
            })
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**Assistant:** {message['content']}")
    
    # Chat input
    user_question = st.text_input("Ask a question:", placeholder="e.g., What precautions should I take?", key="chat_input")
    
    if st.button("Send", type="primary") or user_question:
        if user_question:
            # Add user message to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_question
            })
            
            # Generate response
            response = answer_question(
                user_question, 
                st.session_state.forecast_data,
                st.session_state.current_site,
                st.session_state.current_element
            )
            
            # Add assistant response to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response
            })
            
            # Rerun to update chat display
            st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Show current forecast summary in sidebar
    if st.session_state.forecast_data is not None:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Current Forecast")
        st.sidebar.markdown(f"**Site:** {st.session_state.current_site}")
        st.sidebar.markdown(f"**Pollutant:** {st.session_state.current_element}")
        
        avg_val = st.session_state.forecast_data[f"{st.session_state.current_element} (µg/m³)"].mean()
        category, emoji = get_air_quality_category(avg_val, st.session_state.current_element)
        st.sidebar.markdown(f"{emoji} **{category}**")
        st.sidebar.markdown(f"Avg: {avg_val:.2f} µg/m³")

# Footer
st.markdown("---")
st.markdown("**Note:** This model uses Bidirectional GRU with attention mechanism and 48-hour sequences for forecasting.")
