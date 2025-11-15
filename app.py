"""
Streamlit app: Clear Cast
- Reads data from ./data
- Loads models from ./model
- If model exists for pollutant -> use it for prediction
- Otherwise fall back to simple linear forecast
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import glob
import io
import joblib
import traceback
import streamlit.components.v1 as components

# Optional: keras is used only if user has keras models (.h5)
try:
    from tensorflow.keras.models import load_model as keras_load_model
    KERAS_AVAILABLE = True
except Exception:
    KERAS_AVAILABLE = False

st.set_page_config(page_title="Clear Cast", layout="wide")

ROOT = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "model")

# ---------- Utilities: file discovery and loaders ----------

def find_data_files(data_dir=DATA_DIR):
    """Return list of CSV/Excel files in data folder."""
    patterns = [os.path.join(data_dir, "*.csv"), os.path.join(data_dir, "*.xlsx"), os.path.join(data_dir, "*.xls")]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    return files

def load_tabular_file(path):
    """Try to load CSV/Excel into DataFrame. Returns None on failure."""
    try:
        if path.lower().endswith(".csv"):
            return pd.read_csv(path, parse_dates=True, low_memory=False)
        else:
            return pd.read_excel(path, engine="openpyxl")
    except Exception:
        try:
            # fallback - read without parsing
            return pd.read_csv(path, low_memory=False)
        except Exception:
            return None

def standardize_station_data(df, filename_hint=None):
    """
    Try to produce a dictionary mapping pollutant -> {'years': [..], 'values': [..]}
    Supported common forms:
      - df with columns: 'date' (or 'year'), 'O3', 'NO2', ...
      - df where first column is 'Year' and other columns pollutants
      - df with monthly/daily dates: we'll aggregate to yearly mean
    Returns dict or None on failure.
    """
    df_orig = df.copy()
    df = df.copy()
    # find a date-like column
    date_col = None
    for c in df.columns:
        if "date" in c.lower() or "day" in c.lower():
            date_col = c
            break
        if c.lower() in ("year", "yr"):
            date_col = c
            break

    # pollutants candidate columns: typical names
    pollutant_cols = [c for c in df.columns if c.lower() in ("o3", "no2", "pm2.5", "pm25", "pm10", "so2", "co")]
    # fallback: any numeric columns other than index/date
    if not pollutant_cols:
        pollutant_cols = [c for c in df.select_dtypes(include=[np.number]).columns]

    if date_col:
        # if date_col is year-like numeric, use directly
        if df[date_col].dtype in [np.int64, np.float64] or df[date_col].astype(str).str.len().max() <= 4:
            # treat as year
            df['__year__'] = df[date_col].astype(int)
        else:
            # try parse datetime
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df['__year__'] = df[date_col].dt.year
            except Exception:
                df['__year__'] = pd.to_datetime(df[date_col], errors='coerce').dt.year

        if '__year__' not in df.columns:
            return None

        grouped = df.groupby('__year__')
        output = {}
        for p in pollutant_cols:
            vals = grouped[p].mean().dropna()
            if vals.empty:
                continue
            years = [int(y) for y in vals.index.tolist()]
            values = [float(v) for v in vals.values.tolist()]
            output[p if isinstance(p, str) else str(p)] = {"years": years, "values": values}
        return output if output else None
    else:
        # If no date column, check if first column is Year
        first_col = df.columns[0]
        if first_col.lower() in ("year", "yr"):
            output = {}
            for p in pollutant_cols:
                vals = df[[first_col, p]].dropna()
                years = vals[first_col].astype(int).tolist()
                values = vals[p].astype(float).tolist()
                output[p] = {"years": years, "values": values}
            return output if output else None

    # as last fallback, if dataframe has two columns (year, value)
    if df.shape[1] == 2:
        col1, col2 = df.columns[:2]
        try:
            years = df[col1].astype(int).tolist()
            values = df[col2].astype(float).tolist()
            key = filename_hint or "series"
            return {key: {"years": years, "values": values}}
        except Exception:
            return None

    return None

def build_data_from_folder(data_dir=DATA_DIR):
    """
    Explore data files and produce a dict with top-level keys = station names.
    Each station maps to pollutants -> {'years': [...], 'values': [...]}
    Heuristics:
      - If file name contains station name: use that
      - If file contains columns O3/NO2 etc: use them
    """
    files = find_data_files(data_dir)
    stations = {}
    for f in files:
        df = load_tabular_file(f)
        if df is None:
            continue
        fname = os.path.splitext(os.path.basename(f))[0]
        standardized = standardize_station_data(df, filename_hint=fname)
        if standardized:
            # if filename looks like "MukherjeeNagar_O3" try to parse station/pollutant
            # We'll treat fname as station if there are multiple pollutants, otherwise if single series use fname/pollutant
            station_name = fname.replace("_", " ").strip()
            # if file contains explicit 'station' column, prefer that (first unique)
            station_col = None
            for c in df.columns:
                if "station" in c.lower():
                    station_col = c
                    break
            if station_col:
                possible = df[station_col].dropna().astype(str).unique()
                if possible.size > 0:
                    station_name = possible[0]

            # merge into stations dict
            if station_name not in stations:
                stations[station_name] = {}
            # standardized keys are pollutant names; if key collides, suffix with filename
            for p, series in standardized.items():
                key = p
                # normalize pollutant names: e.g. 'o3' -> 'O3'
                if isinstance(key, str):
                    key_norm = key.upper().replace(".", "").replace(" ", "")
                else:
                    key_norm = str(key)
                # keep only O3/NO2 canonical for your use-case, but keep others too
                stations[station_name][key_norm] = series
    return stations

# ---------- Model loading and prediction helpers ----------

def find_model_for(pollutant, model_dir=MODEL_DIR):
    """
    Search model_dir for a model file that mentions pollutant.
    Supports .h5 (keras), .pkl/.joblib (sklearn), or generic model files.
    Returns path or None.
    """
    if not os.path.exists(model_dir):
        return None
    patterns = [
        os.path.join(model_dir, f"*{pollutant}*.h5"),
        os.path.join(model_dir, f"*{pollutant}*.keras"),
        os.path.join(model_dir, f"*{pollutant}*.pkl"),
        os.path.join(model_dir, f"*{pollutant}*.joblib"),
        os.path.join(model_dir, f"*{pollutant}*.*"),
    ]
    for p in patterns:
        matches = glob.glob(p)
        if matches:
            # prefer keras if available
            for m in matches:
                if m.endswith(".h5") and KERAS_AVAILABLE:
                    return m
            return matches[0]
    # try any model if nothing matched
    all_models = glob.glob(os.path.join(model_dir, "*"))
    return all_models[0] if all_models else None

def load_model_and_optional_scaler(model_path):
    """
    Load model and any scaler found alongside (same basename + '_scaler.*').
    Returns (model_object, scaler_object_or_None, format_str)
    """
    scaler = None
    fmt = None
    if model_path.endswith(".h5") or model_path.endswith(".keras"):
        fmt = "keras"
        try:
            model = keras_load_model(model_path)
            # look for scaler same basename
            base = os.path.splitext(model_path)[0]
            for ext in (".pkl", ".joblib"):
                sfile = base + "_scaler" + ext
                if os.path.exists(sfile):
                    scaler = joblib.load(sfile)
                    break
            return model, scaler, fmt
        except Exception:
            return None, None, fmt
    elif model_path.endswith(".pkl") or model_path.endswith(".joblib"):
        fmt = "sklearn"
        try:
            model = joblib.load(model_path)
            base = os.path.splitext(model_path)[0]
            for ext in (".pkl", ".joblib"):
                sfile = base + "_scaler" + ext
                if os.path.exists(sfile):
                    scaler = joblib.load(sfile)
                    break
            return model, scaler, fmt
        except Exception:
            return None, None, fmt
    else:
        # try to load as joblib/pickle
        try:
            model = joblib.load(model_path)
            return model, None, "sklearn"
        except Exception:
            return None, None, None

def model_predict(model, model_format, scaler, history_values, horizon=1, lookback=12):
    """
    Generic wrapper to produce forecasts using the provided model.
    - For keras LSTM expect shape (1, lookback, 1) and that the model outputs horizon predictions or 1-step iteratively.
    - For sklearn, attempt direct prediction on features created from last lookback values (flattened).
    If anything fails, return None.
    """
    try:
        arr = np.array(history_values, dtype=float)
        if len(arr) < 2:
            return None
        # If scaler present, use it on the series
        if scaler is not None:
            arr_scaled = scaler.transform(arr.reshape(-1, 1)).flatten()
        else:
            arr_scaled = arr.copy()

        # Keras LSTM
        if model_format == "keras" and KERAS_AVAILABLE:
            # If model expects (batch, time_steps, features)
            preds = []
            # If model outputs multi-step directly
            try:
                last_seq = arr_scaled[-lookback:].reshape((1, min(lookback, len(arr_scaled)), 1))
                # pad if needed
                if last_seq.shape[1] < lookback:
                    pad = np.zeros((1, lookback - last_seq.shape[1], 1))
                    last_seq = np.concatenate([pad, last_seq], axis=1)
                out = model.predict(last_seq, verbose=0)
                out = np.array(out).flatten()
                # if scaler present, inverse transform
                if scaler is not None:
                    inv = scaler.inverse_transform(out.reshape(-1, 1)).flatten()
                else:
                    inv = out
                return [float(x) for x in inv[:horizon]]
            except Exception:
                # iterative prediction
                seq = list(arr_scaled[-lookback:])
                for _ in range(horizon):
                    inp = np.array(seq[-lookback:]).reshape((1, lookback, 1))
                    out = model.predict(inp, verbose=0).flatten()
                    next_val = out[-1] if out.size > 0 else out[0]
                    seq.append(next_val)
                predicted = seq[-horizon:]
                if scaler is not None:
                    predicted = scaler.inverse_transform(np.array(predicted).reshape(-1, 1)).flatten()
                return [float(x) for x in predicted]

        # sklearn / classical estimator
        elif model_format == "sklearn":
            # build simple feature: last `lookback` values flattened (pad with last if short)
            if len(arr_scaled) < lookback:
                padded = np.concatenate([np.full((lookback - len(arr_scaled),), arr_scaled[-1]), arr_scaled])
            else:
                padded = arr_scaled[-lookback:]
            X = padded.reshape(1, -1)
            try:
                out = model.predict(X)
                # If model returns multiple horizon steps as array-like
                out = np.array(out).flatten()
                if scaler is not None:
                    out = scaler.inverse_transform(out.reshape(-1, 1)).flatten()
                return [float(x) for x in out[:horizon]]
            except Exception:
                # maybe model supports predict_proba? or only single-step
                single = model.predict(X)
                val = float(single.flatten()[0])
                # naive: repeat the single-step prediction for horizon
                return [val] * horizon
        else:
            return None
    except Exception:
        # unexpected error in model prediction
        st.write("Model prediction error:", traceback.format_exc())
        return None

# ---------- Simple linear forecast fallback ----------

def simple_forecast(values, horizon):
    """Simple linear forecast (same idea as your sample)"""
    values = np.array(values, dtype=float)
    n = len(values)
    if n < 2:
        return [float(values[-1]) if n >= 1 else 0.0] * horizon
    x = np.arange(n)
    A = np.vstack([x, np.ones(n)]).T
    slope, intercept = np.linalg.lstsq(A, values, rcond=None)[0]
    preds = []
    for i in range(horizon):
        pred = slope * (n + i) + intercept
        noise = np.random.normal(0, np.std(values) * 0.05)
        preds.append(float(max(0.0, pred + noise)))
    return preds

# ---------- Build dataset used by the HTML component ----------

@st.cache_data(ttl=300)
def load_data_and_models():
    stations = build_data_from_folder(DATA_DIR)
    # If no data files found, return None so UI can display fallback.
    return stations

# Load data
with st.spinner("Loading data from ./data ..."):
    data = load_data_and_models()

# If no data found, create an example fallback like your original sample
if not data:
    st.warning("No data files found in ./data folder (or files couldn't be parsed). Using built-in sample data. "
               "Place station CSVs (or Excel) in the data/ folder. See README for expected formats.")
    # small sample (same as your template) to ensure UI still works
    def _sample():
        stations = {
            'Mukherjee Nagar': {
                'O3': {
                    'years': list(range(2015, 2025)),
                    'values': [45.2, 47.1, 46.8, 48.5, 49.2, 50.1, 51.3, 52.0, 53.5, 54.2]
                },
                'NO2': {
                    'years': list(range(2015, 2025)),
                    'values': [38.5, 39.2, 40.1, 41.5, 42.8, 43.5, 44.2, 45.0, 46.1, 47.3]
                }
            },
            'Uttam Nagar': {
                'O3': {
                    'years': list(range(2015, 2025)),
                    'values': [42.1, 43.5, 44.2, 45.8, 46.5, 47.2, 48.1, 49.0, 50.2, 51.5]
                },
                'NO2': {
                    'years': list(range(2015, 2025)),
                    'values': [35.2, 36.1, 37.5, 38.8, 39.5, 40.2, 41.0, 42.1, 43.5, 44.8]
                }
            }
        }
        return stations
    data = _sample()

# Sidebar data export and info
with st.sidebar:
    st.markdown("### 📊 Data Management")
    if st.button("Export All Data as JSON"):
        st.download_button(
            label="Download JSON",
            data=json.dumps(data, indent=2),
            file_name="air_quality_data.json",
            mime="application/json"
        )
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("""
    **Clear Cast** loads station data from ./data and models from ./model.
    
    - Data files: CSV or Excel. The app will try to detect 'date'/'year' and numeric pollutant columns (O3, NO2, ...).
    - Models: name a model file containing pollutant name (e.g. 'model_O3.h5' or 'model_NO2.pkl').
    - If a model + scaler are present, they'll be used; otherwise a linear fallback forecast is used.
    """)

# Main UI controls (we'll embed the Chart.js HTML component below)
stations_list = sorted(list(data.keys()))
selected_station = st.selectbox("Select Station", stations_list)

# available pollutants at the station
available_pollutants = sorted(list(data[selected_station].keys()))
if not available_pollutants:
    st.error("No pollutant series found for selected station.")
    st.stop()

selected_pollutant = st.radio("Select Pollutant", available_pollutants, horizontal=True)

forecast_days = st.number_input("Forecast Years Ahead", min_value=1, max_value=5, value=1, step=1)

# Run prediction
if st.button("Run Forecast"):
    # prepare historical series
    series = data[selected_station].get(selected_pollutant)
    if not series:
        st.error("No historical data for selected station/pollutant.")
    else:
        years = series['years']
        values = series['values']
        # attempt to find model
        model_path = find_model_for(selected_pollutant, MODEL_DIR)
        model_used = None
        predictions = None
        model_info_msg = ""
        if model_path:
            st.info(f"Attempting to load model: `{os.path.basename(model_path)}`")
            model_obj, scaler_obj, fmt = load_model_and_optional_scaler(model_path)
            if model_obj is not None:
                model_used = os.path.basename(model_path)
                try:
                    preds = model_predict(model_obj, fmt, scaler_obj, values, horizon=forecast_days)
                    if preds is not None:
                        predictions = preds
                        model_info_msg = f"Used model `{os.path.basename(model_path)}` ({fmt})"
                except Exception:
                    st.warning("Model loaded but failed to predict; falling back to linear forecast.")
            else:
                st.warning("Found a model file but couldn't load it (incompatible format). Falling back to linear forecast.")

        if predictions is None:
            # fallback
            predictions = simple_forecast(values, horizon=forecast_days)
            model_info_msg = "Used fallback linear forecast."

        # Build forecast years (assuming data years are consecutive-ish)
        last_year = max(years) if years else pd.Timestamp.now().year
        forecast_years = [int(last_year + i + 1) for i in range(forecast_days)]

        # Prepare data JSON for embedding in the HTML component
        # convert pollutant label to canonical form like "O3"
        hist = {"years": years, "values": values}
        station_payload = {selected_station: {selected_pollutant: hist}}
        # We'll embed only the selected station/pollutant to keep HTML light
        payload_json = json.dumps({selected_station: {selected_pollutant: hist}})

        # Build JS-friendly version of predictions and HTML (we reuse your provided HTML template but substitute the data)
        # Use the same HTML template you provided, but replace the 'DATA' placeholder with payload_json.
        # Keep height flexible.
        html_template = r"""
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>/* same styles as before - trimmed for brevity in template*/ body{font-family:Segoe UI, Tahoma, Geneva, Verdana,sans-serif;padding:12px;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%)}.container{max-width:1200px;margin:0 auto;background:#fff;border-radius:12px;padding:20px} .header{color:#fff;background:linear-gradient(135deg,#2eafcc 0%,#27ae60 100%);padding:20px;border-radius:8px;text-align:center}.chart-container{padding:10px} table{width:100%;border-collapse:collapse} th,td{padding:8px;text-align:left}</style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>☁️ Clear Cast ⛈️</h2>
            <p>{station} — {pollutant}</p>
        </div>

        <div style="margin-top:15px;">
            <div class="chart-container">
                <canvas id="mainChart"></canvas>
            </div>

            <div style="margin-top:16px;">
                <h3>Forecast Details</h3>
                <table border="1">
                    <thead><tr><th>Index</th><th>Year</th><th>Predicted Value (µg/m³)</th></tr></thead>
                    <tbody id="tableBody"></tbody>
                </table>
            </div>
        </div>
    </div>

<script>
const historicalData = JSON.parse('{{DATA}}');
const station = "{{STATION}}";
const pollutant = "{{POLLUTANT}}";
const hist = historicalData[station][pollutant];
const histYears = hist.years;
const histValues = hist.values;
const forecastYears = {{F_YEARS}};
const predictions = {{PREDICTIONS}};

function renderChart() {
    const ctx = document.getElementById('mainChart').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: [...histYears, ...forecastYears],
            datasets: [
                {
                    label: 'Historical',
                    data: [...histValues, ...Array(forecastYears.length).fill(null)],
                    borderWidth: 2,
                    tension: 0.25
                },
                {
                    label: 'Forecast',
                    data: [...Array(histYears.length).fill(null), histValues[histValues.length-1], ...predictions],
                    borderDash: [6,4],
                    borderWidth: 2,
                    tension: 0.25
                }
            ]
        },
        options: {
            responsive: true,
            plugins: { title: { display: true, text: pollutant + " Levels Forecast" } },
            scales: { x: { title: { display: true, text: 'Year' } }, y: { title: { display: true, text: 'Concentration (µg/m³)' } } }
        }
    });
}

function populateTable() {
    const tbody = document.getElementById('tableBody');
    tbody.innerHTML = "";
    for (let i = 0; i < forecastYears.length; i++){
        const row = `<tr><td>${i+1}</td><td>${forecastYears[i]}</td><td>${predictions[i].toFixed(2)}</td></tr>`;
        tbody.innerHTML += row;
    }
}

renderChart();
populateTable();
</script>
</body>
</html>
"""
        # Format replacements
        f_html = html_template.replace("{{DATA}}", payload_json.replace("\\", "\\\\"))
        f_html = f_html.replace("{{STATION}}", selected_station.replace('"', '\\"'))
        f_html = f_html.replace("{{POLLUTANT}}", selected_pollutant.replace('"', '\\"'))
        f_html = f_html.replace("{{F_YEARS}}", json.dumps(forecast_years))
        f_html = f_html.replace("{{PREDICTIONS}}", json.dumps(predictions))

        # show model info and stats in Streamlit
        st.success(model_info_msg)
        col1, col2, col3 = st.columns(3)
        col1.metric("Current (last)", f"{values[-1]:.2f} µg/m³")
        col2.metric("Forecast avg", f"{(sum(predictions)/len(predictions)):.2f} µg/m³")
        trend = "Rising ↑" if (sum(predictions)/len(predictions)) > values[-1] else "Falling ↓"
        col3.metric("Trend", trend)

        # show the component (height tuned)
        components.html(f_html, height=800, scrolling=True)

        # add CSV download
        csv_buf = io.StringIO()
        csv_buf.write("Year,Predicted_Value\n")
        for y, p in zip(forecast_years, predictions):
            csv_buf.write(f"{y},{p:.4f}\n")
        csv_data = csv_buf.getvalue()
        st.download_button("📥 Download Forecast CSV", data=csv_data, file_name=f"forecast_{selected_station}_{selected_pollutant}.csv", mime="text/csv")

else:
    st.markdown("### Instructions")
    st.markdown("""
    1. Put station data files (CSV or Excel) in the `data/` folder.
       - The app will try to detect `date`/`year` columns and numeric pollutant columns like `O3`, `NO2`.
       - Acceptable formats:
         * `date,O3,NO2` (daily/monthly) — app will aggregate to yearly mean
         * `Year,O3,NO2` — yearly series
         * `Year,Value` — single pollutant series (filename or inferred pollutant name used)
    2. Put trained models in `model/`. Name models to include pollutant (e.g. `model_O3.h5`, `model_NO2.pkl`).
       - Supported model types: Keras `.h5` (requires tensorflow installed), sklearn joblib/pickle `.pkl`/`.joblib`.
       - Optional: a scaler file next to the model named `<model_basename>_scaler.pkl` will be loaded automatically.
    3. Click **Run Forecast** to generate predictions and visualize.
    """)
