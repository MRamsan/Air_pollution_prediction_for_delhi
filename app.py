"""
Streamlit app with embedded HTML component for Clear Cast
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import streamlit.components.v1 as components

st.set_page_config(page_title="Clear Cast", layout="wide")

# Sample data structure - replace with your actual data loading
def load_data():
    """Load or generate sample data"""
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
        },
        'Lajpat Nagar': {
            'O3': {
                'years': list(range(2015, 2025)),
                'values': [48.5, 49.2, 50.1, 51.5, 52.8, 53.5, 54.2, 55.0, 56.1, 57.3]
            },
            'NO2': {
                'years': list(range(2015, 2025)),
                'values': [40.2, 41.1, 42.5, 43.8, 44.5, 45.2, 46.0, 47.1, 48.5, 49.8]
            }
        },
        'Alipur': {
            'O3': {
                'years': list(range(2015, 2025)),
                'values': [40.5, 41.2, 42.1, 43.5, 44.8, 45.5, 46.2, 47.0, 48.1, 49.3]
            },
            'NO2': {
                'years': list(range(2015, 2025)),
                'values': [33.2, 34.1, 35.5, 36.8, 37.5, 38.2, 39.0, 40.1, 41.5, 42.8]
            }
        },
        'Mayur Vihar': {
            'O3': {
                'years': list(range(2015, 2025)),
                'values': [46.5, 47.2, 48.1, 49.5, 50.8, 51.5, 52.2, 53.0, 54.1, 55.3]
            },
            'NO2': {
                'years': list(range(2015, 2025)),
                'values': [38.2, 39.1, 40.5, 41.8, 42.5, 43.2, 44.0, 45.1, 46.5, 47.8]
            }
        },
        'Rohini Sector 16': {
            'O3': {
                'years': list(range(2015, 2025)),
                'values': [44.5, 45.2, 46.1, 47.5, 48.8, 49.5, 50.2, 51.0, 52.1, 53.3]
            },
            'NO2': {
                'years': list(range(2015, 2025)),
                'values': [36.2, 37.1, 38.5, 39.8, 40.5, 41.2, 42.0, 43.1, 44.5, 45.8]
            }
        },
        'Yamuna Vihar': {
            'O3': {
                'years': list(range(2015, 2025)),
                'values': [47.5, 48.2, 49.1, 50.5, 51.8, 52.5, 53.2, 54.0, 55.1, 56.3]
            },
            'NO2': {
                'years': list(range(2015, 2025)),
                'values': [39.2, 40.1, 41.5, 42.8, 43.5, 44.2, 45.0, 46.1, 47.5, 48.8]
            }
        }
    }
    return stations

def simple_forecast(values, horizon):
    """Generate simple linear forecast"""
    n = len(values)
    x = np.arange(n)
    
    # Linear regression
    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, values, rcond=None)[0]
    
    # Generate predictions
    predictions = []
    for i in range(horizon):
        pred = slope * (n + i) + intercept
        noise = np.random.normal(0, np.std(values) * 0.05)
        predictions.append(max(0, pred + noise))
    
    return predictions

# Load data
data = load_data()

# HTML Component with JavaScript
html_template = """
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #2eafcc 0%, #27ae60 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .main-content {
            padding: 30px;
        }
        
        .controls {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .form-group {
            display: flex;
            flex-direction: column;
        }
        
        .form-group label {
            margin-bottom: 8px;
            color: #34495e;
            font-weight: 600;
            font-size: 0.95em;
        }
        
        .form-group select,
        .form-group input {
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 1em;
            transition: all 0.3s;
        }
        
        .form-group select:focus,
        .form-group input:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .radio-group {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }
        
        .radio-option {
            display: flex;
            align-items: center;
            padding: 12px 20px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .radio-option:hover {
            border-color: #667eea;
            background: #f8f9ff;
        }
        
        .radio-option input {
            margin-right: 10px;
        }
        
        .radio-option.selected {
            border-color: #667eea;
            background: #f8f9ff;
        }
        
        .btn-primary {
            padding: 15px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
            margin-top: 20px;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
        }
        
        .chart-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-top: 30px;
        }
        
        .alert {
            padding: 15px 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: none;
        }
        
        .alert.success {
            background: #d4edda;
            border-left: 4px solid #28a745;
            color: #155724;
        }
        
        .alert.error {
            background: #f8d7da;
            border-left: 4px solid #dc3545;
            color: #721c24;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        
        .stat-card h3 {
            font-size: 2em;
            margin-bottom: 5px;
        }
        
        .stat-card p {
            opacity: 0.9;
        }
        
        .table-container {
            overflow-x: auto;
            margin-top: 20px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 10px;
            overflow: hidden;
        }
        
        thead {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        th, td {
            padding: 15px;
            text-align: left;
        }
        
        tbody tr:nth-child(even) {
            background: #f8f9fa;
        }
        
        tbody tr:hover {
            background: #e3f2fd;
        }
        
        .btn-download {
            padding: 12px 24px;
            background: #28a745;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            margin-top: 20px;
        }
        
        .btn-download:hover {
            background: #218838;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>☁️ Clear Cast ⛈️</h1>
            <p>AI-Fueled Insights into Tomorrow's Air</p>
        </div>
        
        <div class="main-content">
            <div id="alert" class="alert"></div>
            
            <div class="controls">
                <div class="form-group">
                    <label>Select Station</label>
                    <select id="station">
                        <option value="Mukherjee Nagar">Mukherjee Nagar</option>
                        <option value="Uttam Nagar">Uttam Nagar</option>
                        <option value="Lajpat Nagar">Lajpat Nagar</option>
                        <option value="Alipur">Alipur</option>
                        <option value="Mayur Vihar">Mayur Vihar</option>
                        <option value="Rohini Sector 16">Rohini Sector 16</option>
                        <option value="Yamuna Vihar">Yamuna Vihar</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label>Select Pollutant</label>
                    <div class="radio-group">
                        <label class="radio-option selected">
                            <input type="radio" name="pollutant" value="O3" checked>
                            <span>O₃</span>
                        </label>
                        <label class="radio-option">
                            <input type="radio" name="pollutant" value="NO2">
                            <span>NO₂</span>
                        </label>
                    </div>
                </div>
                
                <div class="form-group">
                    <label>Forecast Days Ahead</label>
                    <input type="number" id="forecast_days" value="0" min="1" max="2">
                </div>
            </div>
            
            <button class="btn-primary" onclick="updateVisualization()">Run Forecast</button>
            
            <div class="stats-grid" id="stats" style="display: none;">
                <div class="stat-card">
                    <h3 id="current_value">--</h3>
                    <p>Current Level (µg/m³)</p>
                </div>
                <div class="stat-card">
                    <h3 id="forecast_avg">--</h3>
                    <p>Forecast Average</p>
                </div>
                <div class="stat-card">
                    <h3 id="trend">--</h3>
                    <p>Trend</p>
                </div>
            </div>
            
            <div class="chart-container" style="display: none;" id="chartContainer">
                <canvas id="mainChart"></canvas>
            </div>
            
            <div class="table-container" style="display: none;" id="tableContainer">
                <h3>Forecast Details</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Day</th>
                            <th>Date</th>
                            <th>Predicted Value (µg/m³)</th>
                        </tr>
                    </thead>
                    <tbody id="tableBody"></tbody>
                </table>
                <button class="btn-download" onclick="downloadCSV()">📥 Download CSV</button>
            </div>
        </div>
    </div>
    
    <script>
        // Receive data from Streamlit
        const historicalData = JSON.parse('{{DATA}}');
        let currentChart = null;
        let forecastResults = null;
        
        // Radio button handling
        document.querySelectorAll('.radio-option').forEach(option => {
            option.addEventListener('click', function() {
                document.querySelectorAll('.radio-option').forEach(o => o.classList.remove('selected'));
                this.classList.add('selected');
                this.querySelector('input[type="radio"]').checked = true;
            });
        });
        
        function showAlert(message, type) {
            const alert = document.getElementById('alert');
            alert.textContent = message;
            alert.className = `alert ${type}`;
            alert.style.display = 'block';
            setTimeout(() => alert.style.display = 'none', 5000);
        }
        
        function updateVisualization() {
            const station = document.getElementById('station').value;
            const pollutant = document.querySelector('input[name="pollutant"]:checked').value;
            const forecastDays = parseInt(document.getElementById('forecast_days').value);
            
            if (!historicalData[station] || !historicalData[station][pollutant]) {
                showAlert('No data available for selected combination', 'error');
                return;
            }
            
            const data = historicalData[station][pollutant];
            const years = data.years;
            const values = data.values;
            
            // Simple forecast
            const predictions = simpleForecast(values, forecastDays);
            const forecastYears = Array.from({length: forecastDays}, (_, i) => years[years.length - 1] + i + 1);
            
            // Store results
            forecastResults = {
                station, pollutant, forecastYears, predictions
            };
            
            // Update stats
            updateStats(values, predictions);
            
            // Update chart
            updateChart(years, values, forecastYears, predictions, pollutant);
            
            // Update table
            updateTable(forecastYears, predictions);
            
            showAlert('Forecast generated successfully!', 'success');
        }
        
        function simpleForecast(values, horizon) {
            const n = values.length;
            const predictions = [];
            
            // Calculate linear trend
            let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
            for (let i = 0; i < n; i++) {
                sumX += i;
                sumY += values[i];
                sumXY += i * values[i];
                sumX2 += i * i;
            }
            
            const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
            const intercept = (sumY - slope * sumX) / n;
            
            for (let i = 0; i < horizon; i++) {
                const trendValue = slope * (n + i) + intercept;
                const noise = (Math.random() - 0.5) * (values[n-1] * 0.03);
                predictions.push(Math.max(0, trendValue + noise));
            }
            
            return predictions;
        }
        
        function updateStats(historical, forecast) {
            document.getElementById('stats').style.display = 'grid';
            document.getElementById('current_value').textContent = historical[historical.length - 1].toFixed(1);
            
            const avg = forecast.reduce((a, b) => a + b, 0) / forecast.length;
            document.getElementById('forecast_avg').textContent = avg.toFixed(1);
            
            const trend = avg > historical[historical.length - 1] ? '↑ Rising' : '↓ Falling';
            document.getElementById('trend').textContent = trend;
        }
        
        function updateChart(histYears, histValues, foreYears, foreValues, pollutant) {
            document.getElementById('chartContainer').style.display = 'block';
            const ctx = document.getElementById('mainChart').getContext('2d');
            
            if (currentChart) currentChart.destroy();
            
            currentChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [...histYears, ...foreYears],
                    datasets: [
                        {
                            label: 'Historical',
                            data: [...histValues, ...Array(foreYears.length).fill(null)],
                            borderColor: '#3498db',
                            backgroundColor: 'rgba(52, 152, 219, 0.1)',
                            borderWidth: 3,
                            pointRadius: 5,
                            tension: 0.3
                        },
                        {
                            label: 'Forecast',
                            data: [...Array(histYears.length).fill(null), histValues[histValues.length - 1], ...foreValues],
                            borderColor: '#e74c3c',
                            backgroundColor: 'rgba(231, 76, 60, 0.1)',
                            borderWidth: 3,
                            borderDash: [10, 5],
                            pointRadius: 5,
                            tension: 0.3
                        }
                    ]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { display: true },
                        title: {
                            display: true,
                            text: `${pollutant} Levels Forecast`,
                            font: { size: 18, weight: 'bold' }
                        }
                    },
                    scales: {
                        x: { title: { display: true, text: 'Year' } },
                        y: { title: { display: true, text: 'Concentration (µg/m³)' } }
                    }
                }
            });
        }
        
        function updateTable(years, values) {
            document.getElementById('tableContainer').style.display = 'block';
            const tbody = document.getElementById('tableBody');
            tbody.innerHTML = '';
            
            years.forEach((year, idx) => {
                const row = `<tr>
                    <td>Day ${idx + 1}</td>
                    <td>Year ${year}</td>
                    <td>${values[idx].toFixed(2)}</td>
                </tr>`;
                tbody.innerHTML += row;
            });
        }
        
        function downloadCSV() {
            if (!forecastResults) return;
            
            let csv = 'Year,Predicted_Value\\n';
            forecastResults.forecastYears.forEach((year, idx) => {
                csv += `${year},${forecastResults.predictions[idx].toFixed(2)}\\n`;
            });
            
            const blob = new Blob([csv], { type: 'text/csv' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `forecast_${forecastResults.station}_${forecastResults.pollutant}.csv`;
            a.click();
            URL.revokeObjectURL(url);
        }
    </script>
</body>
</html>
"""

# Convert data to JSON for JavaScript
data_json = json.dumps(data)

# Replace placeholder with actual data
html_code = html_template.replace('{{DATA}}', data_json)

# Display the component
components.html(html_code, height=1200, scrolling=True)

# Add download button in Streamlit sidebar for data
with st.sidebar:
    st.markdown("### 📊 Data Management")
    
    if st.button("Export All Data as JSON"):
        st.download_button(
            label="Download JSON",
            data=data_json,
            file_name="air_quality_data.json",
            mime="application/json"
        )
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("""
    **Clear Cast** uses LSTM models to forecast air quality levels.
    
    Select a station and pollutant to view predictions.
    """)
    # models.py
"""
Model loader + prediction helpers for pollution forecasting.
Expect model files in ./models/ with names:
    <site>_<model>_<pollutant>.<ext>
Examples:
    mukherjee_nagar_gru_o3.h5
    mukherjee_nagar_randomforest_no2.pkl

Model loading rules:
 - GRU, SLTM -> Keras .h5
 - RandomForest, XGBoost -> joblib/pickle .pkl

If model file missing, this returns a deterministic dummy fallback 24-hour forecast.
Replace prepare_features_for(...) with your real feature engineering.
"""

import os
import numpy as np
import pandas as pd
import joblib
from typing import Tuple
from datetime import datetime, timedelta

# Optional import for Keras models
try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

def safe_name(s: str) -> str:
    """normalise site or other names into filename-friendly strings"""
    return str(s).strip().lower().replace(" ", "_").replace("-", "_")

def model_filename(site: str, model_name: str, pollutant: str) -> str:
    site_f = safe_name(site)
    model_f = model_name.strip().lower()
    pollutant_f = pollutant.strip().lower()
    # choose extension depending on model
    if model_f in ("gru", "sltm"):
        ext = "h5"
    else:
        ext = "pkl"
    return f"{site_f}_{model_f}_{pollutant_f}.{ext}"

def load_model_file(site: str, model_name: str, pollutant: str):
    fname = model_filename(site, model_name, pollutant)
    path = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(path):
        return None, path
    model_key = model_name.strip().lower()
    try:
        if model_key in ("gru", "sltm"):
            if load_model is None:
                raise RuntimeError("TensorFlow/Keras not available to load .h5 models.")
            m = load_model(path, compile=False)
            return m, path
        else:
            m = joblib.load(path)
            return m, path
    except Exception as e:
        # on any load error return None
        print(f"Error loading {path}: {e}")
        return None, path

def make_dummy_prediction(site: str, pollutant: str, base: float = 50.0, seed: int = 0) -> np.ndarray:
    """Return deterministic 24-hour fallback values (no negative)."""
    rng = np.random.RandomState(abs(hash(site + pollutant)) % 2**32 + seed)
    hours = np.arange(24)
    trend = (np.sin(hours / 24 * 2 * np.pi) * 5.0)  # diurnal pattern
    vals = base + trend + rng.normal(scale=3.0, size=24)
    return np.clip(vals, 0.0, None)

def prepare_features_for(site: str):
    """
    Placeholder feature builder for each of next 24 hours.
    Real pipeline should use past observations + meteorology + time features.
    We'll return a 24 x F numpy array where F=2 (hour_norm, site_idx).
    """
    hours = np.arange(24)
    hour_norm = (hours / 23.0).reshape(-1, 1)
    site_idx = (abs(hash(site)) % 100) / 100.0
    site_col = np.full((24, 1), site_idx)
    X = np.hstack([hour_norm, site_col])  # shape (24,2)
    return X

def predict_with_model_obj(model_obj, X: np.ndarray) -> np.ndarray:
    """
    Attempt to get predictions from model_obj for 24 rows in X.
    Handles: sklearn-like models (predict), Keras models (predict).
    If model_obj is None, raise ValueError.
    """
    if model_obj is None:
        raise ValueError("model_obj is None")
    try:
        # Keras models sometimes expect 3D shape (samples, timesteps, features)
        import numpy as _np
        if hasattr(model_obj, "predict") and len(X.shape) == 2:
            # Try sklearn-style predict
            y = model_obj.predict(X)
            return np.ravel(y)[:24]
    except Exception:
        pass

    # fallback: try keras predict with reshape
    try:
        # reshape to (samples, timesteps=1, features)
        X3 = X.reshape((X.shape[0], 1, X.shape[1]))
        y = model_obj.predict(X3, verbose=0)
        return np.ravel(y)[:24]
    except Exception as e:
        raise RuntimeError(f"Model predict failed: {e}")

def predict_24h(site: str, model_name: str) -> Tuple[pd.DataFrame, dict]:
    """
    Return DataFrame indexed by next 24 hourly timestamps with columns ['o3','no2'].
    Also returns a dict of model_paths used for each pollutant.
    """
    pollutants = ["o3", "no2"]
    preds = {}
    model_paths = {}
    X = prepare_features_for(site)  # shape (24, F)
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    idx = [now + timedelta(hours=i+1) for i in range(24)]

    for p in pollutants:
        model_obj, path = load_model_file(site, model_name, p)
        model_paths[p] = path
        if model_obj is None:
            # fallback dummy — base different for pollutants
            base = 60.0 if p == "o3" else 30.0
            preds[p] = make_dummy_prediction(site, p, base=base, seed=1)
        else:
            try:
                y = predict_with_model_obj(model_obj, X)
                # ensure length 24
                if len(y) < 24:
                    y = np.pad(y, (0, 24 - len(y)), "edge")
                preds[p] = np.clip(np.array(y[:24], dtype=float), 0.0, None)
            except Exception as e:
                print(f"Prediction failed for {path}: {e}")
                base = 60.0 if p == "o3" else 30.0
                preds[p] = make_dummy_prediction(site, p, base=base, seed=2)

    df = pd.DataFrame(preds, index=pd.to_datetime(idx))
    df.index.name = "timestamp"
    return df, model_paths

