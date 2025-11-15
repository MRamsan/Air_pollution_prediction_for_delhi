# """
# Streamlit app with embedded HTML component for Clear Cast
# Run with: streamlit run app.py
# """

# import streamlit as st
# import pandas as pd
# import numpy as np
# import json
# import streamlit.components.v1 as components

# st.set_page_config(page_title="Clear Cast", layout="wide")

# # Sample data structure - replace with your actual data loading
# def load_data():
#     """Load or generate sample data"""
#     stations = {
#         'Mukherjee Nagar': {
#             'O3': {
#                 'years': list(range(2015, 2025)),
#                 'values': [45.2, 47.1, 46.8, 48.5, 49.2, 50.1, 51.3, 52.0, 53.5, 54.2]
#             },
#             'NO2': {
#                 'years': list(range(2015, 2025)),
#                 'values': [38.5, 39.2, 40.1, 41.5, 42.8, 43.5, 44.2, 45.0, 46.1, 47.3]
#             }
#         },
#         'Uttam Nagar': {
#             'O3': {
#                 'years': list(range(2015, 2025)),
#                 'values': [42.1, 43.5, 44.2, 45.8, 46.5, 47.2, 48.1, 49.0, 50.2, 51.5]
#             },
#             'NO2': {
#                 'years': list(range(2015, 2025)),
#                 'values': [35.2, 36.1, 37.5, 38.8, 39.5, 40.2, 41.0, 42.1, 43.5, 44.8]
#             }
#         },
#         'Lajpat Nagar': {
#             'O3': {
#                 'years': list(range(2015, 2025)),
#                 'values': [48.5, 49.2, 50.1, 51.5, 52.8, 53.5, 54.2, 55.0, 56.1, 57.3]
#             },
#             'NO2': {
#                 'years': list(range(2015, 2025)),
#                 'values': [40.2, 41.1, 42.5, 43.8, 44.5, 45.2, 46.0, 47.1, 48.5, 49.8]
#             }
#         },
#         'Alipur': {
#             'O3': {
#                 'years': list(range(2015, 2025)),
#                 'values': [40.5, 41.2, 42.1, 43.5, 44.8, 45.5, 46.2, 47.0, 48.1, 49.3]
#             },
#             'NO2': {
#                 'years': list(range(2015, 2025)),
#                 'values': [33.2, 34.1, 35.5, 36.8, 37.5, 38.2, 39.0, 40.1, 41.5, 42.8]
#             }
#         },
#         'Mayur Vihar': {
#             'O3': {
#                 'years': list(range(2015, 2025)),
#                 'values': [46.5, 47.2, 48.1, 49.5, 50.8, 51.5, 52.2, 53.0, 54.1, 55.3]
#             },
#             'NO2': {
#                 'years': list(range(2015, 2025)),
#                 'values': [38.2, 39.1, 40.5, 41.8, 42.5, 43.2, 44.0, 45.1, 46.5, 47.8]
#             }
#         },
#         'Rohini Sector 16': {
#             'O3': {
#                 'years': list(range(2015, 2025)),
#                 'values': [44.5, 45.2, 46.1, 47.5, 48.8, 49.5, 50.2, 51.0, 52.1, 53.3]
#             },
#             'NO2': {
#                 'years': list(range(2015, 2025)),
#                 'values': [36.2, 37.1, 38.5, 39.8, 40.5, 41.2, 42.0, 43.1, 44.5, 45.8]
#             }
#         },
#         'Yamuna Vihar': {
#             'O3': {
#                 'years': list(range(2015, 2025)),
#                 'values': [47.5, 48.2, 49.1, 50.5, 51.8, 52.5, 53.2, 54.0, 55.1, 56.3]
#             },
#             'NO2': {
#                 'years': list(range(2015, 2025)),
#                 'values': [39.2, 40.1, 41.5, 42.8, 43.5, 44.2, 45.0, 46.1, 47.5, 48.8]
#             }
#         }
#     }
#     return stations

# def simple_forecast(values, horizon):
#     """Generate simple linear forecast"""
#     n = len(values)
#     x = np.arange(n)
    
#     # Linear regression
#     A = np.vstack([x, np.ones(len(x))]).T
#     slope, intercept = np.linalg.lstsq(A, values, rcond=None)[0]
    
#     # Generate predictions
#     predictions = []
#     for i in range(horizon):
#         pred = slope * (n + i) + intercept
#         noise = np.random.normal(0, np.std(values) * 0.05)
#         predictions.append(max(0, pred + noise))
    
#     return predictions

# # Load data
# data = load_data()

# # HTML Component with JavaScript
# html_template = """
# <!DOCTYPE html>
# <html>
# <head>
#     <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
#     <style>
#         * {
#             margin: 0;
#             padding: 0;
#             box-sizing: border-box;
#         }
        
#         body {
#             font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#             padding: 20px;
#         }
        
#         .container {
#             max-width: 1400px;
#             margin: 0 auto;
#             background: white;
#             border-radius: 20px;
#             box-shadow: 0 20px 60px rgba(0,0,0,0.3);
#             overflow: hidden;
#         }
        
#         .header {
#             background: linear-gradient(135deg, #2eafcc 0%, #27ae60 100%);
#             color: white;
#             padding: 30px;
#             text-align: center;
#         }
        
#         .header h1 {
#             font-size: 2.5em;
#             margin-bottom: 10px;
#         }
        
#         .main-content {
#             padding: 30px;
#         }
        
#         .controls {
#             display: grid;
#             grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
#             gap: 20px;
#             margin-bottom: 30px;
#         }
        
#         .form-group {
#             display: flex;
#             flex-direction: column;
#         }
        
#         .form-group label {
#             margin-bottom: 8px;
#             color: #34495e;
#             font-weight: 600;
#             font-size: 0.95em;
#         }
        
#         .form-group select,
#         .form-group input {
#             padding: 12px;
#             border: 2px solid #e0e0e0;
#             border-radius: 8px;
#             font-size: 1em;
#             transition: all 0.3s;
#         }
        
#         .form-group select:focus,
#         .form-group input:focus {
#             outline: none;
#             border-color: #667eea;
#             box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
#         }
        
#         .radio-group {
#             display: flex;
#             gap: 15px;
#             flex-wrap: wrap;
#         }
        
#         .radio-option {
#             display: flex;
#             align-items: center;
#             padding: 12px 20px;
#             border: 2px solid #e0e0e0;
#             border-radius: 8px;
#             cursor: pointer;
#             transition: all 0.3s;
#         }
        
#         .radio-option:hover {
#             border-color: #667eea;
#             background: #f8f9ff;
#         }
        
#         .radio-option input {
#             margin-right: 10px;
#         }
        
#         .radio-option.selected {
#             border-color: #667eea;
#             background: #f8f9ff;
#         }
        
#         .btn-primary {
#             padding: 15px 30px;
#             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#             color: white;
#             border: none;
#             border-radius: 10px;
#             font-size: 1.1em;
#             font-weight: 600;
#             cursor: pointer;
#             transition: transform 0.2s;
#             margin-top: 20px;
#         }
        
#         .btn-primary:hover {
#             transform: translateY(-2px);
#             box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
#         }
        
#         .chart-container {
#             background: white;
#             border-radius: 10px;
#             padding: 20px;
#             box-shadow: 0 2px 10px rgba(0,0,0,0.1);
#             margin-top: 30px;
#         }
        
#         .alert {
#             padding: 15px 20px;
#             border-radius: 10px;
#             margin-bottom: 20px;
#             display: none;
#         }
        
#         .alert.success {
#             background: #d4edda;
#             border-left: 4px solid #28a745;
#             color: #155724;
#         }
        
#         .alert.error {
#             background: #f8d7da;
#             border-left: 4px solid #dc3545;
#             color: #721c24;
#         }
        
#         .stats-grid {
#             display: grid;
#             grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
#             gap: 20px;
#             margin: 20px 0;
#         }
        
#         .stat-card {
#             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#             color: white;
#             padding: 20px;
#             border-radius: 10px;
#             text-align: center;
#         }
        
#         .stat-card h3 {
#             font-size: 2em;
#             margin-bottom: 5px;
#         }
        
#         .stat-card p {
#             opacity: 0.9;
#         }
        
#         .table-container {
#             overflow-x: auto;
#             margin-top: 20px;
#         }
        
#         table {
#             width: 100%;
#             border-collapse: collapse;
#             background: white;
#             border-radius: 10px;
#             overflow: hidden;
#         }
        
#         thead {
#             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#             color: white;
#         }
        
#         th, td {
#             padding: 15px;
#             text-align: left;
#         }
        
#         tbody tr:nth-child(even) {
#             background: #f8f9fa;
#         }
        
#         tbody tr:hover {
#             background: #e3f2fd;
#         }
        
#         .btn-download {
#             padding: 12px 24px;
#             background: #28a745;
#             color: white;
#             border: none;
#             border-radius: 8px;
#             cursor: pointer;
#             margin-top: 20px;
#         }
        
#         .btn-download:hover {
#             background: #218838;
#         }
#     </style>
# </head>
# <body>
#     <div class="container">
#         <div class="header">
#             <h1>☁️ Clear Cast ⛈️</h1>
#             <p>AI-Fueled Insights into Tomorrow's Air</p>
#         </div>
        
#         <div class="main-content">
#             <div id="alert" class="alert"></div>
            
#             <div class="controls">
#                 <div class="form-group">
#                     <label>Select Station</label>
#                     <select id="station">
#                         <option value="Mukherjee Nagar">Mukherjee Nagar</option>
#                         <option value="Uttam Nagar">Uttam Nagar</option>
#                         <option value="Lajpat Nagar">Lajpat Nagar</option>
#                         <option value="Alipur">Alipur</option>
#                         <option value="Mayur Vihar">Mayur Vihar</option>
#                         <option value="Rohini Sector 16">Rohini Sector 16</option>
#                         <option value="Yamuna Vihar">Yamuna Vihar</option>
#                     </select>
#                 </div>
                
#                 <div class="form-group">
#                     <label>Select Pollutant</label>
#                     <div class="radio-group">
#                         <label class="radio-option selected">
#                             <input type="radio" name="pollutant" value="O3" checked>
#                             <span>O₃</span>
#                         </label>
#                         <label class="radio-option">
#                             <input type="radio" name="pollutant" value="NO2">
#                             <span>NO₂</span>
#                         </label>
#                     </div>
#                 </div>
                
#                 <div class="form-group">
#                     <label>Forecast Days Ahead</label>
#                     <input type="number" id="forecast_days" value="0" min="1" max="2">
#                 </div>
#             </div>
            
#             <button class="btn-primary" onclick="updateVisualization()">Run Forecast</button>
            
#             <div class="stats-grid" id="stats" style="display: none;">
#                 <div class="stat-card">
#                     <h3 id="current_value">--</h3>
#                     <p>Current Level (µg/m³)</p>
#                 </div>
#                 <div class="stat-card">
#                     <h3 id="forecast_avg">--</h3>
#                     <p>Forecast Average</p>
#                 </div>
#                 <div class="stat-card">
#                     <h3 id="trend">--</h3>
#                     <p>Trend</p>
#                 </div>
#             </div>
            
#             <div class="chart-container" style="display: none;" id="chartContainer">
#                 <canvas id="mainChart"></canvas>
#             </div>
            
#             <div class="table-container" style="display: none;" id="tableContainer">
#                 <h3>Forecast Details</h3>
#                 <table>
#                     <thead>
#                         <tr>
#                             <th>Day</th>
#                             <th>Date</th>
#                             <th>Predicted Value (µg/m³)</th>
#                         </tr>
#                     </thead>
#                     <tbody id="tableBody"></tbody>
#                 </table>
#                 <button class="btn-download" onclick="downloadCSV()">📥 Download CSV</button>
#             </div>
#         </div>
#     </div>
    
#     <script>
#         // Receive data from Streamlit
#         const historicalData = JSON.parse('{{DATA}}');
#         let currentChart = null;
#         let forecastResults = null;
        
#         // Radio button handling
#         document.querySelectorAll('.radio-option').forEach(option => {
#             option.addEventListener('click', function() {
#                 document.querySelectorAll('.radio-option').forEach(o => o.classList.remove('selected'));
#                 this.classList.add('selected');
#                 this.querySelector('input[type="radio"]').checked = true;
#             });
#         });
        
#         function showAlert(message, type) {
#             const alert = document.getElementById('alert');
#             alert.textContent = message;
#             alert.className = `alert ${type}`;
#             alert.style.display = 'block';
#             setTimeout(() => alert.style.display = 'none', 5000);
#         }
        
#         function updateVisualization() {
#             const station = document.getElementById('station').value;
#             const pollutant = document.querySelector('input[name="pollutant"]:checked').value;
#             const forecastDays = parseInt(document.getElementById('forecast_days').value);
            
#             if (!historicalData[station] || !historicalData[station][pollutant]) {
#                 showAlert('No data available for selected combination', 'error');
#                 return;
#             }
            
#             const data = historicalData[station][pollutant];
#             const years = data.years;
#             const values = data.values;
            
#             // Simple forecast
#             const predictions = simpleForecast(values, forecastDays);
#             const forecastYears = Array.from({length: forecastDays}, (_, i) => years[years.length - 1] + i + 1);
            
#             // Store results
#             forecastResults = {
#                 station, pollutant, forecastYears, predictions
#             };
            
#             // Update stats
#             updateStats(values, predictions);
            
#             // Update chart
#             updateChart(years, values, forecastYears, predictions, pollutant);
            
#             // Update table
#             updateTable(forecastYears, predictions);
            
#             showAlert('Forecast generated successfully!', 'success');
#         }
        
#         function simpleForecast(values, horizon) {
#             const n = values.length;
#             const predictions = [];
            
#             // Calculate linear trend
#             let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
#             for (let i = 0; i < n; i++) {
#                 sumX += i;
#                 sumY += values[i];
#                 sumXY += i * values[i];
#                 sumX2 += i * i;
#             }
            
#             const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
#             const intercept = (sumY - slope * sumX) / n;
            
#             for (let i = 0; i < horizon; i++) {
#                 const trendValue = slope * (n + i) + intercept;
#                 const noise = (Math.random() - 0.5) * (values[n-1] * 0.03);
#                 predictions.push(Math.max(0, trendValue + noise));
#             }
            
#             return predictions;
#         }
        
#         function updateStats(historical, forecast) {
#             document.getElementById('stats').style.display = 'grid';
#             document.getElementById('current_value').textContent = historical[historical.length - 1].toFixed(1);
            
#             const avg = forecast.reduce((a, b) => a + b, 0) / forecast.length;
#             document.getElementById('forecast_avg').textContent = avg.toFixed(1);
            
#             const trend = avg > historical[historical.length - 1] ? '↑ Rising' : '↓ Falling';
#             document.getElementById('trend').textContent = trend;
#         }
        
#         function updateChart(histYears, histValues, foreYears, foreValues, pollutant) {
#             document.getElementById('chartContainer').style.display = 'block';
#             const ctx = document.getElementById('mainChart').getContext('2d');
            
#             if (currentChart) currentChart.destroy();
            
#             currentChart = new Chart(ctx, {
#                 type: 'line',
#                 data: {
#                     labels: [...histYears, ...foreYears],
#                     datasets: [
#                         {
#                             label: 'Historical',
#                             data: [...histValues, ...Array(foreYears.length).fill(null)],
#                             borderColor: '#3498db',
#                             backgroundColor: 'rgba(52, 152, 219, 0.1)',
#                             borderWidth: 3,
#                             pointRadius: 5,
#                             tension: 0.3
#                         },
#                         {
#                             label: 'Forecast',
#                             data: [...Array(histYears.length).fill(null), histValues[histValues.length - 1], ...foreValues],
#                             borderColor: '#e74c3c',
#                             backgroundColor: 'rgba(231, 76, 60, 0.1)',
#                             borderWidth: 3,
#                             borderDash: [10, 5],
#                             pointRadius: 5,
#                             tension: 0.3
#                         }
#                     ]
#                 },
#                 options: {
#                     responsive: true,
#                     plugins: {
#                         legend: { display: true },
#                         title: {
#                             display: true,
#                             text: `${pollutant} Levels Forecast`,
#                             font: { size: 18, weight: 'bold' }
#                         }
#                     },
#                     scales: {
#                         x: { title: { display: true, text: 'Year' } },
#                         y: { title: { display: true, text: 'Concentration (µg/m³)' } }
#                     }
#                 }
#             });
#         }
        
#         function updateTable(years, values) {
#             document.getElementById('tableContainer').style.display = 'block';
#             const tbody = document.getElementById('tableBody');
#             tbody.innerHTML = '';
            
#             years.forEach((year, idx) => {
#                 const row = `<tr>
#                     <td>Day ${idx + 1}</td>
#                     <td>Year ${year}</td>
#                     <td>${values[idx].toFixed(2)}</td>
#                 </tr>`;
#                 tbody.innerHTML += row;
#             });
#         }
        
#         function downloadCSV() {
#             if (!forecastResults) return;
            
#             let csv = 'Year,Predicted_Value\\n';
#             forecastResults.forecastYears.forEach((year, idx) => {
#                 csv += `${year},${forecastResults.predictions[idx].toFixed(2)}\\n`;
#             });
            
#             const blob = new Blob([csv], { type: 'text/csv' });
#             const url = URL.createObjectURL(blob);
#             const a = document.createElement('a');
#             a.href = url;
#             a.download = `forecast_${forecastResults.station}_${forecastResults.pollutant}.csv`;
#             a.click();
#             URL.revokeObjectURL(url);
#         }
#     </script>
# </body>
# </html>
# """

# # Convert data to JSON for JavaScript
# data_json = json.dumps(data)

# # Replace placeholder with actual data
# html_code = html_template.replace('{{DATA}}', data_json)

# # Display the component
# components.html(html_code, height=1200, scrolling=True)

# # Add download button in Streamlit sidebar for data
# with st.sidebar:
#     st.markdown("### 📊 Data Management")
    
#     if st.button("Export All Data as JSON"):
#         st.download_button(
#             label="Download JSON",
#             data=data_json,
#             file_name="air_quality_data.json",
#             mime="application/json"
#         )
    
#     st.markdown("---")
#     st.markdown("### ℹ️ About")
#     st.info("""
#     **Clear Cast** uses LSTM models to forecast air quality levels.
    
#     Select a station and pollutant to view predictions.
#     """)
#     # models.py
# """
# Model loader + prediction helpers for pollution forecasting.
# Expect model files in ./models/ with names:
#     <site>_<model>_<pollutant>.<ext>
# Examples:
#     mukherjee_nagar_gru_o3.h5
#     mukherjee_nagar_randomforest_no2.pkl

# Model loading rules:
#  - GRU, SLTM -> Keras .h5
#  - RandomForest, XGBoost -> joblib/pickle .pkl

# If model file missing, this returns a deterministic dummy fallback 24-hour forecast.
# Replace prepare_features_for(...) with your real feature engineering.
# """

# import os
# import numpy as np
# import pandas as pd
# import joblib
# from typing import Tuple
# from datetime import datetime, timedelta

# # Optional import for Keras models
# try:
#     from tensorflow.keras.models import load_model
# except Exception:
#     load_model = None

# MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# def safe_name(s: str) -> str:
#     """normalise site or other names into filename-friendly strings"""
#     return str(s).strip().lower().replace(" ", "_").replace("-", "_")

# def model_filename(site: str, model_name: str, pollutant: str) -> str:
#     site_f = safe_name(site)
#     model_f = model_name.strip().lower()
#     pollutant_f = pollutant.strip().lower()
#     # choose extension depending on model
#     if model_f in ("gru", "sltm"):
#         ext = "h5"
#     else:
#         ext = "pkl"
#     return f"{site_f}_{model_f}_{pollutant_f}.{ext}"

# def load_model_file(site: str, model_name: str, pollutant: str):
#     fname = model_filename(site, model_name, pollutant)
#     path = os.path.join(MODELS_DIR, fname)
#     if not os.path.exists(path):
#         return None, path
#     model_key = model_name.strip().lower()
#     try:
#         if model_key in ("gru", "sltm"):
#             if load_model is None:
#                 raise RuntimeError("TensorFlow/Keras not available to load .h5 models.")
#             m = load_model(path, compile=False)
#             return m, path
#         else:
#             m = joblib.load(path)
#             return m, path
#     except Exception as e:
#         # on any load error return None
#         print(f"Error loading {path}: {e}")
#         return None, path

# def make_dummy_prediction(site: str, pollutant: str, base: float = 50.0, seed: int = 0) -> np.ndarray:
#     """Return deterministic 24-hour fallback values (no negative)."""
#     rng = np.random.RandomState(abs(hash(site + pollutant)) % 2**32 + seed)
#     hours = np.arange(24)
#     trend = (np.sin(hours / 24 * 2 * np.pi) * 5.0)  # diurnal pattern
#     vals = base + trend + rng.normal(scale=3.0, size=24)
#     return np.clip(vals, 0.0, None)

# def prepare_features_for(site: str):
#     """
#     Placeholder feature builder for each of next 24 hours.
#     Real pipeline should use past observations + meteorology + time features.
#     We'll return a 24 x F numpy array where F=2 (hour_norm, site_idx).
#     """
#     hours = np.arange(24)
#     hour_norm = (hours / 23.0).reshape(-1, 1)
#     site_idx = (abs(hash(site)) % 100) / 100.0
#     site_col = np.full((24, 1), site_idx)
#     X = np.hstack([hour_norm, site_col])  # shape (24,2)
#     return X

# def predict_with_model_obj(model_obj, X: np.ndarray) -> np.ndarray:
#     """
#     Attempt to get predictions from model_obj for 24 rows in X.
#     Handles: sklearn-like models (predict), Keras models (predict).
#     If model_obj is None, raise ValueError.
#     """
#     if model_obj is None:
#         raise ValueError("model_obj is None")
#     try:
#         # Keras models sometimes expect 3D shape (samples, timesteps, features)
#         import numpy as _np
#         if hasattr(model_obj, "predict") and len(X.shape) == 2:
#             # Try sklearn-style predict
#             y = model_obj.predict(X)
#             return np.ravel(y)[:24]
#     except Exception:
#         pass

#     # fallback: try keras predict with reshape
#     try:
#         # reshape to (samples, timesteps=1, features)
#         X3 = X.reshape((X.shape[0], 1, X.shape[1]))
#         y = model_obj.predict(X3, verbose=0)
#         return np.ravel(y)[:24]
#     except Exception as e:
#         raise RuntimeError(f"Model predict failed: {e}")

# def predict_24h(site: str, model_name: str) -> Tuple[pd.DataFrame, dict]:
#     """
#     Return DataFrame indexed by next 24 hourly timestamps with columns ['o3','no2'].
#     Also returns a dict of model_paths used for each pollutant.
#     """
#     pollutants = ["o3", "no2"]
#     preds = {}
#     model_paths = {}
#     X = prepare_features_for(site)  # shape (24, F)
#     now = datetime.now().replace(minute=0, second=0, microsecond=0)
#     idx = [now + timedelta(hours=i+1) for i in range(24)]

#     for p in pollutants:
#         model_obj, path = load_model_file(site, model_name, p)
#         model_paths[p] = path
#         if model_obj is None:
#             # fallback dummy — base different for pollutants
#             base = 60.0 if p == "o3" else 30.0
#             preds[p] = make_dummy_prediction(site, p, base=base, seed=1)
#         else:
#             try:
#                 y = predict_with_model_obj(model_obj, X)
#                 # ensure length 24
#                 if len(y) < 24:
#                     y = np.pad(y, (0, 24 - len(y)), "edge")
#                 preds[p] = np.clip(np.array(y[:24], dtype=float), 0.0, None)
#             except Exception as e:
#                 print(f"Prediction failed for {path}: {e}")
#                 base = 60.0 if p == "o3" else 30.0
#                 preds[p] = make_dummy_prediction(site, p, base=base, seed=2)

#     df = pd.DataFrame(preds, index=pd.to_datetime(idx))
#     df.index.name = "timestamp"
#     return df, model_paths


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
