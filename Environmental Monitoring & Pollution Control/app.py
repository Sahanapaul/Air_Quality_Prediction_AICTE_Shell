import streamlit as st
import numpy as np
import joblib
import os

# --- Load models and scalers ---
@st.cache_resource
def load_models():
    reg = joblib.load("model/aqi_regressor.pkl")
    reg_scaler = joblib.load("model/scaler_reg.pkl")
    clf = joblib.load("model/aqi_classifier.pkl")
    clf_scaler = joblib.load("model/scaler_clf.pkl")
    iso = joblib.load("model/anomaly_detector.pkl")
    return reg, reg_scaler, clf, clf_scaler, iso

regressor, reg_scaler, classifier, clf_scaler, anomaly_detector = load_models()

# --- Page config ---
st.set_page_config(page_title="Air Quality Prediction", layout="centered")

# --- Title & Subtitle ---
st.markdown(
    "<h1 style='text-align: center; color: #FF5733;'>🌬️ Smart Air Quality Predictor</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; font-size:16px;'>Predict AQI, classify air quality, and detect anomalies using environmental sensor data</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# --- Sensor Input Panel ---
st.markdown(
    "<h3 style='color: #2980B9;'>📊 Sensor Input Panel</h3>"
    "<p style='font-size:15px;'>Provide sensor readings to predict AQI and detect anomalies.</p>",
    unsafe_allow_html=True
)

# --- Sensor names ---
sensor_names = [
    'PM2.5','PM10','NO2','CO','O3','SO2','Temperature','Humidity','Wind Speed',
    'pH','Turbidity','Dissolved Oxygen'
]

sensor_values = []
for sensor in sensor_names:
    # Set realistic ranges for each sensor
    if sensor in ['pH']:
        val = st.number_input(f"{sensor} value", min_value=0.0, max_value=14.0, value=7.0, step=0.01)
    elif sensor in ['Turbidity']:
        val = st.number_input(f"{sensor} value", min_value=0.0, max_value=1000.0, value=1.0, step=0.1)
    elif sensor in ['Dissolved Oxygen']:
        val = st.number_input(f"{sensor} value", min_value=0.0, max_value=20.0, value=7.0, step=0.01)
    elif sensor in ['Temperature']:
        val = st.number_input(f"{sensor} value (°C)", min_value=-50.0, max_value=60.0, value=25.0, step=0.1)
    elif sensor in ['Humidity']:
        val = st.number_input(f"{sensor} value (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    elif sensor in ['Wind Speed']:
        val = st.number_input(f"{sensor} value (m/s)", min_value=0.0, max_value=50.0, value=2.0, step=0.1)
    else:
        val = st.number_input(f"{sensor} value", min_value=0.0, max_value=500.0, value=50.0, step=0.1)
    sensor_values.append(val)
    st.markdown("<div style='height:2px; background-color:#eee; margin:5px 0 10px 0;'></div>", unsafe_allow_html=True)

# --- Predict button ---
if st.button("🔍 Predict AQI & Quality"):
    input_array = np.array(sensor_values).reshape(1, -1)

    # --- Anomaly Detection ---
    anomaly = anomaly_detector.predict(input_array)[0]
    if anomaly == -1:
        st.warning("⚠️ Sensor readings detected as unusual or anomalous!")

    # --- Regression Prediction ---
    scaled_input_r = reg_scaler.transform(input_array)
    aqi_pred = regressor.predict(scaled_input_r)[0]

    # --- Classification Prediction ---
    scaled_input_c = clf_scaler.transform(input_array)
    category_pred = classifier.predict(scaled_input_c)[0]

    # --- AQI Categories & Emoji ---
    if aqi_pred <= 50:
        cat = "Good"; emoji = "🟢"
    elif aqi_pred <= 100:
        cat = "Moderate"; emoji = "🟡"
    elif aqi_pred <= 150:
        cat = "Unhealthy for Sensitive Groups"; emoji = "🟠"
    elif aqi_pred <= 200:
        cat = "Unhealthy"; emoji = "🔴"
    elif aqi_pred <= 300:
        cat = "Very Unhealthy"; emoji = "🟣"
    else:
        cat = "Hazardous"; emoji = "⚫"

    st.markdown("---")
    st.markdown("<h3 style='color: #27AE60;'>🌱 Prediction Results</h3>", unsafe_allow_html=True)
    st.success(f"{emoji} AQI (Regression): {aqi_pred:.2f} — Category: {cat}")
    st.info(f"✅ AQI Classifier predicts category: {category_pred}")

# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align:center; padding-top:10px;'>"
    "<p style='font-size:14px; color:gray;'>Created by <strong>Sahana Paul</strong></p>"
    "<p style='font-size:13px; color:#888;'>Part of AICTE Internship Project — Environmental Monitoring</p>"
    "</div>",
    unsafe_allow_html=True
)
