# fire_predictor_app.py

import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("fire_model.pkl")
scaler = joblib.load("scaler.pkl")

# Web App Title
st.set_page_config(page_title="Forest Fire Predictor", layout="centered")
st.title("🔥 Forest Fire Risk Predictor")
st.write("Enter environmental conditions to check if there's a fire risk.")

# Input fields
temp = st.number_input("🌡️ Temperature (°C)", min_value=-10.0, max_value=50.0, value=20.0)
rh = st.number_input("💧 Relative Humidity (%)", min_value=0.0, max_value=100.0, value=40.0)
wind = st.number_input("💨 Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=5.0)
rain = st.number_input("🌧️ Rain (mm)", min_value=0.0, max_value=50.0, value=0.0)
ffmc = st.number_input("🔥 FFMC Index", min_value=0.0, max_value=100.0, value=85.0)
dmc = st.number_input("🔥 DMC Index", min_value=0.0, max_value=200.0, value=100.0)
dc = st.number_input("🔥 DC Index", min_value=0.0, max_value=800.0, value=600.0)
isi = st.number_input("🔥 ISI Index", min_value=0.0, max_value=50.0, value=10.0)

# Predict Button
if st.button("Predict Fire Risk"):
    input_data = np.array([[temp, rh, wind, rain, ffmc, dmc, dc, isi]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)

    if prediction[0] == 1:
        st.error("🔥 Fire Risk Detected! Take Preventive Measures.")
    else:
        st.success("✅ No Fire Risk Detected. Conditions appear safe.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Namal University Project")
