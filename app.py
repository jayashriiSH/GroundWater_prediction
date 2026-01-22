import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="Groundwater Dashboard", layout="wide")

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_model():
    return joblib.load("groundwater_model.pkl")

model = load_model()

zone_map = {
    "North": 0,
    "Central": 1,
    "South": 2,
    "West": 3
}

# ======================
# PREDICTION FUNCTION
# ======================
def predict_groundwater(zone, month, year):
    zone_id = zone_map[zone]
    X = np.array([[zone_id, month, year]])
    return round(float(model.predict(X)[0]), 2)

# ======================
# UI
# ======================
st.title("🌊 Chennai Groundwater Dashboard")

zones = list(zone_map.keys())

# ======================
# ZONE STATUS
# ======================
st.subheader("Zone Status Overview")
cols = st.columns(4)

for i, zone in enumerate(zones):
    level = predict_groundwater(zone, 1, 2026)

    if level < 2.5:
        badge = "🔴 HIGH"
    elif level < 5:
        badge = "🟡 MED"
    else:
        badge = "🟢 LOW"

    cols[i].metric(zone, f"{level} m", badge)

# ======================
# FORECAST
# ======================
st.subheader("12-Month Forecast")

zone = st.selectbox("Select Zone", zones)
year = st.selectbox("Select Year", list(range(2025, 2031)))

levels = []
for m in range(1, 13):
    levels.append(predict_groundwater(zone, m, year))

months = pd.date_range("2025-01-01", periods=12, freq="M").strftime("%b")

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(months, levels, marker="o")
ax.set_ylabel("Groundwater Level (m)")
ax.set_title(f"{zone} Forecast {year}")
st.pyplot(fig)

# ======================
# MANUAL PREDICTION
# ======================
st.subheader("Manual Prediction")

c1, c2, c3, c4 = st.columns(4)

zone2 = c1.selectbox("Zone", zones)
month2 = c2.selectbox("Month", range(1, 13))
year2 = c3.selectbox("Year", range(2025, 2035))

if c4.button("Predict"):
    value = predict_groundwater(zone2, month2, year2)
    st.success(f"Predicted groundwater level: {value} m")
