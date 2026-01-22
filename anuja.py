import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

API = "http://127.0.0.1:5000"

st.set_page_config(page_title="GroundWater  Dashboard", layout="wide")
st.title(" Groundwater Dashboard")

# =========================
# ZONE STATUS ROW
# =========================
st.subheader("Zone Status Overview")

zones = ["North", "Central", "South", "West"]
cols = st.columns(4)

for i, zone in enumerate(zones):
    r = requests.post(f"{API}/predict", json={"zone": zone, "month": 1, "year": 2026})
    level = round(r.json()["groundwater_level"], 2)

    if level < 2.5:
        badge = "🔴 HIGH"
    elif level < 5:
        badge = "🟡 MED"
    else:
        badge = "🟢 LOW"

    cols[i].metric(zone, f"{level} m", badge)

# =========================
# FORECAST SECTION
# =========================
st.subheader("12-Month Forecast")

zone = st.selectbox("Select Zone", zones, index=1)
year = st.selectbox("Select Forecast Year", list(range(2025, 2031)), index=1)

levels = []
for m in range(1, 13):
    r = requests.post(f"{API}/predict", json={"zone": zone, "month": m, "year": year})
    levels.append(r.json()["groundwater_level"])

months = pd.date_range("2025-01-01", periods=12, freq="M").strftime("%b")

fig, ax = plt.subplots(figsize=(8,3))
ax.plot(months, levels, marker="o")
ax.set_ylabel("Groundwater Level (m)")
ax.set_title(f"{zone} Forecast ({year})")
st.pyplot(fig)

# =========================
# MANUAL PREDICTION PANEL
# =========================
st.subheader("Manual Prediction")

col1, col2, col3, col4 = st.columns(4)
zone2 = col1.selectbox("Zone", zones)
month2 = col2.selectbox("Month", list(range(1,13)))
year2 = col3.selectbox("Year", list(range(2025, 2035)))

if col4.button("Predict"):
    r = requests.post(f"{API}/predict", json={"zone": zone2, "month": month2, "year": year2})
    p = round(r.json()["groundwater_level"], 2)
    st.success(f"{zone2} {month2}/{year2} → {p} m")
