from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load("groundwater_model.pkl")

# Map Chennai zones to location codes
ZONE_MAP = {
    "North": 0,
    "South": 1,
    "Central": 2,
    "West": 3
}

@app.route("/")
def home():
    return jsonify({"status": "API Running"})

# -------------------------------
# SINGLE MONTH/YEAR PREDICT API
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    year = data["year"]
    month = data["month"]
    zone = data["zone"]

    if zone not in ZONE_MAP:
        return jsonify({"error": "Invalid zone name"}), 400

    loc_code = ZONE_MAP[zone]

    df = pd.DataFrame([[year, month, loc_code]], columns=["Year", "Month", "LocCode"])
    pred = model.predict(df)[0]

    return jsonify({
        "zone": zone,
        "groundwater_level": round(float(pred), 2)
    })

# -------------------------------
# MULTI-YEAR FORECAST API (2025-2030)
# -------------------------------
@app.route("/forecast_multi", methods=["POST"])
def forecast_multi():
    data = request.json
    zone = data["zone"]
    start = data["start"]
    end = data["end"]

    if zone not in ZONE_MAP:
        return jsonify({"error": "Invalid zone name"}), 400

    loc_code = ZONE_MAP[zone]

    results = []
    for year in range(start, end + 1):
        for month in range(1, 13):
            df = pd.DataFrame([[year, month, loc_code]], columns=["Year", "Month", "LocCode"])
            pred = model.predict(df)[0]
            results.append({
                "year": year,
                "month": month,
                "level": round(float(pred), 2)
            })

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
