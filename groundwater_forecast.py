import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# Files with corresponding year
files = [
    ("data/groundwater_1.csv", 2021),
    ("data/groundwater_2.csv", 2022),
    ("data/groundwater_3.csv", 2023),
    ("data/groundwater_4.csv", 2024)
]

dfs = []

# Months found in Chennai datasets
all_months = [
    'Jan ', 'Feb', 'Mar', 'Apr ', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
]

for f, year in files:
    df = pd.read_csv(f)

    # ---- DROP NON-RELEVANT COLUMNS ----
    drop_cols = [c for c in df.columns if "S.No" in c or "Area" in c or "Dept" in c]
    df = df.drop(columns=drop_cols, errors="ignore")

    # ---- DETECT WHICH MONTH COLUMNS ARE IN THIS FILE ----
    available_months = [m for m in all_months if m in df.columns]

    # ---- KEEP ONLY LOCATION + MONTHS ----
    df = df[['Location'] + available_months]

    # ---- MELT WIDE → LONG FORMAT ----
    df_long = df.melt(id_vars=["Location"], value_name="Level", var_name="Month")

    # ---- ADD YEAR ----
    df_long["Year"] = year
    dfs.append(df_long)

# ---- STACK EVERYTHING ----
df_all = pd.concat(dfs, ignore_index=True)

# ---- MAP MONTHS TO NUMBERS ----
month_map = {
    'Jan ': 1, 'Feb': 2, 'Mar': 3, 'Apr ': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}
df_all["Month"] = df_all["Month"].map(month_map)

# ---- DROP MISSING MONTHS ----
df_all = df_all.dropna(subset=["Month"])

# ---- CONVERT LEVEL TO NUMERIC & DROP NON-NUMERIC (like "removed") ----
df_all["Level"] = pd.to_numeric(df_all["Level"], errors="coerce")
df_all = df_all.dropna(subset=["Level"])

# ---- ENCODE LOCATION ----
df_all["LocCode"] = df_all["Location"].astype("category").cat.codes

# ---- SELECT FEATURES ----
X = df_all[["Year", "Month", "LocCode"]]
y = df_all["Level"]

# ---- TRAIN MODEL ----
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X, y)

# ---- SAVE MODEL ----
joblib.dump(model, "groundwater_model.pkl")

print("✅ MODEL TRAINED SUCCESSFULLY - groundwater_forecast.py:72")
print(f"Total usable samples: {len(df_all)} - groundwater_forecast.py:73")
print(df_all.head())
