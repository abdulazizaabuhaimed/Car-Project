"""Saudi used car price prediction application."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_REFERENCE_YEAR = 2026  # The same reference year used in training.

def first_existing(*names):
    for name in names:
        p = BASE_DIR / name
        if p.is_file():
            return p
    return None

MODEL_PATH = first_existing("rf_model.joblib", "rf_model(6).joblib")
DATA_PATH = first_existing("saudi_used_cars_csv_.csv", "saudi_used_cars.csv", "saudi_used_cars_csv_(6).csv")

st.set_page_config(page_title="Saudi Used Car Price Prediction", page_icon="🚗", layout="centered")
st.markdown("""<style>
[data-testid="stForm"] {border:1px solid rgba(100,116,139,.28);border-radius:14px;padding:20px;}
[data-testid="stMetric"] {background:rgba(120,120,120,.07);padding:14px;border-radius:14px;}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_data(path):
    return pd.read_csv(path, sep=";", encoding="utf-8-sig")

def make_input(make, car_type, year, origin, color, options, engine_size, fuel, gear, mileage, region):
    return pd.DataFrame([{
        "Make": make, "Type": car_type, "Year": int(year),
        "Origin": origin, "Color": color, "Options": options,
        "Engine_Size": float(engine_size), "Fuel_Type": fuel,
        "Gear_Type": gear, "Mileage": int(mileage), "Region": region,
        "Car_Age": MODEL_REFERENCE_YEAR - int(year),
    }])

def choices(df, column):
    return sorted(df[column].dropna().astype(str).unique().tolist())

st.title("🚗 Saudi Used Car Price Prediction")
st.caption("Machine Learning | Random Forest Regression")
st.write("Enter the vehicle details below to estimate its market price using our trained machine learning model.")

if MODEL_PATH is None or DATA_PATH is None:
    st.error("Required files are missing. Place app.py alongside rf_model.joblib and saudi_used_cars_csv_.csv (or saudi_used_cars.csv).")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
    data = load_data(str(DATA_PATH))
except Exception as exc:
    st.error("Could not load the model or dataset. Check your installed dependencies.")
    st.exception(exc)
    st.stop()

# Mirror the training-time trimming when populating dropdown options.
train_df = data.loc[data["Price"] > 0].copy()
low, high = train_df["Price"].quantile([0.01, 0.99])
train_df = train_df.loc[train_df["Price"].between(low, high)]
train_df = train_df.loc[train_df["Mileage"] < train_df["Mileage"].quantile(0.995)]

with st.form("vehicle_details"):
    st.subheader("Vehicle Details")
    makes = choices(train_df, "Make")
    make = st.selectbox("Car Make", makes, index=makes.index("Toyota") if "Toyota" in makes else 0)
    make_rows = train_df.loc[train_df["Make"] == make]
    car_type = st.selectbox("Car Model", choices(make_rows, "Type"))
    left, right = st.columns(2)
    with left:
        min_year, max_year = int(train_df["Year"].min()), int(train_df["Year"].max())
        year = st.number_input("Model Year", min_value=min_year, max_value=max_year, value=min(max(2020, min_year), max_year), step=1)
        mileage = st.number_input("Mileage (km)", min_value=0, max_value=2_000_000, value=100_000, step=5_000)
        engine_size = st.selectbox("Engine Size (L)", sorted(make_rows["Engine_Size"].dropna().unique().tolist()))
        origin = st.selectbox("Vehicle Origin", choices(train_df, "Origin"))
    with right:
        options = st.selectbox("Trim / Options", choices(train_df, "Options"))
        fuel = st.selectbox("Fuel Type", choices(train_df, "Fuel_Type"))
        gear = st.selectbox("Transmission", choices(train_df, "Gear_Type"))
        color = st.selectbox("Color", choices(train_df, "Color"))
        region = st.selectbox("Region", choices(train_df, "Region"))
    submitted = st.form_submit_button("🔍 Predict Price", use_container_width=True, type="primary")

if submitted:
    vehicle = make_input(make, car_type, year, origin, color, options, engine_size, fuel, gear, mileage, region)
    try:
        log_price = float(model.predict(vehicle)[0])
        predicted_price = max(0.0, float(np.expm1(log_price)))
        st.success("Price prediction completed.")
        st.metric("Estimated Price", f"SAR {predicted_price:,.0f}")
        st.caption("This is an estimate, not a guaranteed selling price or an official vehicle valuation.")
        same_model = make_rows.loc[make_rows["Type"] == car_type]
        if same_model.loc[same_model["Year"] == year].empty:
            st.info("Few or no matching listings for this make, model, and year were found in the training data. Interpret this prediction cautiously.")
        if make in {"Maserati", "Porsche", "Land Rover", "Mercedes", "Aston Martin", "Bentley"}:
            st.warning("Predictions for rare or luxury vehicles may be less accurate due to limited training examples.")
    except Exception as exc:
        st.error("Prediction failed. Verify that the saved model is compatible with the installed scikit-learn version.")
        st.exception(exc)

with st.expander("About This Project"):
    st.write("This project estimates used-car prices in Saudi Arabia based on vehicle specifications. The saved Random Forest pipeline processes the inputs and predicts log-transformed price; the app converts the result back to Saudi riyals.")
    st.write("Final held-out test performance: R² ≈ 0.799 and mean absolute error (MAE) ≈ SAR 13,200. Prediction errors vary by vehicle.")
    st.write("The model uses historical listings and does not access live market prices.")
