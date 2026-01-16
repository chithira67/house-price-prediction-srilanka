import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("models/best_house_price_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Load data to get feature names
df = pd.read_csv("data/processed/cleaned_data.csv")
feature_cols = [col for col in df.columns if col != 'price_lkr']

st.title("Sri Lanka House Price Prediction")

# Input fields
district = st.selectbox("District", df['district'].unique() if 'district' in df.columns else ["Polonnaruwa", "Matale"])  # Dynamic
area = st.selectbox("Area", df['area'].unique() if 'area' in df.columns else ["Central"])
perch = st.number_input("Perch", min_value=0.0)
bedrooms = st.number_input("Bedrooms", min_value=1)
bathrooms = st.number_input("Bathrooms", min_value=1)
kitchen_area = st.number_input("Kitchen Area (sqft)", min_value=0.0)
parking = st.number_input("Parking Spots", min_value=0)
has_garden = st.checkbox("Has Garden")
has_ac = st.checkbox("Has AC")
water_supply = st.selectbox("Water Supply", df['water_supply'].unique() if 'water_supply' in df.columns else ["Pipe-borne"])
electricity = st.selectbox("Electricity", df['electricity'].unique() if 'electricity' in df.columns else ["Single phase"])
floors = st.number_input("Floors", min_value=1)
house_age = st.number_input("House Age", min_value=0)

if st.button("Predict Price"):
    # Prepare input
    input_data = {
        'perch': perch,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'kitchen_area_sqft': kitchen_area,
        'parking_spots': parking,
        'has_garden': int(has_garden),
        'has_ac': int(has_ac),
        'floors': floors,
        'house_age': house_age,
        'total_rooms': bedrooms + bathrooms,
        'area_per_room': perch / (bedrooms + bathrooms) if bedrooms + bathrooms > 0 else 0,
        'luxury_score': int(has_garden) + int(has_ac) + int(parking > 1)
    }
    
    # One-hot encode
    for col in feature_cols:
        if col.startswith('district_'):
            input_data[col] = 1 if col == f"district_{district}" else 0
        elif col.startswith('area_'):
            input_data[col] = 1 if col == f"area_{area}" else 0
        elif col.startswith('water_supply_'):
            input_data[col] = 1 if col == f"water_supply_{water_supply}" else 0
        elif col.startswith('electricity_'):
            input_data[col] = 1 if col == f"electricity_{electricity}" else 0
        else:
            input_data.setdefault(col, 0)
    
    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_cols]
    
    # Predict
    if hasattr(model, 'n_estimators'):  # Tree-based
        prediction = model.predict(input_df)
    else:
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
    
    st.success(f"Predicted Price: LKR {prediction[0]:,.0f}")