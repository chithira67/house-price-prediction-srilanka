import pandas as pd
import joblib
import numpy as np


model = joblib.load('models/best_random_forest_model.pkl')
scaler = joblib.load('models/scaler.pkl')
df_encoded = pd.read_csv('data/processed/cleaned_data.csv')

districts = [col.replace('district_', '') for col in df_encoded.columns if col.startswith('district_')]
areas = [col.replace('area_', '') for col in df_encoded.columns if col.startswith('area_')]
water_supplies = [col.replace('water_supply_', '') for col in df_encoded.columns if col.startswith('water_supply_')]
electricity_types = [col.replace('electricity_', '') for col in df_encoded.columns if col.startswith('electricity_')]

feature_cols = [col for col in df_encoded.columns if col != 'price_lkr']

print(f'Districts: {len(districts)} options')
print(f'Areas: {len(areas)} options')
print(f'Water supplies: {len(water_supplies)} options')
print(f'Electricity types: {len(electricity_types)} options')
print('App data loading test: PASSED')