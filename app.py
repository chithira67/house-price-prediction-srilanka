import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="🏠 Sri Lanka House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(45deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 2rem 0;
    }
    .prediction-amount {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .input-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #007bff;
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        transition: transform 0.2s;
    }
    .feature-card:hover {
        transform: translateY(-2px);
    }
    .stButton>button {
        background: linear-gradient(45deg, #007bff, #0056b3);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #0056b3, #004085);
        transform: scale(1.05);
    }
    .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Load model and data
@st.cache_resource
def load_model():
    model = joblib.load("models/best_xgboost_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    df_encoded = pd.read_csv("data/processed/cleaned_data.csv")
    feature_cols = [col for col in df_encoded.columns if col != 'price_lkr']
    return model, scaler, df_encoded, feature_cols

model, scaler, df_encoded, feature_cols = load_model()

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.title("🏠 About")
    st.markdown("""
    **Sri Lanka House Price Predictor**

    This AI-powered tool predicts house prices in Sri Lanka based on various property features using machine learning.

    **Features:**
    - 🤖 Advanced ML algorithms
    - 📊 Real-time predictions
    - 🎯 High accuracy
    - 🌍 Sri Lanka focused
    """)

    st.markdown("---")
    st.markdown("**📈 Model Performance:**")
    st.info("R² Score: ~0.85\nMAE: ~2.1M LKR")

    st.markdown("---")
    st.markdown("**🛠️ Built with:**")
    st.markdown("Streamlit • Scikit-learn • XGBoost")
    st.markdown('</div>', unsafe_allow_html=True)

# Main content
st.markdown('<h1 class="main-header">🏠 Sri Lanka House Price Predictor</h1>', unsafe_allow_html=True)
st.markdown("### Enter your property details below to get an instant price prediction!")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("### 📍 Location Details")
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)

    district = st.selectbox(
        "🏛️ District",
        sorted(df_encoded['district'].unique()) if 'district' in df_encoded.columns else ["Polonnaruwa", "Matale"],
        help="Select the district where the property is located"
    )

    area = st.selectbox(
        "📍 Area",
        sorted(df_encoded['area'].unique()) if 'area' in df_encoded.columns else ["Central"],
        help="Specific area within the district"
    )

    house_age = st.slider(
        "📅 House Age (years)",
        min_value=0,
        max_value=50,
        value=20,
        help="Age of the house in years"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("### 🏠 Property Features")
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)

    floors = st.selectbox(
        "🏢 Number of Floors",
        [1, 2, 3],
        help="Total floors in the building"
    )

    bedrooms = st.slider(
        "🛏️ Bedrooms",
        min_value=1,
        max_value=7,
        value=3,
        help="Number of bedrooms"
    )

    bathrooms = st.slider(
        "🛁 Bathrooms",
        min_value=1,
        max_value=5,
        value=2,
        help="Number of bathrooms"
    )

    perch = st.number_input(
        "📐 Land Area (Perch)",
        min_value=0.0,
        value=10.0,
        step=0.5,
        help="Land area in perch (1 perch ≈ 25.29 m²)"
    )

    kitchen_area = st.number_input(
        "🍳 Kitchen Area (sqft)",
        min_value=0.0,
        value=100.0,
        step=10.0,
        help="Kitchen area in square feet"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("### 🚗 Amenities & Utilities")
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)

    parking = st.slider(
        "🚗 Parking Spots",
        min_value=0,
        max_value=3,
        value=1,
        help="Number of parking spaces"
    )

    col2a, col2b = st.columns(2)
    with col2a:
        has_garden = st.checkbox("🌳 Garden", help="Property has a garden")
    with col2b:
        has_ac = st.checkbox("❄️ Air Conditioning", help="Property has AC")

    water_supply = st.selectbox(
        "💧 Water Supply",
        sorted(df_encoded['water_supply'].unique()) if 'water_supply' in df_encoded.columns else ["Pipe-borne"],
        help="Type of water supply"
    )

    electricity = st.selectbox(
        "⚡ Electricity",
        sorted(df_encoded['electricity'].unique()) if 'electricity' in df_encoded.columns else ["Single phase"],
        help="Type of electrical supply"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Prediction section
st.markdown("---")
col_pred, col_btn = st.columns([2, 1])

with col_btn:
    predict_button = st.button("🔮 Predict House Price", use_container_width=True)

with col_pred:
    if predict_button:
        with st.spinner("🔍 Analyzing property features..."):
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
            prediction = model.predict(input_df)

        # Display prediction
        st.markdown(f"""
        <div class="prediction-card">
            <h2>💰 Estimated House Price</h2>
            <div class="prediction-amount">LKR {prediction[0]:,.0f}</div>
            <p>Based on your property specifications</p>
        </div>
        """, unsafe_allow_html=True)

        # Additional insights
        st.markdown("### 📊 Property Insights")
        col_ins1, col_ins2, col_ins3 = st.columns(3)

        with col_ins1:
            st.metric("Total Rooms", f"{bedrooms + bathrooms}")
        with col_ins2:
            st.metric("Area per Room", f"{perch / (bedrooms + bathrooms):.1f} perch" if bedrooms + bathrooms > 0 else "N/A")
        with col_ins3:
            st.metric("Luxury Score", f"{int(has_garden) + int(has_ac) + int(parking > 1)}/3")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🏠 Sri Lanka House Price Predictor | Built with ❤️ using Machine Learning</p>
    <p><small>Data last updated: January 2026</small></p>
</div>
""", unsafe_allow_html=True)