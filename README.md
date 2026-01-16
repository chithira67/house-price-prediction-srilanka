# House Price Prediction - Sri Lanka

A complete end-to-end machine learning project for predicting house prices in Sri Lanka.

## Project Structure

- `data/raw/`: Raw dataset
- `data/processed/`: Cleaned and processed data
- `models/`: Trained models
- `notebooks/`: Jupyter notebooks for analysis and training
- `app.py`: Streamlit web app for predictions

## Notebooks

1. `01_data_overview.ipynb`: Initial data inspection
2. `02_data_cleaning_feature_engineering.ipynb`: Data preprocessing and feature engineering
3. `03_eda.ipynb`: Exploratory data analysis
4. `04_model_training.ipynb`: Training multiple ML models
5. `05_model_evaluation_insights.ipynb`: Model evaluation and insights
6. `06_model_deployment.ipynb`: Deployment setup

## Setup

1. Create virtual environment: `python -m venv venv`
2. Activate: `venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`

## Usage

- Run notebooks in order for data processing and model training
- Launch app: `streamlit run app.py`

## Models

- Linear Regression
- Ridge Regression
- Lasso Regression
- Support Vector Regression
- Random Forest
- Gradient Boosting
- XGBoost

Best model saved as `models/best_house_price_model.pkl`