# 🏠 Sri Lanka House Price Prediction

A comprehensive end-to-end machine learning project for predicting house prices in Sri Lanka using advanced ML algorithms and modern web technologies.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📊 Project Overview

This project implements a complete machine learning pipeline for house price prediction in Sri Lanka, featuring:

- **Advanced ML Models**: 7 different algorithms with cross-validation
- **Comprehensive EDA**: 20+ high-resolution visualizations
- **Web Application**: Interactive Streamlit app for real-time predictions
- **Production Ready**: Model evaluation, hyperparameter tuning, and deployment

### 🎯 Key Features

- **High Accuracy**: R² Score of 0.9485 (94.8% variance explained)
- **Real-time Predictions**: Web app with instant price estimates
- **Data-Driven Insights**: Comprehensive analysis of Sri Lankan real estate market
- **Scalable Architecture**: Modular code structure for easy maintenance

## 🏗️ Project Structure

```
house-price-prediction-srilanka/
│
├── 📁 data/
│   ├── 📁 raw/
│   │   └── house_prices_srilanka.csv          # Raw dataset
│   └── 📁 processed/
│       └── cleaned_data.csv                   # Processed dataset
│
├── 📁 models/
│   ├── best_random_forest_model.pkl          # Best performing model
│   ├── best_xgboost_model.pkl               # Alternative model
│   ├── scaler.pkl                           # Feature scaler
│   ├── feature_names.pkl                    # Feature names
│   └── evaluation_results.pkl               # Model performance metrics
│
├── 📁 notebooks/
│   ├── 01_data_overview.ipynb               # Dataset inspection
│   ├── 02_data_cleaning_feature_engineering.ipynb  # Data preprocessing
│   ├── 03_eda.ipynb                         # Exploratory data analysis
│   ├── 04_model_training.ipynb              # Model training & comparison
│   └── 05_model_evaluation_insights.ipynb   # Model evaluation & insights
│
├── 📁 plots/
│   └── 📊 20+ visualization files (PNG)     # All EDA plots
│
├── app.py                                   # Streamlit web application
├── requirements.txt                         # Python dependencies
├── test_app.py                             # App testing script
└── README.md                               # Project documentation
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/house-price-prediction-srilanka.git
   cd house-price-prediction-srilanka
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Option 1: Run the Web App (Recommended)
```bash
streamlit run app.py
```
Navigate to `http://localhost:8501` in your browser for the interactive web application.

#### Option 2: Run Notebooks (For Analysis)
Execute the Jupyter notebooks in order:
1. `01_data_overview.ipynb` - Dataset inspection
2. `02_data_cleaning_feature_engineering.ipynb` - Data preprocessing
3. `03_eda.ipynb` - Exploratory data analysis
4. `04_model_training.ipynb` - Model training
5. `05_model_evaluation_insights.ipynb` - Model evaluation

## 📈 Model Performance

### Best Model: Random Forest Regressor
- **R² Score**: 0.9485 (94.8% variance explained)
- **MAE**: LKR 1,502,912 (average prediction error)
- **RMSE**: LKR 2,121,223
- **Prediction Interval Coverage**: 93.3% at 95% confidence

### Model Comparison Results
| Model | R² Score | Cross-Validation Std |
|-------|----------|---------------------|
| Random Forest | **0.9458** | ±0.0033 |
| XGBoost | 0.9518 | ±0.0023 |
| Gradient Boosting | 0.9431 | ±0.0033 |
| Linear Regression | 0.9177 | ±0.0031 |
| Ridge | 0.9177 | ±0.0031 |
| Lasso | 0.9177 | ±0.0031 |
| SVR | -0.0707 | ±0.0277 |

### Key Predictors
1. **Land Area (Perch)** - 58.32% importance
2. **Kitchen Area** - 22.28% importance
3. **District Location** - Top districts: Colombo, Kandy, Gampaha
4. **House Age** - 1.27% importance
5. **Number of Floors** - 0.43% importance

## 📊 Exploratory Data Analysis

The project includes comprehensive EDA with **20+ visualizations**:

### Distribution Analysis
- Price distribution histograms
- Feature distributions (bedrooms, bathrooms, land area)
- Categorical variable analysis

### Relationship Analysis
- Scatter plots: Price vs numerical features
- Correlation heatmaps
- Box plots by categories

### Geographic Insights
- Average price by district
- District-wise property distribution
- Location-based price variations

### Advanced Analytics
- Feature importance analysis
- Residual analysis plots
- Prediction interval visualizations
- Error analysis by price ranges

All plots are automatically saved as high-resolution PNG files in the `plots/` directory.

## 🛠️ Technical Implementation

### Data Processing Pipeline
1. **Data Cleaning**: Missing value handling, outlier removal
2. **Feature Engineering**: Total rooms, area per room, luxury score
3. **Encoding**: One-hot encoding for categorical variables
4. **Scaling**: StandardScaler for numerical features

### Machine Learning Pipeline
1. **Model Training**: 7 algorithms with 5-fold cross-validation
2. **Hyperparameter Tuning**: Grid search for optimal parameters
3. **Model Selection**: Best model based on R² score
4. **Model Persistence**: Joblib for model serialization

### Web Application Features
- **Interactive UI**: Modern Streamlit interface
- **Real-time Predictions**: Instant price estimates
- **Input Validation**: Proper data type checking
- **Responsive Design**: Mobile-friendly layout
- **Property Insights**: Additional metrics display

## 📋 Dataset Information

- **Source**: Sri Lankan real estate market data
- **Size**: 19,051 properties
- **Features**: 107 engineered features
- **Target**: House price in Sri Lankan Rupees (LKR)

### Key Features
- Location (District, Area)
- Property specs (Bedrooms, Bathrooms, Floors)
- Land details (Perch, Garden)
- Amenities (AC, Parking, Water supply, Electricity)
- Age and condition

## 🔧 Dependencies

```
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.1.0
joblib>=1.1.0
streamlit>=1.10.0
xgboost>=1.6.0
scipy>=1.7.0
```

## 📈 Model Interpretability

### Feature Importance Analysis
The Random Forest model provides clear feature importance rankings, helping users understand which property characteristics most influence price predictions.

### Prediction Confidence
- 95% prediction intervals provided
- Error analysis by price ranges
- Residual analysis for model diagnostics

## 🚀 Deployment & Production

### Local Deployment
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Docker Deployment (Future)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Sri Lankan real estate market data
- Scikit-learn, XGBoost, and Streamlit communities
- Open source ML community

## 📞 Contact

For questions or suggestions, please open an issue on GitHub.

---

**⭐ Star this repository if you found it helpful!**

*Built with ❤️ for the Sri Lankan real estate market*