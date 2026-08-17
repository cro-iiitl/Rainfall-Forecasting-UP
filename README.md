# ⛈️ UP Rainfall Prediction Project
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://up-rainfall-prediction.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Overview
The **UP Rainfall Prediction System** is a sophisticated meteorological forecasting tool designed to predict daily rainfall patterns across the 72+ districts of Uttar Pradesh, India. By leveraging high-resolution atmospheric data from NASA and ECMWF, combined with real-time synchronization via Open-Meteo, the system provides a 24-hour lead time forecast with exceptional precision.

### 🎥 Watch the Dashboard in Action
> Experience the precision of our real-time precipitation forecasting and interactive mapping.

### Demo Video

To view the dashboard demonstration, download:

**[demo.mp4](demo.mp4)**

and play it locally.
---

## 🚀 Key Features
- **🌍 High-Resolution Spatial Coverage**: Predictive analytics for all districts in Uttar Pradesh using nearest-grid meteorological mapping.
- **📡 Real-Time Data Sync**: Integrated `WeatherService` fetches current atmospheric parameters (Temperature, Dewpoint, Wind) via the Open-Meteo API.
- **🧪 Advanced Feature Engineering**: Utilizes 24+ curated meteorological features including dewpoint depression, wind vectors (U10/V10), and rolling precipitation trends.
- **🖥️ Premium Dashboard**: A glassmorphism-inspired UI featuring interactive **Folium** maps and probability-based decision support.

---

## 🛠️ Technical Pipeline

### 1. Data Acquisition & Processing
- **Historical Data**: Aggregated from **NASA POWER** (Solar/Precipitation) and **ECMWF ERA5** (Atmospheric Reanalysis).
- **Processing**: Automated mapping of grid-based NetCDF data to district coordinates using spatial proximity algorithms.
- **Real-Time Layer**: Live meteorological parameters are synchronized through the `app/weather_service.py` module.

### 2. Feature Engineering
The model processes a multidimensional feature space to capture complex atmospheric instabilities:
- **Atmospheric**: Temperature (T2M), Dewpoint (D2M), Surface Pressure (SP), Relative Humidity (RH2M).
- **Wind Dynamics**: Max Wind Speed (WS10M), Dominant Direction (WD10M), and calculated U/V vectors.
- **Lag Features**: Rainfall totals from yesterday (Lag 1), 3 days ago, and 7 days ago.
- **Rolling Stats**: 3-day and 7-day moving averages of precipitation.

### 3. Machine Learning 
The system employs a **Random Forest Classifier** optimized for the imbalanced nature of precipitation events:
- **Accuracy**: 86.73%
- **ROC-AUC**: 0.920 (Excellent discriminative capability)
- **Precision/Recall**: 78.5% / 72.8% (Highly reliable for identifying actual rain events)

---

## 📊 Results & Visualizations
Comprehensive evaluation artifacts are stored in the `results/` directory:
- **Confusion Matrix**: Detailed analysis of True Positives (Rain) vs. True Negatives (No Rain).
- **ROC Curve**: Visual proof of the model's 0.92 AUC performance.
- **Threshold Analysis**: Decision boundary optimization to maximize the F1-Score.

---

## 📁 Project Structure
```text
UP_Rainfall_Prediction_Project/
├── app/
│   ├── frontend/           # Streamlit UI & Styling
│   │   ├── app.py          # Dashboard entry point
│   │   └── style.css       # Custom styles
│   └── weather_service.py  # Real-time weather API logic
├── data/
│   ├── raw/                # Historical datasets (NASA, ERA5)
│   └── features/           # Processed ML-ready data (.npy)
├── models/
│   ├── phase7_rain_classifier/ # Final classification models
│   └── scalers/            # Preprocessors & scalers
├── notebooks/              # Research & Model Development
│   ├── rainfall_1.ipynb    # Data Acquisition
│   ├── rainfall_2.ipynb    # Data Processing
│   ├── rainfall_3.ipynb    # Feature Engineering
│   └── rainfall_4.ipynb    # Model Training & Evaluation
├── results/
│   ├── metrics/            # Evaluation metrics (CSV/JSON)
│   └── plots/              # Visualization artifacts
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 📓 Notebooks (Open in Colab)
Access the detailed development phases directly via Google Colab:
- 🛰️ **Phase 1: Data Acquisition** - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gunjan-Bansal1/UP-Rainfall-Classifier/blob/main/notebooks/rainfall_1.ipynb)
- ⚙️ **Phase 2: Data Processing** - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gunjan-Bansal1/UP-Rainfall-Classifier/blob/main/notebooks/rainfall_2.ipynb)
- 🧪 **Phase 3: Feature Engineering** - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gunjan-Bansal1/UP-Rainfall-Classifier/blob/main/notebooks/rainfall_3.ipynb)
- 🤖 **Phase 4: Model Training** - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gunjan-Bansal1/UP-Rainfall-Classifier/blob/main/notebooks/rainfall_4.ipynb) 
 
--- 
 
## 🚦 Getting Started 
 
### 1. Installation 
```powershell 
pip install -r requirements.txt 
``` 
 
### 2. Model Compatibility (Important) 
If you encounter a blank screen or `AttributeError`, run the retraining script to ensure the model is compatible with your local `scikit-learn` version: 
```powershell 
python scripts/retrain_model.py 
``` 
 
### 3. Launch the Application 
```powershell 
streamlit run app/frontend/app.py 
``` 
 
--- 
 
## 🏗️ Future Scope 
- Integration of higher-frequency (hourly) data for flash flood warnings. 
- Expansion to include neighboring states (Bihar, Uttarakhand). 
- Deployment via Docker for scalable cloud hosting. 
 
--- 
