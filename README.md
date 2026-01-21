# UP Rainfall Prediction Project

## Overview
This project aims to predict daily rainfall for various districts in Uttar Pradesh (UP), India. It utilizes a comprehensive machine learning pipeline that aggregates meteorological data from multiple sources (NASA POWER, GPM IMERG, ECMWF ERA5) to forecast the occurrence of rainfall (Classification).

The project is structured to handle data acquisition, processing, feature engineering, and model training/evaluation across a series of Jupyter Notebooks.

## Project Structure
```
UP_Rainfall_Prediction_Project/
├── app/                # (Placeholder for future web application)
├── config.py           # Configuration settings
├── data/               # Data storage
│   ├── raw/            # Raw downloaded data (NASA, ERA5, etc.)
│   ├── cleaned/        # Processed and merged datasets
│   └── features/       # Final features for modeling
├── logs/               # Execution logs
├── models/             # Saved trained models
├── notebooks/          # Analysis and modeling notebooks
│   ├── rainfall_1.ipynb # Data Acquisition (NASA POWER, GPM, ERA5)
│   ├── rainfall_2.ipynb # Data Processing & Merging
│   ├── rainfall_3.ipynb # Feature Engineering & Selection
│   └── rainfall_4.ipynb # Model Training & Evaluation
└── results/            # Analysis results and outputs
```

## Data Sources
1.  **NASA POWER**: Provides daily meteorological parameters like Probability of Precipitation, Relative Humidity, Dew Point, Wind Speed, etc.
2.  **GPM IMERG**: Integrated Multi-satellitE Retrievals for GPM (Global Precipitation Measurement) for precise rainfall estimates.
3.  **ERA5**: The fifth generation ECMWF atmospheric reanalysis of the global climate, providing detailed atmospheric data (Temperature, Pressure, Wind components).

## Pipeline Workflow

### 1. Data Acquisition ([rainfall_1.ipynb](notebooks/rainfall_1.ipynb))
-   Sets up authentication for NASA Earthdata and Copernicus Climate Data Store (CDS).
-   Downloads raw data:
    -   **NASA POWER**: Fetches daily weather data for UP districts via API.
    -   **GPM IMERG**: Downloads granular satellite precipitation data.
    -   **ERA5**: Retrieves historical weather data (NetCDF format) using `cdsapi`.

### 2. Data Processing ([rainfall_2.ipynb](notebooks/rainfall_2.ipynb))
-   **ERA5 Processing**: Merges "instant" and "accumulated" data streams from NetCDF files.
-   **Aggregation**: Maps grid-based weather data to specific districts by finding the nearest grid points.
-   **Cleaning**: Handles missing values and merges datasets from different sources into a unified district-daily format.

### 3. Feature Engineering ([rainfall_3.ipynb](notebooks/rainfall_3.ipynb))
-   **Target Creation**: Generates the target variable `rain_t_plus_1` (Next Day Rainfall).
-   **Time Features**: Extracts year, month, day, and day-of-week.
-   **Feature Selection**:
    -   Calculates correlation with the target variable.
    -   Removes features with low correlation (< 0.01).
    -   Removes highly collinear features (> 0.95 correlation) to reduce redundancy.
-   **Output**: Saves the final feature set to `data/features/phase3_features.csv`.

### 4. Modeling & Evaluation ([rainfall_4.ipynb](notebooks/rainfall_4.ipynb))
The project employs a modeling approach to capture the likelihood of rainfall.

#### A. Classification (Predicting Rain vs. No-Rain)
This model predicts whether it will rain (> 0mm) on the next day.
-   **Target**: Binary variable (1 for Rain, 0 for No-Rain).
-   **Class Distribution**: ~27% Rain events vs ~73% No-Rain events.
-   **Models Evaluated**:
    -   **Logistic Regression**: A linear baseline.
    -   **Random Forest Classifier**: An ensemble method using 200 estimators.
-   **Metrics**: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.


## Performance Results

### Classification Results (Validation Set)
The Random Forest model outperformed Logistic Regression across all key metrics.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **86.73%** | **78.55%** | **72.81%** | **0.756** | **0.920** |
| Logistic Regression | 86.40% | 78.52% | 71.24% | 0.747 | 0.903 |

*   **ROC-AUC of 0.92** indicates excellent capability in distinguishing between rain and no-rain days.
*   **Recall of 72.8%** means the model successfully identifies nearly 73% of actual rainfall events, which is crucial for forecasting.

## Key Dependencies
-   `pandas`, `numpy`: Data manipulation.
-   `xarray`, `netCDF4`: Handling multi-dimensional weather data.
-   `requests`, `cdsapi`: API data fetching.
-   `scikit-learn`: Model training, preprocessing, and evaluation.
-   `matplotlib`, `seaborn`: Visualization.

## Setup & Usage
1.  **Install Dependencies**: Ensure Python and required libraries are installed.
2.  **API Keys**: Set up `.netrc` for Earthdata and `.cdsapirc` for ERA5 access as shown in `rainfall_1.ipynb`.
3.  **Run Notebooks**: Execute the notebooks in order (1 to 4) to reproduce the dataset and models.

## Notebook Links
- [rainfall_1.ipynb](notebooks/rainfall_1.ipynb): Data Acquisition
- [rainfall_2.ipynb](notebooks/rainfall_2.ipynb): Data Processing & Merging
- [rainfall_3.ipynb](notebooks/rainfall_3.ipynb): Feature Engineering & Selection
- [rainfall_4.ipynb](notebooks/rainfall_4.ipynb): Model Training & Evaluation

