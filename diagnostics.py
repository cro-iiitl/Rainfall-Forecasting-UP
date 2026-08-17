import sys
import os

print("Starting diagnostics...")

try:
    import streamlit as st
    print("Streamlit imported")
    import pandas as pd
    print("Pandas imported")
    import joblib
    print("Joblib imported")
    import numpy as np
    print("Numpy imported")
    import folium
    print("Folium imported")
    from streamlit_folium import st_folium
    print("Streamlit-folium imported")
    
    # Check if files exist
    model_path = "models/phase7_rain_classifier/rain_no_rain_model.pkl"
    coords_path = "data/raw/district_coordinates.csv"
    
    print(f"Checking {model_path}: {os.path.exists(model_path)}")
    print(f"Checking {coords_path}: {os.path.exists(coords_path)}")
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("Model loaded successfully")
        
    if os.path.exists(coords_path):
        df = pd.read_csv(coords_path)
        print("Coordinates loaded successfully")

    print("All core components verified successfully.")

except Exception as e:
    print(f"DIAGNOSTIC FAILURE: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
