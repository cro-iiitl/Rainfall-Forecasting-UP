import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class WeatherService:
    def __init__(self, districts_coord_path):
        self.coords_df = pd.read_csv(districts_coord_path)
        # Normalize district names for matching
        self.coords_df['district_key'] = self.coords_df['district'].str.lower().str.replace(' ', '_').str.replace('-', '_')

    def get_district_coords(self, district_name):
        dist_key = district_name.lower().replace(' ', '_').replace('-', '_')
        row = self.coords_df[self.coords_df['district_key'] == dist_key]
        if not row.empty:
            return row.iloc[0]['latitude'], row.iloc[0]['longitude']
        return 26.8467, 80.9462  # Default to Lucknow

    def fetch_live_features(self, district_name):
        lat, lon = self.get_district_coords(district_name)
        
        # End date is today
        end_date = datetime.now().date()
        # Start date 10 days ago to handle 7-day rolling means and lags
        start_date = end_date - timedelta(days=10)
        
        # Open-Meteo Archive API
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&daily=precipitation_sum,temperature_2m_mean,dew_point_2m_mean,wind_speed_10m_max,wind_direction_10m_dominant,surface_pressure_mean,relative_humidity_2m_mean&timezone=auto"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            daily = data.get('daily', {})
            
            if not daily:
                return None

            df = pd.DataFrame({
                'date': pd.to_datetime(daily['time']),
                'PRECTOTCORR': daily['precipitation_sum'],
                't2m_c': daily['temperature_2m_mean'],
                'd2m_c': daily['dew_point_2m_mean'],
                'wind_speed_10m': daily['wind_speed_10m_max'],
                'WS10M': daily['wind_speed_10m_max'], 
                'WD10M': daily['wind_direction_10m_dominant'],
                'sp_hpa': daily['surface_pressure_mean'],
                'RH2M': daily['relative_humidity_2m_mean']
            })

            # Hand-calculate rolling means and lags
            df['rain_roll_3_mean'] = df['PRECTOTCORR'].rolling(3).mean()
            df['rain_roll_7_mean'] = df['PRECTOTCORR'].rolling(7).mean()
            df['rain_lag_1'] = df['PRECTOTCORR'].shift(1)
            df['rain_lag_3'] = df['PRECTOTCORR'].shift(3)
            df['rain_lag_7'] = df['PRECTOTCORR'].shift(7)
            
            # Derived Meteorological Features (per training scripts)
            df['dewpoint_depression'] = df['t2m_c'] - df['d2m_c']
            df['sp'] = df['sp_hpa'] * 100 # hPa to Pa
            df['T2MDEW'] = df['d2m_c']
            df['tp'] = df['PRECTOTCORR'] 
            df['moisture_pressure_ratio'] = df['RH2M'] / df['sp_hpa']
            
            # Wind Vectors (approximated for u10/v10)
            rad = np.deg2rad(df['WD10M'])
            df['u10'] = -df['wind_speed_10m'] * np.sin(rad)
            df['v10'] = -df['wind_speed_10m'] * np.cos(rad)
            df['wind_dir_10m'] = np.arctan2(df['v10'], df['u10'])
            
            # Temporal Features
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['dayofweek'] = df['date'].dt.dayofweek
            df['district'] = district_name
            
            # The model was trained with 'rain_t_plus_1' as a feature (likely unintentional leak)
            # In live mode, we set it to 0 as we don't know tomorrow's rain.
            df['rain_t_plus_1'] = 0.0 
            
            # Return the latest record (Today)
            return df.iloc[-1].to_dict()

        except Exception as e:
            print(f"Error fetching weather: {e}")
            return None
