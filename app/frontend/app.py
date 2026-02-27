import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium

# Add parent directory to path to import weather_service
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from weather_service import WeatherService

# --- CONFIGURATION ---
st.set_page_config(
    page_title="UP Rainfall Prediction System",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD CSS ---
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css(os.path.join(os.path.dirname(__file__), "style.css"))

# --- LOAD MODEL ---
@st.cache_resource
def load_assets():
    model = joblib.load("models/phase7_rain_classifier/rain_no_rain_model.pkl")
    threshold = 0.4
    if os.path.exists("models/phase7_rain_classifier/threshold.txt"):
        with open("models/phase7_rain_classifier/threshold.txt", "r") as f:
            threshold = float(f.read().strip())
    return model, threshold

model_pipeline, prob_threshold = load_assets()
weather_svc = WeatherService("data/raw/district_coordinates.csv")

# --- DISTRICTS ---
DISTRICTS = ['agra', 'aligarh', 'ambedkar_nagar', 'amethi', 'amroha', 'auraiya', 'ayodhya', 'azamgarh', 'baghpat', 'bahraich', 'ballia', 'balrampur', 'banda', 'barabanki', 'bareilly', 'basti', 'bijnor', 'budaun', 'bulandshahr', 'chandauli', 'chitrakoot', 'deoria', 'etah', 'etawah', 'farrukhabad', 'fatehpur', 'firozabad', 'gautam_buddha_nagar', 'ghaziabad', 'ghazipur', 'gonda', 'gorakhpur', 'hamirpur', 'hapur', 'hardoi', 'hathras', 'jalaun', 'jaunpur', 'jhansi', 'kannauj', 'kanpur_nagar', 'kasganj', 'kaushambi', 'kushinagar', 'lakhimpur_kheri', 'lalitpur', 'lucknow', 'maharajganj', 'mahoba', 'mainpuri', 'mathura', 'mau', 'meerut', 'mirzapur', 'moradabad', 'muzaffarnagar', 'pilibhit', 'prayagraj', 'raebareli', 'rampur', 'saharanpur', 'sambhal', 'sant_kabir_nagar', 'shahjahanpur', 'shamli', 'shravasti', 'siddharthnagar', 'sitapur', 'sonbhadra', 'sultanpur', 'unnao', 'varanasi']

# --- SESSION STATE ---
if 'fetched_data' not in st.session_state:
    st.session_state.fetched_data = None

# --- SIDEBAR (INPUTS) ---
with st.sidebar:
    st.markdown("<div class='sidebar-info'>ℹ️ <b>How to use:</b> Enter TODAY'S weather conditions below to predict rainfall for TOMORROW.</div>", unsafe_allow_html=True)
    
    st.markdown("### 📍 Location & Date")
    selected_district = st.selectbox("Select District (Today)", sorted(DISTRICTS), index=DISTRICTS.index('lucknow'))
    selected_date = st.date_input("Select Date (Today)", datetime.now())
    
    if st.button("🌐 Fetch Real-Time Weather Data", use_container_width=True):
        with st.spinner("Fetching..."):
            st.session_state.fetched_data = weather_svc.fetch_live_features(selected_district)
            st.session_state.last_prediction = None # Clear old prediction when new data is fetched

    fd = st.session_state.fetched_data if st.session_state.fetched_data else {}

    with st.expander("🌧️ Rainfall History", expanded=True):
        r1 = st.number_input("Rainfall Yesterday (mm)", value=float(fd.get('rain_lag_1', 0.0)))
        r3 = st.number_input("Rainfall 3 Days Ago (mm)", value=float(fd.get('rain_lag_3', 0.0)))
        r7 = st.number_input("Rainfall 7 Days Ago (mm)", value=float(fd.get('rain_lag_7', 0.0)))
        ravg3 = st.number_input("Avg Rain (Last 3 Days)", value=float(fd.get('rain_roll_3_mean', 0.0)))
        ravg7 = st.number_input("Avg Rain (Last 7 Days)", value=float(fd.get('rain_roll_3_mean', 0.0)))
        st.number_input("Total Rain (Last 3 Days)", value=float(ravg3 * 3))
        st.number_input("Total Rain (Last 7 Days)", value=float(ravg7 * 7))

    with st.expander("💨 Atmosphere & Wind (Today)", expanded=True):
        temp = st.number_input("Temperature (°C)", value=float(fd.get('t2m_c', 25.0)))
        dew = st.number_input("Dewpoint (°C)", value=float(fd.get('d2m_c', 15.0)))
        press = st.number_input("Pressure (Pa)", value=float(fd.get('sp', 100000.0)))
        precip = st.number_input("Precipitation (mm/day)", value=float(fd.get('PRECTOTCORR', 0.0)))
        ws = st.number_input("Wind Speed (m/s)", value=float(fd.get('WS10M', 2.0)))
        wd = st.number_input("Wind Direction (°)", value=float(fd.get('WD10M', 180.0)))
        u = st.number_input("Wind U-Vector", value=float(fd.get('u10', 0.0)))
        v = st.number_input("Wind V-Vector", value=float(fd.get('v10', 0.0)))
        moist = st.number_input("Moisture Ratio", value=float(fd.get('moisture_pressure_ratio', 0.03)), format="%.4f")

# --- MAIN CONTENT ---
st.markdown("""
    <div class="header-banner">
        <div class="badge-container">
            <div class="badge-live">LIVE SYSTEM</div>
        </div>
        <h1><span style="font-size: 3rem;">⛈️</span> UP Rainfall Prediction</h1>
        <h3>Scientific Meteorological Forecasting System</h3>
        <p>Utilizing high-resolution data and machine learning to predict precipitation patterns across Uttar Pradesh districts with 24-hour lead time.</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("<div class='tip-box'>📡 <b>System Advisor:</b> Select a district from the workspace and click 'Fetch Live Data' to synchronize current atmospheric parameters with the forecasting engine.</div>", unsafe_allow_html=True)

# --- MAP SECTION ---
st.markdown("<div class='map-container'>", unsafe_allow_html=True)
lat, lon = weather_svc.get_district_coords(selected_district)
m = folium.Map(location=[lat, lon], zoom_start=8, tiles='cartodbpositron')

if st.session_state.fetched_data:
    fd = st.session_state.fetched_data
    popup_html = f"""
        <div style="font-family: 'Inter', sans-serif; min-width: 250px; background-color: #1e293b; color: #f8fafc; padding: 20px; border-radius: 16px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);">
            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 15px;">
                <h3 style="margin: 0; color: #60a5fa; font-size: 1.2rem; font-weight: 700;">{selected_district.title()}</h3>
                <span style="background: #334155; padding: 4px 8px; border-radius: 6px; font-size: 0.7rem; color: #94a3b8;">LIVE</span>
            </div>
            
            <div style="display: flex; flex-direction: column; gap: 12px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.5rem;">🌡️</span>
                    <div style="flex-grow: 1;">
                        <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Temperature</div>
                        <div style="font-size: 1.1rem; font-weight: 600;">{fd.get('t2m_c', 0):.1f}°C</div>
                    </div>
                </div>
                
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.5rem;">💧</span>
                    <div style="flex-grow: 1;">
                        <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Humidity</div>
                        <div style="font-size: 1.1rem; font-weight: 600;">{fd.get('RH2M', 0):.1f}%</div>
                    </div>
                </div>
                
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.5rem;">🍃</span>
                    <div style="flex-grow: 1;">
                        <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Wind Speed</div>
                        <div style="font-size: 1.1rem; font-weight: 600;">{fd.get('WS10M', 0):.1f} m/s</div>
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid #334155; text-align: center;">
                <p style="margin: 0; font-size: 0.85rem; color: #94a3b8;">Ready for Prediction</p>
                <div style="margin-top: 10px; font-size: 0.75rem; color: #3b82f6; font-style: italic;">Scroll down and click 'Predict'</div>
            </div>
        </div>
    """
    folium.Marker(
        [lat, lon],
        popup=folium.Popup(popup_html, max_width=300),
        tooltip=f"Current Weather: {selected_district.title()}",
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)

st_folium(m, width="100%", height=450, key="up_weather_map")
st.markdown("</div>", unsafe_allow_html=True)

col_info, col_pred = st.columns([0.8, 2.2])

# Feature ordering to match model training exactly
input_data = {
    'PRECTOTCORR': precip,
    'rain_roll_3_mean': ravg3,
    'rain_roll_7_mean': ravg7,
    'T2MDEW': dew,
    'd2m_c': dew,
    'moisture_pressure_ratio': moist,
    'rain_lag_1': r1,
    'tp': precip,
    't2m_c': temp,
    'rain_lag_3': r3,
    'sp': press,
    'WD10M': wd,
    'u10': u,
    'rain_lag_7': r7,
    'dewpoint_depression': temp - dew,
    'wind_speed_10m': ws,
    'month': selected_date.month,
    'v10': v,
    'WS10M': ws,
    'year': selected_date.year,
    'wind_dir_10m': wd, # Linked to manual input wd
    'dayofweek': selected_date.weekday(),
    'day': selected_date.day,
    'district': selected_district,
    'rain_t_plus_1': 0.0
}

# --- REPORTING SECTION ---
st.markdown("<h4 style='color: #475569; letter-spacing: 0.1em; text-transform: uppercase; font-size: 0.8rem; margin-bottom: 15px;'>Forecasting Intelligence Report</h4>", unsafe_allow_html=True)

col_info, col_pred = st.columns([1, 2.2])

with col_info:
    st.markdown(f"""
        <div style="text-align: center; padding: 20px;">
            <img src="https://cdn-icons-png.flaticon.com/512/4834/4834559.png" width="140" style="margin-bottom: 20px;">
            <h2 style="font-size: 2.5rem !important; color: #0f172a !important; margin-bottom: 10px !important;">{selected_district.title()}</h2>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 1rem; color: #475569; background: #f1f5f9; padding: 8px 15px; border-radius: 10px; display: inline-block;">📅 {selected_date}</div>
        </div>
    """, unsafe_allow_html=True)

# --- PERSISTENCE LOGIC ---
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

with col_pred:
    # Placeholder for prediction outcome
    result_placeholder = st.empty()
    
    # Initialize with either the last prediction or the empty state
    if st.session_state.last_prediction:
        result_placeholder.markdown(st.session_state.last_prediction, unsafe_allow_html=True)
    else:
        result_placeholder.markdown("""
            <div class='empty-state'>
                <div class='empty-state-icon'>🧬</div>
                <h4>Model Ready for Simulation</h4>
                <p>Atmospheric parameters synchronized. Awaiting execution of the forecasting engine.</p>
            </div>
        """, unsafe_allow_html=True)

# The Predictive Execution Button
if st.button("Generate Forecast ⛈️", type="primary", use_container_width=True):
    try:
        df_pred = pd.DataFrame([input_data])
        prob = model_pipeline.predict_proba(df_pred)[0][1]
        is_rain = prob >= prob_threshold
        tomorrow = selected_date + timedelta(days=1)
        
        icon = "🌧️" if is_rain else "☀️"
        status = "Rain Expected" if is_rain else "No Rain Expected"
        
        # Refined Prediction Result Card
        html_result = f"<div class='prediction-box'><div class='status-line'>{icon} {status}</div><div class='prediction-desc'>Analytical model processing indicates <b>{'unstable' if is_rain else 'stable'}</b> conditions for <b>{selected_district.title()}</b> on <b>{tomorrow}</b>.</div><div class='metric-row'><span class='metric-label-simple'>Probability of Rain:</span><span class='metric-value-simple'>{prob*100:.1f}%</span></div><div class='progress-bar-container'><div class='progress-bar-fill' style='width: {prob*100}%;'></div></div><div class='metric-row'><span class='metric-label-simple'>Model Confidence:</span><span class='metric-value-simple'>{max(prob, 1-prob)*100:.1f}%</span></div><div class='metric-row'><span class='metric-label-simple'>Decision Threshold:</span><span class='metric-value-simple'>{prob_threshold}</span></div></div>"
        
        # Save to session state to prevent disappearing
        st.session_state.last_prediction = html_result
        result_placeholder.markdown(html_result, unsafe_allow_html=True)
        
        if is_rain: st.balloons()
        
    except Exception as e:
        st.error(f"Execution Error: {e}")

# --- FOOTER ---
st.markdown("<br><hr><p style='text-align: center; opacity: 0.6; font-size: 0.8rem;'>Developed for UP Rainfall Prediction Project | Powered by Open-Meteo </p>", unsafe_allow_html=True)
