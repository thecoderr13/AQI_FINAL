import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()  # This loads variables from .env file

API_KEY = os.getenv("API_KEY")  # Make sure the variable name matches exactly

# Load pre-trained models
rf_model = joblib.load('../trained_models/random_forest_model.pkl')
xgb_model = joblib.load('../trained_models/xgboost_model.pkl')
poly_model = joblib.load('../trained_models/polynomial_regression_model.pkl')
linear_model = joblib.load('../trained_models/linear_regression_model.pkl')

# Define model options at the top
model_options = {
    "Random Forest": rf_model,
    "XGBoost": xgb_model,
    "Polynomial Regression": poly_model,
    "Linear Regression": linear_model
}

# Load dataset to calculate mean values and additional data (replace with your dataset path)
df = pd.read_csv('training_dataset.csv')  # Ensure this file exists and contains the features
mean_values = df[['PM2.5', 'PM10', 'NO', 'NO2', 'NH3', 'CO', 'SO2', 'O3']].mean().to_dict()

# OpenWeather API configuration
BASE_URL = "http://api.openweathermap.org/data/2.5/air_pollution"

# City coordinates
city_coordinates = {
    'Delhi': (28.7041, 77.1025),
    'Bengaluru': (12.9716, 77.5946),
    'Hyderabad': (17.3850, 78.4867),
    'Chennai': (13.0827, 80.2707),
    'Lucknow': (26.8467, 80.9462),
    'Mumbai': (19.0760, 72.8777),
    'Patna': (25.5941, 85.1376),
    'Gurugram': (28.4595, 77.0266),
    'Jaipur': (26.9124, 75.7873),
    'Ahmedabad': (23.0225, 72.5714)
}

# Function to map month to season
def get_season_number(month):
    if pd.isnull(month):
        return -1  # Unknown
    if month in [3, 4, 5, 6]:
        return 0  # Summer
    elif month in [7, 8, 9, 10]:
        return 1  # Monsoon
    else:
        return 2  # Winter

# Function to get real-time AQI data from OpenWeather API and fill zeros/missing with mean
def get_real_time_aqi(lat, lon):
    params = {
        'lat': lat,
        'lon': lon,
        'appid': API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        components = data['list'][0]['components']
        fetched_data = {
            'PM2.5': components.get('pm2_5', 0),
            'PM10': components.get('pm10', 0),
            'NO': components.get('no', 0),
            'NO2': components.get('no2', 0),
            'NH3': components.get('nh3', 0),
            'CO': components.get('co', 0),
            'SO2': components.get('so2', 0),
            'O3': components.get('o3', 0)
        }
        # Fill zero or missing values with dataset mean
        for key in fetched_data:
            if fetched_data[key] == 0 or pd.isna(fetched_data[key]):
                fetched_data[key] = mean_values.get(key, 0)
        return fetched_data
    else:
        st.error("Failed to fetch real-time AQI data. Check API key or connection.")
        return None

# Function to prepare data for prediction
def prepare_data(features, month):
    df = pd.DataFrame([features])
    df['Season'] = get_season_number(month)
    required_columns = ['PM2.5', 'PM10', 'NO', 'NO2', 'NH3', 'CO', 'SO2', 'O3', 'Season']
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0  # Default to 0 if missing
    return df[required_columns]

# Function to predict AQI
def predict_aqi(model, features, month):
    prepared_data = prepare_data(features, month)
    prediction = model.predict(prepared_data)
    return prediction[0]

# Function to classify AQI
def classify_aqi(aqi):
    if aqi <= 50:
        return "Good", "green"
    elif 51 <= aqi <= 100:
        return "Satisfactory", "lightgreen"
    elif 101 <= aqi <= 200:
        return "Moderate", "yellow"
    elif 201 <= aqi <= 300:
        return "Poor", "orange"
    elif 301 <= aqi <= 400:
        return "Very Poor", "red"
    else:
        return "Severe", "darkred"

# Custom CSS with Font Awesome, spacing, and enhanced styling
st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    <style>
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 2px solid rgba(0, 196, 180, 0.5); /* Teal border */
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.3);
        color: white;
        margin-bottom: 20px;
    }
    .gradient-text {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
    .pollutant-card {
        background: linear-gradient(135deg, #1e3c72, #2a5298); /* Dark blue gradient */
        border-radius: 10px;
        padding: 15px;
        margin: 0 20px 15px 20px; /* Increased horizontal spacing to 20px */
        text-align: center;
        border: 2px solid #00C4B4; /* Teal border */
        display: inline-block;
        width: 180px;
        height: 120px;
        vertical-align: middle;
        transition: transform 0.2s, box-shadow 0.2s;
        color: white;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    .pollutant-card:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 18px rgba(0, 196, 180, 0.5);
    }
    .pollutant-card i {
        font-size: 24px;
        margin-bottom: 5px;
        color: #00C4B4; /* Teal icon color */
    }
    .stButton>button {
        background: linear-gradient(135deg, #2c3e50, #3498db); /* Darker blue gradient */
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        color: white;
        font-weight: bold;
        cursor: pointer;
        backdrop-filter: blur(5px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s, background 0.3s;
        margin: 5px;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        background: linear-gradient(135deg, #3498db, #2c3e50); /* Reversed gradient on hover */
        color: white; /* White text on hover */
        font-weight: bold; /* Bold text on hover */
    }
    .stButton>button:active {
        color: white; /* White text on click */
        font-weight: bold; /* Bold text on click */
    }
    .input-group {
        background: linear-gradient(135deg, #1e3c72, #2a5298); /* Matching pollutant card gradient */
        backdrop-filter: blur(5px);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        color: white;
    }
    .input-group label {
        color: #00C4B4; /* Teal accent for labels */
        font-weight: bold;
    }
    .stNumberInput input {
        background: rgba(255, 255, 255, 0.1); /* Light glass effect */
        border: 2px solid #00C4B4; /* Thicker teal border */
        border-radius: 5px;
        color: white;
        padding: 8px;
        transition: border-color 0.3s, box-shadow 0.3s;
    }
    .stNumberInput input:focus {
        border-color: #4ecdc4; /* Lighter teal on focus */
        box-shadow: 0 0 8px rgba(0, 196, 180, 0.5);
        outline: none;
    }
    @media (max-width: 768px) {
        .pollutant-card {
            width: 140px;
            height: 100px;
            margin: 0 10px 10px 10px; /* Adjusted spacing for mobile */
        }
        .stButton>button {
            padding: 8px 15px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI with enhanced design
st.markdown('<h1 class="gradient-text">AQI Prediction Dashboard</h1>', unsafe_allow_html=True)

# Input section with glassmorphism
with st.container():
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        city = st.selectbox("City", list(city_coordinates.keys()), key="city_select")
    with col2:
        date = st.date_input("Date", datetime(2025, 4, 20), key="date_input")
    with col3:
        selected_model = st.selectbox("Model", list(model_options.keys()), key="model_select")
    month = date.month

# Buttons with glassmorphism and gradient
col1, col2, col3 = st.columns(3)
with col1:
    fetch_button = st.button("Fetch AQI", key="fetch_button")
with col2:
    autofill_button = st.button("Autofill to Mean", key="autofill_button")
with col3:
    manual_button = st.button("Manual Enter", key="manual_button")

# Initialize features dictionary
if 'features' not in st.session_state:
    st.session_state.features = {
        'PM2.5': 0.0,
        'PM10': 0.0,
        'NO': 0.0,
        'NO2': 0.0,
        'NH3': 0.0,
        'CO': 0.0,
        'SO2': 0.0,
        'O3': 0.0
    }

# Handle button actions
if fetch_button:
    if city:
        lat, lon = city_coordinates[city]
        data = get_real_time_aqi(lat, lon)
        if data:
            st.session_state.features.update(data)
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.write("Fetched Pollutant Levels (Zeros/Missing filled with Mean):")
            st.dataframe(pd.DataFrame([st.session_state.features]))
            st.markdown('</div>', unsafe_allow_html=True)

if autofill_button:
    st.session_state.features.update(mean_values)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.write("Mean Pollutant Levels:")
    st.dataframe(pd.DataFrame([st.session_state.features]))
    st.markdown('</div>', unsafe_allow_html=True)

if manual_button or not (fetch_button or autofill_button):
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="input-group">', unsafe_allow_html=True)
    st.write("Manual Entry:")
    st.session_state.features['PM2.5'] = st.number_input("PM2.5", value=st.session_state.features['PM2.5'], format="%.2f", step=1.0)
    st.session_state.features['PM10'] = st.number_input("PM10", value=st.session_state.features['PM10'], format="%.2f", step=1.0)
    st.session_state.features['NO'] = st.number_input("NO", value=st.session_state.features['NO'], format="%.2f", step=1.0)
    st.session_state.features['NO2'] = st.number_input("NO2", value=st.session_state.features['NO2'], format="%.2f", step=1.0)
    st.session_state.features['NH3'] = st.number_input("NH3", value=st.session_state.features['NH3'], format="%.2f", step=1.0)
    st.session_state.features['CO'] = st.number_input("CO", value=st.session_state.features['CO'], format="%.2f", step=1.0)
    st.session_state.features['SO2'] = st.number_input("SO2", value=st.session_state.features['SO2'], format="%.2f", step=1.0)
    st.session_state.features['O3'] = st.number_input("O3", value=st.session_state.features['O3'], format="%.2f", step=1.0)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Display current features in pollutant cards with icons and uniform styling
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.write("Current Pollutant Levels:")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="pollutant-card"><i class="fas fa-cloud"></i>Particulate Matter (PM2.5)<br>{st.session_state.features["PM2.5"]:.2f} µg/m³</div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="pollutant-card"><i class="fas fa-cloud"></i>Particulate Matter (PM10)<br>{st.session_state.features["PM10"]:.2f} µg/m³</div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="pollutant-card"><i class="fas fa-fire"></i>Carbon Monoxide (CO)<br>{st.session_state.features["CO"]:.2f} ppb</div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="pollutant-card"><i class="fas fa-flask"></i>Sulfur Dioxide (SO2)<br>{st.session_state.features["SO2"]:.2f} ppb</div>', unsafe_allow_html=True)
col5, col6, col7, col8 = st.columns(4)
with col5:
    st.markdown(f'<div class="pollutant-card"><i class="fas fa-wind"></i>Nitrogen Dioxide (NO2)<br>{st.session_state.features["NO2"]:.2f} ppb</div>', unsafe_allow_html=True)
with col6:
    st.markdown(f'<div class="pollutant-card"><i class="fas fa-sun"></i>Ozone (O3)<br>{st.session_state.features["O3"]:.2f} ppb</div>', unsafe_allow_html=True)
with col7:
    st.markdown(f'<div class="pollutant-card"><i class="fas fa-cloud-rain"></i>NO<br>{st.session_state.features["NO"]:.2f} ppb</div>', unsafe_allow_html=True)
with col8:
    st.markdown(f'<div class="pollutant-card"><i class="fas fa-leaf"></i>NH3<br>{st.session_state.features["NH3"]:.2f} ppb</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Predict AQI button and display
if st.button("Predict AQI"):
    if city and date:
        aqi = predict_aqi(model_options[selected_model], st.session_state.features, month)
        classification, color = classify_aqi(aqi)
        st.markdown(f'<div class="glass-card" style="background: linear-gradient(135deg, rgba(0,0,0,0.7), rgba(0,0,0,0.3)); text-align: center;">', unsafe_allow_html=True)
        st.markdown(f'<h2 style="color: {color};">Live AQI: {aqi:.2f}</h2>', unsafe_allow_html=True)
        st.markdown(f'<h3 style="color: white;">Air Quality is {classification}</h3>', unsafe_allow_html=True)
        st.markdown(f'<div style="background-color: {color}; height: 20px; width: 100%; border-radius: 5px;"></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("Please fill all fields and fetch or enter pollutant data.")

# Run with: streamlit run your_script.py