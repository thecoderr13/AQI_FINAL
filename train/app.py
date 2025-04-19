import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import os
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Function to load the trained model
def load_model(model_name):
    model_path = f"{model_name}_model.pkl"
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            print(f"Loaded {model_name} model type: {type(model)}, has predict: {hasattr(model, 'predict')}")
            return model
        except Exception as e:
            st.error(f"Error loading model from {model_path}: {str(e)}")
            return None
    else:
        st.error(f"Model file {model_path} not found! Current directory: {os.getcwd()}")
        return None

# Function to map AQI to AQI_Bucket
def get_aqi_bucket(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

# Function to preprocess input data
def preprocess_input(city, date_time, pm25, pm10, no, no2, nox, co, cities, mean_values):
    # Extract datetime features
    features = {
        'Year': date_time.year,
        'Month': date_time.month,
        'Day': date_time.day,
        'DayOfWeek': date_time.weekday(),
        'Hour': date_time.hour,
        'PM2.5': pm25,
        'PM10': pm10,
        'NO': no,
        'NO2': no2,
        'NOx': nox,
        'CO': co,
        'City': city
    }
    
    # Label encode city
    le = LabelEncoder()
    le.classes_ = cities  # Use the same city list as training
    features['City'] = le.transform([city])[0]
    
    # Convert to DataFrame
    input_df = pd.DataFrame([features])
    
    # Ensure all expected columns are present
    expected_columns = ['City', 'Year', 'Month', 'Day', 'DayOfWeek', 'Hour', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'CO']
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]
    
    # Fill missing values with mean values
    input_df.fillna(mean_values, inplace=True)
    
    return input_df

# Function to plot historical AQI
def plot_historical_aqi(df, city):
    city_data = df[df['City'] == city].copy()
    if city_data.empty:
        st.warning(f"No historical data available for {city}.")
        return
    
    city_data['Datetime'] = pd.to_datetime(city_data['Date'] + ' ' + city_data['Time'])
    city_data = city_data.sort_values('Datetime')
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(city_data['Datetime'], city_data['AQI'], label='AQI', color='blue')
    ax.set_title(f"Historical AQI for {city}")
    ax.set_xlabel("Date and Time")
    ax.set_ylabel("AQI")
    ax.grid(True)
    ax.legend(loc="upper left")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

# Streamlit app
st.title("AQI Prediction App")

# Load training data to get city list, mean values, and historical data
@st.cache_data
def load_training_data():
    df = pd.read_csv(os.path.join('..', 'data_clean', 'filtered_dataset.csv'))
    df.dropna(subset=["AQI"], inplace=True)
    cities = df['City'].unique()
    mean_values = df[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'CO']].mean()
    return df, cities, mean_values

df, cities, mean_values = load_training_data()

# User inputs
st.header("Enter Details")
city = st.selectbox("Select City", options=cities)
date = st.date_input("Select Date")
time = st.time_input("Select Time")

# Pollutant inputs with autofill option
st.subheader("Pollutant Levels (Optional)")
st.write("Enter pollutant values or click 'Autofill' to use average values from the dataset.")

# Initialize session state for pollutant values and autofill status
if 'pollutant_values' not in st.session_state:
    st.session_state.pollutant_values = {
        'PM2.5': float(mean_values['PM2.5']),
        'PM10': float(mean_values['PM10']),
        'NO': float(mean_values['NO']),
        'NO2': float(mean_values['NO2']),
        'NOx': float(mean_values['NOx']),
        'CO': float(mean_values['CO'])
    }
if 'autofill_used' not in st.session_state:
    st.session_state.autofill_used = False

# Autofill button
if st.button("Autofill Pollutant Values"):
    st.session_state.pollutant_values = {
        'PM2.5': float(mean_values['PM2.5']),
        'PM10': float(mean_values['PM10']),
        'NO': float(mean_values['NO']),
        'NO2': float(mean_values['NO2']),
        'NOx': float(mean_values['NOx']),
        'CO': float(mean_values['CO'])
    }
    st.session_state.autofill_used = True
    st.success("Pollutant values set to dataset averages.")

# Pollutant input fields
pm25 = st.number_input("PM2.5", min_value=0.0, value=st.session_state.pollutant_values['PM2.5'], step=0.1, key='pm25')
pm10 = st.number_input("PM10", min_value=0.0, value=st.session_state.pollutant_values['PM10'], step=0.1, key='pm10')
no = st.number_input("NO", min_value=0.0, value=st.session_state.pollutant_values['NO'], step=0.01, key='no')
no2 = st.number_input("NO2", min_value=0.0, value=st.session_state.pollutant_values['NO2'], step=0.01, key='no2')
nox = st.number_input("NOx", min_value=0.0, value=st.session_state.pollutant_values['NOx'], step=0.01, key='nox')
co = st.number_input("CO", min_value=0.0, value=st.session_state.pollutant_values['CO'], step=0.01, key='co')

# Update session state when user changes input
st.session_state.pollutant_values = {
    'PM2.5': pm25,
    'PM10': pm10,
    'NO': no,
    'NO2': no2,
    'NOx': nox,
    'CO': co
}

# Reset autofill status if user modifies inputs
if st.session_state.autofill_used:
    if any(st.session_state.pollutant_values[key] != mean_values[key] for key in st.session_state.pollutant_values):
        st.session_state.autofill_used = False

# Model selection
model_choice = st.selectbox("Select Model", options=[
    "Polynomial Regression",
    "Linear Regression",
    "Lasso Regression",
    "Random Forest",
    "XGBoost"
])

# Combine date and time
date_time = datetime.combine(date, time)

# Display historical AQI graph
st.subheader("Historical AQI for Selected City")
plot_historical_aqi(df, city)

# Predict button
if st.button("Predict AQI"):
    # Map model choice to model file name
    model_mapping = {
        "Polynomial Regression": "polynomial_regression",
        "Linear Regression": "linear_regression",
        "Lasso Regression": "lasso_regression",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost"
    }
    
    # Load the selected model
    model_name = model_mapping[model_choice]
    model = load_model(model_name)
    
    if model is not None and hasattr(model, 'predict'):
        # Preprocess input
        with st.spinner("Predicting..."):
            input_data = preprocess_input(
                city, date_time, pm25, pm10, no, no2, nox, co,
                cities, mean_values
            )
            print(f"Input data shape: {input_data.shape}, columns: {input_data.columns.tolist()}")
            # Make prediction
            try:
                prediction = model.predict(input_data)[0]
                aqi_bucket = get_aqi_bucket(prediction)
                
                # Display prediction and AQI bucket
                st.success(f"Predicted AQI: {prediction:.2f}")
                st.write(f"Predicted AQI Bucket: {aqi_bucket}")
                
                # Display pollutant values used
                st.subheader("Pollutant Values Used")
                st.write(f"PM2.5: {pm25:.2f}")
                st.write(f"PM10: {pm10:.2f}")
                st.write(f"NO: {no:.2f}")
                st.write(f"NO2: {no2:.2f}")
                st.write(f"NOx: {nox:.2f}")
                st.write(f"CO: {co:.2f}")
                
                # Indicate if autofill was used
                if st.session_state.autofill_used:
                    st.info("Pollutant values were autofilled with dataset averages.")
                    
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
    else:
        st.error("Invalid model loaded. Please check the model file or re-save the model.")

# App info
st.write("This app predicts AQI and AQI Bucket based on city, date, time, and pollutant levels using pre-trained models. You can enter pollutant values or use the 'Autofill' button to set them to dataset averages. The graph shows historical AQI for the selected city.")