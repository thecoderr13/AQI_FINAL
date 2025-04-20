#code to checking working of the API key
import requests
import json
import os 
from dotenv import load_dotenv

# Replace with your OpenWeather API key
load_dotenv()  # This loads variables from .env file
API_KEY = os.getenv("API_KEY")  # Make sure the variable name matches exactly


# Coordinates for a sample city (Delhi)
lat = 28.6139
lon = 77.2090

# Air Pollution API endpoint
url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"

# Send request
response = requests.get(url)

# Print response status and data
print("Status Code:", response.status_code)

if response.status_code == 200:
    data = response.json()
    print(json.dumps(data, indent=4))  # Pretty print the JSON response
else:
    print("Error:", response.text)