### 🌫️ AQI Prediction System
This project aims to predict the Air Quality Index (AQI) using various regression models including Linear Regression, Lasso, Random Forest, XGBoost, and LSTM. It features data cleaning, model training, preprocessing pipelines, and a Streamlit-based frontend for user interaction.

### 🚀 Getting Started
### ✅ Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/AQI_FINAL.git
cd AQI_FINAL
```


### ✅ Step 2: Add the Filtered Dataset

```bash
Ensure the filtered_dataset.csv is placed in the data_clean/ folder:
AQI_FINAL/data_clean/filtered_dataset.csv
```

### ✅ Step 3: Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### ✅ Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### ✅ Step 5: Run the Application

```bash
streamlit run app.py
```


### 📊 Models Included

✅ Linear Regression
✅ Lasso Regression
✅ Random Forest Regressor
✅ XGBoost Regressor
✅ LSTM (Sequential)
✅ Support Vector Regression
✅ ARIMA (for time series)

All models are trained using cleaned AQI datasets and stored in the output_trained/ folder as .pkl files for fast inference.
