import streamlit as st
import pandas as pd
import importlib
from utils.preprocessing import preprocess_data
from utils.metrics import evaluate_classification, evaluate_regression, get_confusion
from utils.visualizations import plot_confusion_matrix, plot_error_distribution, plot_feature_importance
from utils.helpers import is_classification, is_timeseries_model, get_model_summary

st.set_page_config(page_title="AQI Prediction App", layout="wide")
st.title("🌫️ AQI Prediction App")

# Sidebar Settings
st.sidebar.header("⚙️ Settings")
smote_enabled = st.sidebar.checkbox("Enable SMOTE (for class imbalance)", value=True)
split_ratio = st.sidebar.slider("Train-Test Split Ratio (Test size %)", min_value=10, max_value=50, value=20, step=5)

# Upload dataset
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    target_column = st.selectbox("🎯 Select Target Column", df.columns)

    model_options = {
        "Linear Regression": "linear_regression",
        "Logistic Regression": "logistic_regression",
        "Decision Tree": "decision_tree",
        "Random Forest": "random_forest",
        "SVR": "svr_model",
        "XGBoost": "xgboost_model",
        "LSTM": "lstm_model",
        "ARIMA": "arima_model"
    }
    selected_model = st.selectbox("🧠 Select Model", list(model_options.keys()))
    model_key = model_options[selected_model]

    if st.button("🚀 Train Model"):
        if is_timeseries_model(model_key):
            _, forecast = importlib.import_module(f"models.{model_key}").train(df[target_column])
            st.subheader("🔮 Forecast (ARIMA)")
            st.line_chart(forecast)
        else:
            test_size = split_ratio / 100.0
            X_train, X_test, y_train, y_test = preprocess_data(df, target_column, smote=smote_enabled)

            model_module = importlib.import_module(f"models.{model_key}")
            model, y_pred = model_module.train(X_train, X_test, y_train, y_test)

            if is_classification(y_train):
                st.subheader("📈 Classification Metrics")
                metrics = evaluate_classification(y_test, y_pred)
                cm = get_confusion(y_test, y_pred)
                st.json(metrics)
                plot_confusion_matrix(cm)
            else:
                st.subheader("📉 Regression Metrics")
                metrics = evaluate_regression(y_test, y_pred)
                st.json(metrics)
                plot_error_distribution(y_test, y_pred)

            # Feature Importance (if supported)
            if hasattr(model, "feature_importances_"):
                st.subheader("📊 Feature Importance")
                plot_feature_importance(model, X_train.columns)

            # Model Summary (if available)
            summary_text = get_model_summary(model)
            if summary_text:
                st.subheader("📘 Model Summary")
                st.code(summary_text, language="text")

            # Download Results
            result_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Prediction Results",
                data=csv,
                file_name='predictions.csv',
                mime='text/csv'
            )
