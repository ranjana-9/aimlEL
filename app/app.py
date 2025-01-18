import streamlit as st
import numpy as np
import pandas as pd
from easypreprocessing import EasyPreProcessing
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import xgboost

# Streamlit App Title
st.title("Product Demand Forecasting Using Machine Learning")

# File Uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    # Load and preprocess data
    prep = EasyPreProcessing(uploaded_file)
    st.write("Data Information:")
    st.write(prep.info)
    
    # Display dataset
    st.write("First 5 rows of the dataset:")
    st.dataframe(prep.df.head())
    
    # Data preprocessing
    prep.df['key'] = prep.df['week'].astype(str) + '_' + prep.df['store_id'].astype(str)
    prep.df = prep.df.drop(['record_ID', 'week', 'store_id', 'sku_id', 'total_price', 'base_price', 'is_featured_sku', 'is_display_sku'], axis=1)
    prep.df = prep.df.groupby('key').sum()
    
    # Feature Engineering
    for i in range(1, 5):
        prep.df[f'day_{i}'] = prep.df['units_sold'].shift(-i)
    
    df = prep.df.dropna()
    
    # Train-Test Split
    split_percentage = 15
    test_split = int(len(df) * (split_percentage / 100))
    x = np.array(df[[f'day_{i}' for i in range(1, 5)]])
    y = np.array(df['units_sold'])
    X_train, X_test = x[:-test_split], x[-test_split:]
    y_train, y_test = y[:-test_split], y[-test_split:]
    
    # Random Forest Model
    rf_regressor = RandomForestRegressor()
    rf_regressor.fit(X_train, y_train)
    rf_pred = rf_regressor.predict(X_test)
    rf_score = rf_regressor.score(X_test, y_test)
    
    # Display Random Forest Results
    st.write(f"R-Squared Score for Random Forest: {rf_score:.2f}")
    st.line_chart({"Predictions": rf_pred[-100:], "Actual": y_test[-100:]})
    
    # XGBoost Model
    xgb_regressor = xgboost.XGBRegressor()
    xgb_regressor.fit(X_train, y_train)
    xgb_pred = xgb_regressor.predict(X_test)
    xgb_score = xgb_regressor.score(X_test, y_test)
    
    # Display XGBoost Results
    st.write(f"R-Squared Score for XGBoost: {xgb_score:.2f}")
    st.line_chart({"Predictions": xgb_pred[-100:], "Actual": y_test[-100:]})
