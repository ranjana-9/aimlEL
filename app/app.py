import streamlit as st
import numpy as np
import pandas as pd
from easypreprocessing import EasyPreProcessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import xgboost

# Title of the App
st.title("Product Demand Forecasting Using Machine Learning")

# File Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")
if uploaded_file is not None:
    try:
        # Try reading the file
        data = pd.read_csv(uploaded_file)
        
        # Check if the file is empty
        if data.empty:
            st.error("Uploaded file is empty. Please upload a valid CSV file with data.")
        else:
            st.write("Uploaded Data Preview:")
            st.dataframe(data.head())

            # Preprocessing Section
            st.header("Data Preprocessing")
            prep = EasyPreProcessing(uploaded_file)
            st.write("Dataset Info:")
            st.text(prep.info)
            st.write("Categorical Fields:", prep.categorical.fields)
            st.write("Numerical Fields:", prep.numerical.fields)

            # Handling Missing Values
            st.write("Handling Missing Values")
            prep.numerical.impute()
            st.write("Preprocessed Dataset Preview:")
            st.dataframe(prep.df.head())

            # Feature Engineering
            st.header("Feature Engineering")
            prep.df['key'] = prep.df['week'].astype(str) + '_' + prep.df['store_id'].astype(str)
            prep.dataset = prep.df.drop(['record_ID', 'week', 'store_id', 'sku_id', 'total_price', 'base_price', 'is_featured_sku', 'is_display_sku'], axis=1)
            prep.dataset = prep.df.groupby('key').sum()
            st.write("Feature-Engineered Dataset Preview:")
            st.dataframe(prep.dataset.head())

            # Time-Series Lag Features
            st.write("Generating Lag Features")
            prep.df['day_1'] = prep.df['units_sold'].shift(-1)
            prep.df['day_2'] = prep.df['units_sold'].shift(-2)
            prep.df['day_3'] = prep.df['units_sold'].shift(-3)
            prep.df['day_4'] = prep.df['units_sold'].shift(-4)
            st.dataframe(prep.df.head())

            # Data Splitting
            df = prep.df.dropna()
            x1, x2, x3, x4, y = df['day_1'], df['day_2'], df['day_3'], df['day_4'], df['units_sold']
            x1, x2, x3, x4, y = np.array(x1), np.array(x2), np.array(x3), np.array(x4), np.array(y)
            x1, x2, x3, x4, y = x1.reshape(-1, 1), x2.reshape(-1, 1), x3.reshape(-1, 1), x4.reshape(-1, 1), y.reshape(-1, 1)

            split_percentage = 15
            test_split = int(len(df) * (split_percentage / 100))
            x = np.concatenate((x1, x2, x3, x4), axis=1)
            X_train, X_test, y_train, y_test = x[:-test_split], x[-test_split:], y[:-test_split], y[-test_split:]

            st.write("Train/Test Split:")
            st.write("Training Set:", X_train.shape)
            st.write("Testing Set:", X_test.shape)

            # Random Forest Model
            rf_regressor = RandomForestRegressor()
            rf_regressor.fit(X_train, y_train)
            y_pred = rf_regressor.predict(X_test)
            st.write("Random Forest R² Score:", rf_regressor.score(X_test, y_test))

            # Visualization of Predictions
            st.header("Prediction Visualization")
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(y_pred[-100:], label="Predictions")
            ax.plot(y_test[-100:], label="Actual Sales")
            ax.legend(loc="upper left")
            st.pyplot(fig)

            # Hyperparameter Tuning
            st.header("Hyperparameter Tuning (RandomizedSearchCV)")
            random_grid = {
                'n_estimators': [int(x) for x in np.linspace(start=50, stop=250, num=10)],
                'max_features': ['auto', 'sqrt'],
                'max_depth': [int(x) for x in np.linspace(0, 120, num=20)] + [None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            }

            rf = RandomForestRegressor()
            rf_random = RandomizedSearchCV(
                estimator=rf,
                param_distributions=random_grid,
                n_iter=10,
                cv=3,
                verbose=2,
                random_state=0,
                n_jobs=-1
            )
            rf_random.fit(X_train, y_train)
            best_random = rf_random.best_estimator_
            y_pred = best_random.predict(X_test)
            st.write("Best Hyperparameters:", rf_random.best_params_)
            st.write("Tuned Random Forest R² Score:", best_random.score(X_test, y_test))

            # Final Visualization
            fig, ax = plt.subplots(figsize=(30, 8))
            ax.plot(y_pred[500:800], label="Predictions")
            ax.plot(y_test[500:800], label="Actual Sales")
            ax.legend(loc="upper left")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error reading the CSV file: {str(e)}")
