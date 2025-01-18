import streamlit as st
import numpy as np
import joblib
import os
from datetime import datetime
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Sales Forecasting App",
    page_icon="📊",
    layout="wide"
)

# Define the path for Google Drive
DRIVE_PATH = '/content/drive/MyDrive/ForecastData'

def load_model_and_scaler():
    """Load the model and scaler"""
    try:
        model = joblib.load(os.path.join(DRIVE_PATH, 'sales_forecast_model.joblib'))
        scaler = joblib.load(os.path.join(DRIVE_PATH, 'sales_scaler.joblib'))
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def predict_sales(sales_data, model, scaler):
    """Make sales prediction"""
    try:
        # Scale the input data
        scaled_input = scaler.transform(sales_data)
        # Make prediction
        prediction = model.predict(scaled_input)[0]
        return prediction
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def main():
    # Header
    st.title("📊 Sales Forecasting Application")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info(
        "This application predicts future sales based on the "
        "previous 4 days of sales data using a Random Forest model."
    )
    
    # Load model
    model, scaler = load_model_and_scaler()
    
    if model is None or scaler is None:
        st.error("Failed to load model. Please check if model files exist in the correct location.")
        return
    
    # Main content
    st.subheader("Enter Sales Data")
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Manual Entry", "CSV Upload"]
        )
        
        if input_method == "Manual Entry":
            # Manual input fields
            day1 = st.number_input("Sales from 1 day ago", min_value=0.0, value=100.0)
            day2 = st.number_input("Sales from 2 days ago", min_value=0.0, value=100.0)
            day3 = st.number_input("Sales from 3 days ago", min_value=0.0, value=100.0)
            day4 = st.number_input("Sales from 4 days ago", min_value=0.0, value=100.0)
            
            if st.button("Make Prediction"):
                # Prepare input data
                input_data = np.array([[day1, day2, day3, day4]])
                
                # Get prediction
                prediction = predict_sales(input_data, model, scaler)
                
                if prediction is not None:
                    # Calculate statistics
                    avg_input = np.mean([day1, day2, day3, day4])
                    pct_change = ((prediction - avg_input) / avg_input * 100)
                    
                    # Display results
                    with col2:
                        st.subheader("Prediction Results")
                        st.metric(
                            label="Predicted Sales",
                            value=f"{prediction:.2f} units",
                            delta=f"{pct_change:.1f}% from average"
                        )
                        
                        # Additional metrics
                        st.markdown("**Additional Metrics:**")
                        col3, col4 = st.columns(2)
                        with col3:
                            st.metric("Average of Previous 4 Days", f"{avg_input:.2f} units")
                        with col4:
                            st.metric("Percentage Change", f"{pct_change:.1f}%")
                        
                        # Create a small dataframe for visualization
                        dates = pd.date_range(end=datetime.now(), periods=5, freq='D')
                        data = {
                            'Date': dates,
                            'Sales': [day4, day3, day2, day1, prediction]
                        }
                        df = pd.DataFrame(data)
                        
                        # Plot
                        st.line_chart(df.set_index('Date')['Sales'])
        
        else:  # CSV Upload
            st.info("Upload a CSV file with columns for the last 4 days of sales")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    if len(df.columns) < 4:
                        st.error("CSV must have at least 4 columns for the last 4 days of sales")
                    else:
                        # Make predictions for each row
                        predictions = []
                        for _, row in df.iloc[:, :4].iterrows():
                            input_data = np.array([row.values])
                            pred = predict_sales(input_data, model, scaler)
                            predictions.append(pred)
                        
                        # Add predictions to dataframe
                        df['Predicted_Sales'] = predictions
                        
                        # Display results
                        with col2:
                            st.subheader("Batch Predictions")
                            st.dataframe(df)
                            
                            # Download predictions
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="Download Predictions as CSV",
                                data=csv,
                                file_name="sales_predictions.csv",
                                mime="text/csv"
                            )
                
                except Exception as e:
                    st.error(f"Error processing CSV: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown("Made with ❤️ by Your Team")

if __name__ == "__main__":
    main()
