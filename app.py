import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Set page config first
st.set_page_config(page_title="Stock Market Predictor", layout="wide")

# Add title and description
st.title("📈 Stock Market Financial Health Predictor")
st.write("This app predicts whether a company is financially healthy based on its property and debt values.")

# Load data function with error handling
@st.cache_data
def load_data():
    try:
        # Sample data - replace with your actual data loading
        data = {
            'property': [58.01, 64.54, 39.69, 25.84, 12.75],
            'debt': [31.37, 30.83, 9.33, 11.00, 0.62],
            'status': [1, 0, 0, 1, 0]
        }
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

df = load_data()

# Show raw data if checkbox is selected
if st.checkbox('Show raw data'):
    st.subheader('Raw Data')
    st.write(df)

# Create input sliders in sidebar
st.sidebar.header('Input Parameters')
property_val = st.sidebar.slider(
    'Property Value', 
    float(df['property'].min()), 
    float(df['property'].max()), 
    float(df['property'].mean())
)
debt_val = st.sidebar.slider(
    'Debt Value', 
    float(df['debt'].min()), 
    float(df['debt'].max()), 
    float(df['debt'].mean())
)

# Prediction function
def predict(property_val, debt_val):
    try:
        # In a real app, you would load your trained model here
        # model = joblib.load('model.pkl')
        # scaler = joblib.load('scaler.pkl')
        
        # For demo purposes, we'll use a simple rule-based approach
        ratio = property_val / (debt_val + 1e-6)  # Avoid division by zero
        prediction = 1 if ratio > 1.5 else 0  # Simple threshold
        confidence = min(0.95, abs(ratio - 1.5) / 2)  # Fake confidence score
        
        return prediction, confidence
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

# Make and display prediction
if st.button('Predict Financial Health'):
    prediction, confidence = predict(property_val, debt_val)
    
    if prediction is not None:
        st.subheader('Prediction Result')
        
        if prediction == 1:
            st.success('✅ Healthy (confidence: {:.1%})'.format(confidence))
            st.write("The company appears financially stable based on the inputs.")
        else:
            st.error('❌ At-Risk (confidence: {:.1%})'.format(confidence))
            st.write("The company shows signs of financial distress based on the inputs.")
        
        # Show some visualizations
        col1, col2 = st.columns(2)
        with col1:
            st.write("Property vs Debt Ratio")
            chart_data = pd.DataFrame({
                'Metric': ['Property', 'Debt'],
                'Value': [property_val, debt_val]
            })
            st.bar_chart(chart_data.set_index('Metric'))
        
        with col2:
            st.write("Financial Health Indicator")
            gauge_data = pd.DataFrame({
                'Status': ['Healthy Threshold', 'Current Ratio'],
                'Value': [1.5, property_val/(debt_val + 1e-6)]
            })
            st.bar_chart(gauge_data.set_index('Status'))
    else:
        st.warning("Could not make prediction")

# Add some explanations
st.markdown("""
### How It Works
- Adjust the property and debt values using the sliders in the sidebar
- Click the 'Predict Financial Health' button
- The model evaluates the ratio of property to debt
- Results show whether the company is likely healthy or at-risk
""")