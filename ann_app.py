import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("📊 Stock Market Financial Health Classifier")

# Load and cache data
@st.cache_data
def load_data():
    # Load both datasets
    df1 = pd.read_csv("ann_stock_market_data.csv")
    df2 = pd.read_csv("stock_market_ann_dataset.csv")
    
    # Rename columns for consistency
    df1.columns = ['property_value', 'debt_value', 'status']
    df2 = df2[['property_value', 'debt_value', 'status']]
    
    # Convert scientific notation to float for df2
    df2['property_value'] = df2['property_value'].astype(float)
    df2['debt_value'] = df2['debt_value'].astype(float)
    
    # Normalize df2 values to match df1 scale (0-100 range)
    df2['property_value'] = (df2['property_value'] - df2['property_value'].min()) / (df2['property_value'].max() - df2['property_value'].min()) * 100
    df2['debt_value'] = (df2['debt_value'] - df2['debt_value'].min()) / (df2['debt_value'].max() - df2['debt_value'].min()) * 100
    
    # Combine datasets
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    # Encode status (0 = At-Risk, 1 = Healthy)
    le = LabelEncoder()
    combined_df['status_encoded'] = le.fit_transform(combined_df['status'])
    
    X = combined_df[['property_value', 'debt_value']]
    y = combined_df['status_encoded']
    status_names = ['At-Risk', 'Healthy']  # Simplified to binary classification
    
    return X, y, status_names, combined_df

X, y, status_names, combined_df = load_data()

# Split data and train model
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

model, model_accuracy = train_model(X, y)

# Sidebar inputs
st.sidebar.header("Financial Metrics Input")

# Get min, max, and mean values for sliders
property_min = float(X['property_value'].min())
property_max = float(X['property_value'].max())
property_mean = float(X['property_value'].mean())

debt_min = float(X['debt_value'].min())
debt_max = float(X['debt_value'].max())
debt_mean = float(X['debt_value'].mean())

# Create sliders
property_value = st.sidebar.slider(
    "Property Value (0-100 scale)", 
    property_min, 
    property_max, 
    property_mean,
    help="Normalized property value (0-100 scale)"
)

debt_value = st.sidebar.slider(
    "Debt Value (0-100 scale)", 
    debt_min, 
    debt_max, 
    debt_mean,
    help="Normalized debt value (0-100 scale)"
)

input_data = pd.DataFrame([[property_value, debt_value]], 
                         columns=X.columns)

# Main content
st.subheader("Model Information")
st.write(f"Model Accuracy: {model_accuracy:.2%}")
st.write("Dataset Overview:")
st.dataframe(combined_df.head())

# Visualizations
col1, col2 = st.columns(2)
with col1:
    st.write("Property Value Distribution")
    st.bar_chart(combined_df['property_value'].value_counts())
with col2:
    st.write("Debt Value Distribution")
    st.bar_chart(combined_df['debt_value'].value_counts())

# Predict
if st.button("Assess Financial Health"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    
    # Display results
    st.subheader("Assessment Result")
    
    if status_names[prediction] == "Healthy":
        st.success(f"✅ Status: **{status_names[prediction]}**")
    else:
        st.error(f"❌ Status: **{status_names[prediction]}**")
    
    st.subheader("Probability Distribution")
    
    # Create a more informative chart
    prob_df = pd.DataFrame({
        'Status': status_names,
        'Probability': proba
    }).sort_values('Probability', ascending=False)
    
    st.bar_chart(prob_df.set_index('Status'))
    
    # Add some interpretation
    st.subheader("Interpretation")
    if status_names[prediction] == "Healthy":
        st.info("The company appears to be in good financial health based on the provided metrics.")
    else:
        st.warning("The company shows concerning financial indicators that warrant closer examination.")
else:
    st.info("⬅️ Adjust the financial metrics and click 'Assess Financial Health'")

# Add data exploration section
st.subheader("Data Exploration")
st.write("Relationship between Property and Debt Values with Status")
st.scatter_chart(combined_df, x='property_value', y='debt_value', color='status')
