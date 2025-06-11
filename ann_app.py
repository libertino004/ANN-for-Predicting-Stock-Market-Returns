import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

st.title("📈 ANN Stock Market Return Prediction")

@st.cache_data
def load_csv_data():
    file_option = st.selectbox("Choose dataset file", ["ann_stock_market_data.csv", "stock_market_ann_dataset.csv"])
    return pd.read_csv(file_option)

df = load_csv_data()
st.subheader("Preview of Data")
st.write(df.head())

@st.cache_resource
def load_ann_model():
    model = load_model("ann_stock_model.h5")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_ann_model()

st.sidebar.header("Input Stock Features")
input_data = {}
for col in df.columns[:-1]:  # exclude target column
    val = st.sidebar.number_input(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
    input_data[col] = val

input_df = pd.DataFrame([input_data])

if st.button("Predict Return"):
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0][0]
    st.subheader("📊 Prediction Result")
    st.success(f"Predicted Return: **{prediction:.4f}**")
else:
    st.info("⬅️ Input features and click Predict")
