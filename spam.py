import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="SMS Spam Detection", page_icon="📨")
st.title("📨 SMS Spam Detection App")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("SMSSpamCollection.txt", sep="\t", header=None, names=["label", "message"])
    return df

# Train and save model if not already saved
@st.cache_resource
def train_and_save_model(df):
    X = df["message"]
    y = df["label"].map({"ham": 0, "spam": 1})

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", MultinomialNB())
    ])

    pipeline.fit(X, y)

    joblib.dump(pipeline, "spam_model.pkl")
    return pipeline

# Load or train model
if os.path.exists("spam_model.pkl"):
    model = joblib.load("spam_model.pkl")
else:
    df = load_data()
    model = train_and_save_model(df)

# Show data sample
if st.checkbox("Show raw dataset"):
    df = load_data()
    st.write(df.head())

# User input
st.subheader("Check if your SMS is spam:")
user_input = st.text_area("Enter your SMS message:")

if st.button("Detect"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        prediction = model.predict([user_input])[0]
        proba = model.predict_proba([user_input])[0]

        label = "🚫 Spam" if prediction == 1 else "✅ Not Spam"
        st.success(f"Prediction: {label}")
        st.write(f"Spam Probability: {proba[1]:.2f}")
