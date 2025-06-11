import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.title("📈 Stock Market Financial Health Classifier")

# Load and cache data
@st.cache_data
def load_data():
    # Load the ANN dataset
    df = pd.read_csv("stock_market_ann_dataset.csv")
    
    # Convert scientific notation to float
    df['property_value'] = df['property_value'].astype(float)
    df['debt_value'] = df['debt_value'].astype(float)
    
    # Encode the status labels
    le = LabelEncoder()
    df['status_encoded'] = le.fit_transform(df['status'])
    
    X = df[['property_value', 'debt_value', 'revenue_growth', 'profit_margin']]
    y = df['status_encoded']
    status_names = df['status'].unique()
    
    return X, y, status_names

X, y, status_names = load_data()

# Train and cache model
@st.cache_resource
def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model(X, y)

# Sidebar inputs
st.sidebar.header("Financial Metrics Input")

# Get min, max, and mean values for sliders
property_min = float(X['property_value'].min())
property_max = float(X['property_value'].max())
property_mean = float(X['property_value'].mean())

debt_min = float(X['debt_value'].min())
debt_max = float(X['debt_value'].max())
debt_mean = float(X['debt_value'].mean())

growth_min = float(X['revenue_growth'].min())
growth_max = float(X['revenue_growth'].max())
growth_mean = float(X['revenue_growth'].mean())

margin_min = float(X['profit_margin'].min())
margin_max = float(X['profit_margin'].max())
margin_mean = float(X['profit_margin'].mean())

# Create sliders
property_value = st.sidebar.slider(
    "Property Value (USD)", 
    property_min, 
    property_max, 
    property_mean,
    help="Total value of company assets"
)

debt_value = st.sidebar.slider(
    "Debt Value (USD)", 
    debt_min, 
    debt_max, 
    debt_mean,
    help="Total company debt"
)

revenue_growth = st.sidebar.slider(
    "Revenue Growth Rate", 
    growth_min, 
    growth_max, 
    growth_mean,
    help="Percentage change in revenue year-over-year",
    format="%.4f"
)

profit_margin = st.sidebar.slider(
    "Profit Margin", 
    margin_min, 
    margin_max, 
    margin_mean,
    help="Net profit as percentage of revenue",
    format="%.4f"
)

input_data = pd.DataFrame([[property_value, debt_value, revenue_growth, profit_margin]], 
                         columns=X.columns)

# Predict
if st.button("Assess Financial Health"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    
    # Display results
    st.subheader("Assessment Result")
    
    if status_names[prediction] == "Healthy":
        st.success(f"✅ Status: **{status_names[prediction]}**")
    elif status_names[prediction] == "At-Risk":
        st.warning(f"⚠️ Status: **{status_names[prediction]}**")
    else:  # Critical
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
    elif status_names[prediction] == "At-Risk":
        st.warning("The company shows some concerning financial indicators that warrant closer examination.")
    else:
        st.error("The company's financial metrics indicate critical issues that require immediate attention.")
else:
    st.info("⬅️ Adjust the financial metrics and click 'Assess Financial Health'")
