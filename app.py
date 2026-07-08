import streamlit as st
import pandas as pd
import joblib

# Load saved objects

model = joblib.load("gradient_boosting_model.pkl")
scaler = joblib.load("scaler.pkl")
pt = joblib.load("power_transformer_age.pkl")
features_names = joblib.load("features_names.pkl")

# Page Configuration
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

st.markdown("""
<style>

/* ===========================
   Main App
=========================== */

[data-testid="stAppViewContainer"]{
    background-color:#F4F7FC;
}

/* ===========================
   Sidebar
=========================== */

[data-testid="stSidebar"]{
    background-color:#1E3A5F;
}

/* Sidebar Header */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4{
    color:white !important;
}

/* Sidebar Labels */
[data-testid="stSidebar"] label{
    color:white !important;
    font-weight:600;
}

/* Sidebar Normal Text */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span{
    color:white !important;
}

/* ===========================
   Number Input
=========================== */

[data-testid="stSidebar"] input{
    background-color:white !important;
    color:black !important;
    border-radius:8px;
}

/* ===========================
   Select Box
=========================== */

[data-testid="stSidebar"] div[data-baseweb="select"]{
    background-color:white !important;
    border-radius:8px;
}

[data-testid="stSidebar"] div[data-baseweb="select"] *{
    color:black !important;
}

/* ===========================
   Main Text
=========================== */

h1,h2,h3,h4,p{
    color:#222222;
}

/* ===========================
   Button
=========================== */

.stButton > button {
    width: 100%;
    height: 60px;
    background-color: #1565C0;
    color: white !important;
    border: none;
    border-radius: 12px;
    font-size: 22px !important;
    font-weight: 700;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    background-color: #0D47A1;
    color: white !important;
    transform: scale(1.02);
}

.stButton > button p {
    color: white !important;
    font-size: 22px !important;
    font-weight: 700 !important;
}

/* ===========================
   Metric Cards
=========================== */

[data-testid="metric-container"]{
    background-color:white;
    border-radius:15px;
    padding:18px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.15);
}

/* ===========================
   Success Message
=========================== */

.stSuccess{
    border-radius:12px;
}

/* ===========================
   Error Message
=========================== */

.stError{
    border-radius:12px;
}

/* ===========================
   Divider
=========================== */

hr{
    border:1px solid #d9d9d9;
}

</style>
""", unsafe_allow_html=True)

st.title("Bank Customer Churn Prediction")
st.markdown("""Predict whether a customer is likely to churn.
            Fill in the customer's details from the left panel and click **Predict Custmomer Churn**
            """)

# User Input

st.sidebar.header("Customer Information")
credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=850, value=650)
Geography = st.sidebar.selectbox("Geography", options=["France", "Spain", "Germany"])
Gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=35)
tenure = st.sidebar.number_input("Tenure (in years)", min_value=0, max_value=10, value=5)
balance = st.sidebar.number_input("Balance", min_value=0.0, value=50000.0)
NumOfProducts = st.sidebar.selectbox("Number of Products", options=[1, 2, 3, 4])
HasCrCard = st.sidebar.selectbox("Has Credit Card", options=["No", "Yes"])
IsActiveMember = st.sidebar.selectbox("Is Active Member", options=["No", "Yes"])
EstimatedSalary = st.sidebar.number_input("Salary", min_value=0.0, value=50000.0)



# Prediction

predict = st.button(
    "🔍 Predict Customer Churn",
    use_container_width=True
)

if predict:

    st.subheader("Customer Summary")

    col1, col2 = st.columns(2)

    with col1:
      st.write(f"**Credit Score:** {credit_score}")
      st.write(f"**Age:** {age}")
      st.write(f"**Gender:** {Gender}")
      st.write(f"**Geography:** {Geography}")
      st.write(f"**Tenure:** {tenure}")

    with col2:
      st.write(f"**Balance:** ₹{balance:,.2f}")
      st.write(f"**Salary:** ₹{EstimatedSalary:,.2f}")
      st.write(f"**Products:** {NumOfProducts}")
      st.write(f"**Credit Card:** {HasCrCard}")
      st.write(f"**Active Member:** {IsActiveMember}")
    st.markdown("---")


    input_df = pd.DataFrame({
    "CreditScore": [credit_score],
    "Geography": [Geography],
    "Gender": [Gender],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [NumOfProducts],
    "HasCrCard": [1 if HasCrCard == "Yes" else 0],
    "IsActiveMember": [1 if IsActiveMember == "Yes" else 0],
    "EstimatedSalary": [EstimatedSalary]
})

    # Power Transformation

    input_df["Age"] = pt.transform(input_df[["Age"]])

    # Encode

    input_df = pd.get_dummies(input_df, drop_first = True, dtype= int)

    # Reorder

    input_df = input_df.reindex(columns=features_names, fill_value=0)

    # Scaling

    input_scaled = scaler.transform(input_df)

    # Prediction

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0][1]


    st.markdown("---")

    col1, col2, = st.columns(2)

    col1.metric(
    "Prediction",
    "⚠️ Customer is likely to Churn" if prediction==1 else "✅ Customer is likely to stay"
)

    col2.metric(
    "Probability",
    f"{probability*100:.2f}%"
)
    


