import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('outputs/model.pkl')

st.set_page_config(page_title="CreditSense AI", layout="centered")

st.title("💳 CreditSense - AI Loan Approval System")

st.markdown("Enter applicant details below:")

# User Inputs
income = st.number_input("Annual Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
credit_score = st.slider("Credit Score", 300, 900, 650)
age = st.number_input("Age", min_value=18, max_value=100)

# Predict Button
if st.button("Check Loan Eligibility"):

    # Create DataFrame
    data = pd.DataFrame({
        'income': [income],
        'loan_amount': [loan_amount],
        'credit_score': [credit_score],
        'age': [age]
    })

    # Prediction
    prob = model.predict_proba(data)[0][1]

    st.subheader(f"Risk Score: {round(prob, 2)}")

    # Decision Logic
    if prob < 0.3:
        decision = "✅ Approved"
        st.success(decision)
    elif prob < 0.6:
        decision = "⚠️ Under Review"
        st.warning(decision)
    else:
        decision = "❌ Rejected"
        st.error(decision)