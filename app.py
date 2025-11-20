import streamlit as st
import pickle
import numpy as np

# Load Model & Scaler
model = pickle.load(open("../model/loan_model.pkl", "rb"))
scaler = pickle.load(open("../model/scaler.pkl", "rb"))

st.title("🏦 Bank Loan Eligibility Prediction App")
st.write("Enter the customer details below to check loan approval probability.")

# Inputs
income = st.number_input("Income (per year)", min_value=10000, max_value=2000000)
credit_score = st.number_input("Credit Score (300-900)", min_value=300, max_value=900)
loan_amount = st.number_input("Loan Amount", min_value=10000, max_value=2000000)
loan_term = st.number_input("Loan Term (Months)", min_value=6, max_value=360)
age = st.number_input("Age", min_value=18, max_value=80)
existing_debt = st.number_input("Existing Debt", min_value=0, max_value=1000000)
employment = st.selectbox("Employment Type", ["Salaried", "Self-employed"])

# Convert employment to dummy variable
employment_self = 1 if employment == "Self-employed" else 0

if st.button("Predict"):
    row = np.array([
        income, credit_score, loan_amount, loan_term,
        age, existing_debt, employment_self
    ]).reshape(1, -1)

    row_scaled = scaler.transform(row)
    prediction = model.predict(row_scaled)[0]
    probability = model.predict_proba(row_scaled)[0][1]

    if prediction == 1:
        st.success(f"✔ Loan Approved! Probability: {probability:.2f}")
    else:
        st.error(f"✘ Loan Not Approved. Probability: {probability:.2f}")
