import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("simple_salary_model.pkl")

st.title("Employer Salary Predictor")

# User inputs (must match training features)
age = st.slider("Age", 18, 70, 30)
education_num = st.slider("Education Number (1–16)", 1, 16, 10)
marital_status = st.selectbox("Marital Status", ["Never-married", "Married-civ-spouse", "Divorced"])
occupation = st.selectbox("Occupation", ["Tech-support", "Exec-managerial", "Sales"])
capital_gain = st.number_input("Capital Gain", min_value=0)
hours_per_week = st.slider("Hours per Week", 1, 100, 40)

# Manual encoding (simplified for matching training)
marital_map = {"Never-married": 4, "Married-civ-spouse": 2, "Divorced": 0}
occupation_map = {"Tech-support": 11, "Exec-managerial": 0, "Sales": 9}

# Construct input
input_data = np.array([[age, education_num, marital_map[marital_status], occupation_map[occupation], capital_gain, hours_per_week]])

# Prediction
if st.button("Predict Salary"):
    prediction = model.predict(input_data)[0]
    result = ">50K" if prediction == 1 else "<=50K"
    st.success(f"💰 Predicted Salary: {result}")
