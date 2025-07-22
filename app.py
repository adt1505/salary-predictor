import streamlit as st
import joblib
import numpy as np

model = joblib.load("logistic_salary_model.pkl")

st.title("💼 Employer Salary Predictor")

age = st.slider("Age", 18, 70, 30)
education_num = st.slider("Education Number", 1, 16, 10)
marital = st.selectbox("Marital Status", ["Married", "Single"])
occupation = st.selectbox("Occupation", ["Tech", "Exec", "Sales"])
capital_gain = st.number_input("Capital Gain")
capital_loss = st.number_input("Capital Loss")
hours = st.slider("Hours/Week", 1, 100, 40)

marital_encoded = 1 if marital == "Married" else 0
occupation_map = {"Tech": 0, "Exec": 1, "Sales": 2}
occupation_encoded = occupation_map[occupation]

input_data = np.array([[age, education_num, marital_encoded, occupation_encoded, capital_gain, capital_loss, hours]])

if st.button("Predict Salary"):
    prediction = model.predict(input_data)[0]
    result = ">50K" if prediction == 1 else "<=50K"
    st.success(f"Predicted Salary: {result}")
