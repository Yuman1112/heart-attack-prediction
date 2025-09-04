import streamlit as st
from joblib import load
import numpy as np
import pandas as pd  # import pandas for DataFrame

# Load model and scaler
model = load('heart_attack_model.joblib')  # Use correct model
scaler = load('scaler.joblib')            # Load scaler as in Part 19

st.title("Heart Attack Risk Prediction")

# Collect user inputs
age = st.number_input("Age", min_value=15, max_value=100, step=1)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=200, step=1)
chol = st.number_input("Cholesterol (chol)", min_value=100, max_value=600, step=1)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate (thalach)", min_value=60, max_value=220, step=1)
exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, step=0.1)
slope = st.selectbox("Slope of peak exercise ST (slope)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (thal)", [0, 1, 2], format_func=lambda x: ["Normal", "Fixed defect", "Reversible defect"][x])

# Create DataFrame with feature names matching training data
user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
input_df = pd.DataFrame([user_input], columns=[
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
])

# Scale the input
input_scaled = scaler.transform(input_df)

if st.button("Predict Heart Attack Risk"):
    predicted_class = model.predict(input_scaled)[0]
    predicted_prob = model.predict_proba(input_scaled)[0][1]

    st.write(f"Predicted Class: {predicted_class} (0 = Low Risk, 1 = High Risk)")
    st.write(f"Probability of Heart Attack: {predicted_prob:.2f}")

    if predicted_class == 1:
        st.error("⚠ High chance of heart attack!")
    else:
        st.success("✅ Low chance of heart attack.")