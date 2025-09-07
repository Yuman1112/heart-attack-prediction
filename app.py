import streamlit as st
from joblib import load
import pandas as pd

# Load model and scaler
model = load('heart_attack_model.joblib')
scaler = load('scaler.joblib')

st.title("Heart Attack Risk Prediction")

# Collect user inputs with explicit types
age = st.number_input("Age (15-100)", min_value=15, max_value=100, step=1)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (80-200)", min_value=80, max_value=200, step=1)
chol = st.number_input("Cholesterol mg/dl (100-600)", min_value=100, max_value=600, step=1)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate (60-220)", min_value=60, max_value=220, step=1)
exang = st.selectbox("Exercise Induced Angina (0=no, 1=yes)", [0, 1])
oldpeak = st.number_input("ST Depression (0.0-10.0)", min_value=0.0, max_value=10.0, step=0.1)
slope = st.selectbox("Slope of peak exercise ST (0-2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", [0, 1, 2], format_func=lambda x: ["Normal", "Fixed defect", "Reversible defect"][x])

# Cast all inputs to correct types explicitly
user_input = [
    int(age), int(sex), int(cp), int(trestbps), int(chol), int(fbs), int(restecg),
    int(thalach), int(exang), float(oldpeak), int(slope), int(ca), int(thal)
]

# Create DataFrame with same column order as training
input_df = pd.DataFrame([user_input], columns=[
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
])

# Scale input
input_scaled = scaler.transform(input_df)

# Predict when button clicked
if st.button("Predict Heart Attack Risk"):
    predicted_class = model.predict(input_scaled)[0]
    predicted_prob = model.predict_proba(input_scaled)[0][1]

    st.write(f"Predicted Class: {predicted_class} (0 = Low Risk, 1 = High Risk)")
    st.write(f"Probability of Heart Attack: {predicted_prob:.2f}")

    if predicted_class == 1:
        st.error("⚠ High chance of heart attack! Please consult a doctor.")
    else:
        st.success("✅ Low chance of heart attack. Maintain a healthy lifestyle.")

