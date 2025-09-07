import streamlit as st
from joblib import load
import pandas as pd

# Load model and scaler
model = load('heart_attack_model.joblib')
scaler = load('scaler.joblib')

st.set_page_config(page_title="Heart Attack Risk Predictor", layout="wide")
st.title("❤️ Heart Attack Risk Prediction")
st.markdown("Fill in the details below to assess your risk of heart attack.")

# -------------------------
# Layout: Columns for inputs
# -------------------------
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age (15-100)", 15, 100, step=1)
    cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
    chol = st.number_input("Cholesterol mg/dl (100-600)", 100, 600, step=1)
    restecg = st.selectbox("Resting ECG (0-2)", [0,1,2])
    slope = st.selectbox("Slope of peak exercise ST (0-2)", [0,1,2])

with col2:
    sex = st.selectbox("Sex", [0,1], format_func=lambda x: "Male" if x==1 else "Female")
    trestbps = st.number_input("Resting Blood Pressure (80-200)", 80, 200, step=1)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    thalach = st.number_input("Max Heart Rate (60-220)", 60, 220, step=1)
    ca = st.selectbox("Number of Major Vessels (0-3)", [0,1,2,3])

# Other inputs in an expander
with st.expander("Advanced Inputs"):
    exang = st.selectbox("Exercise Induced Angina", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    oldpeak = st.number_input("ST Depression (0.0-10.0)", 0.0, 10.0, step=0.1)
    thal = st.selectbox("Thalassemia", [0,1,2], format_func=lambda x: ["Normal","Fixed defect","Reversible defect"][x])

# -------------------------
# Cast inputs explicitly
# -------------------------
user_input = [
    int(age), int(sex), int(cp), int(trestbps), int(chol), int(fbs),
    int(restecg), int(thalach), int(exang), float(oldpeak),
    int(slope), int(ca), int(thal)
]

input_df = pd.DataFrame([user_input], columns=[
    'age','sex','cp','trestbps','chol','fbs','restecg',
    'thalach','exang','oldpeak','slope','ca','thal'
])

# Scale input
input_scaled = scaler.transform(input_df)

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Heart Attack Risk"):
    predicted_class = model.predict(input_scaled)[0]
    predicted_prob = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")
    st.subheader("Prediction Result")
    st.metric("Predicted Class", "High Risk ⚠️" if predicted_class==1 else "Low Risk ✅")
    
    st.progress(predicted_prob)
    st.write(f"Probability of Heart Attack: {predicted_prob:.2f}")

    if predicted_class == 1:
        st.error("⚠ High chance of heart attack! Please consult a doctor.")
    else:
        st.success("✅ Low chance of heart attack. Maintain a healthy lifestyle.")
