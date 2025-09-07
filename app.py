import streamlit as st
from joblib import load
import pandas as pd

# Load model and scaler
model = load('heart_attack_model.joblib')
scaler = load('scaler.joblib')

# Page setup
st.set_page_config(page_title="Heart Attack Risk Predictor", layout="centered")
st.title("❤️ Heart Attack Risk Prediction")
st.markdown("Fill in your details below to assess your risk of heart attack.")

# -------------------------
# Section 1: Personal Info
# -------------------------
st.markdown("### 👤 Personal Information")
age = st.number_input("Age (15-100)", min_value=15, max_value=100, step=1, help="Enter your age in years")
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female", help="Select your biological sex")

st.divider()

# -------------------------
# Section 2: Vitals / Heart Symptoms
# -------------------------
st.markdown("### 💓 Vitals / Heart Symptoms")
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3], help="Type of chest pain")
trestbps = st.number_input("Resting Blood Pressure (80-200)", min_value=80, max_value=200, step=1, help="Blood pressure at rest (mm Hg)")
chol = st.number_input("Cholesterol mg/dl (100-600)", min_value=100, max_value=600, step=1, help="Serum cholesterol in mg/dl")
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1], format_func=lambda x: "No" if x==0 else "Yes", help="Fasting blood sugar greater than 120 mg/dl?")
restecg = st.selectbox("Resting ECG (0-2)", [0,1,2], help="Resting electrocardiographic results")

st.divider()

# -------------------------
# Section 3: Exercise Test
# -------------------------
st.markdown("### 🏃 Exercise Test")
thalach = st.number_input("Max Heart Rate (60-220)", min_value=60, max_value=220, step=1, help="Highest heart rate achieved during exercise")
exang = st.selectbox("Exercise Induced Angina", [0,1], format_func=lambda x: "No" if x==0 else "Yes", help="Did chest pain occur with exercise?")
oldpeak = st.number_input("ST Depression (0.0-10.0)", min_value=0.0, max_value=10.0, step=0.1, help="ST depression induced by exercise relative to rest")

st.divider()

# -------------------------
# Section 4: Advanced Inputs
# -------------------------
st.markdown("### ⚙️ Advanced Inputs")
slope = st.selectbox("Slope of peak exercise ST (0-2)", [0, 1, 2], help="Slope of the peak exercise ST segment")
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3], help="Number of major vessels colored by fluoroscopy")
thal = st.selectbox("Thalassemia", [0, 1, 2], format_func=lambda x: ["Normal", "Fixed defect", "Reversible defect"][x], help="Thalassemia type")

st.divider()

# -------------------------
# Cast inputs explicitly
# -------------------------
user_input = [
    int(age), int(sex), int(cp), int(trestbps), int(chol), int(fbs),
    int(restecg), int(thalach), int(exang), float(oldpeak),
    int(slope), int(ca), int(thal)
]

# -------------------------
# Build DataFrame
# -------------------------
input_df = pd.DataFrame([user_input], columns=[
    'age','sex','cp','trestbps','chol','fbs','restecg',
    'thalach','exang','oldpeak','slope','ca','thal'
])

# Scale input
input_scaled = scaler.transform(input_df)

# -------------------------
# Predict
# -------------------------
if st.button("Predict Heart Attack Risk"):
    predicted_class = model.predict(input_scaled)[0]
    predicted_prob = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")
    st.subheader("🩺 Prediction Result")
    st.metric("Predicted Class", "High Risk ⚠️" if predicted_class==1 else "Low Risk ✅")
    st.progress(predicted_prob)
    st.write(f"Probability of Heart Attack: {predicted_prob:.2f}")

    if predicted_class == 1:
        st.error("⚠ High chance of heart attack! Please consult a doctor.")
    else:
        st.success("✅ Low chance of heart attack. Maintain a healthy lifestyle.")
