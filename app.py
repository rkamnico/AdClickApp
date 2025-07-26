import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("social_ad_click_model.pkl")
scaler = joblib.load("scaler.pkl")  # Make sure scaler.pkl is in same folder

# App title
st.title("🧠 Social Network Ad Click Prediction")
st.markdown("Predict whether a user will click an ad based on Age, Gender, and Salary.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 60, 30)
salary = st.slider("Estimated Salary (₹)", 10000, 200000, 50000, step=1000)
threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.5)

# Predict button
if st.button("Predict"):
    # Encode gender
    gender_encoded = 1 if gender.lower() == "male" else 0

    # Prepare and scale input
    input_data = np.array([[gender_encoded, age, salary]])
    input_scaled = scaler.transform(input_data)

    # Predict probabilities
    proba = model.predict_proba(input_scaled)[0]
    prediction = 1 if proba[1] >= threshold else 0

    # Output
    st.subheader("🔍 Prediction Result")
    st.success("🟢 Clicked the Ad ✅" if prediction == 1 else "🔴 Did NOT Click the Ad ❌")
    st.write(f"**Probability (Clicked):** {proba[1]*100:.2f}%")
    st.write(f"**Probability (Not Clicked):** {proba[0]*100:.2f}%")
    st.write(f"**Threshold Used:** {threshold}")

    # Debug information
    st.markdown("---")
    st.subheader("🧪 Debug Info")
    st.write("Raw Input:", input_data)
    st.write("Scaled Input:", input_scaled)
    st.write("Raw Probabilities:", proba)
