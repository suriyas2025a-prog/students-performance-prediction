%%writefile app.py
import streamlit as st
import joblib
import numpy as np

st.title("🎓 Student Performance Prediction")

# Corrected file paths
model = joblib.load("student_model.pkl")
scaler = joblib.load("scaler.pkl")

study_time = st.slider("Study Time (hours)", 0, 10)
absences = st.slider("Absences", 0, 30)
previous_score = st.slider("Previous Score", 0, 100)

if st.button("Predict"):
    data = np.array([[study_time, absences, previous_score]])
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)

    if prediction[0] == 1:
        st.success("✅ Student Will Pass")
    else:
        st.error("❌ Student May Fail")
