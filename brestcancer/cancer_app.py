import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model & scaler
model = joblib.load("breast_cancer.pkl")
scaler = joblib.load("scaler2.pkl")

st.title("🩺 Breast Cancer Prediction")
st.write("Enter all tumor feature values to predict the target.")

# Feature list
features = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension"
]

# Create 2-column layout for better UI
col1, col2 = st.columns(2)

input_values = []

for i, feature in enumerate(features):
    if i % 2 == 0:
        value = col1.number_input(feature, value=0.0, format="%.5f")
    else:
        value = col2.number_input(feature, value=0.0, format="%.5f")
    input_values.append(value)

# Predict button
if st.button("🔍 Predict"):
    data = np.array(input_values).reshape(1, -1)
    scaled_data = scaler.transform(data)
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][prediction]

    st.subheader("Result")

    if prediction == 1:
        st.error(f"⚠️ Malignant (Cancer Detected)\n\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Benign (No Cancer)\n\nProbability: {probability:.2f}")

    st.write(f"Predicted Target: **{prediction}**")