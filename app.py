import streamlit as st
import pickle
import joblib
import numpy as np

# Load model & scaler
with open('thyroid_risk_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load encoders
le_gender = joblib.load('Gender_encoder.pkl')
le_country = joblib.load('Country_encoder.pkl')

st.title("🔬 Thyroid Cancer Risk Predictor")

# Inputs
age = st.number_input("Age", 0, 120, 30)
gender = st.selectbox("Gender", le_gender.classes_)
country = st.selectbox("Country", le_country.classes_)
family_history = st.selectbox("Family History?", ["Yes", "No"])
radiation = st.selectbox("Radiation Exposure?", ["Yes", "No"])
iodine = st.selectbox("Iodine Deficiency?", ["Yes", "No"])
smoking = st.selectbox("Smoking?", ["Yes", "No"])
obesity = st.selectbox("Obesity?", ["Yes", "No"])
diabetes = st.selectbox("Diabetes?", ["Yes", "No"])
tsh_level = st.number_input("TSH Level", 0.0, step=0.1, value=1.0)

# Encode inputs
gender_enc = le_gender.transform([gender])[0]
country_enc = le_country.transform([country])[0]
family_history_enc = 1 if family_history == "Yes" else 0
radiation_enc = 1 if radiation == "Yes" else 0
iodine_enc = 1 if iodine == "Yes" else 0
smoking_enc = 1 if smoking == "Yes" else 0
obesity_enc = 1 if obesity == "Yes" else 0
diabetes_enc = 1 if diabetes == "Yes" else 0

# Predict
if st.button("Predict Risk"):
    X = np.array([[
        age, gender_enc, country_enc, family_history_enc, radiation_enc,
        iodine_enc, smoking_enc, obesity_enc, diabetes_enc, tsh_level
    ]])
    X_scaled = scaler.transform(X)
    result = model.predict(X_scaled)[0]
    # Map numeric to label
    risk_labels = {1: 'Low', 2: 'Medium', 0: 'High'}
    risk_label = risk_labels.get(result, "Unknown")

    st.success(f"Predicted Risk Level: {risk_label}")
    #st.success(f"Predicted Risk Level: {result}")
