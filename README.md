# 🧬 Thyroid Cancer Risk Prediction App

This is a simple Streamlit web app that predicts thyroid cancer risk levels (**Low**, **Medium**, **High**) based on user inputs.

## 🚀 How it works
- Built with Python and Streamlit.
- Uses a trained machine learning model (`thyroid_risk_model.pkl`).
- Inputs are scaled using `scaler.pkl`.
- Categorical features are encoded with saved encoders.

## 📊 Input Features
- Age
- Gender
- Country
- Family History
- Radiation Exposure
- Iodine Deficiency
- Smoking
- Obesity
- Diabetes
- TSH Level

## ✅ How to run locally
1. Clone the repo.
2. Install dependencies:
   pip install -r requirements.txt
3. Run the app:
streamlit run app.py
