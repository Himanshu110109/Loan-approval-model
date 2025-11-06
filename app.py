import streamlit as st
import pandas as pd
import numpy as np
import joblib

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")
expected_columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Loan Approval Predictor", page_icon="💰", layout="centered")

st.markdown("""
    <style>
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
            font-family: 'Inter', sans-serif;
        }
        .stTextInput > div > div > input,
        .stNumberInput input,
        .stSelectbox > div > div {
            background-color: #1E2228;
            color: white;
        }
        .stButton>button {
            background-color: #00BFFF;
            color: white;
            border-radius: 8px;
            height: 3em;
            width: 100%;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #0099CC;
        }
        h1, h2, h3 { color: #00BFFF; }
    </style>
""", unsafe_allow_html=True)

st.title("💰 Loan Approval Prediction")
st.caption("Made by sanskari coder.")

no_of_dependents = st.number_input("Number of Dependents", min_value=0, step=1, value=1)
education = st.selectbox("Education", ["Graduate", "Not Graduate"], index=0)
self_employed = st.selectbox("Self Employed", ["No", "Yes"], index=0)
income_annum = st.number_input("Annual Income (₹)", min_value=0, step=10000, value=9500000)
loan_amount = st.number_input("Loan Amount (₹)", min_value=0, step=10000, value=12000000)
loan_term = st.number_input("Loan Term (months)", min_value=1, step=1, value=15)
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, step=1, value=810)
residential_assets_value = st.number_input("Residential Assets Value (₹)", min_value=0, step=10000, value=5000000)
commercial_assets_value = st.number_input("Commercial Assets Value (₹)", min_value=0, step=10000, value=4000000)
luxury_assets_value = st.number_input("Luxury Assets Value (₹)", min_value=0, step=10000, value=3000000)
bank_asset_value = st.number_input("Bank Asset Value (₹)", min_value=0, step=10000, value=2500000)

edu_val = 0 if education == "Graduate" else 1
emp_val = 0 if self_employed == "No" else 1

input_data = pd.DataFrame([[
    no_of_dependents,
    edu_val,
    emp_val,
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value
]], columns=[
    'no_of_dependents', 'education', 'self_employed', 'income_annum',
    'loan_amount', 'loan_term', 'cibil_score',
    'residential_assets_value', 'commercial_assets_value',
    'luxury_assets_value', 'bank_asset_value'
])

input_df = input_data.reindex(columns=expected_columns, fill_value=0)

if st.button("Predict"):
    try:
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        prob = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(scaled_input)[0][1] * 100

        st.subheader("Prediction Result:")
        if prediction == 1:
            if prob:
                st.success(f"✅ Loan Approved ({prob:.2f}% confidence)")
            else:
                st.success("✅ Loan Approved")
        else:
            if prob:
                st.error(f"❌ Loan Rejected ({prob:.2f}% confidence)")
            else:
                st.error("❌ Loan Rejected")
    except Exception as e:
        st.error(f"Error: {e}")
