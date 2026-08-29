import streamlit as st
import pandas as pd
import joblib
import requests
from streamlit_lottie import st_lottie

# -----------------------------------------
# 1. Page Configuration & Custom CSS
# -----------------------------------------
st.set_page_config(page_title="Intelligent Financial AI", page_icon="🚀", layout="centered")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
    .tech-font { font-family: 'Share Tech Mono', monospace; color: #00FF41; text-align: center; }
    .big-prediction { font-family: 'Share Tech Mono', monospace; font-size: 24px; color: #FFFFFF; text-align: center; background-color: #1E1E1E; padding: 15px; border-radius: 10px; border: 1px solid #00FF41; }
    .risk-metric { font-family: 'Share Tech Mono', monospace; font-size: 16px; color: #FFA500; margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------
# 2. Helper Functions
# -----------------------------------------
def load_lottieurl(url: str):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

@st.cache_resource
def load_all_assets():
    models = {
        'expense': joblib.load('expense_model_pipeline.pkl'),
        'revenue': joblib.load('revenue_model_pipeline.pkl'),
        'default': joblib.load('default_model_pipeline.pkl'),
        'credit': joblib.load('credit_risk_model.pkl'),
        'profit': joblib.load('profitability_model.pkl'),
        'audit': joblib.load('audit_risk_model.pkl')
    }
    encoders = {
        'expense': joblib.load('expense_label_encoder.pkl'),
        'risk': joblib.load('risk_label_encoder.pkl')
    }
    return models, encoders

assets, encoders = load_all_assets()
lottie_ai = load_lottieurl("https://lottie.host/42d6bc52-e8e3-4716-a17e-273816e6a15e/5rXGQFkIcg.json")

# -----------------------------------------
# 3. UI Layout
# -----------------------------------------
col1, col2 = st.columns([1, 4])
with col1:
    if lottie_ai: st_lottie(lottie_ai, height=100)
with col2:
    st.markdown("<h1 class='tech-font'>Finance Intelligence Hub</h1>", unsafe_allow_html=True)
    st.write("Multi-model AI for Expense, Revenue, and Risk Analysis.")

st.divider()

# -----------------------------------------
# 4. Inputs
# -----------------------------------------
with st.container():
    col_a, col_b = st.columns(2)
    with col_a:
        vendor = st.text_input("Vendor Name", "Amazon Web Services")
        division = st.selectbox("Division", ["Software Division", "VLSI Division", "Corporate / Shared"])
        dept = st.selectbox("Department", ["IT & Telecom", "Software Engineering & Dev", "Finance & Legal"])
        cc = st.selectbox("Cost Center", ["CC-CORP-IT", "CC-SW-DEV", "CC-CORP-FIN"])
    with col_b:
        amount = st.number_input("Transaction Amount", min_value=0.0, value=50000.0)
        tax = st.number_input("Tax", min_value=0.0, value=amount*0.18)
        currency = st.selectbox("Currency", ["INR", "USD", "EUR"])
        pay_method = st.selectbox("Method", ["Bank Transfer (NEFT)", "Corporate Credit Card", "Wire Transfer"])

# -----------------------------------------
# 5. Analysis Popup
# -----------------------------------------
@st.dialog("AI Financial Audit Complete")
def show_audit(data_row):
    # 1. Expense Prediction
    exp_pred = encoders['expense'].inverse_transform(assets['expense'].predict(data_row))[0]
    # 2. Revenue & Profit
    rev_pred = assets['revenue'].predict(data_row)[0]
    prof_pred = assets['profit'].predict(data_row)[0]
    # 3. Risks
    risk_lvl = encoders['risk'].inverse_transform(assets['credit'].predict(data_row))[0]
    audit_flag = "🚨 HIGH AUDIT RISK" if assets['audit'].predict(data_row)[0] == 1 else "✅ AUDIT PASSED"
    def_risk = "⚠️ HIGH" if assets['default'].predict(data_row)[0] == 1 else "🟢 LOW"

    st.markdown(f"<div class='big-prediction'>{exp_pred}</div>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Forecasted Economics**")
        st.info(f"Est. Revenue: {rev_pred:,.2f}\n\nNet Profit: {prof_pred:,.2f}")
    with c2:
        st.write("**Risk Assessment**")
        st.warning(f"Credit: {risk_lvl}\n\nDefault: {def_risk}")
    
    st.subheader(audit_flag)
    if st.button("Dismiss"): st.rerun()

if st.button("🚀 RUN COMPLETE FINANCIAL AUDIT", use_container_width=True, type="primary"):
    input_df = pd.DataFrame([{ 
        "Vendor": vendor, "Division": division, "Department": dept, "Cost Center": cc, 
        "Amount": amount, "Tax": tax, "Currency": currency, "Payment Method": pay_method 
    }])
    show_audit(input_df)
