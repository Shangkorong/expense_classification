import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# 1. Load All Models and Encoders
@st.cache_resource
def load_models():
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

# 2. Page Configuration
st.set_page_config(page_title="AI Financial Dashboard", layout="wide")
st.title("🚀 Enterprise Financial Intelligence Dashboard")

# 3. Sidebar for Prediction
st.sidebar.header("New Transaction Input")
vendor = st.sidebar.text_input("Vendor Name", "Amazon Web Services")
amount = st.sidebar.number_input("Amount", value=5000.0)
division = st.sidebar.selectbox("Division", ["Software Division", "Corporate / Shared", "VLSI Division"])

# Prepare Input for Prediction
input_df = pd.DataFrame([{
    "Vendor": vendor, "Amount": amount, "Division": division,
    "Department": "IT & Telecom", "Cost Center": "CC-CORP-IT", 
    "Currency": "USD", "Payment Method": "Wire Transfer", "Tax": amount * 0.18
}])

# 4. Dashboard Metrics
models, encoders = load_models()

if st.sidebar.button("Run Financial Audit"):
    # Run Predictions
    exp_cat = encoders['expense'].inverse_transform(models['expense'].predict(input_df))[0]
    pred_rev = models['revenue'].predict(input_df)[0]
    is_def = "🔴 HIGH RISK" if models['default'].predict(input_df)[0] == 1 else "🟢 LOW RISK"
    risk_lvl = encoders['risk'].inverse_transform(models['credit'].predict(input_df))[0]
    
    # Display Results in Columns
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Expense Category", exp_cat)
    col2.metric("Forecasted Revenue", f"${pred_rev:,.2f}")
    col3.metric("Payment Default Risk", is_def)
    col4.metric("Credit Score Level", risk_lvl)

    st.success("Transaction Audit Completed!")

else:
    st.info("Enter transaction details in the sidebar and click 'Run Financial Audit' to see ML insights.")
