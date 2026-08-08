import streamlit as st
import pandas as pd
import joblib
import requests
from streamlit_lottie import st_lottie

# -----------------------------------------
# 1. Page Configuration & Custom CSS
# -----------------------------------------
st.set_page_config(page_title="Intelligent Expense AI", page_icon="🤖", layout="centered")

# Custom CSS for tech-oriented fonts and aesthetics
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
    
    .tech-font {
        font-family: 'Share Tech Mono', monospace;
        color: #00FF41; /* Hacker Green */
        text-align: center;
    }
    .big-prediction {
        font-family: 'Share Tech Mono', monospace;
        font-size: 28px;
        color: #FFFFFF;
        text-align: center;
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #00FF41;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------
# 2. Helper Functions (Lottie & Model Load)
# -----------------------------------------
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

@st.cache_resource
def load_model():
    pipeline = joblib.load('expense_model_pipeline.pkl')
    encoder = joblib.load('expense_label_encoder.pkl')
    return pipeline, encoder

model_pipeline, label_encoder = load_model()

# Load a tech/AI Lottie animation (Replace URL with any you like from Lottiefiles)
#lottie_ai = load_lottieurl("https://lottie.host/bf841617-9d37-40ba-91b4-f4ffacf9a035/lXDjyLgI8V.json")
lottie_ai = load_lottieurl("https://lottie.host/42d6bc52-e8e3-4716-a17e-273816e6a15e/5rXGQFkIcg.json")

# -----------------------------------------
# 3. UI Header
# -----------------------------------------
col1, col2 = st.columns([1, 4])
with col1:
    if lottie_ai:
        st_lottie(lottie_ai, height=100, key="header_anim")
with col2:
    st.markdown("<h1 class='tech-font'>Expense Classifier</h1>", unsafe_allow_html=True)
    st.write("ML Classification for enterprise accounts payable.")

st.divider()

# -----------------------------------------
# 4. Form Inputs
# -----------------------------------------
# Note: In Streamlit, to allow typing a new vendor, we use a conditional text input.
known_vendors = ["Synopsys", "AWS", "WeWork", "LinkedIn", "Dell", "Other (Type New)"]

with st.container():
    col_a, col_b = st.columns(2)
    
    with col_a:
        selected_vendor = st.selectbox("Vendor", known_vendors)
        if selected_vendor == "Other (Type New)":
            final_vendor = st.text_input("Enter New Vendor Name", placeholder="e.g., Nexus Tech")
        else:
            final_vendor = selected_vendor
            
        division = st.selectbox("Division", ["Software Division", "VLSI Division", "Corporate / Shared"])
        department = st.selectbox("Department", ["Software Engineering & Dev", "VLSI Engineering & Design", "Facilities & HR", "Sales & Marketing", "Finance & Legal", "IT & Telecom"])
        cost_center = st.selectbox("Cost Center", ["CC-SW-DEV", "CC-VLSI-ENG", "CC-CORP-HR", "CC-CORP-MKT", "CC-CORP-FIN", "CC-CORP-IT"])

    with col_b:
        currency = st.selectbox("Currency", ["INR", "USD", "EUR", "GBP"])
        payment_method = st.selectbox("Payment Method", ["Bank Transfer (RTGS)", "Bank Transfer (NEFT)", "Wire Transfer", "Corporate Credit Card", "Corporate UPI"])
        amount = st.number_input("Amount", min_value=0.0, value=15000.0, step=1000.0)
        tax = st.number_input("Tax", min_value=0.0, value=2700.0, step=100.0)

# -----------------------------------------
# 5. Prediction Popup Logic
# -----------------------------------------
# Using Streamlit's new dialog feature for a true popup
@st.dialog("Prediction Complete")
def show_prediction_popup(category):
    # Success Lottie Animation
    lottie_success = load_lottieurl("https://lottie.host/6b1ee372-4e76-4c33-a2d0-8cd709f4c717/fqrItzBy5I.json")
    if lottie_success:
        st_lottie(lottie_success, height=150, key="success_anim")
    
    st.markdown(f"<div class='big-prediction'>{category}</div>", unsafe_allow_html=True)
    
    if st.button("Close"):
        st.rerun()

st.write("") # Spacing
# Send Button
if st.button("🚀 Process Invoice", use_container_width=True, type="primary"):
    if not final_vendor:
        st.error("Please provide a Vendor name.")
    else:
        # Construct the DataFrame exactly as the model expects
        input_data = pd.DataFrame([{
            "Vendor": final_vendor,
            "Division": division,
            "Department": department,
            "Cost Center": cost_center,
            "Currency": currency,
            "Payment Method": payment_method,
            "Amount": amount,
            "Tax": tax
        }])
        
        # Make Prediction
        pred_numeric = model_pipeline.predict(input_data)
        pred_category = label_encoder.inverse_transform(pred_numeric)[0]
        
        # Trigger Popup
        show_prediction_popup(pred_category)
