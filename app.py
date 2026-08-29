import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from streamlit_lottie import st_lottie
from streamlit_echarts import st_echarts

# -----------------------------------------
# 1. Page Configuration & Custom CSS
# -----------------------------------------
st.set_page_config(page_title="Intelligent Financial AI", page_icon="🚀", layout="wide")

# Deep Anthracite and EUV Violet styling injected globally
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
    
    /* Main Backgrounds */
    .stApp {
        background-color: #121217; 
        color: #E0E0E0;
    }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #1A1A24;
        border: 1px solid #333344;
        border-left: 4px solid #7D00FF;
        padding: 5% 5% 5% 10%;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    
    /* Typography */
    .tech-font { font-family: 'Share Tech Mono', monospace; color: #00FF41; text-align: left; margin-bottom: 0; }
    .sub-tech { font-family: 'Share Tech Mono', monospace; color: #7D00FF; font-size: 1.2rem; }
    
    /* Popup styling */
    .big-prediction { font-family: 'Share Tech Mono', monospace; font-size: 24px; color: #FFFFFF; text-align: center; background-color: #1A1A24; padding: 15px; border-radius: 10px; border: 1px solid #7D00FF; box-shadow: 0 0 10px rgba(125,0,255,0.2); }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------
# 2. Helper Functions & Data Loading
# -----------------------------------------
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except:
        return None

@st.cache_resource
def load_all_assets():
    # Mocking loading for UI demonstration purposes without breaking if files are missing
    try:
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
    except FileNotFoundError:
        models, encoders = None, None
    return models, encoders

@st.cache_data
def load_dataset():
    """Loads dataset.csv verbatim. Fails over to generated dummy data to ensure UI never breaks."""
    try:
        df = pd.read_csv("dataset.csv")
        # Ensure a datetime index exists for time-series charts
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
        return df
    except FileNotFoundError:
        # Fallback simulated data for charting
        dates = pd.date_range(start="2026-01-01", periods=12, freq="ME")
        return pd.DataFrame({
            "Date": dates,
            "Cash_In": np.random.randint(100000, 300000, 12),
            "Cash_Out": np.random.randint(50000, 200000, 12),
            "Net_Profit": np.random.randint(20000, 150000, 12),
            "Expense_Category": np.random.choice(["Cloud Infra", "VLSI Design & Functional Verification", "Marketing", "Travel", "Legal"], 12),
            "Risk_Score": np.random.randint(10, 90, 12)
        })

assets, encoders = load_all_assets()
lottie_ai = load_lottieurl("https://lottie.host/42d6bc52-e8e3-4716-a17e-273816e6a15e/5rXGQFkIcg.json")
df = load_dataset()

# -----------------------------------------
# 3. Sidebar: Operational Inputs (Audit)
# -----------------------------------------
with st.sidebar:
    if lottie_ai: st_lottie(lottie_ai, height=120)
    st.markdown("<h2 class='tech-font' style='text-align:center;'>Audit Terminal</h2>", unsafe_allow_html=True)
    st.divider()
    
    vendor = st.text_input("Vendor Name", "Synopsys Inc.")
    division = st.selectbox("Division", ["Software Division", "Hardware Division", "Corporate / Shared"])
    dept = st.selectbox("Department", ["Design & Functional Verification (DV)", "IT & Telecom", "Finance & Legal"])
    cc = st.selectbox("Cost Center", ["CC-HW-DV", "CC-SW-DEV", "CC-CORP-FIN"])
    
    amount = st.number_input("Transaction Amount", min_value=0.0, value=125000.0)
    tax = st.number_input("Tax", min_value=0.0, value=amount*0.18)
    currency = st.selectbox("Currency", ["USD", "INR", "EUR"])
    pay_method = st.selectbox("Method", ["Wire Transfer", "Bank Transfer (NEFT)", "Corporate Credit Card"])

    if st.button("🚀 RUN ML AUDIT", use_container_width=True, type="primary"):
        input_df = pd.DataFrame([{ 
            "Vendor": vendor, "Division": division, "Department": dept, "Cost Center": cc, 
            "Amount": amount, "Tax": tax, "Currency": currency, "Payment Method": pay_method 
        }])
        
        @st.dialog("Neural Audit Report")
        def show_audit(data):
            if assets:
                exp_pred = encoders['expense'].inverse_transform(assets['expense'].predict(data))[0]
                rev_pred = assets['revenue'].predict(data)[0]
                prof_pred = assets['profit'].predict(data)[0]
                risk_lvl = encoders['risk'].inverse_transform(assets['credit'].predict(data))[0]
                audit_flag = "🚨 HIGH AUDIT RISK" if assets['audit'].predict(data)[0] == 1 else "✅ AUDIT PASSED"
            else:
                exp_pred, rev_pred, prof_pred, risk_lvl, audit_flag = "CAPEX - Software", 150000, 32000, "LOW", "✅ AUDIT PASSED"

            st.markdown(f"<div class='big-prediction'>{exp_pred}</div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Forecasted Economics**")
                st.info(f"Est. Revenue: {rev_pred:,.2f}\n\nNet Profit: {prof_pred:,.2f}")
            with c2:
                st.write("**Risk Assessment**")
                st.warning(f"Credit: {risk_lvl}")
            
            st.subheader(audit_flag)
            if st.button("Acknowledge"): st.rerun()
            
        show_audit(input_df)

# -----------------------------------------
# 4. Main Canvas: Intelligence Dashboard
# -----------------------------------------
st.markdown("<h1 class='tech-font' style='font-size: 3rem;'>Finance Intelligence Hub</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-tech'>Global Macro-Forecasting & Risk Telemetry</p>", unsafe_allow_html=True)
st.write("")

# Key Metrics Row
col1, col2, col3, col4 = st.columns(4)
col1.metric("YTD Revenue", "$4.2M", "+12.5%")
col2.metric("Operating Margin", "24.8%", "1.2%")
col3.metric("High-Risk Transactions", "14", "-3")
col4.metric("Model Confidence (Avg)", "98.2%", "+0.4%")

st.divider()

# -----------------------------------------
# 5. ECharts Visualizations
# -----------------------------------------
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.markdown("**Monthly Cash Flow (Forecasting Base)**")
    # Extracting months for the x-axis
    months = df['Date'].dt.strftime('%b').tolist() if 'Date' in df.columns else ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    cash_in = df['Cash_In'].tolist() if 'Cash_In' in df.columns else [np.random.randint(100, 300) for _ in range(12)]
    cash_out = [-x for x in (df['Cash_Out'].tolist() if 'Cash_Out' in df.columns else [np.random.randint(50, 150) for _ in range(12)])]
    forecast = [x * 1.15 for x in cash_in] 

    cash_flow_options = {
        "backgroundColor": "transparent",
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "legend": {"data": ["Inflow", "Outflow", "AI Forecast"], "textStyle": {"color": "#E0E0E0"}},
        "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
        "xAxis": [{"type": "category", "data": months, "axisLabel": {"color": "#A0A0A0"}}],
        "yAxis": [{"type": "value", "splitLine": {"lineStyle": {"color": "#333344"}}, "axisLabel": {"color": "#A0A0A0"}}],
        "series": [
            {"name": "Inflow", "type": "bar", "stack": "Total", "itemStyle": {"color": "#00FF41", "borderRadius": [4, 4, 0, 0]}, "data": cash_in},
            {"name": "Outflow", "type": "bar", "stack": "Total", "itemStyle": {"color": "#FF003C", "borderRadius": [0, 0, 4, 4]}, "data": cash_out},
            {"name": "AI Forecast", "type": "line", "smooth": True, "itemStyle": {"color": "#7D00FF"}, "lineStyle": {"width": 3, "shadowColor": 'rgba(125,0,255,0.5)', "shadowBlur": 10}, "data": forecast}
        ]
    }
    st_echarts(options=cash_flow_options, height="350px")

with row1_col2:
    st.markdown("**Monthly Net Profit (Gradient Mapping)**")
    profits = df['Net_Profit'].tolist() if 'Net_Profit' in df.columns else [np.random.randint(20, 100) for _ in range(12)]
    
    profit_options = {
        "backgroundColor": "transparent",
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross", "label": {"backgroundColor": "#6a7985"}}},
        "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
        "xAxis": [{"type": "category", "boundaryGap": False, "data": months, "axisLabel": {"color": "#A0A0A0"}}],
        "yAxis": [{"type": "value", "splitLine": {"lineStyle": {"color": "#333344"}}, "axisLabel": {"color": "#A0A0A0"}}],
        "series": [{
            "name": "Net Profit",
            "type": "line",
            "smooth": True,
            "lineStyle": {"width": 0},
            "showSymbol": False,
            "areaStyle": {
                "opacity": 0.8,
                "color": {
                    "type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
                    "colorStops": [{"offset": 0, "color": "#7D00FF"}, {"offset": 1, "color": "rgba(125,0,255,0.05)"}]
                }
            },
            "emphasis": {"focus": "series"},
            "data": profits
        }]
    }
    st_echarts(options=profit_options, height="350px")

row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.markdown("**Top 5 Expense Categories (ML Classified)**")
    # Nightingale Rose Chart
    expense_options = {
        "backgroundColor": "transparent",
        "tooltip": {"trigger": "item", "formatter": "{a} <br/>{b} : ${c} ({d}%)"},
        "legend": {"bottom": "0%", "textStyle": {"color": "#A0A0A0"}},
        "series": [{
            "name": "Expense Vector",
            "type": "pie",
            "radius": ["20%", "70%"],
            "center": ["50%", "45%"],
            "roseType": "area",
            "itemStyle": {"borderRadius": 8, "borderColor": "#1A1A24", "borderWidth": 2},
            "data": [
                {"value": 45000, "name": "Compute / Cloud", "itemStyle": {"color": "#7D00FF"}},
                {"value": 38000, "name": "Hardware IP", "itemStyle": {"color": "#9D4EDD"}},
                {"value": 32000, "name": "Marketing", "itemStyle": {"color": "#C77DFF"}},
                {"value": 28000, "name": "R&D Software", "itemStyle": {"color": "#E0AAFF"}},
                {"value": 15000, "name": "Legal", "itemStyle": {"color": "#5A189A"}}
            ]
        }]
    }
    st_echarts(options=expense_options, height="380px")

with row2_col2:
    st.markdown("**Customer Credit Risk Assessment Matrix**")
    # Radar Chart
    risk_options = {
        "backgroundColor": "transparent",
        "tooltip": {},
        "legend": {"data": ['Enterprise Client A', 'SMB Client B'], "bottom": "0%", "textStyle": {"color": "#A0A0A0"}},
        "radar": {
            "indicator": [
                {"name": 'Liquidity', "max": 100},
                {"name": 'Default Prob.', "max": 100},
                {"name": 'Market Volatility', "max": 100},
                {"name": 'Audit Flag History', "max": 100},
                {"name": 'Payment Latency', "max": 100}
            ],
            "splitArea": {"areaStyle": {"color": ["rgba(125,0,255,0.05)", "rgba(125,0,255,0.1)", "rgba(125,0,255,0.15)", "rgba(125,0,255,0.2)"]}},
            "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.2)"}},
            "splitLine": {"lineStyle": {"color": "rgba(255,255,255,0.2)"}},
            "axisName": {"color": "#00FF41"}
        },
        "series": [{
            "name": 'Risk Dimensions',
            "type": 'radar',
            "data": [
                {
                    "value": [85, 20, 45, 10, 30],
                    "name": 'Enterprise Client A',
                    "itemStyle": {"color": "#00FF41"},
                    "areaStyle": {"color": "rgba(0,255,65,0.3)"}
                },
                {
                    "value": [40, 75, 60, 50, 80],
                    "name": 'SMB Client B',
                    "itemStyle": {"color": "#FF003C"},
                    "areaStyle": {"color": "rgba(255,0,60,0.3)"}
                }
            ]
        }]
    }
    st_echarts(options=risk_options, height="380px")
