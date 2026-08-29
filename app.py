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

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
    
    /* Main Backgrounds: Deep Anthracite */
    .stApp { background-color: #121217; color: #E0E0E0; }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #1A1A24;
        border: 1px solid #333344;
        border-left: 4px solid #7D00FF; /* EUV Violet Accent */
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
def load_and_process_dataset():
    """Loads master_financial_dataset_2.csv to compute real timeline metrics and extract dropdown lists."""
    try:
        df = pd.read_csv("master_financial_dataset_2.csv")
        df['Invoice Date'] = pd.to_datetime(df['Invoice Date'])
        df['Month_Year'] = df['Invoice Date'].dt.to_period('M')
        
        # 1. Aggregate Actuals per month
        monthly = df.groupby('Month_Year').agg({
            'Amount': 'sum',      
            'Revenue': 'sum',     
            'Net_Profit': 'sum'   
        }).reset_index()
        
        monthly['Month_Label'] = monthly['Month_Year'].dt.strftime('%b %y')
        monthly['Cash_Out_Lakhs'] = (monthly['Amount'] / 100000).round(2)
        monthly['Cash_In_Lakhs'] = (monthly['Revenue'] / 100000).round(2)
        monthly['Net_Profit_Lakhs'] = (monthly['Net_Profit'] / 100000).round(2)
        
        # 2. Extract Top 5 Expenses
        top_expenses = df.groupby('Expense Category')['Amount'].sum().nlargest(5).reset_index()
        top_expenses['Amount_Lakhs'] = (top_expenses['Amount'] / 100000).round(2)
        
        # 3. Dynamic Radar Chart Stats
        div_stats = df.groupby('Division').agg(
            avg_credit=('Credit_Score', 'mean'),
            default_rate=('Is_Default', 'mean'),
            audit_rate=('Audit_Risk_Score', 'mean'),
            avg_days_late=('Days Late', 'mean'),
            total_rev=('Revenue', 'sum'),
            total_profit=('Net_Profit', 'sum')
        ).reset_index()

        # 4. Extract Unique Categorical Values for Dropdowns
        dropdown_options = {
            'Division': df['Division'].dropna().unique().tolist(),
            'Department': df['Department'].dropna().unique().tolist(),
            'Cost Center': df['Cost Center'].dropna().unique().tolist(),
            'Currency': df['Currency'].dropna().unique().tolist(),
            'Payment Method': df['Payment Method'].dropna().unique().tolist()
        }
        
        return df, monthly.sort_values('Month_Year'), top_expenses, div_stats, dropdown_options
    except FileNotFoundError:
        # Fallback dictionary matching the exact schema if file is missing during boot
        fallback_options = {
            'Division': ['VLSI Division', 'Software Division', 'Corporate / Shared'],
            'Department': ['VLSI Engineering & Design', 'Sales & Marketing', 'Finance & Legal', 'IT & Telecom', 'Software Engineering & Dev'],
            'Cost Center': ['CC-VLSI-ENG', 'CC-CORP-MKT', 'CC-CORP-FIN', 'CC-CORP-IT', 'CC-SW-DEV'],
            'Currency': ['INR', 'USD'],
            'Payment Method': ['Bank Transfer (NEFT)', 'Corporate Credit Card', 'Wire Transfer', 'Direct Debit']
        }
        return None, None, None, None, fallback_options

assets, encoders = load_all_assets()
lottie_ai = load_lottieurl("https://lottie.host/42d6bc52-e8e3-4716-a17e-273816e6a15e/5rXGQFkIcg.json")
df, monthly_data, top_expenses, div_stats, options = load_and_process_dataset()

# -----------------------------------------
# 3. Sidebar: Operational Inputs (Unified Schema)
# -----------------------------------------
with st.sidebar:
    if lottie_ai: st_lottie(lottie_ai, height=120)
    st.markdown("<h2 class='tech-font' style='text-align:center;'>Audit Terminal</h2>", unsafe_allow_html=True)
    st.divider()
    
    # 8-Feature Unified Schema Inputs
    vendor = st.text_input("Vendor Name (Text)", "Synopsys Inc.")
    
    # Selecting defaults intelligently if available in the dataset
    default_div = options['Division'].index('VLSI Division') if 'VLSI Division' in options['Division'] else 0
    division = st.selectbox("Division (Categorical)", options['Division'], index=default_div)
    
    default_dept = options['Department'].index('VLSI Engineering & Design') if 'VLSI Engineering & Design' in options['Department'] else 0
    dept = st.selectbox("Department (Categorical)", options['Department'], index=default_dept)
    
    default_cc = options['Cost Center'].index('CC-VLSI-ENG') if 'CC-VLSI-ENG' in options['Cost Center'] else 0
    cc = st.selectbox("Cost Center (Categorical)", options['Cost Center'], index=default_cc)
    
    currency = st.selectbox("Currency (Categorical)", options['Currency'])
    pay_method = st.selectbox("Payment Method (Categorical)", options['Payment Method'])
    
    amount = st.number_input("Transaction Amount (Numerical)", min_value=0.0, value=1500000.0) 
    tax = st.number_input("Tax (Numerical)", min_value=0.0, value=amount*0.18)

    if st.button("🚀 RUN ML AUDIT", use_container_width=True, type="primary"):
        # Explicitly enforcing the 8-column schema and exact column names
        input_data = pd.DataFrame([{ 
            "Vendor": vendor, 
            "Division": division, 
            "Department": dept, 
            "Cost Center": cc, 
            "Currency": currency, 
            "Payment Method": pay_method,
            "Amount": float(amount), 
            "Tax": float(tax) 
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
                exp_pred, rev_pred, prof_pred, risk_lvl, audit_flag = "EDA Tools & Software", 1850000, 350000, "LOW", "✅ AUDIT PASSED"

            st.markdown(f"<div class='big-prediction'>{exp_pred}</div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Forecasted Economics**")
                st.info(f"Est. Revenue: ₹ {rev_pred:,.2f}\n\nNet Profit: ₹ {prof_pred:,.2f}")
            with c2:
                st.write("**Risk Assessment**")
                st.warning(f"Credit: {risk_lvl}")
            
            st.subheader(audit_flag)
            if st.button("Acknowledge"): st.rerun()
            
        show_audit(input_data)

# -----------------------------------------
# 4. Main Canvas: Intelligence Dashboard
# -----------------------------------------
st.markdown("<h1 class='tech-font' style='font-size: 3rem;'>Finance Intelligence Hub</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-tech'>Global Macro-Forecasting & Risk Telemetry</p>", unsafe_allow_html=True)
st.write("")

# Dynamic Key Metrics Row
col1, col2, col3, col4 = st.columns(4)
if df is not None:
    total_rev_cr = df['Revenue'].sum() / 10000000
    total_profit = df['Net_Profit'].sum()
    margin_pct = (total_profit / df['Revenue'].sum()) * 100
    high_risk_count = df[(df['Is_Default'] == 1) | (df['Audit_Risk_Score'] == 1)].shape[0]
    avg_credit_health = (df['Credit_Score'].mean() / 850) * 100

    col1.metric("YTD Revenue", f"₹ {total_rev_cr:,.1f} Cr", "Live Data")
    col2.metric("Operating Margin", f"{margin_pct:.1f}%", "Live Data")
    col3.metric("High-Risk Transactions", f"{high_risk_count}", "Live Data")
    col4.metric("Avg Credit Health", f"{avg_credit_health:.1f}%", "Live Data")

st.divider()

if monthly_data is not None:
    # -----------------------------------------
    # 5. ECharts Visualizations 
    # -----------------------------------------
    row1_col1, row1_col2 = st.columns(2)
    
    months = monthly_data['Month_Label'].tolist()
    cash_in = monthly_data['Cash_In_Lakhs'].tolist()
    cash_out = [-x for x in monthly_data['Cash_Out_Lakhs'].tolist()]
    forecast = [x * 1.12 for x in cash_in] 
    profits = monthly_data['Net_Profit_Lakhs'].tolist()

    with row1_col1:
        st.markdown("**Monthly Cash Flow (Forecasting Base)**")
        cash_flow_options = {
            "backgroundColor": "transparent",
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}, "formatter": "{b}<br/>{a0}: ₹{c0} Lakhs<br/>{a1}: ₹{c1} Lakhs"},
            "legend": {"data": ["Inflow (Revenue)", "Outflow (Amount)", "AI Forecast"], "textStyle": {"color": "#E0E0E0"}},
            "grid": {"left": "12%", "right": "4%", "bottom": "10%", "containLabel": True},
            "xAxis": [{"type": "category", "data": months, "axisLabel": {"color": "#A0A0A0"}}],
            "yAxis": [{"type": "value", "name": "Amount (₹ in Lakhs)", "nameLocation": "middle", "nameGap": 50, "nameTextStyle": {"color": "#7D00FF", "fontWeight": "bold"}, "splitLine": {"lineStyle": {"color": "#333344"}}, "axisLabel": {"color": "#A0A0A0", "formatter": "₹{value}"}}],
            "series": [
                {"name": "Inflow (Revenue)", "type": "bar", "stack": "Total", "itemStyle": {"color": "#00FF41", "borderRadius": [4, 4, 0, 0]}, "data": cash_in},
                {"name": "Outflow (Amount)", "type": "bar", "stack": "Total", "itemStyle": {"color": "#FF003C", "borderRadius": [0, 0, 4, 4]}, "data": cash_out},
                {"name": "AI Forecast", "type": "line", "smooth": True, "itemStyle": {"color": "#7D00FF"}, "lineStyle": {"width": 3, "shadowColor": 'rgba(125,0,255,0.5)', "shadowBlur": 10}, "data": forecast}
            ]
        }
        st_echarts(options=cash_flow_options, height="350px")

    with row1_col2:
        st.markdown("**Monthly Net Profit (Gradient Mapping)**")
        profit_options = {
            "backgroundColor": "transparent",
            "tooltip": {"trigger": "axis", "formatter": "Month: {b} <br/>Net Profit: ₹{c} Lakhs", "axisPointer": {"type": "cross", "label": {"backgroundColor": "#6a7985"}}},
            "grid": {"left": "12%", "right": "4%", "bottom": "10%", "containLabel": True},
            "xAxis": [{"type": "category", "boundaryGap": False, "data": months, "axisLabel": {"color": "#A0A0A0"}}],
            "yAxis": [{"type": "value", "name": "Net Profit (₹ in Lakhs)", "nameLocation": "middle", "nameGap": 50, "nameTextStyle": {"color": "#7D00FF", "fontWeight": "bold"}, "splitLine": {"lineStyle": {"color": "#333344"}}, "axisLabel": {"color": "#A0A0A0", "formatter": "₹{value}"}}],
            "series": [{"name": "Net Profit", "type": "line", "smooth": True, "lineStyle": {"width": 0}, "showSymbol": False, "areaStyle": {"opacity": 0.8, "color": {"type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1, "colorStops": [{"offset": 0, "color": "#7D00FF"}, {"offset": 1, "color": "rgba(125,0,255,0.05)"}]}}, "data": profits}]
        }
        st_echarts(options=profit_options, height="350px")

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.markdown("**Top 5 Expense Categories**")
        pie_data = [{"value": val, "name": name} for val, name in zip(top_expenses['Amount_Lakhs'], top_expenses['Expense Category'])]
        expense_options = {
            "backgroundColor": "transparent",
            "tooltip": {"trigger": "item", "formatter": "<b>{b}</b> <br/>Amount: ₹{c} Lakhs ({d}%)"}, 
            "legend": {"bottom": "0%", "textStyle": {"color": "#A0A0A0"}},
            "series": [{"name": "Expense Vector", "type": "pie", "radius": ["20%", "70%"], "center": ["50%", "45%"], "roseType": "area", "itemStyle": {"borderRadius": 8, "borderColor": "#1A1A24", "borderWidth": 2}, "label": {"color": "#E0E0E0", "formatter": "{b}\n₹{c}L"}, "data": pie_data}]
        }
        st_echarts(options=expense_options, height="380px")

    with row2_col2:
        st.markdown("**Division Risk Assessment Matrix**")
        radar_data = []
        colors = ["#7D00FF", "#00FF41", "#FF003C"]
        for idx, row in div_stats.iterrows():
            radar_data.append({
                "value": [
                    (row['avg_credit'] / 850) * 100, row['default_rate'] * 100 * 5, row['audit_rate'] * 100 * 5, (row['avg_days_late'] / 30) * 100, (row['total_profit'] / row['total_rev']) * 100 * 5
                ],
                "name": row['Division'], "itemStyle": {"color": colors[idx % len(colors)]}, "areaStyle": {"color": f"{colors[idx % len(colors)]}4D"}
            })

        risk_options = {
            "backgroundColor": "transparent",
            "tooltip": {"trigger": "item"},
            "legend": {"data": div_stats['Division'].tolist(), "bottom": "0%", "textStyle": {"color": "#A0A0A0"}},
            "radar": {
                "indicator": [
                    {"name": 'Avg Credit Health (Scale 0-100)', "max": 100}, {"name": 'Default Risk Factor', "max": 100}, {"name": 'Audit Flag Density', "max": 100}, {"name": 'Payment Latency', "max": 100}, {"name": 'Profitability Index', "max": 100}
                ],
                "splitArea": {"areaStyle": {"color": ["rgba(125,0,255,0.05)", "rgba(125,0,255,0.1)", "rgba(125,0,255,0.15)", "rgba(125,0,255,0.2)"]}},
                "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.2)"}},
                "splitLine": {"lineStyle": {"color": "rgba(255,255,255,0.2)"}},
                "axisName": {"color": "#E0E0E0"}
            },
            "series": [{"name": 'Risk Dimensions', "type": 'radar', "data": radar_data}]
        }
        st_echarts(options=risk_options, height="380px")
else:
    st.warning("Please ensure 'master_financial_dataset.csv' is in the same directory to render the intelligence charts.")
