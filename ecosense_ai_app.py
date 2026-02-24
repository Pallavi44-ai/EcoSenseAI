"""
╔══════════════════════════════════════════════════════╗
║     EcoSense AI – Smart Carbon Intelligence Platform  ║
║     Streamlit + ML/DL Powered | Deployable App        ║
╚══════════════════════════════════════════════════════╝

HOW TO RUN:
    pip install streamlit pandas numpy plotly scikit-learn tensorflow
    streamlit run ecosense_ai_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EcoSense AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── GLOBAL STYLE ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
        color: white;
    }
    .metric-card {
        background: #1a1a2e;
        border: 1px solid #00ff88;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .tip-box {
        background: #0f3460;
        border-left: 4px solid #00ff88;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
    }
    div[data-testid="stChatMessage"] {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & MODEL TRAINING (Integrated with Real Datasets)
# ══════════════════════════════════════════════════════════════════════════════

INDUSTRIES = {
    0: "Technology",    1: "Manufacturing", 2: "Retail",
    3: "Finance",       4: "Healthcare",    5: "Energy",
    6: "Agriculture",   7: "Transport",     8: "Construction",
    9: "Hospitality"
}

INDUSTRY_INTENSITY = {  # tCO2e per $M revenue
    0: 2.1, 1: 45.2, 2: 8.5, 3: 1.8, 4: 6.2,
    5: 85.4, 6: 62.1, 7: 48.3, 8: 35.6, 9: 12.4
}

@st.cache_data
def load_emission_dataset():
    """Loads the real carbon credits/emissions dataset."""
    try:
        df =pd.read_csv("data/carbon_credits_cal.csv")
        # Ensure we drop any NA values in the columns we need to train on
        df = df.dropna(subset=['Industry_Type', 'Energy_Demand_MWh', 'Emission_Produced_tCO2'])
        return df
    except FileNotFoundError:
        st.error("Dataset 'carbon_credits_cal.csv' not found. Please ensure it is in the same directory.")
        st.stop()

@st.cache_data
def load_climate_risk_dataset():
    """Loads the real climate risk dataset."""
    try:
        df = pd.read_csv("data/climate_risk.csv")
        df = df.dropna(subset=['temperature_change_c', 'rainfall_change_mm', 'avg_temperature_c', 'flood_risk'])
        return df
    except FileNotFoundError:
        st.error("Dataset 'climate_risk.csv' not found. Please ensure it is in the same directory.")
        st.stop()

@st.cache_data
def load_esg_dataset():
    """Loads and cleans the real ESG dataset (handles European number formats)."""
    try:
        df = pd.read_csv("data/ESG_Data_Emerging new.csv")
        feats = ['E', 'S', 'G', 'ESGS']
        # Clean the string formatting (replace commas with dots) and convert to float
        for col in feats:
            df[col] = df[col].astype(str).str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=feats)
        return df
    except FileNotFoundError:
        st.error("Dataset 'ESG_Data_Emerging new.csv' not found. Please ensure it is in the same directory.")
        st.stop()

@st.cache_resource
def train_emission_model():
    df = load_emission_dataset()
    le_ind = LabelEncoder()
    # Safely convert to string before encoding
    df['Industry_Encoded'] = le_ind.fit_transform(df['Industry_Type'].astype(str))
    
    # We predict emissions primarily from Energy Demand and Industry 
    # to maintain compatibility with the UI sliders
    X = df[['Industry_Encoded', 'Energy_Demand_MWh']]
    y = df['Emission_Produced_tCO2']
    
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(Xtr, ytr)
    
    r2 = model.score(Xte, yte)
    feats = ['Industry_Encoded', 'Energy_Demand_MWh']
    return model, r2, le_ind, feats

@st.cache_resource
def train_risk_model():
    df = load_climate_risk_dataset()
    le = LabelEncoder()
    
    X = df[['temperature_change_c', 'rainfall_change_mm', 'avg_temperature_c']]
    y = le.fit_transform(df['flood_risk'].astype(str))
    
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(Xtr, ytr)
    
    acc = model.score(Xte, yte)
    feats = ['temperature_change_c', 'rainfall_change_mm', 'avg_temperature_c']
    return model, acc, feats, le

@st.cache_resource
def train_esg_model():
    df = load_esg_dataset()
    
    X = df[['E', 'S', 'G']]
    y = df['ESGS']
    
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    model.fit(Xtr, ytr)
    
    r2 = model.score(Xte, yte)
    feats = ['E', 'S', 'G']
    return model, r2, feats


# ══════════════════════════════════════════════════════════════════════════════
#  ECOBOT CHATBOT
# ══════════════════════════════════════════════════════════════════════════════

ECOBOT_KB = {
    "carbon credit":        "A carbon credit = permit to emit 1 tonne of CO₂. Buy verified credits (Gold Standard / VCS) to offset unavoidable emissions. Rule: **reduce first, offset last!** Current market price: $15–$90/tonne depending on type.",
    "scope 1":              "**Scope 1** = Direct emissions from sources you own/control — factories, company vehicles, on-site fuel burning. Easiest to measure but often not the biggest slice.",
    "scope 2":              "**Scope 2** = Indirect emissions from purchased electricity/heat. Reduce by switching to renewable energy contracts (PPAs) or buying RECs.",
    "scope 3":              "**Scope 3** = All other indirect emissions — supply chain, employee commuting, product use & disposal. Can be 70–90% of total footprint! Hardest to measure but most impactful.",
    "esg":                  "ESG = **Environmental, Social, Governance**. Investors use ESG scores to assess sustainability risk. High ESG = lower capital cost + better long-term returns. Key frameworks: GRI, TCFD, SASB, ISSB.",
    "net zero":             "**Net Zero** = your total GHG emissions equal the amount you remove from the atmosphere. Most large companies target 2050. Use Science-Based Targets (SBTi) to set credible, Paris-aligned goals.",
    "tcfd":                 "TCFD = Task Force on Climate-related Financial Disclosures. It asks companies to report 4 areas: Governance, Strategy, Risk Management, Metrics & Targets. Now mandatory in UK, EU & increasingly globally.",
    "csrd":                 "CSRD = EU Corporate Sustainability Reporting Directive. Affects 50,000+ EU companies. Requires detailed sustainability reporting under ESRS standards from 2024–2028 rollout.",
    "carbon tax":           "A carbon tax puts a direct price on GHG emissions. Currently 70+ countries have one. EU ETS price ≈ €60–80/tonne. Canada: CAD $65/tonne. Higher taxes expected by 2030.",
    "renewable energy":     "Switch to renewables via: (1) On-site solar/wind, (2) Power Purchase Agreements (PPAs), (3) Renewable Energy Certificates (RECs). This directly cuts Scope 2 emissions.",
    "supply chain":         "Supply chain = Scope 3 Category 1 (purchased goods) + Category 4 (upstream transport). Audit suppliers, request emission data, switch to low-carbon alternatives. Use EcoVadis or CDP supply chain programs.",
    "paris agreement":      "The Paris Agreement (2015) targets limiting global warming to 1.5°C above pre-industrial levels. Requires countries to submit NDCs (Nationally Determined Contributions) every 5 years.",
    "reduce emission":      "Top strategies: 1) Switch to renewable energy, 2) EV fleet transition, 3) Supply chain audit, 4) Energy efficiency upgrades, 5) Reduce business travel (replace with video), 6) Circular economy practices.",
    "carbon offset":        "Carbon offsets = projects that remove/reduce CO₂ elsewhere to compensate for your emissions. Types: reforestation 🌳, soil carbon, cookstoves, direct air capture. Buy from Gold Standard or Verra-certified projects only.",
    "greenwashing":         "Greenwashing = making false or exaggerated sustainability claims. Red flags: vague language, no data backing, offsetting without reduction, cherry-picked metrics. Regulators (FCA, SEC) are cracking down hard.",
    "default":              "I can help with: **carbon emissions** (Scope 1/2/3), **ESG reporting**, **carbon credits**, **net zero strategy**, **regulations** (TCFD, CSRD, carbon tax), **renewable energy**, and **supply chain emissions**. Try asking about any of these! 🌱"
}

def ecobot_respond(query: str) -> str:
    q = query.lower()
    for keyword, answer in ECOBOT_KB.items():
        if keyword in q:
            return f"🤖 **EcoBot:** {answer}"
    for keyword, answer in ECOBOT_KB.items():
        for word in keyword.split():
            if len(word) > 4 and word in q:
                return f"🤖 **EcoBot:** {answer}"
    return f"🤖 **EcoBot:** {ECOBOT_KB['default']}"


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🌍 EcoSense AI")
    st.markdown("*Smart Carbon Intelligence*")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠  Home Dashboard",
        "📊  Carbon Calculator",
        "🌊  Climate Risk Predictor",
        "📈  ESG Score Generator",
        "🔄  Carbon Credit Estimator",
        "🤖  EcoBot – AI Advisor",
        "📉  Trends & Insights",
    ])
    st.markdown("---")
    st.caption("Models trained on real uploaded corporate & climate datasets. All ML runs in-browser.")

# Initialize the real-data models
em_model, em_r2, le_ind, em_feats = train_emission_model()
risk_model, risk_acc, r_feats, le_risk = train_risk_model()
esg_model, esg_r2, esg_feats = train_esg_model()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: HOME DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠  Home Dashboard":
    st.markdown("""
    <div class="main-header">
        <h1>🌍 EcoSense AI</h1>
        <p style="font-size:18px; margin:0">Smart Carbon Intelligence Platform — ML-Powered, Streamlit Deployed</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🧠 Emission Model R²",    f"{em_r2*100:.1f}%",   "+Real Data")
    c2.metric("⚡ Risk Model Accuracy",  f"{risk_acc*100:.1f}%", "+Real Data")
    c3.metric("📉 ESG Model R²",         f"{esg_r2*100:.1f}%",  "+Real Data")
    c4.metric("🌱 Features Used",        "Dynamic",              "per model")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌐 Global CO₂ Emission Map (Simulated)")
        world = pd.DataFrame({
            "country":  ["USA","China","India","Germany","Brazil","UK","Japan","Australia","Canada","France","Russia","South Africa"],
            "lat":      [38,   35,     20,     51,       -15,     55,  36,     -25,        56,      46,      60,     -29],
            "lon":      [-97,  105,    77,     10,       -47,     -3,  138,    134,        -96,     2,       90,     25],
            "gt_co2":   [5.0,  11.5,   2.7,    0.7,      0.5,     0.4, 1.1,    0.4,        0.7,     0.3,     1.7,    0.5],
            "intensity":[15.2, 7.4,    1.9,    9.4,      2.3,     5.8, 9.8,    16.9,       15.4,    4.8,     10.1,   8.9],
        })
        fig = px.scatter_geo(
            world, lat="lat", lon="lon", size="gt_co2",
            color="intensity", hover_name="country",
            color_continuous_scale="RdYlGn_r",
            projection="natural earth",
            title="CO₂ by Country (bubble = Gt, color = intensity)"
        )
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🏭 Global Emissions by Sector")
        sec = pd.DataFrame({
            "Sector":   ["Energy", "Industry", "Transport", "Agriculture", "Buildings", "Waste"],
            "Share_%":  [34,       24,          16,           11,            9,           6],
        })
        fig2 = px.pie(sec, values="Share_%", names="Sector",
                      color_discrete_sequence=px.colors.qualitative.Safe,
                      title="Global GHG Breakdown (%)")
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("🚀 What EcoSense AI Can Do")
    features = [
        ("📊", "Carbon Calculator",      "ML model predicts your Scope 1/2/3 footprint from company inputs"),
        ("🌊", "Climate Risk AI",        "GBM classifier scores physical climate risk for any location"),
        ("📈", "ESG Auto-Score",         "Generate compliance-ready ESG score with regulatory gap analysis"),
        ("🔄", "Carbon Credit Calc",     "Estimate offset volumes needed + cost projections to 2050"),
        ("🤖", "EcoBot Chatbot",         "Knowledge-base chatbot for carbon strategy & climate regulations"),
        ("📉", "Trends & Insights",      "Interactive charts on global CO₂, renewables & industry benchmarks"),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.info(f"**{icon} {title}**\n\n{desc}")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: CARBON CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊  Carbon Calculator":
    st.title("📊 AI Carbon Footprint Calculator")
    st.caption("Model: Random Forest (R² ≈ {:.1f}%) | Scope 1 + 2 + 3 breakdown".format(em_r2 * 100))

    with st.form("carbon_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🏢 Company")
            employees  = st.slider("Employees",             10,   50000, 500)
            revenue    = st.number_input("Revenue ($M)",    1.0,  10000., 100.)
            industry   = st.selectbox("Industry", list(INDUSTRIES.values()))
            ind_code   = [k for k, v in INDUSTRIES.items() if v == industry][0]
            renewables = st.slider("Renewables Mix (%)",    0,    100,   20)

        with col2:
            st.subheader("⚡ Energy & Ops")
            energy_kwh  = st.number_input("Energy (kWh/yr)",   1000, 2_000_000, 80_000)
            fleet       = st.slider("Fleet Vehicles",          0,    1000,      15)
            office_sqft = st.number_input("Office Area (sqft)",500,  800_000,   8_000)

        with col3:
            st.subheader("✈️ Travel & Supply")
            travel_miles = st.number_input("Business Travel (miles)", 0, 500_000, 12_000)
            supply_spend = st.number_input("Supply Chain Spend ($)",  500, 5_000_000, 200_000)

        submitted = st.form_submit_button("🚀 Calculate Emissions", type="primary", use_container_width=True)

    if submitted:
        # Map UI Inputs to ML Features trained on the Carbon dataset
        energy_mwh = energy_kwh / 1000.0
        
        # Safely handle industry encoding in case of mismatch
        try:
            ind_enc = le_ind.transform([industry])[0]
        except ValueError:
            # Fallback to the first class if the UI industry name isn't in the dataset
            ind_enc = 0 
            
        # Predict base emissions from the ML Model trained on real dataset
        X_in = np.array([[ind_enc, energy_mwh]])
        predicted_base_emissions = max(0, em_model.predict(X_in)[0])

        # Layer in UI specific metrics so the interface remains fully interactive
        scope1 = max(0, fleet * 4.6 + predicted_base_emissions * 0.1) 
        scope2 = max(0, predicted_base_emissions * 0.9 * (1 - renewables / 200))
        scope3 = max(0, supply_spend * 0.0004 + travel_miles * 0.000158)
        
        total = scope1 + scope2 + scope3

        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🌍 Total Emissions",       f"{total:,.0f} tCO₂e")
        m2.metric("🏭 Scope 1 (Direct)",      f"{scope1:,.0f} t")
        m3.metric("⚡ Scope 2 (Energy)",      f"{scope2:,.0f} t")
        m4.metric("🔗 Scope 3 (Supply Chain)",f"{scope3:,.0f} t")

        col_a, col_b = st.columns(2)

        with col_a:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=total,
                title={"text": "Total Emissions (tCO₂e)"},
                gauge={
                    "axis": {"range": [0, max(5000, total * 1.3)]},
                    "bar": {"color": "#00ff88"},
                    "steps": [
                        {"range": [0,               total * 0.33],  "color": "#1a472a"},
                        {"range": [total * 0.33,    total * 0.66],  "color": "#f6c90e"},
                        {"range": [total * 0.66,    total * 1.3],   "color": "#c0392b"},
                    ],
                    "threshold": {"line": {"color": "red", "width": 3}, "value": total * 1.2}
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            scope_df = pd.DataFrame({
                "Scope":    ["Scope 1 (Direct)", "Scope 2 (Energy)", "Scope 3 (Indirect)"],
                "Emissions": [scope1, scope2, scope3]
            })
            fig2 = px.pie(scope_df, values="Emissions", names="Scope",
                          color_discrete_sequence=["#2ecc71", "#3498db", "#e74c3c"],
                          title="Emission Scope Breakdown")
            fig2.update_layout(height=300)
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("💡 AI Recommendations")
        if scope3 > (scope1 + scope2):
            st.warning("⚠️ **Scope 3 dominates.** Prioritize supply chain sustainability audit. Consider SBTi FLAG targets.")
        if energy_kwh > 200_000 and renewables < 50:
            st.info("💡 **High energy usage + low renewables.** Sign a Power Purchase Agreement (PPA) to cut Scope 2 by 80%+.")
        if fleet > 50:
            st.info("🚗 **Large fleet detected.** EV transition can eliminate ~4.6 tCO₂e per vehicle per year.")
        if total < 500:
            st.success("✅ **Strong carbon profile!** Focus on maintaining supplier standards and publishing annual ESG report.")

        # Feature importance (updated for the new real data features)
        with st.expander("🔍 Model Feature Importance"):
            imp_df = pd.DataFrame({
                "Feature":    em_feats,
                "Importance": em_model.feature_importances_
            }).sort_values("Importance", ascending=True)
            fig3 = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                          title="What drives emission prediction most?",
                          color="Importance", color_continuous_scale="Greens")
            st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: CLIMATE RISK PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🌊  Climate Risk Predictor":
    st.title("🌊 Climate Risk Predictor")
    st.caption(f"Model: Gradient Boosting Classifier | Accuracy ≈ {risk_acc*100:.1f}%")

    with st.form("risk_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📍 Location")
            lat          = st.slider("Latitude",           -60.0, 70.0,  28.6)
            lon          = st.slider("Longitude",          -180.0, 180.0, 77.2)
            elevation    = st.number_input("Elevation (m)",0, 5000, 220)
            coastal_km   = st.number_input("Distance from Coast (km)", 0, 800, 80)
        with col2:
            st.subheader("🌡️ Climate Projections")
            temp_change  = st.slider("Temp Rise by 2050 (°C)", 0.5, 4.5, 1.5)
            rain_change  = st.slider("Rainfall Change (%)",    -45,  45,   0)
            gdp          = st.number_input("Regional GDP/capita ($)", 500, 100_000, 20_000)
            infra        = st.slider("Infrastructure Score (0–100)", 0, 100, 65)

        run_risk = st.form_submit_button("🔮 Predict Climate Risk", type="primary", use_container_width=True)

    if run_risk:
        # Map UI Inputs to ML Features trained on the Climate Risk dataset
        # Estimating average temperature heavily based on latitude for the model 
        estimated_avg_temp = max(0, 35 - abs(lat) * 0.4) 
        
        # Ensure array matches the ['temperature_change_c', 'rainfall_change_mm', 'avg_temperature_c'] features
        X_r = np.array([[temp_change, rain_change, estimated_avg_temp]])
        
        risk_pred_enc = risk_model.predict(X_r)[0]
        risk_proba    = risk_model.predict_proba(X_r)[0]
        risk_label    = le_risk.inverse_transform([risk_pred_enc])[0]
        classes       = le_risk.inverse_transform(risk_model.classes_)

        badge = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(risk_label, "⚪")
        st.markdown(f"## Risk Level: {badge} **{risk_label}**")

        col_a, col_b = st.columns(2)
        with col_a:
            prob_df = pd.DataFrame({"Risk Level": classes, "Probability %": risk_proba * 100})
            fig = px.bar(prob_df, x="Risk Level", y="Probability %",
                         color="Risk Level",
                         color_discrete_map={"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"},
                         title="Risk Class Probability Distribution")
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            # Radar chart of risk dimensions based on the UI inputs
            dims = ["Heat Stress", "Flood Risk", "Drought Risk", "Storm Risk", "Sea-Level Rise"]
            scores = [
                min(100, temp_change * 22),
                min(100, max(0, (500 - coastal_km) / 5 + (elevation < 20) * 30)),
                min(100, max(0, -rain_change * 1.5 + 10)),
                min(100, temp_change * 12 + 20),
                min(100, max(0, (20 - coastal_km) * 3 + 10)),
            ]
            fig2 = go.Figure(go.Scatterpolar(
                r=scores + [scores[0]], theta=dims + [dims[0]],
                fill="toself", fillcolor="rgba(231,76,60,0.3)",
                line_color="#e74c3c"
            ))
            fig2.update_layout(
                polar=dict(radialaxis=dict(range=[0, 100])),
                title="Multi-Hazard Risk Profile",
                height=320
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("⚠️ Key Risk Alerts")
        alerts = []
        if temp_change > 2.5:   alerts.append(("🌡️ Extreme Heat Risk",   f"+{temp_change}°C — operational disruption likely"))
        if coastal_km < 15:     alerts.append(("🌊 Coastal Flood Risk",   f"Only {coastal_km:.0f} km from coast"))
        if elevation < 10:      alerts.append(("📉 Low Elevation Risk",   f"{elevation}m — storm surge vulnerability"))
        if rain_change < -20:   alerts.append(("🏜️ Drought Risk",        f"{rain_change}% rainfall decrease"))
        if rain_change > 25:    alerts.append(("🌧️ Flood / Landslide Risk", f"+{rain_change}% rainfall increase"))
        if infra < 40:          alerts.append(("🏗️ Weak Infrastructure", "Low resilience score reduces adaptive capacity"))

        if alerts:
            for title, desc in alerts:
                st.warning(f"**{title}**: {desc}")
        else:
            st.success("✅ No critical risk flags for this location. Continue monitoring projections annually.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: ESG SCORE GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📈  ESG Score Generator":
    st.title("📈 ESG Score Generator")
    st.caption(f"Rule-based sub-scoring + ML Regressor (R² ≈ {esg_r2*100:.1f}%) trained on emerging market dataset.")

    with st.form("esg_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🌿 Environmental (E)")
            net_zero      = st.checkbox("Has Net Zero Target (SBTi)")
            renewables_50 = st.checkbox("Renewables > 50% energy mix")
            scope_track   = st.checkbox("Tracks Scope 1, 2 & 3 Emissions")
            circular      = st.checkbox("Circular Economy Practices")
            water_mgmt    = st.checkbox("Water Usage Management Plan")
            biodiversity  = st.checkbox("Biodiversity / Land-Use Policy")
            e_score = (net_zero*20 + renewables_50*18 + scope_track*22 + circular*15 + water_mgmt*13 + biodiversity*12)

        with col2:
            st.subheader("👥 Social (S)")
            diversity     = st.checkbox("Diversity & Inclusion Program")
            living_wage   = st.checkbox("Pays Living Wage")
            community     = st.checkbox("Community Investment Program")
            health_safety = st.checkbox("ISO 45001 Health & Safety Cert")
            human_rights  = st.checkbox("Human Rights Policy in Supply Chain")
            training      = st.checkbox("Employee Training & Development")
            s_score = (diversity*18 + living_wage*22 + community*15 + health_safety*20 + human_rights*17 + training*8)

        with col3:
            st.subheader("🏛️ Governance (G)")
            board_div     = st.checkbox("Board Diversity > 30%")
            anti_corrupt  = st.checkbox("Anti-Corruption Policy")
            sr_report     = st.checkbox("Annual Sustainability Report")
            audit         = st.checkbox("Third-Party ESG Audit")
            exec_pay      = st.checkbox("ESG-linked Executive Pay")
            whistleblower = st.checkbox("Whistleblower Protection Policy")
            g_score = (board_div*18 + anti_corrupt*20 + sr_report*22 + audit*20 + exec_pay*10 + whistleblower*10)

        gen_esg = st.form_submit_button("📊 Generate ESG Score", type="primary", use_container_width=True)

    if gen_esg:
        # Normalize each pillar to /100
        e_norm = min(100, e_score)
        s_norm = min(100, s_score)
        g_norm = min(100, g_score)
        
        # Use ML Model to predict TOTAL ESG score from real data correlations instead of manual weights
        X_in = np.array([[e_norm, s_norm, g_norm]])
        total_esg = max(0, min(100, esg_model.predict(X_in)[0]))

        grade = "A+" if total_esg>=88 else "A" if total_esg>=78 else "B+" if total_esg>=66 else "B" if total_esg>=55 else "C" if total_esg>=40 else "D"

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🌿 Environmental", f"{e_norm:.0f}/100")
        m2.metric("👥 Social",        f"{s_norm:.0f}/100")
        m3.metric("🏛️ Governance",   f"{g_norm:.0f}/100")
        m4.metric("📊 Overall ESG",  f"{total_esg:.0f}/100", f"Grade: {grade}")

        col_a, col_b = st.columns(2)
        with col_a:
            comp_df = pd.DataFrame({
                "Pillar":      ["Environmental", "Social", "Governance"],
                "Your Score":  [e_norm, s_norm, g_norm],
                "Industry Avg":[58,      61,      64],
                "Best in Class":[88,     85,      90],
            })
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Your Score",     x=comp_df["Pillar"], y=comp_df["Your Score"],     marker_color="#00ff88"))
            fig.add_trace(go.Bar(name="Industry Avg",   x=comp_df["Pillar"], y=comp_df["Industry Avg"],   marker_color="#3498db"))
            fig.add_trace(go.Bar(name="Best in Class",  x=comp_df["Pillar"], y=comp_df["Best in Class"],  marker_color="#9b59b6"))
            fig.update_layout(barmode="group", title="ESG Score Benchmark", yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("📋 Regulatory Compliance")
            checks = {
                "GRI Standards":   "✅" if sr_report else "❌",
                "TCFD":            "✅" if (scope_track and sr_report) else "⚠️ Partial",
                "EU CSRD":         "✅" if total_esg > 70 else "❌ Not Ready",
                "SEC Climate":     "✅" if scope_track else "❌",
                "ISO 14001 (Env)": "✅" if (net_zero and scope_track) else "⚠️ Partial",
                "UN SDGs Aligned": "✅" if total_esg > 65 else "❌",
            }
            for framework, status in checks.items():
                st.write(f"**{framework}**: {status}")

        st.subheader("🎯 Priority Actions")
        if not scope_track:
            st.error("🚨 **Critical Gap:** Implement Scope 1, 2 & 3 tracking — required by TCFD & SEC climate rules")
        if not sr_report:
            st.warning("⚠️ Publish an annual sustainability report (GRI Standards) — investor requirement")
        if not audit:
            st.info("💡 Get a third-party ESG audit for credibility and investor confidence")
        if total_esg > 75:
            st.success(f"🏆 Strong ESG performance (Grade {grade})! Consider submitting to CDP disclosure for top rating.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: CARBON CREDIT ESTIMATOR
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔄  Carbon Credit Estimator":
    st.title("🔄 Carbon Credit Estimator")
    st.caption("Estimate offset requirements + cost projections for your decarbonization roadmap")

    with st.form("credit_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Emission Profile")
            total_em      = st.number_input("Annual Emissions (tCO₂e)", 0, 2_000_000, 1_500)
            baseline_year = 2024
            reduce_target = st.slider("Reduction Target by 2035 (%)",  0, 100, 50)
            offset_pct    = st.slider("% Offset via Credits",          0, 100, 30)
        with col2:
            st.subheader("💳 Credit Market")
            credit_type = st.selectbox("Credit Type", [
                "Voluntary (VCS / Gold Standard)  ~$22/t",
                "Nature-Based Solutions           ~$35/t",
                "Tech Removal (BECCS / DAC)       ~$200/t",
                "EU ETS Compliance                ~$75/t",
            ])
            price_map = {
                "Voluntary (VCS / Gold Standard)  ~$22/t":  22,
                "Nature-Based Solutions           ~$35/t":  35,
                "Tech Removal (BECCS / DAC)       ~$200/t": 200,
                "EU ETS Compliance                ~$75/t":  75,
            }
            credit_price = price_map[credit_type]
            include_shadow = st.checkbox("Apply Shadow Carbon Price ($50/t for internal risk)", True)

        run_credit = st.form_submit_button("💰 Calculate Credit Need", type="primary", use_container_width=True)

    if run_credit:
        reduced_em    = total_em * (1 - reduce_target / 100)
        offset_amount = reduced_em * (offset_pct / 100)
        residual      = reduced_em - offset_amount
        credit_cost   = offset_amount * credit_price
        shadow_cost   = total_em * 50 if include_shadow else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🎯 Post-Reduction Emissions", f"{reduced_em:,.0f} t")
        m2.metric("🌱 Credits Needed",            f"{offset_amount:,.0f} t")
        m3.metric("💵 Credit Cost",               f"${credit_cost:,.0f}")
        m4.metric("⚠️ Shadow Carbon Liability",  f"${shadow_cost:,.0f}")

        # Pathway chart to 2050
        years   = list(range(2024, 2051))
        n_yrs   = len(years)
        gross   = [total_em * (1 - (i / (n_yrs - 1)) * reduce_target / 100) for i in range(n_yrs)]
        offsets = [g * offset_pct / 100 for g in gross]
        net     = [g - o for g, o in zip(gross, offsets)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=gross,   name="Gross Emissions",     fill="tozeroy", line_color="#e74c3c",  fillcolor="rgba(231,76,60,0.15)"))
        fig.add_trace(go.Scatter(x=years, y=offsets, name="Offset Credits",      fill="tozeroy", line_color="#3498db",  fillcolor="rgba(52,152,219,0.15)"))
        fig.add_trace(go.Scatter(x=years, y=net,     name="Net Emissions",        fill="tozeroy", line_color="#2ecc71",  fillcolor="rgba(46,204,113,0.15)"))
        fig.add_hline(y=0, line_dash="dash", line_color="white", annotation_text="Net Zero", annotation_position="bottom right")
        fig.update_layout(title="Decarbonization Pathway to 2050",
                           xaxis_title="Year", yaxis_title="tCO₂e",
                           legend=dict(orientation="h", yanchor="top", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

        # Cumulative cost table
        st.subheader("📅 5-Year Cost Projection")
        years5  = list(range(2024, 2030))
        price_growth = 1.08  # 8% annual price increase assumed
        rows = []
        for i, yr in enumerate(years5):
            g = total_em * (1 - (i / n_yrs) * reduce_target / 100)
            o = g * offset_pct / 100
            p = credit_price * (price_growth ** i)
            rows.append({"Year": yr, "Gross Emissions": f"{g:,.0f}", "Credits Needed": f"{o:,.0f}", "Price/t ($)": f"{p:.0f}", "Annual Cost ($)": f"{o*p:,.0f}"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: ECOBOT CHATBOT
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🤖  EcoBot – AI Advisor":
    st.title("🤖 EcoBot – Carbon Intelligence Advisor")
    st.caption("Knowledge-base chatbot covering emissions, ESG, credits, regulations & strategy")

    QUICK_QUESTIONS = [
        "What is a carbon credit?",
        "What is Scope 3?",
        "How do I reach net zero?",
        "What is ESG reporting?",
        "What is the Paris Agreement?",
        "How to reduce supply chain emissions?",
    ]

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{
            "role": "assistant",
            "content": "👋 **Hi! I'm EcoBot.** Ask me anything about carbon emissions, ESG reporting, climate regulations, net zero strategy, or carbon credits! 🌱"
        }]

    # Quick-question buttons
    st.markdown("**💡 Quick Questions:**")
    cols = st.columns(3)
    for i, q in enumerate(QUICK_QUESTIONS):
        if cols[i % 3].button(q, key=f"qq_{i}"):
            st.session_state.chat_history.append({"role": "user",      "content": q})
            st.session_state.chat_history.append({"role": "assistant", "content": ecobot_respond(q)})

    st.markdown("---")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Ask EcoBot a carbon / ESG question..."):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        response = ecobot_respond(user_input)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: TRENDS & INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📉  Trends & Insights":
    st.title("📉 Emission Trends & Market Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📅 Global CO₂ (2000–2024)")
        years = list(range(2000, 2025))
        co2 = [25.0 + i * 0.52 + np.random.normal(0, 0.2) for i in range(25)]
        co2[20] = 36.3   # COVID dip 2020
        co2[21] = 36.7
        co2[22] = 37.1
        co2[23] = 37.4
        co2[24] = 37.8
        fig = px.area(x=years, y=co2, title="Global CO₂ (Gt)",
                      labels={"x": "Year", "y": "Gt CO₂"},
                      color_discrete_sequence=["#e74c3c"])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🌱 Renewable Energy Capacity (TW)")
        yrs2  = list(range(2010, 2025))
        solar = [0.04 * (1.33 ** i) for i in range(15)]
        wind  = [0.20 * (1.15 ** i) for i in range(15)]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=yrs2, y=solar, name="Solar 🌞", fill="tozeroy", line_color="#f39c12"))
        fig2.add_trace(go.Scatter(x=yrs2, y=wind,  name="Wind 💨",  fill="tozeroy", line_color="#3498db"))
        fig2.update_layout(title="Renewable Growth (TW Capacity)")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🏭 Emission Intensity by Industry (tCO₂e per $M Revenue)")
    ind_df = pd.DataFrame({
        "Industry":   list(INDUSTRIES.values()),
        "Intensity":  list(INDUSTRY_INTENSITY.values()),
        "YoY Change": [-5, -3, -2, -8, -4, -1, -6, -2, -3, -5]
    })
    fig3 = px.bar(ind_df, x="Industry", y="Intensity",
                   color="YoY Change", color_continuous_scale="RdYlGn",
                   title="Carbon Intensity vs YoY Improvement (%)",
                   text="Intensity")
    fig3.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("💵 Carbon Credit Price Forecast ($/t)")
    yrs3 = list(range(2024, 2036))
    voluntary  = [22 * (1.08 ** i) for i in range(12)]
    compliance = [75 * (1.10 ** i) for i in range(12)]
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=yrs3, y=voluntary,  name="Voluntary (VCS)", line_color="#2ecc71"))
    fig4.add_trace(go.Scatter(x=yrs3, y=compliance, name="EU ETS Compliance", line_color="#e74c3c", line_dash="dash"))
    fig4.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.05, annotation_text="Current Voluntary Range")
    fig4.update_layout(title="Carbon Price Forecast to 2035", xaxis_title="Year", yaxis_title="$/tonne")
    st.plotly_chart(fig4, use_container_width=True)


# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("EcoSense AI v1.0 | ML: RandomForest + GradientBoosting | Built with Streamlit | Educational purposes only")