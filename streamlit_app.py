import sys
import time
from pathlib import Path
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure model directory is on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from model import (
    predict_price,
    load_artifacts,
    get_known_cars,
    get_car_defaults,
    get_feature_importances,
    get_dataset
)

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="AutoValuate AI | Used Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    
    /* Header styling */
    .hero-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 24px 30px;
        border-radius: 16px;
        color: white;
        margin-bottom: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
        background: linear-gradient(90deg, #60a5fa 0%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 6px;
        margin-bottom: 0px;
    }
    
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-success {
        background-color: #dcfce7;
        color: #15803d;
        border: 1px solid #86efac;
    }
    
    .badge-info {
        background-color: #e0f2fe;
        color: #0369a1;
        border: 1px solid #7dd3fc;
    }

    .valuation-box {
        background: #f8fafc;
        border: 2px solid #e2e8f0;
        border-radius: 16px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }

    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Backend Connectivity Helper
# ---------------------------------------------------------
API_DEFAULT_URL = "http://127.0.0.1:8000"

@st.cache_data(ttl=5)
def check_api_health(url: str):
    try:
        res = requests.get(f"{url.rstrip('/')}/health", timeout=1.5)
        if res.status_code == 200:
            return True, res.json()
        return False, None
    except Exception:
        return False, None

# ---------------------------------------------------------
# Sidebar Configuration & Quick Presets
# ---------------------------------------------------------
st.sidebar.image("https://images.unsplash.com/photo-1503376780353-7e6692767b70?auto=format&fit=crop&w=400&q=80", use_container_width=True)
st.sidebar.markdown("### ⚙️ Engine Settings")

api_endpoint = st.sidebar.text_input(
    "FastAPI Base URL",
    value=API_DEFAULT_URL,
    help="FastAPI backend host address"
)

is_api_online, health_data = check_api_health(api_endpoint)

if is_api_online:
    st.sidebar.success("🟢 FastAPI Backend: **Connected**")
else:
    st.sidebar.warning("⚡ Direct Engine: **In-Process ML**")
    st.sidebar.caption("FastAPI server offline; utilizing high-speed direct in-memory ML model.")

execution_mode = st.sidebar.radio(
    "Inference Route",
    ["Auto (API if online)", "Always Direct ML Model", "Always FastAPI Endpoint"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Quick Presets")

popular_presets = {
    "Maruti Swift 2015 (Diesel)": {
        "car": "swift", "year": 2015, "present_price": 6.87, "kms": 42000,
        "fuel": "Diesel", "seller": "Dealer", "trans": "Manual", "owner": 0
    },
    "Honda City 2017 (Petrol)": {
        "car": "city", "year": 2017, "present_price": 9.85, "kms": 25000,
        "fuel": "Petrol", "seller": "Dealer", "trans": "Manual", "owner": 0
    },
    "Toyota Fortuner 2017 (Auto)": {
        "car": "fortuner", "year": 2017, "present_price": 31.0, "kms": 55000,
        "fuel": "Diesel", "seller": "Dealer", "trans": "Automatic", "owner": 0
    },
    "Hyundai Verna 2016 (Diesel)": {
        "car": "verna", "year": 2016, "present_price": 9.4, "kms": 38000,
        "fuel": "Diesel", "seller": "Dealer", "trans": "Manual", "owner": 0
    },
    "Maruti Ciaz 2016 (Petrol)": {
        "car": "ciaz", "year": 2016, "present_price": 8.89, "kms": 31000,
        "fuel": "Petrol", "seller": "Dealer", "trans": "Manual", "owner": 0
    },
    "Hyundai i20 2016 (2nd Owner)": {
        "car": "i20", "year": 2016, "present_price": 7.5, "kms": 48000,
        "fuel": "Petrol", "seller": "Individual", "trans": "Manual", "owner": 1
    }
}

selected_preset = st.sidebar.selectbox("Apply Test Car Preset:", ["-- Select Preset --"] + list(popular_presets.keys()))

preset_vals = None
if selected_preset != "-- Select Preset --":
    preset_vals = popular_presets[selected_preset]

# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                <span class="badge badge-info">Random Forest Regressor • R² 0.965</span>
                <span class="badge badge-success">Production Ready</span>
            </div>
            <h1 class="hero-title">AutoValuate AI — Used Car Valuation</h1>
            <p class="hero-subtitle">Intelligent pricing engine, depreciation analytics & market intelligence.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Main Tabs Navigation
# ---------------------------------------------------------
tab_val, tab_analytics, tab_compare, tab_emi, tab_data, tab_api = st.tabs([
    "🚗 Car Valuation",
    "📊 Market Analytics",
    "⚖️ Variant Compare",
    "💳 EMI Calculator",
    "📋 Dataset Explorer",
    "⚡ Developer & API"
])


# =========================================================
# TAB 1: CAR VALUATION
# =========================================================
with tab_val:
    col_input, col_result = st.columns([1.1, 0.9], gap="large")

    all_cars = get_known_cars()

    with col_input:
        st.subheader("1. Enter Vehicle Details")

        # Determine default values from preset or standard
        def_car = preset_vals["car"] if preset_vals else "swift"
        def_year = preset_vals["year"] if preset_vals else 2015
        def_present = preset_vals["present_price"] if preset_vals else 6.87
        def_kms = preset_vals["kms"] if preset_vals else 40000
        def_fuel = preset_vals["fuel"] if preset_vals else "Diesel"
        def_seller = preset_vals["seller"] if preset_vals else "Dealer"
        def_trans = preset_vals["trans"] if preset_vals else "Manual"
        def_owner = preset_vals["owner"] if preset_vals else 0

        # Car selection
        car_name_idx = all_cars.index(def_car) if def_car in all_cars else 0
        car_name = st.selectbox(
            "Car Brand / Model",
            options=all_cars,
            index=car_name_idx,
            help="Select vehicle name from the trained model dataset"
        )

        # Quick auto-defaults on car change (if no preset is active)
        if not preset_vals:
            car_specs = get_car_defaults(car_name)
        else:
            car_specs = preset_vals

        c1, c2 = st.columns(2)
        with c1:
            year = st.slider(
                "Manufacturing Year",
                min_value=2004,
                max_value=2026,
                value=int(def_year),
                help="Year when the vehicle was manufactured"
            )
            car_age = max(0, 2026 - year)
            st.caption(f"Vehicle Age: **{car_age} years**")

        with c2:
            present_price = st.number_input(
                "Present/Showroom Price (₹ Lakhs)",
                min_value=0.1,
                max_value=120.0,
                value=float(def_present),
                step=0.25,
                help="Original or current showroom price in Lakhs (1 Lakh = ₹100,000)"
            )
            st.caption(f"Approx: **₹ {present_price * 100000:,.0f}**")

        kms_driven = st.number_input(
            "Kilometers Driven (Odometer)",
            min_value=0,
            max_value=500000,
            value=int(def_kms),
            step=2500,
            help="Total distance traveled in kilometers"
        )

        c3, c4 = st.columns(2)
        with c3:
            fuel_options = ["Petrol", "Diesel", "CNG"]
            fuel_type = st.selectbox("Fuel Type", fuel_options, index=fuel_options.index(def_fuel) if def_fuel in fuel_options else 0)

            transmission_options = ["Manual", "Automatic"]
            transmission = st.selectbox("Transmission", transmission_options, index=transmission_options.index(def_trans) if def_trans in transmission_options else 0)

        with c4:
            seller_options = ["Dealer", "Individual"]
            seller_type = st.selectbox("Seller Type", seller_options, index=seller_options.index(def_seller) if def_seller in seller_options else 0)

            owner_map = {0: "0 (1st Owner)", 1: "1 (2nd Owner)", 3: "3 (3rd+ Owner)"}
            owner_options = list(owner_map.keys())
            owner = st.selectbox(
                "Previous Owners",
                options=owner_options,
                format_func=lambda x: owner_map[x],
                index=owner_options.index(def_owner) if def_owner in owner_options else 0
            )

        payload = {
            "Car_Name": str(car_name),
            "Year": int(year),
            "Present_Price": float(present_price),
            "Kms_Driven": int(kms_driven),
            "Fuel_Type": str(fuel_type),
            "Seller_Type": str(seller_type),
            "Transmission": str(transmission),
            "Owner": int(owner),
        }

        predict_clicked = st.button("Calculate Resale Value 🚀", type="primary", use_container_width=True)

    with col_result:
        st.subheader("2. AI Valuation Summary")

        # Execute prediction
        predicted_val = None
        meta_res = {}
        error_msg = None
        latency_ms = 0

        should_use_api = (execution_mode == "Always FastAPI Endpoint") or (execution_mode == "Auto (API if online)" and is_api_online)

        if predict_clicked or True:  # Default live calculation
            start_t = time.perf_counter()
            if should_use_api and is_api_online:
                try:
                    res = requests.post(f"{api_endpoint.rstrip('/')}/predict", json=payload, timeout=5)
                    if res.status_code == 200:
                        data = res.json()
                        predicted_val = data.get("prediction_price")
                        meta_res = data
                    else:
                        error_msg = f"API Error: HTTP {res.status_code}"
                except Exception as e:
                    # Fallback to direct model
                    predicted_val, meta_res = predict_price(payload)
            else:
                predicted_val, meta_res = predict_price(payload)
            latency_ms = round((time.perf_counter() - start_t) * 1000, 1)

        if predicted_val is not None:
            # Display Valuation Cards
            inr_val = int(predicted_val * 100000)
            low_val = meta_res.get("fair_price_low", round(predicted_val * 0.94, 2))
            high_val = meta_res.get("fair_price_high", round(predicted_val * 1.06, 2))
            deprec = meta_res.get("depreciation_pct")
            if deprec is None and present_price > 0:
                deprec = round(max(0.0, (1 - (predicted_val / present_price)) * 100), 1)

            st.markdown(f"""
            <div class="valuation-box">
                <span style="font-size: 0.85rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px;">
                    Estimated Resale Value
                </span>
                <div style="font-size: 3rem; font-weight: 800; color: #1e293b; margin: 4px 0;">
                    ₹ {predicted_val:.2f} <span style="font-size: 1.5rem; color: #64748b;">Lakhs</span>
                </div>
                <div style="font-size: 1.15rem; font-weight: 600; color: #2563eb;">
                    ₹ {inr_val:,.0f}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("####")

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Fair Price Range", f"₹ {low_val:.2f}L - {high_val:.2f}L")
            with m2:
                st.metric("Value Depreciation", f"{deprec:.1f}%")
            with m3:
                retention = max(0.0, 100.0 - deprec)
                st.metric("Value Retained", f"{retention:.1f}%")

            # Depreciation progress
            st.progress(min(1.0, max(0.0, deprec / 100.0)))

            # Valuation Insights
            with st.expander("📌 Key Valuation Drivers", expanded=True):
                st.write(f"- **Vehicle Age**: {car_age} years on road has depreciated approx ~{min(85, car_age * 7)}% of initial showroom capital.")
                st.write(f"- **Usage Profile**: {kms_driven:,} km recorded on odometer reflects {'normal' if kms_driven < 60000 else 'elevated'} wear-and-tear.")
                st.write(f"- **Fuel & Transmission**: {fuel_type} with {transmission} transmission variant.")
                st.write(f"- **Ownership History**: {owner_map[owner]}.")
                st.caption(f"⚡ Calculated in {latency_ms} ms via {'FastAPI Endpoint' if should_use_api and is_api_online else 'Direct ML Pipeline'}")

            # Projected 5-Year Depreciation Curve
            st.markdown("#### 📉 Projected Resale Value (Next 5 Years)")
            years_ahead = [0, 1, 2, 3, 4, 5]
            projected_prices = []
            for y_off in years_ahead:
                decay_factor = (1 - 0.08) ** y_off
                projected_prices.append(round(max(0.2, predicted_val * decay_factor), 2))

            df_proj = pd.DataFrame({
                "Timeline": ["Current"] + [f"+{y} Year" for y in years_ahead[1:]],
                "Projected Value (₹ Lakhs)": projected_prices
            })
            fig_proj = px.line(
                df_proj,
                x="Timeline",
                y="Projected Value (₹ Lakhs)",
                markers=True,
                title=f"Depreciation Forecast for {car_name.title()}",
                color_discrete_sequence=["#2563eb"]
            )
            fig_proj.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_proj, use_container_width=True)

        elif error_msg:
            st.error(error_msg)


# =========================================================
# TAB 2: MARKET ANALYTICS
# =========================================================
with tab_analytics:
    st.subheader("📊 Automotive Resale Market Insights")
    df_raw = get_dataset()

    if not df_raw.empty:
        c_an1, c_an2 = st.columns(2)

        with c_an1:
            fig_fuel = px.box(
                df_raw,
                x="Fuel_Type",
                y="Selling_Price",
                color="Fuel_Type",
                title="Resale Price Distribution by Fuel Type (₹ Lakhs)",
                labels={"Selling_Price": "Selling Price (₹ Lakhs)", "Fuel_Type": "Fuel"}
            )
            fig_fuel.update_layout(height=340, showlegend=False)
            st.plotly_chart(fig_fuel, use_container_width=True)

        with c_an2:
            fig_trans = px.box(
                df_raw,
                x="Transmission",
                y="Selling_Price",
                color="Transmission",
                title="Transmission Impact on Resale Value (₹ Lakhs)",
                labels={"Selling_Price": "Selling Price (₹ Lakhs)"}
            )
            fig_trans.update_layout(height=340, showlegend=False)
            st.plotly_chart(fig_trans, use_container_width=True)

        c_an3, c_an4 = st.columns(2)

        with c_an3:
            df_plot = df_raw.copy()
            df_plot["Car_Age"] = 2026 - df_plot["Year"]
            fig_scatter = px.scatter(
                df_plot,
                x="Kms_Driven",
                y="Selling_Price",
                color="Fuel_Type",
                size="Present_Price",
                hover_data=["Car_Name", "Year"],
                title="Mileage (Kms Driven) vs Resale Price",
                labels={"Selling_Price": "Resale Price (₹ Lakhs)", "Kms_Driven": "Odometer (km)"}
            )
            fig_scatter.update_layout(height=360)
            st.plotly_chart(fig_scatter, use_container_width=True)

        with c_an4:
            # Feature Importance
            feat_imp = get_feature_importances()
            if feat_imp:
                df_fi = pd.DataFrame(feat_imp).sort_values("importance", ascending=True)
                # Clean up feature names for display
                df_fi["clean_feature"] = df_fi["feature"].str.replace("_", " ").str.title()
                fig_fi = px.bar(
                    df_fi,
                    x="importance",
                    y="clean_feature",
                    orientation="h",
                    title="Top Random Forest Model Price Drivers",
                    labels={"importance": "Importance Weight", "clean_feature": "Feature"},
                    color="importance",
                    color_continuous_scale="Blues"
                )
                fig_fi.update_layout(height=360, coloraxis_showscale=False)
                st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("Dataset not loaded.")


# =========================================================
# TAB 3: VARIANT COMPARISON
# =========================================================
with tab_compare:
    st.subheader("⚖️ Side-by-Side Car Resale Valuation")
    st.caption("Compare two different configurations (e.g. Manual vs Automatic, Diesel vs Petrol, or different years)")

    col_car_a, col_car_b = st.columns(2, gap="large")

    with col_car_a:
        st.markdown("### 🚘 Vehicle A")
        car_a_name = st.selectbox("Car Model A", all_cars, index=all_cars.index("swift") if "swift" in all_cars else 0, key="ca_name")
        car_a_year = st.slider("Year A", 2005, 2026, 2016, key="ca_yr")
        car_a_price = st.number_input("Present Price A (₹ L)", 0.1, 100.0, 7.0, step=0.5, key="ca_pr")
        car_a_kms = st.number_input("Kms Driven A", 0, 300000, 35000, step=5000, key="ca_km")
        car_a_fuel = st.selectbox("Fuel A", ["Diesel", "Petrol", "CNG"], index=0, key="ca_fl")
        car_a_trans = st.selectbox("Transmission A", ["Manual", "Automatic"], index=0, key="ca_tr")
        car_a_seller = st.selectbox("Seller A", ["Dealer", "Individual"], index=0, key="ca_sl")
        car_a_owner = st.selectbox("Owner A", [0, 1, 3], index=0, key="ca_ow")

        payload_a = {
            "Car_Name": car_a_name, "Year": car_a_year, "Present_Price": car_a_price,
            "Kms_Driven": car_a_kms, "Fuel_Type": car_a_fuel, "Seller_Type": car_a_seller,
            "Transmission": car_a_trans, "Owner": car_a_owner
        }
        val_a, meta_a = predict_price(payload_a)

    with col_car_b:
        st.markdown("### 🚙 Vehicle B")
        car_b_name = st.selectbox("Car Model B", all_cars, index=all_cars.index("swift") if "swift" in all_cars else 0, key="cb_name")
        car_b_year = st.slider("Year B", 2005, 2026, 2014, key="cb_yr")
        car_b_price = st.number_input("Present Price B (₹ L)", 0.1, 100.0, 7.0, step=0.5, key="cb_pr")
        car_b_kms = st.number_input("Kms Driven B", 0, 300000, 65000, step=5000, key="cb_km")
        car_b_fuel = st.selectbox("Fuel B", ["Petrol", "Diesel", "CNG"], index=0, key="cb_fl")
        car_b_trans = st.selectbox("Transmission B", ["Manual", "Automatic"], index=0, key="cb_tr")
        car_b_seller = st.selectbox("Seller B", ["Dealer", "Individual"], index=0, key="cb_sl")
        car_b_owner = st.selectbox("Owner B", [0, 1, 3], index=0, key="cb_ow")

        payload_b = {
            "Car_Name": car_b_name, "Year": car_b_year, "Present_Price": car_b_price,
            "Kms_Driven": car_b_kms, "Fuel_Type": car_b_fuel, "Seller_Type": car_b_seller,
            "Transmission": car_b_trans, "Owner": car_b_owner
        }
        val_b, meta_b = predict_price(payload_b)

    st.markdown("---")
    st.subheader("Comparison Result")

    diff = val_a - val_b
    col_res_a, col_res_b, col_diff = st.columns(3)
    with col_res_a:
        st.metric(f"Vehicle A ({car_a_name.title()})", f"₹ {val_a:.2f} Lakhs", f"₹ {val_a*100000:,.0f}")
    with col_res_b:
        st.metric(f"Vehicle B ({car_b_name.title()})", f"₹ {val_b:.2f} Lakhs", f"₹ {val_b*100000:,.0f}")
    with col_diff:
        prefix = "+" if diff >= 0 else ""
        st.metric("Price Delta (A vs B)", f"{prefix}₹ {diff:.2f} Lakhs", delta=f"{prefix}₹ {diff*100000:,.0f}")


# =========================================================
# TAB 4: EMI CALCULATOR
# =========================================================
with tab_emi:
    st.subheader("💳 Used Car Loan & Financing Calculator")

    c_emi1, c_emi2 = st.columns([1, 1.2], gap="large")

    with c_emi1:
        loan_car_price = st.number_input(
            "Vehicle Valuation (₹ Lakhs)",
            min_value=0.5,
            max_value=100.0,
            value=float(predicted_val if predicted_val else 5.0),
            step=0.25
        )
        total_loan_amount_inr = loan_car_price * 100000

        down_payment_pct = st.slider("Down Payment (%)", 10, 50, 20, step=5)
        down_payment_inr = total_loan_amount_inr * (down_payment_pct / 100.0)
        principal = total_loan_amount_inr - down_payment_inr

        interest_rate = st.slider("Annual Interest Rate (%)", 7.0, 18.0, 10.5, step=0.25)
        tenure_years = st.slider("Loan Tenure (Years)", 1, 7, 3, step=1)
        months = tenure_years * 12

        # Monthly EMI Calculation
        r = (interest_rate / 12) / 100
        if r > 0 and months > 0:
            emi = (principal * r * ((1 + r) ** months)) / (((1 + r) ** months) - 1)
        else:
            emi = principal / months

        total_payment = emi * months
        total_interest = total_payment - principal

    with c_emi2:
        st.markdown(f"""
        <div class="valuation-box" style="margin-top: 15px;">
            <span style="font-size: 0.85rem; font-weight: 700; color: #64748b; text-transform: uppercase;">
                Monthly Installment (EMI)
            </span>
            <div style="font-size: 2.8rem; font-weight: 800; color: #16a34a; margin: 4px 0;">
                ₹ {round(emi):,.0f} <span style="font-size: 1rem; color: #64748b;">/ month</span>
            </div>
            <div style="font-size: 0.95rem; color: #64748b;">
                For {tenure_years} Years ({months} months) at {interest_rate}% p.a.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("####")
        e_m1, e_m2, e_m3 = st.columns(3)
        with e_m1:
            st.metric("Down Payment", f"₹ {round(down_payment_inr):,.0f}")
        with e_m2:
            st.metric("Principal Loan", f"₹ {round(principal):,.0f}")
        with e_m3:
            st.metric("Total Interest", f"₹ {round(total_interest):,.0f}")

        # Pie chart
        df_loan_breakdown = pd.DataFrame({
            "Component": ["Principal Loan", "Total Interest"],
            "Amount": [round(principal), round(total_interest)]
        })
        fig_loan = px.pie(
            df_loan_breakdown,
            names="Component",
            values="Amount",
            hole=0.45,
            color_discrete_sequence=["#2563eb", "#f59e0b"]
        )
        fig_loan.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_loan, use_container_width=True)


# =========================================================
# TAB 5: DATASET EXPLORER
# =========================================================
with tab_data:
    st.subheader("📋 Cardekho Training Dataset")
    df_explorer = get_dataset()

    if not df_explorer.empty:
        f1, f2, f3 = st.columns(3)
        with f1:
            fuel_filter = st.multiselect("Filter by Fuel Type", options=df_explorer["Fuel_Type"].unique(), default=df_explorer["Fuel_Type"].unique())
        with f2:
            trans_filter = st.multiselect("Filter by Transmission", options=df_explorer["Transmission"].unique(), default=df_explorer["Transmission"].unique())
        with f3:
            search_query = st.text_input("Search Model Name", "")

        df_filtered = df_explorer[
            df_explorer["Fuel_Type"].isin(fuel_filter) &
            df_explorer["Transmission"].isin(trans_filter)
        ]
        if search_query:
            df_filtered = df_filtered[df_filtered["Car_Name"].str.lower().str.contains(search_query.lower())]

        st.caption(f"Showing **{len(df_filtered)}** of **{len(df_explorer)}** records")
        st.dataframe(df_filtered, use_container_width=True, height=400)

        # Download CSV
        csv_bytes = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Data (CSV) 📥",
            data=csv_bytes,
            file_name="cardekho_filtered.csv",
            mime="text/csv"
        )
    else:
        st.warning("Dataset not found.")


# =========================================================
# TAB 6: DEVELOPER & API
# =========================================================
with tab_api:
    st.subheader("⚡ FastAPI Developer Health & Tester")

    d1, d2 = st.columns([1, 1], gap="large")

    with d1:
        st.markdown("#### API Health & Connection")
        st.write(f"- **Configured URL**: `{api_endpoint}`")
        st.write(f"- **Server Live**: {'✅ Online' if is_api_online else '❌ Offline'}")
        if health_data:
            st.json(health_data)

        st.markdown("#### Swagger & Interactive Docs")
        st.markdown(f"- [Open Swagger UI Docs]({api_endpoint.rstrip('/')}/docs)")
        st.markdown(f"- [Open ReDoc Interface]({api_endpoint.rstrip('/')}/redoc)")
        st.markdown(f"- [Open Web UI Dashboard]({api_endpoint.rstrip('/')}/)")

    with d2:
        st.markdown("#### Live Request Payload")
        st.json(payload)
        st.markdown("#### Latest Prediction Response")
        st.json(meta_res if meta_res else {"status": "Pending execution"})
