import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Agriculture Price Prediction",
    page_icon="🌾",
    layout="wide"
)

# ---------------- LOAD FILES ----------------
@st.cache_resource
def load_assets():
    price_model = joblib.load("price_model.pkl")
    encoders = joblib.load("encoders.pkl")
    df = joblib.load("df.pkl")
    return price_model, encoders, df

try:
    price_model, encoders, df = load_assets()
except Exception as e:
    st.error(f"❌ Files load nahi ho paaye: {e}")
    st.stop()

# ---------------- DATA PREPARATION FOR DYNAMIC FILTERING ----------------
# Agar df mein integers saved hain, toh unhe strings/classes mein map kar lete hain
df_raw = df.copy()

# Encoders se Original String names extract karna for filtering
state_classes = list(encoders["STATE"].classes_)
district_classes = list(encoders["District Name"].classes_)
market_classes = list(encoders["Market Name"].classes_)
commodity_classes = list(encoders["Commodity"].classes_)

# ---------------- HELPER FUNCTIONS ----------------
def get_season_details(month_num):
    seasons = {
        "Winter (सर्दी / Rabi Harvesting)": [12, 1, 2],
        "Summer / Spring (गर्मी / Zaid Season)": [3, 4, 5],
        "Monsoon (बरसात / Rainy Season)": [6, 7, 8],
        "Autumn / Post-Monsoon (शरद ऋतु / Peak Mandi Demand)": [9, 10, 11]
    }
    
    months_map = {
        1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
        7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
    }
    
    month_name = months_map.get(month_num, "Unknown")
    season_name = "Unknown Season"
    
    for season, months in seasons.items():
        if month_num in months:
            season_name = season
            break
            
    return month_name, season_name

def get_best_selling_time(commodity_name, state_name):
    try:
        comm_encoded = encoders["Commodity"].transform([commodity_name])[0]
        state_encoded = encoders["STATE"].transform([state_name])[0]

        filtered_df = df[
            ((df["Commodity"] == comm_encoded) | (df["Commodity"] == commodity_name)) &
            ((df["STATE"] == state_encoded) | (df["STATE"] == state_name))
        ]

        if filtered_df.empty:
            return None, None, "No Data Available for this State & Commodity combination."

        monthly_avg = (
            filtered_df.groupby("Month")["Modal_Price"]
            .mean()
            .reset_index()
        )

        best_row = monthly_avg.loc[monthly_avg["Modal_Price"].idxmax()]
        best_month_num = int(best_row["Month"])
        avg_price = best_row["Modal_Price"]

        month_name, season_name = get_season_details(best_month_num)
        return month_name, season_name, avg_price

    except Exception as e:
        return None, None, f"Error calculating best time: {e}"

# ---------------- TITLE ----------------
st.title("🌾 Agriculture Price & Seasonality Intelligence")
st.success("Model & Dynamic Filters Loaded Successfully ✅")

# =====================================================
# PRICE PREDICTION (WITH DYNAMIC FILTERING & WARNING)
# =====================================================
st.header("💰 Price Prediction")

col1, col2 = st.columns(2)

with col1:
    # 1. State Selectbox
    state = st.selectbox("State", state_classes, key="p_state")

    # 2. Dynamic District Filtering based on State
    state_enc_val = encoders["STATE"].transform([state])[0]
    
    # Check if df has encoded or string values
    filtered_dist_df = df[(df["STATE"] == state) | (df["STATE"] == state_enc_val)]
    
    if not filtered_dist_df.empty:
        # Extract unique district codes/names
        available_dist_codes = filtered_dist_df["District Name"].unique()
        
        # Convert codes back to string names if encoded
        if isinstance(available_dist_codes[0], (int, np.integer)):
            available_districts = sorted(encoders["District Name"].inverse_transform(available_dist_codes))
        else:
            available_districts = sorted(list(available_dist_codes))
    else:
        available_districts = district_classes

    district = st.selectbox("District", available_districts, key="p_dist")

    # 3. Dynamic Market Filtering based on Selected District
    dist_enc_val = encoders["District Name"].transform([district])[0]
    filtered_mkt_df = filtered_dist_df[(filtered_dist_df["District Name"] == district) | (filtered_dist_df["District Name"] == dist_enc_val)]
    
    if not filtered_mkt_df.empty:
        available_mkt_codes = filtered_mkt_df["Market Name"].unique()
        if isinstance(available_mkt_codes[0], (int, np.integer)):
            available_markets = sorted(encoders["Market Name"].inverse_transform(available_mkt_codes))
        else:
            available_markets = sorted(list(available_mkt_codes))
    else:
        available_markets = market_classes

    market = st.selectbox("Market", available_markets, key="p_mkt")

with col2:
    commodity = st.selectbox("Commodity", commodity_classes, key="p_comm")
    variety = st.selectbox("Variety", encoders["Variety"].classes_, key="p_var")
    month = st.slider("Month", 1, 12, 6, key="p_month")

# Validation check before prediction
is_valid_combo = not filtered_dist_df.empty and not filtered_mkt_df.empty

if not is_valid_combo:
    st.warning(f"⚠️ **Warning:** '{district}' (Market: '{market}') aapke dataset mein '{state}' state ke under nahi mil raha hai.")

if st.button("Predict Price"):
    if not is_valid_combo:
        st.error("❌ Invalid combination! Kripya sahi State, District aur Market chunein.")
    else:
        try:
            s_enc = encoders["STATE"].transform([state])[0]
            d_enc = encoders["District Name"].transform([district])[0]
            m_enc = encoders["Market Name"].transform([market])[0]
            c_enc = encoders["Commodity"].transform([commodity])[0]
            v_enc = encoders["Variety"].transform([variety])[0]

            input_data = [[s_enc, d_enc, m_enc, c_enc, v_enc, month]]

            raw_pred = price_model.predict(input_data)[0]
            prediction = np.expm1(raw_pred) if raw_pred < 15 else raw_pred

            st.success(f"💰 **Predicted Modal Price:** ₹ {prediction:,.2f} / Quintal")
        
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# =====================================================
# BEST TIME TO SELL (SEASON WISE)
# =====================================================
st.markdown("---")
st.header("📅 Best Season & Month To Sell")

col3, col4 = st.columns(2)

with col3:
    state_sell = st.selectbox("Select State", state_classes, key="state_sell")

with col4:
    commodity_sell = st.selectbox("Select Commodity", commodity_classes, key="commodity_sell")

if st.button("Find Best Selling Season"):
    month_name, season_name, avg_price_or_err = get_best_selling_time(commodity_sell, state_sell)

    if month_name is None:
        st.warning(f"⚠️ {avg_price_or_err}")
    else:
        st.markdown("### 📊 Recommended Selling Window")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(label="🌱 Best Season", value=season_name.split(' (')[0])
        with c2:
            st.metric(label="🗓️ Peak Month", value=month_name)
        with c3:
            st.metric(label="💵 Historical Avg Peak Rate", value=f"₹ {avg_price_or_err:,.2f}")

        st.info(f"💡 **Strategy:** For **{commodity_sell}** in **{state_sell}**, selling during **{season_name}** (specifically **{month_name}**) historically fetches the highest profit margin.")