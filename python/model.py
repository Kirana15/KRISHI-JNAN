# app.py
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import joblib
from datetime import datetime
from dateutil.relativedelta import relativedelta
from keras.models import load_model
from crop_recommendation import get_soil_types_for_district, recommend_crops

# ===== Streamlit Config & Theming =====
st.set_page_config(page_title="Krishi-Jñan : Crop & Price Advisor", page_icon="🌾", layout="centered")

theme = st.sidebar.radio("🌓 Theme", ["Light", "Dark"])
if theme == "Dark":
    bg_color = "#1e1e1e"
    font_color = "#fafafa"
else:
    bg_color = "#f5f5f5"
    font_color = "#1e1e1e"

st.markdown(f"""
    <style>
        html, body, [class*="css"] {{
            background-color: {bg_color};
            color: {font_color};
            font-family: 'Segoe UI', sans-serif;
        }}
        .stButton>button {{
            background-color: #2ecc71;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 0.5em 1em;
            border: none;
        }}
        .stButton>button:hover {{
            background-color: #27ae60;
            color: white;
        }}
        h1, h2, h3, h4 {{
            color: {font_color};
        }}
        footer {{ visibility: hidden; }}
    </style>
""", unsafe_allow_html=True)

# ===== Load Model & Assets =====
@st.cache_resource
def load_assets():
    model = load_model("models/unified_lstm_model.h5")
    scaler = joblib.load("models/price_scaler.pkl")
    crop_encoder = joblib.load("models/crop_encoder.pkl")
    district_encoder = joblib.load("models/district_encoder.pkl")
    harvest_month_encoder = joblib.load("models/harvest_month_encoder.pkl")
    mapping_df = pd.read_excel(r"D:\projects\crop_price_pridicton\District_to_Base_District_Mapping.xlsx")
    price_df = pd.read_excel(r"D:\projects\crop_price_pridicton\price_prediction_avg.xlsx")
    price_df['Price Date'] = pd.to_datetime(price_df['Price Date'])
    return model, scaler, crop_encoder, district_encoder, harvest_month_encoder, mapping_df, price_df

model, scaler, crop_encoder, district_encoder, harvest_month_encoder, mapping_df, price_df = load_assets()

# ===== Header =====
with st.container():
    col1, col2 = st.columns([1, 8])
    with col1:
        logo = Image.open("assets/logo.jpg")
        st.image(logo, width=70)
    with col2:
        st.markdown("<h1 style='font-size:40px; margin-bottom:0;'>Krishi-Jñan</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:gray;'>Empowering Farmers with Crop and Price Intelligence</p>", unsafe_allow_html=True)

st.image("assets/banner.jpg", use_container_width=True)

# ===== User Input Section =====
st.subheader("📍 Enter Farming Details")
district_input = st.text_input("Enter your district (e.g., Udupi)").strip().title()

if district_input:
    soil_types = get_soil_types_for_district(district_input)
    if soil_types:
        soil_type = st.selectbox("🌱 Select Soil Type", soil_types)
        sowing_month = st.selectbox("📅 Select Sowing Month",
                                    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    else:
        st.warning("❌ No soil type data found for this district.")
        st.stop()
else:
    st.stop()

# ===== Prediction Trigger =====
if st.button("🔍 Recommend Crops & Predict Prices"):
    recommendations = recommend_crops(district_input, soil_type, sowing_month)
    if recommendations.empty:
        st.error("❌ No crops found for the given combination.")
        st.stop()

    # Map to base district
    mapping_dict = dict(zip(mapping_df['District'].str.title(), mapping_df['Mapped District'].str.title()))
    base_district = mapping_dict.get(district_input, district_input)
    sowing_date = datetime.strptime(f"1 {sowing_month} {datetime.now().year}", "%d %b %Y")

    result_rows = []

    for _, row in recommendations.iterrows():
        crop = row['Crop'].strip().title()
        try:
            maturity = int(row['Maturity'])
        except:
            continue

        harvest_date = sowing_date + relativedelta(months=maturity)
        harvest_month = harvest_date.strftime("%b")

        # Filter past prices for the crop and district
        filtered = price_df[
            (price_df['Crop'].str.strip().str.title() == crop) &
            (price_df['District Name'].str.strip().str.title() == base_district) &
            (price_df['Price Date'] < harvest_date)
        ].sort_values('Price Date', ascending=False)

        if len(filtered) < 6:
            continue

        last_6 = filtered['Modal Price'].head(6).values[::-1]
        scaled_prices = scaler.transform(pd.DataFrame(last_6, columns=["Smooth Price"]))
        X_price = scaled_prices.reshape((1, 6, 1))

        try:
            X_crop = np.array([crop_encoder.transform([crop])])
            X_dist = np.array([district_encoder.transform([base_district])])
            X_month = np.array([harvest_month_encoder.transform([harvest_month])])
        except ValueError:
            continue

        pred_scaled = model.predict([X_price, X_crop, X_dist, X_month])[0][0]
        pred_price = scaler.inverse_transform([[pred_scaled]])[0][0]

        result_rows.append({
            "Crop": crop,
            "Harvest": harvest_date.strftime("%b %Y"),
            "Predicted Price (₹/quintal)": round(pred_price, 2)
        })

    # Display Results
    if result_rows:
        df_result = pd.DataFrame(result_rows).sort_values("Predicted Price (₹/quintal)", ascending=False)
        st.success(f"✅ Showing predicted prices for {len(df_result)} crops")
        st.dataframe(df_result.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
    else:
        st.warning("⚠ No crops had enough data for prediction.")

# ===== Footer =====
st.markdown("---")
st.markdown("<p style='text-align:center; color: gray;'>© 2025 Krishi-Jñan. Built with ❤ to empower Indian farmers.</p>", unsafe_allow_html=True)