import streamlit as st
import numpy as np
import pickle
import plotly.express as px
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium
model = pickle.load(open("model.pkl", "rb"))
st.set_page_config(page_title="🏡 Smart Predictor", layout="wide")
theme = st.sidebar.radio("🌙 Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("""
    <style>
    body {background-color: #0e1117; color: white;}
    </style>
    """, unsafe_allow_html=True)
st.title("🏡 AI House Price Predictor")
st.write("Next-level real estate intelligence 💡")
location_prices = {
    "Mumbai": 8000,
    "Delhi": 6000,
    "Bangalore": 7000,
    "Nagpur": 4000,
    "Pune": 6500
}
col1, col2 = st.columns(2)
with col1:
    area = st.slider("Area", 500, 5000, 1000)
    bedrooms = st.selectbox("Bedrooms", [1,2,3,4,5])
    bathrooms = st.selectbox("Bathrooms", [1,2,3,4])
    location = st.selectbox("Location", list(location_prices.keys()))
    if st.button("Predict"):
        base = model.predict(np.array([[area, bedrooms, bathrooms]]))[0]
        final = base + (area * location_prices[location])
        st.success(f"💰 Price: ₹ {int(final):,}")
with col2:
    st.subheader("📍 Location Map")
    geolocator = Nominatim(user_agent="app")
    loc = geolocator.geocode(location)
    if loc:
        m = folium.Map(location=[loc.latitude, loc.longitude], zoom_start=10)
        folium.Marker([loc.latitude, loc.longitude], tooltip=location).add_to(m)
        st_folium(m, width=400, height=300)
st.subheader("📊 Interactive Price Chart")
areas = list(range(500, 5000, 100))
prices = []
for a in areas:
    p = model.predict([[a, 3, 2]])[0]
    p += a * location_prices["Nagpur"]
    prices.append(p)
fig = px.line(x=areas, y=prices, labels={'x':'Area', 'y':'Price'}, title="Price Trend")
st.plotly_chart(fig)
st.subheader("🤖 AI Assistant")
user_q = st.text_input("Ask something about house prices:")
if user_q:
    if "cheap" in user_q.lower():
        st.write("👉 Try smaller area or choose Nagpur for lower prices.")
    elif "expensive" in user_q.lower():
        st.write("👉 Mumbai and Bangalore have higher property rates.")
    else:
        st.write("👉 Prices depend on area, location, and features.")
st.write("---")
st.write("🚀 Advanced App | Built with Python & AI")