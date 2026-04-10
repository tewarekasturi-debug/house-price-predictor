import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import plotly.express as px
st.title("🏡 House Price Predictor (Pro App)")
st.subheader("📁 Upload Your Dataset")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
else:
    data = pd.read_csv("data.csv")
st.write("Dataset Preview 👇")
st.dataframe(data)
fig = px.scatter(data, x="area", y="price", color="bedrooms",
                 title="House Price Visualization")
st.plotly_chart(fig)
st.subheader("🧠 Train Model")
if st.button("Train Model"):
    X = data[['area', 'bedrooms', 'bathrooms']]
    y = data['price']
    lr = LinearRegression()
    lr.fit(X, y)
    lr_pred = lr.predict(X)
    lr_score = r2_score(y, lr_pred)
    xgb = XGBRegressor()
    xgb.fit(X, y)
    xgb_pred = xgb.predict(X)
    xgb_score = r2_score(y, xgb_pred)
    if xgb_score > lr_score:
        best_model = xgb
        model_name = "XGBoost"
    else:
        best_model = lr
        model_name = "Linear Regression"
    pickle.dump(best_model, open("model.pkl", "wb"))
    st.success(f"Model trained using {model_name} 🎉")
    st.write("### 📊 Accuracy")
    st.write(f"Linear Regression: {lr_score:.2f}")
    st.write(f"XGBoost: {xgb_score:.2f}")
st.subheader("🏡 Predict Price")
area = st.number_input("Area")
bedrooms = st.number_input("Bedrooms")
bathrooms = st.number_input("Bathrooms")
if st.button("Predict"):
    model = pickle.load(open("model.pkl", "rb"))
    result = model.predict(np.array([[area, bedrooms, bathrooms]]))
    st.success(f"Estimated Price: ₹ {result[0]:,.2f}")