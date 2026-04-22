import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load model
model = pickle.load(open("aqi_model.pkl", "rb"))

# load dataset
df = pd.read_csv("data/city_day.csv")

st.title("AI-Based Smart AQI Prediction Dashboard")

st.subheader("Enter Pollution Values")

PM25 = st.number_input("PM2.5", 0.0)
PM10 = st.number_input("PM10", 0.0)
NO = st.number_input("NO", 0.0)
NO2 = st.number_input("NO2", 0.0)
NOx = st.number_input("NOx", 0.0)
NH3 = st.number_input("NH3", 0.0)
CO = st.number_input("CO", 0.0)
SO2 = st.number_input("SO2", 0.0)
O3 = st.number_input("O3", 0.0)


def aqi_category(aqi):

    if aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Satisfactory"

    elif aqi <= 200:
        return "Moderate"

    elif aqi <= 300:
        return "Poor"

    elif aqi <= 400:
        return "Very Poor"

    else:
        return "Severe"


if st.button("Predict AQI"):

    features = np.array([[PM25, PM10, NO, NO2,
                          NOx, NH3, CO, SO2, O3]])

    prediction = model.predict(features)[0]

    category = aqi_category(prediction)

    st.success(f"Predicted AQI: {prediction:.2f}")

    if category == "Good":
        st.success(category)

    elif category == "Moderate":
        st.warning(category)

    else:
        st.error(category)

    st.subheader("Feature Importance")

    importance = model.feature_importances_

    feature_names = [
        "PM2.5", "PM10", "NO", "NO2",
        "NOx", "NH3", "CO", "SO2", "O3"
    ]

    fig, ax = plt.subplots()

    ax.barh(feature_names, importance)

    st.pyplot(fig)

    st.subheader("Input Pollution Levels")

    fig2, ax2 = plt.subplots()

    ax2.bar(feature_names, features[0])

    plt.xticks(rotation=45)

    st.pyplot(fig2)


# dataset visualization section

st.subheader("AQI Distribution")

fig3, ax3 = plt.subplots()

df["AQI"].hist(bins=50)

st.pyplot(fig3)


st.subheader("Correlation Heatmap")

fig4, ax4 = plt.subplots(figsize=(10,6))

sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")

st.pyplot(fig4)