# AI-Based Smart AQI Prediction System

This project predicts Air Quality Index (AQI) using Machine Learning techniques and provides an interactive Streamlit dashboard for visualization.

## Features

- AQI prediction using Random Forest Regression
- AQI category classification (Good, Moderate, Poor, etc.)
- Feature importance visualization
- Pollution input comparison graphs
- AQI distribution analysis
- Correlation heatmap visualization
- Interactive Streamlit dashboard

## Dataset

Dataset used: city_day.csv  
Contains pollutant concentration values such as:

PM2.5  
PM10  
NO  
NO2  
NOx  
NH3  
CO  
SO2  
O3  

## Tech Stack

Python  
Pandas  
NumPy  
Scikit-learn  
Matplotlib  
Seaborn  
Streamlit  

## How to Run

Step 1:

python train_model.py

Step 2:

streamlit run app.py

## Model Performance

Algorithm Used: Random Forest Regression  
Evaluation Metric: Mean Absolute Error (MAE) ≈ 20 AQI units

## Project Outcome

Built a machine learning pipeline that predicts AQI levels and visualizes pollution trends using an interactive dashboard.
