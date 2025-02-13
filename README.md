****Stock Price Prediction App****

This repository contains a Streamlit web application for stock price prediction using historical data. The project integrates multiple time series forecasting models, including ARIMA, SARIMAX, LSTM, and Facebook Prophet, to analyze stock trends and predict future prices.


**Project Overview**

**1. Data Collection**
Historical stock price data is collected using the yfinance library.

Users can select a stock symbol and retrieve data for analysis and forecasting.

**2. Exploratory Data Analysis (EDA)**

Conducted detailed EDA on stock price data, analyzing trends and seasonality.

Visualizations include:
Close Price vs Year
Candlestick Chart (Plotly)
Moving Averages (Plotly)

**3. Time Series Forecasting Models**

Baseline Models (Not Used in Deployment)

**ARIMA & SARIMAX:** Traditional time series models used to analyze historical patterns and establish benchmark predictions.

**Deep Learning & Advanced Forecasting Models** (Used in Deployment)

**LSTM Model:** A Long Short-Term Memory (LSTM) neural network trained on a 10-year dataset to capture complex patterns in stock prices.
**Facebook Prophet:** A robust time series forecasting model that accounts for trends and seasonality.

**4. Deployment**
The application is deployed using Streamlit Cloud, allowing users to:
Select a stock and define forecast duration (1 to 30 days)
Choose a prediction model (LSTM or Facebook Prophet)
Visualize forecasts with interactive Plotly graphs

