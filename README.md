## Stock Price Forecasting App

A Streamlit app for stock price forecasting, built on 10 years of AAPL data. Five models — Naive Baseline, ARIMA, SARIMAX, LSTM, and Facebook Prophet were trained and compared on the same leakage-free train/test split; LSTM and Prophet are the two deployed in the app.

You can try out the Stock Price Prediction App [here](https://stock-price-analysis-forecasting-and-prediction.streamlit.app/).


## Features
Fetches historical stock market data using yfinance
Supports different stock tickers and customizable date ranges
Displays historical price trends using candlestick charts
Calculates and visualizes moving averages
Displays daily trading volume
Generates 1–30 day forecasts
Allows users to choose between LSTM and Facebook Prophet
Compares forecasts with a Naive Baseline
Provides forecast results as a downloadable CSV file


## Exploratory Data Analysis

The project includes exploratory analysis of:

Historical closing prices
Trading volume
Moving averages
Daily returns
Price trends
Stationarity
ACF and PACF
Seasonal decomposition

The complete methodology, EDA, model development, and evaluation are available in:

notebooks/stock_forecasting_project.ipynb


## Models Used
## 1. Naive Baseline

Uses the previous day's closing price as the prediction for the next day.

This provides a simple benchmark to determine whether the machine learning and statistical models provide meaningful improvement.

## 2. ARIMA

A classical time-series forecasting model used as a benchmark for univariate stock-price forecasting.

## 3. SARIMAX

An extension of ARIMA that can model seasonal patterns and incorporate additional variables when available.

## 4. LSTM

A Long Short-Term Memory neural network designed for sequential and time-series data.

The deployed LSTM model consists of:

lstm_model.keras
scaler.joblib

The model was trained using historical AAPL data.

## 5. Facebook Prophet

A time-series forecasting model developed by Facebook (Meta).

Unlike the saved LSTM model, Prophet refits on the data selected by the user each time a forecast is requested. Therefore, it can be used with different stock tickers and does not depend on a fixed historical price range.





