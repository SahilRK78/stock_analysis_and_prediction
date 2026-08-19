
import json
from datetime import date, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from joblib import load
from prophet import Prophet
from tensorflow.keras.models import load_model
 
st.set_page_config(page_title="Stock Price Forecasting", layout="wide")
 

@st.cache_resource
def load_artifacts():
    lstm_model = load_model("lstm_model.keras")
    scaler = load("scaler.joblib")
    with open("model_metadata.json") as f:
        metadata = json.load(f)
    return lstm_model, scaler, metadata
 
 
try:
    lstm_model, scaler, metadata = load_artifacts()
    MODEL_TICKER = metadata["ticker"]
    N_STEPS = metadata["n_steps"]
    artifacts_ok = True
except Exception as e:
    artifacts_ok = False
    st.error(
        "Could not load the LSTM model, scaler, or metadata. "
        f"Details: {e}"
    )
    st.stop()
 
 
# Data loading

@st.cache_data(ttl=3600)
def load_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        # Retry once 
        import time
        time.sleep(1)
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    return df
 
 
def plot_candlestick(df, ticker):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"]
    )])
    fig.update_layout(title=f"{ticker} Candlestick Chart", xaxis_title="Date", yaxis_title="Price",
                       xaxis_rangeslider_visible=False)
    return fig
 
 
def plot_moving_average(df, ma_days):
    df = df.copy()
    df[f"MA_{ma_days}"] = df["Close"].rolling(window=ma_days).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close Price"))
    fig.add_trace(go.Scatter(x=df.index, y=df[f"MA_{ma_days}"], mode="lines", name=f"{ma_days}-Day MA"))
    fig.update_layout(title=f"{ma_days}-Day Moving Average", xaxis_title="Date", yaxis_title="Price",
                       xaxis_rangeslider_visible=False)
    return fig
 
 
# LSTM forecast

def forecast_lstm(df, days, n_steps=N_STEPS):
    close_prices = df[["Close"]].values
 
    if len(close_prices) < n_steps:
        raise ValueError(
            f"Need at least {n_steps} days of price history for the LSTM model, "
            f"but only {len(close_prices)} are available. Widen the date range."
        )
 
    scaled_data = scaler.transform(close_prices)
    last_sequence = scaled_data[-n_steps:].copy()
    future_predictions = []
 
    for _ in range(days):
        input_seq = last_sequence.reshape(1, n_steps, 1)
        next_pred = lstm_model.predict(input_seq, verbose=0)
        future_predictions.append(next_pred[0, 0])
        last_sequence = np.append(last_sequence[1:], next_pred, axis=0)
 
    future_predictions = scaler.inverse_transform(
        np.array(future_predictions).reshape(-1, 1)
    ).flatten()
 
    future_dates = pd.date_range(df.index[-1], periods=days + 1, freq="B")[1:]
    return pd.DataFrame({"Date": future_dates, "Prediction": future_predictions})
 
 

def forecast_prophet(df, days):
    df_prophet = df.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(df_prophet)
 
    future = model.make_future_dataframe(periods=days, freq="D", include_history=False)
    forecast = model.predict(future)
    forecast = forecast.rename(columns={
        "ds": "Date", "yhat": "Predicted Price",
        "yhat_lower": "Lower Bound", "yhat_upper": "Upper Bound"
    })
    return forecast[["Date", "Predicted Price", "Lower Bound", "Upper Bound"]]
 
 
def naive_baseline(df, days):
    """Persistence baseline shown alongside model forecasts for honest comparison."""
    last_price = df["Close"].iloc[-1]
    future_dates = pd.date_range(df.index[-1], periods=days + 1, freq="B")[1:]
    return pd.DataFrame({"Date": future_dates, "Prediction": [last_price] * days})
 
 
# App layout

st.title("📈 Stock Price Forecasting — LSTM & Prophet")
 
max_end_date = date.today() - timedelta(days=1)
 
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Stock Ticker", value=MODEL_TICKER).upper().strip()
    start_date = st.date_input("Start Date", value=pd.to_datetime("2016-08-02"))
    end_date = st.date_input(
        "End Date",
        value=min(pd.to_datetime("2026-08-02").date(), max_end_date),
        max_value=max_end_date
    )
    model_choice = st.radio("Prediction Model", ["LSTM", "Facebook Prophet"])
    days = st.slider("Days to Forecast", 1, 30, 7)
 
if ticker != MODEL_TICKER:
    st.warning(
        f"⚠️ The LSTM model was trained only on **{MODEL_TICKER}** data. "
        f"It will still run for **{ticker}**, but its predictions will not be meaningful for a "
        f"different stock — the price scale and patterns it learned are specific to {MODEL_TICKER}. "
        f"Facebook Prophet is refit live and works correctly for any ticker."
    )
 
df = load_stock_data(ticker, start_date, end_date)

if df.empty:
    load_stock_data.clear()   
    st.error(
        "No data returned. This can happen if the End Date is today or a non-trading day "
        "(weekend/holiday) with no data synced yet, or if the ticker symbol is invalid. "
        "Try pulling the End Date back by a day or two, or widen the date range."
    )
    st.stop()
 
st.subheader(f"{ticker} Historical Data")
col1, col2 = st.columns([2, 1])
with col1:
    st.plotly_chart(plot_candlestick(df, ticker), use_container_width=True)
with col2:
    st.dataframe(df.tail(10), use_container_width=True)
 
ma_days = st.slider("Moving Average Window (days)", 5, 200, 50)
st.plotly_chart(plot_moving_average(df, ma_days), use_container_width=True)
 
st.subheader(f"{days}-Day Forecast — {model_choice}")
 
try:
    baseline_df = naive_baseline(df, days)
 
    if model_choice == "LSTM":
        if len(df) < N_STEPS:
            st.error(f"Need at least {N_STEPS} days of data in the selected range for the LSTM model.")
        else:
            st.caption(
                f"ℹ️ The LSTM model's weights were trained on data through "
                f"**{metadata.get('trained_through', 'the training cutoff')}**. It uses the most "
                f"recent {N_STEPS} days from your selected range as input, so the forecast starts "
                f"from current prices — but the model's learned patterns don't update automatically "
                f"past that training date. For a forecast that's always fully up to date, use "
                f"Facebook Prophet instead, which refits live on every request."
            )
 
            forecast_df = forecast_lstm(df, days)
 
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Prediction"],
                                      mode="lines+markers", name="LSTM Forecast", line=dict(color="darkorange")))
            fig.add_trace(go.Scatter(x=baseline_df["Date"], y=baseline_df["Prediction"],
                                      mode="lines", name="Naive Baseline (today's price)",
                                      line=dict(color="grey", dash="dot")))
            fig.update_layout(title=f"{ticker} — LSTM {days}-Day Forecast", xaxis_title="Date",
                               yaxis_title="Closing Price ($)")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(forecast_df, use_container_width=True)
            st.caption(
                "This is a recursive multi-day forecast: each predicted day is fed back in to "
                "predict the next, so uncertainty compounds the further out you look. Treat later "
                "days as far less reliable than the first day or two."
            )
            st.download_button("Download forecast as CSV", forecast_df.to_csv(index=False),
                                file_name=f"{ticker}_lstm_forecast.csv")
 
    elif model_choice == "Facebook Prophet":
        with st.spinner("Fitting Prophet on the selected data..."):
            forecast_df = forecast_prophet(df, days)
 
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Predicted Price"],
                                  mode="lines", name="Prophet Forecast", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Upper Bound"],
                                  mode="lines", name="Upper Bound", line=dict(dash="dot", color="green")))
        fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Lower Bound"],
                                  mode="lines", name="Lower Bound", line=dict(dash="dot", color="red")))
        fig.add_trace(go.Scatter(x=baseline_df["Date"], y=baseline_df["Prediction"],
                                  mode="lines", name="Naive Baseline (today's price)",
                                  line=dict(color="grey", dash="dot")))
        fig.update_layout(title=f"{ticker} — Prophet {days}-Day Forecast", xaxis_title="Date",
                           yaxis_title="Closing Price ($)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(forecast_df, use_container_width=True)
        st.download_button("Download forecast as CSV", forecast_df.to_csv(index=False),
                            file_name=f"{ticker}_prophet_forecast.csv")
 
except ValueError as ve:
    st.error(str(ve))
except Exception as e:
    st.error(f"Something went wrong while generating the forecast: {e}")
 
st.markdown("---")
st.caption(
    "⚠️ Educational project, not financial advice. Stock prices are close to a random walk — "
    "no model here reliably predicts exact future prices. Compare each forecast against the grey "
    "naive-baseline line: if a model isn't beating it, it isn't adding real predictive value."
)
