import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import requests
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import pandas_ta as ta

# Function to fetch cryptocurrency data
def fetch_crypto_data(symbol, interval):
    try:
        url = f"https://api.binance.us/api/v3/klines?symbol={symbol}USDT&interval={interval}&limit=100"
        response = requests.get(url)
        data = response.json()
        prices = [float(entry[4]) for entry in data]  # Closing prices
        timestamps = [int(entry[0]) for entry in data]
        return np.array(prices), timestamps
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None

# Load model and make predictions
def load_and_predict(model_filename, symbol, interval):
    model = tf.keras.models.load_model(model_filename)
    prices, timestamps = fetch_crypto_data(symbol, interval)
    if prices is not None:
        scaler = MinMaxScaler()
        scaled_prices = scaler.fit_transform(prices.reshape(-1, 1))
        scaled_prices = scaled_prices.reshape(1, scaled_prices.shape[0], 1)
        prediction = model.predict(scaled_prices)
        predicted_price = scaler.inverse_transform(prediction.reshape(-1, 1))[0][0]
        return float(prices[-1]), timestamps[-1], float(predicted_price), prices
    return None, None, None, None

# Function to calculate technical indicators using pandas_ta
def calculate_technical_indicators(prices):
    df = pd.DataFrame(prices, columns=["Close"])
    
    # Calculate RSI
    rsi = ta.rsi(df["Close"], length=14).iloc[-1]
    
    # Calculate MA 20
    ma_20 = ta.sma(df["Close"], length=20).iloc[-1]
    
    # Calculate MA 50
    ma_50 = ta.sma(df["Close"], length=50).iloc[-1]

    # Calculate MA 100
    ma_100 = ta.sma(df["Close"], length=100).iloc[-1]
    
    return rsi, ma_20, ma_50, ma_100

# Function to calculate Fibonacci retracement levels
def calculate_fibonacci_retracements(prices):
    high = max(prices)
    low = min(prices)
    diff = high - low
    levels = {
        "0.0%": high,
        "23.6%": high - 0.236 * diff,
        "38.2%": high - 0.382 * diff,
        "50.0%": high - 0.5 * diff,
        "61.8%": high - 0.618 * diff,
        "78.6%": high - 0.786 * diff,
        "100.0%": low
    }
    return levels

# Streamlit UI
st.title("Crypto Price Prediction using GRU RNN")

# Select cryptocurrency
crypto_options = ["BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "USUAL", "RENDER", "XLM", "STX", "TIA"]
symbol = st.selectbox("Select Cryptocurrency", crypto_options)

# Select timeframe
interval_options = {"5 Minutes": "5m", "15 Minutes": "15m", "30 Minutes": "30m", "Hourly": "1h", "Daily": "1d", "Weekly": "1w", "Monthly": "1M"}
timeframe = st.selectbox("Select Timeframe", list(interval_options.keys()))
interval = interval_options[timeframe]

# Predict button
if st.button("Predict Price"):
    st.write(f"Fetching and predicting {symbol} prices for {timeframe} interval...")
    
    model_filename = "crypto_model.h5"
    current_price, timestamp, predicted_price, prices = load_and_predict(model_filename, symbol, interval)
    
    if timestamp is not None:
        # Calculate technical indicators
        rsi, ma_20, ma_50, ma_100 = calculate_technical_indicators(prices)
        
        # Calculate Fibonacci retracement levels
        fib_levels = calculate_fibonacci_retracements(prices)
        
        readable_timestamp_utc = datetime.utcfromtimestamp(timestamp / 1000)

        # Adjust to UAE time (UTC +4)
        uae_time = readable_timestamp_utc + timedelta(hours=4)

        # Format the UAE time to the desired format
        readable_timestamp = uae_time.strftime('%d-%m-%Y %H:%M')

        # Display results in a table
        data = {
            "Indicator": ["Model Prediction", "RSI", "MA 20", "MA 50", "MA 100"] + list(fib_levels.keys()),
            "Prediction Price (USD)": [predicted_price, rsi, ma_20, ma_50, ma_100] + list(fib_levels.values()),
            "Timeframe": [timeframe] * (5 + len(fib_levels))
        }

        df = pd.DataFrame(data)

        st.write(f"Latest Actual Price Timestamp: {readable_timestamp}")
        st.write(f"Current Price: ${current_price:.4f} USD")
        st.write(f"Predicted Next Price: ${predicted_price:.4f} USD")
        st.write("Comparison of Price Predictions, Technical Indicators, and Fibonacci Levels:")
        st.dataframe(df, use_container_width=True)  # Display the table without scrolling
    else:
        st.error("Failed to fetch data or make predictions. Please try again later.")