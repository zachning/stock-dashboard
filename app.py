import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from streamlit_autorefresh import st_autorefresh
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# 页面配置
st.set_page_config(page_title="Stock Dashboard", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    "<h1 style='text-align: center; color: white;'>📈 Real-Time Stock Monitoring Dashboard (Pro Version)</h1>",
    unsafe_allow_html=True
)

# Sidebar 控件
with st.sidebar:
    st.header("⚙️ Controls")
    stock = st.text_input("Enter Stock/Crypto Ticker:", value="AAPL")
    period = st.selectbox(
        "Select Period:",
        ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
    )
    interval = st.selectbox(
        "Select Interval:",
        ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1d", "1wk", "1mo"]
    )
    refresh_time = st.slider("Refresh Interval (Seconds):", 10, 300, 60)
    threshold_mode = st.radio(
        "Select Signal Sensitivity:",
        ["High Volatility (0.5%)", "Low Volatility (0.15%)"]
    )
    threshold = 0.005 if threshold_mode.startswith("High") else 0.0015

st_autorefresh(interval=refresh_time * 1000, key="refresh")

@st.cache_data(ttl=refresh_time)
def load_data(ticker, period, interval):
    t = yf.Ticker(ticker)
    hist = t.history(period=period, interval=interval)
    info = t.info
    return hist, info

def validate_combo(period, interval):
    minute_intervals = {
        "1m": 7, "2m": 60, "5m": 60, "15m": 60,
        "30m": 60, "60m": 730, "90m": 60
    }
    if interval in minute_intervals:
        if period.endswith("y") or period in ["max", "ytd", "10y", "5y"]:
            return False
        if interval == "1m" and period not in ["1d", "5d", "7d"]:
            return False
    return True

if not validate_combo(period, interval):
    st.warning("⚠️ This period + interval combination is not supported by Yahoo Finance.")
    st.stop()

data, info = load_data(stock.upper(), period, interval)

# 公司信息
st.subheader("🏢 Company Overview")
try:
    domain = info.get('website','').replace("https://","").replace("http://","").split("/")[0]
    if domain:
        st.image(f"https://logo.clearbit.com/{domain}", width=120)
    st.markdown(f"### {info.get('shortName', stock.upper())}")
    if info.get('website'):
        st.markdown(f"[Visit Official Website]({info['website']})")
except Exception:
    st.warning("⚠️ Unable to load company logo or website.")

# Close / MA10 / VWAP
st.subheader(f"📈 {stock.upper()} Price with MA10 & VWAP")
data['MA10'] = data['Close'].rolling(window=10).mean()
vwap = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()

fig1, ax1 = plt.subplots(figsize=(14,6))
ax1.plot(data.index, data['Close'], label='Close', color='cyan')
ax1.plot(data.index, data['MA10'], label='MA10', color='orange')
ax1.plot(data.index, vwap, label='VWAP', color='green')
ax1.set_title(f"{stock.upper()} Close / MA10 / VWAP", color='white')
ax1.set_facecolor('black')
fig1.patch.set_facecolor('black')
ax1.tick_params(colors='white')
ax1.legend()
st.pyplot(fig1)

# Volume
st.subheader(f"📊 {stock.upper()} Volume")
fig2, ax2 = plt.subplots(figsize=(14,3))
ax2.bar(data.index, data['Volume'], color='purple')
ax2.set_title("Volume", color='white')
ax2.set_facecolor('black')
fig2.patch.set_facecolor('black')
ax2.tick_params(colors='white')
st.pyplot(fig2)

# 🔮 预测模块
st.subheader(f"🔮 {stock.upper()} 30-Minute Price Prediction")
model_choice = st.radio(
    "Select Prediction Model:",
    ["Auto-ARIMA", "LSTM Deep Learning", "Prophet Forecasting"],
    horizontal=True
)

close_data = data['Close'].dropna()
if len(close_data) >= 200:
    recent_data = close_data.tail(300)

    # — Auto-ARIMA 使用 statsmodels —
    if model_choice == "Auto-ARIMA":
        st.info("Training statsmodels ARIMA model...")
        order = (3,1,2)
        arima = ARIMA(recent_data, order=order)
        fit = arima.fit()
        arima_forecast = fit.forecast(steps=30)

        future_index = pd.date_range(start=data.index[-1], periods=30, freq='T')
        change = (arima_forecast.iloc[-1] - recent_data.iloc[-1]) / recent_data.iloc[-1]

        fig, ax = plt.subplots(figsize=(14,6))
        ax.plot(data.index[-300:], recent_data, label='Historical Close', color='cyan')
        ax.plot(future_index, arima_forecast, label='Forecast', color='red')
        if change > threshold:
            ax.annotate('📈 Buy', xy=(future_index[-1], arima_forecast.iloc[-1]),
                        xytext=(future_index[-1], arima_forecast.iloc[-1]+0.5),
                        arrowprops=dict(facecolor='green', shrink=0.05))
            st.success("📈 Buy Signal Detected!")
        elif change < -threshold:
            ax.annotate('📉 Sell', xy=(future_index[-1], arima_forecast.iloc[-1]),
                        xytext=(future_index[-1], arima_forecast.iloc[-1]-0.5),
                        arrowprops=dict(facecolor='red', shrink=0.05))
            st.error("📉 Sell Signal Detected!")
        else:
            st.info("⏸️ Hold (No strong signal)")

        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        ax.tick_params(colors='white')
        ax.legend()
        st.pyplot(fig)

    # — LSTM 深度学习 —
    elif model_choice == "LSTM Deep Learning":
        st.info("Training LSTM deep learning model...")
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(recent_data.values.reshape(-1,1))
        X,y = [],[]
        lookback = 60
        for i in range(lookback, len(scaled)):
            X.append(scaled[i-lookback:i,0])
            y.append(scaled[i,0])
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=10, batch_size=8, verbose=0)

        inputs = scaled[-lookback:]
        preds = []
        for _ in range(30):
            inp = inputs.reshape((1, lookback,1))
            p = model.predict(inp, verbose=0)
            preds.append(p[0,0])
            inputs = np.append(inputs, p)[-lookback:]
        preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()

        future_index = pd.date_range(start=data.index[-1], periods=30, freq='T')
        change = (preds[-1] - recent_data.iloc[-1]) / recent_data.iloc[-1]

        fig, ax = plt.subplots(figsize=(14,6))
        ax.plot(data.index[-300:], recent_data, label='Historical Close', color='cyan')
        ax.plot(future_index, preds, label='Forecast', color='red')
        if change > threshold:
            ax.annotate('📈 Buy', xy=(future_index[-1], preds[-1]),
                        xytext=(future_index[-1], preds[-1]+0.5),
                        arrowprops=dict(facecolor='green', shrink=0.05))
            st.success("📈 Buy Signal Detected!")
        elif change < -threshold:
            ax.annotate('📉 Sell', xy=(future_index[-1], preds[-1]),
                        xytext=(future_index[-1], preds[-1]-0.5),
                        arrowprops=dict(facecolor='red', shrink=0.05))
            st.error("📉 Sell Signal Detected!")
        else:
            st.info("⏸️ Hold (No strong signal)")

        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        ax.tick_params(colors='white')
        ax.legend()
        st.pyplot(fig)

    # — Prophet 预测 —
    elif model_choice == "Prophet Forecasting":
        st.info("Training Prophet model...")
        df_p = pd.DataFrame({
            'ds': data.index[-300:].tz_localize(None),
            'y': recent_data
        })
        m = Prophet(daily_seasonality=True)
        m.fit(df_p)
        future = m.make_future_dataframe(periods=30, freq='min')
        fct = m.predict(future)

        change = (fct['yhat'].iloc[-1] - recent_data.iloc[-1]) / recent_data.iloc[-1]
        fig, ax = plt.subplots(figsize=(14,6))
        ax.plot(df_p['ds'], df_p['y'], label='Historical Close', color='cyan')
        ax.plot(fct['ds'][-30:], fct['yhat'][-30:], label='Forecast', color='red')
        if change > threshold:
            ax.annotate('📈 Buy', xy=(fct['ds'].iloc[-1], fct['yhat'].iloc[-1]),
                        xytext=(fct['ds'].iloc[-1], fct['yhat'].iloc[-1]+0.5),
                        arrowprops=dict(facecolor='green', shrink=0.05))
            st.success("📈 Buy Signal Detected!")
        elif change < -threshold:
            ax.annotate('📉 Sell', xy=(fct['ds'].iloc[-1], fct['yhat'].iloc[-1]),
                        xytext=(fct['ds'].iloc[-1], fct['yhat'].iloc[-1]-0.5),
                        arrowprops=dict(facecolor='red', shrink=0.05))
            st.error("📉 Sell Signal Detected!")
        else:
            st.info("⏸️ Hold (No strong signal)")

        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        ax.tick_params(colors='white')
        ax.legend()
        st.pyplot(fig)

else:
    st.info(f"⌛ Waiting for at least 200 data points (current: {len(close_data)})...")

# 📰 新闻摘要
st.subheader(f"📰 Latest News about {stock.upper()}")
def fetch_news(ticker):
    url = f"https://news.google.com/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US%3Aen"
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    links = soup.select('a.DY5T1d')
    return ["https://news.google.com" + link['href'][1:] for link in links][:3]

def summarize(url):
    try:
        art = Article(url)
        art.download()
        art.parse()
        art.nlp()
        return art.title, art.summary
    except:
        return None, None

for link in fetch_news(stock):
    title, summary = summarize(link)
    if title and summary:
        st.markdown(f"### [{title}]({link})")
        st.write(summary)
        st.markdown("---")


