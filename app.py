import os
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from yfinance.exceptions import YFRateLimitError
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from streamlit_autorefresh import st_autorefresh
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from bs4 import BeautifulSoup
from newspaper import Article
import requests
from scipy.stats import norm

# — Page config —
st.set_page_config(
    page_title="Stock & Options Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(
    "<h1 style='text-align:center; color:white;'>📈 Real-Time Stock & Options Dashboard</h1>",
    unsafe_allow_html=True
)

# — Sidebar controls —
with st.sidebar:
    st.header("⚙️ Controls")
    stock = st.text_input("Stock/Crypto Ticker:", value="AAPL")
    period = st.selectbox(
        "History Period:",
        ["1d","5d","7d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"]
    )
    interval = st.selectbox(
        "History Interval:",
        ["2m","5m","15m","30m","60m","90m","1d","1wk","1mo"]
    )
    refresh_time = st.slider("Refresh Interval (sec):", 300, 3600, 600)
    threshold_mode = st.radio(
        "Signal Sensitivity:",
        ["High Volatility (0.5%)","Low Volatility (0.15%)"]
    )
    threshold = 0.005 if "High" in threshold_mode else 0.0015

    # Options underlying and expiry selection
    options_underlying = st.selectbox("Options Underlying:", ["SPY","QQQ"])
    opt = yf.Ticker(options_underlying)
    expirations = opt.options  # list of ISO date strings
    selected_expiry = st.selectbox("Select Expiry Date:", expirations)

# — Auto-refresh —
st_autorefresh(interval=refresh_time * 1000, key="refresh")

# — Data loader with CSV fallback & extended TTL —
@st.cache_data(ttl=600)
def load_data(tkr, per, intr):
    cache_file = "cache.csv"
    try:
        t = yf.Ticker(tkr)
        hist = t.history(period=per, interval=intr)
        info = t.info
        hist.to_csv(cache_file)
        return hist, info
    except Exception as e:
        print("load_data error:", e)
        if os.path.exists(cache_file):
            try:
                hist = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                return hist, {"shortName": tkr}
            except Exception as e2:
                print("CSV fallback error:", e2)
        return pd.DataFrame(), {"shortName": tkr}

# — Validate period+interval combos —
def validate_combo(p, i):
    minute_limits = {"2m":60,"5m":60,"15m":60,"30m":60,"60m":730,"90m":60}
    if i in minute_limits:
        if p.endswith("y") or p in ["max","ytd","10y","5y"]:
            return False
    return True

if not validate_combo(period, interval):
    st.warning("⚠️ This period+interval combo is not supported.")
    st.stop()

# — Load price & info —
data, info = load_data(stock.upper(), period, interval)
if data.empty:
    st.error("⚠️ Could not fetch price data (rate limiting or other error).")
    st.stop()

# — Company Overview —
st.subheader("🏢 Company Overview")
domain = info.get("website", "").replace("https://","").replace("http://","").split("/")[0]
if domain:
    st.image(f"https://logo.clearbit.com/{domain}", width=100)
st.markdown(f"### {info.get('shortName', stock.upper())}")
if info.get("website"):
    st.markdown(f"[Visit Website]({info['website']})")

# — Price / MA10 / VWAP Chart —
st.subheader(f"📈 {stock.upper()} Price Indicators")
data["MA10"] = data["Close"].rolling(10).mean()
vwap = (data["Close"] * data["Volume"]).cumsum() / data["Volume"].cumsum()
fig1, ax1 = plt.subplots(figsize=(14,6))
ax1.plot(data.index, data["Close"], label="Close", color="cyan")
ax1.plot(data.index, data["MA10"], label="MA10", color="orange")
ax1.plot(data.index, vwap, label="VWAP", color="green")
ax1.set_title(f"{stock.upper()} Close │ MA10 │ VWAP", color="white")
ax1.set_facecolor("black"); fig1.patch.set_facecolor("black")
ax1.tick_params(colors="white"); ax1.legend()
st.pyplot(fig1)

# — Volume Chart —
st.subheader(f"📊 {stock.upper()} Volume")
fig2, ax2 = plt.subplots(figsize=(14,3))
ax2.bar(data.index, data["Volume"], color="purple")
ax2.set_title("Volume", color="white")
ax2.set_facecolor("black"); fig2.patch.set_facecolor("black")
ax2.tick_params(colors="white")
st.pyplot(fig2)

# — Prediction Module —
st.subheader(f"🔮 {stock.upper()} 30-Minute Prediction")
model_choice = st.radio("Model:", ["Auto-ARIMA","LSTM","Prophet"], horizontal=True)
close_data = data["Close"].dropna()

if len(close_data) >= 200:
    recent = close_data.tail(300)

    if model_choice == "Auto-ARIMA":
        st.info("Training ARIMA…")
        arima_model = ARIMA(recent, order=(3,1,2)).fit()
        forecast = arima_model.forecast(steps=30)
        future_idx = pd.date_range(start=data.index[-1], periods=30, freq="T")
        change = (forecast.iloc[-1] - recent.iloc[-1]) / recent.iloc[-1]
        fig, ax = plt.subplots(figsize=(14,6))
        ax.plot(data.index[-300:], recent, label="Hist Close", color="cyan")
        ax.plot(future_idx, forecast, label="Forecast", color="red")
        if change > threshold:
            ax.annotate("📈 Buy", xy=(future_idx[-1], forecast.iloc[-1]),
                        xytext=(future_idx[-1], forecast.iloc[-1]+0.5),
                        arrowprops=dict(facecolor="green", shrink=0.05))
            st.success("📈 Buy Signal")
        elif change < -threshold:
            ax.annotate("📉 Sell", xy=(future_idx[-1], forecast.iloc[-1]),
                        xytext=(future_idx[-1], forecast.iloc[-1]-0.5),
                        arrowprops=dict(facecolor="red", shrink=0.05))
            st.error("📉 Sell Signal")
        else:
            st.info("⏸️ Hold")
        st.pyplot(fig)
        rmse = np.sqrt(mean_squared_error(recent[-30:].values, forecast.values))
        mape = mean_absolute_percentage_error(recent[-30:].values, forecast.values)
        st.markdown(f"**RMSE:** {rmse:.4f}   **MAPE:** {mape*100:.2f}%")

    elif model_choice == "LSTM":
        st.info("Training LSTM…")
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(recent.values.reshape(-1,1))
        X, y = [], []
        lb = 60
        for i in range(lb, len(scaled)):
            X.append(scaled[i-lb:i,0]); y.append(scaled[i,0])
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        lstm = Sequential([
            LSTM(50, return_sequences=True, input_shape=(lb,1)),
            LSTM(50),
            Dense(1)
        ])
        lstm.compile("adam", "mean_squared_error")
        lstm.fit(X, y, epochs=10, batch_size=8, verbose=0)
        seq = scaled[-lb:]; preds = []
        for _ in range(30):
            p = lstm.predict(seq.reshape(1,lb,1), verbose=0)[0,0]
            preds.append(p); seq = np.append(seq, p)[-lb:]
        preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
        future_idx = pd.date_range(start=data.index[-1], periods=30, freq="T")
        change = (preds[-1] - recent.iloc[-1]) / recent.iloc[-1]
        fig, ax = plt.subplots(figsize=(14,6))
        ax.plot(data.index[-300:], recent, label="Hist Close", color="cyan")
        ax.plot(future_idx, preds, label="Forecast", color="red")
        if change > threshold:
            ax.annotate("📈 Buy", xy=(future_idx[-1], preds[-1]),
                        xytext=(future_idx[-1], preds[-1]+0.5),
                        arrowprops=dict(facecolor="green", shrink=0.05))
            st.success("📈 Buy Signal")
        elif change < -threshold:
            ax.annotate("📉 Sell", xy=(future_idx[-1], preds[-1]),
                        xytext=(future_idx[-1], preds[-1]-0.5),
                        arrowprops=dict(facecolor="red", shrink=0.05))
            st.error("📉 Sell Signal")
        else:
            st.info("⏸️ Hold")
        st.pyplot(fig)
        rmse = np.sqrt(mean_squared_error(recent[-30:].values, preds))
        mape = mean_absolute_percentage_error(recent[-30:].values, preds)
        st.markdown(f"**RMSE:** {rmse:.4f}   **MAPE:** {mape*100:.2f}%")

    else:
        st.info("Training Prophet…")
        dfp = pd.DataFrame({
            "ds": data.index[-300:].tz_localize(None),
            "y": recent
        })
        m = Prophet(daily_seasonality=True).fit(dfp)
        fut = m.make_future_dataframe(periods=30, freq="min")
        fc = m.predict(fut)
        change = (fc["yhat"].iloc[-1] - recent.iloc[-1]) / recent.iloc[-1]
        fig, ax = plt.subplots(figsize=(14,6))
        ax.plot(dfp["ds"], dfp["y"], label="Hist Close", color="cyan")
        ax.plot(fc["ds"][-30:], fc["yhat"][-30:], label="Forecast", color="red")
        if change > threshold:
            ax.annotate("📈 Buy", xy=(fc["ds"].iloc[-1], fc["yhat"].iloc[-1]),
                        xytext=(fc["ds"].iloc[-1], fc["yhat"].iloc[-1]+0.5),
                        arrowprops=dict(facecolor="green", shrink=0.05))
            st.success("📈 Buy Signal")
        elif change < -threshold:
            ax.annotate("📉 Sell", xy=(fc["ds"].iloc[-1], fc["yhat"].iloc[-1]),
                        xytext=(fc["ds"].iloc[-1], fc["yhat"].iloc[-1]-0.5),
                        arrowprops=dict(facecolor="red", shrink=0.05))
            st.error("📉 Sell Signal")
        else:
            st.info("⏸️ Hold")
        st.pyplot(fig)
        rmse = np.sqrt(mean_squared_error(recent[-30:].values, fc["yhat"].values[-30:]))
        mape = mean_absolute_percentage_error(recent[-30:].values, fc["yhat"].values[-30:])
        st.markdown(f"**RMSE:** {rmse:.4f}   **MAPE:** {mape*100:.2f}%")

else:
    st.info(f"⌛ Waiting for 200+ data points (current: {len(close_data)})...")

# — Intraday Options & Large Trades —
st.subheader("⚡ Intraday Options & Large Trades")
chain = opt.option_chain(selected_expiry)
calls = chain.calls.assign(type="call", expiration=selected_expiry)
puts  = chain.puts.assign(type="put",  expiration=selected_expiry)
df_opts = pd.concat([calls, puts], ignore_index=True)
df_opts["volume"] = df_opts["volume"].fillna(0)
thresh = df_opts["volume"].quantile(0.95)
large  = df_opts[df_opts["volume"] >= thresh]

st.markdown(
    f"### Large Option Trades (Volume ≥ {thresh:.0f}, 95th percentile) "
    f"for {options_underlying} exp {selected_expiry}"
)
if large.empty:
    st.info("No large option trades today.")
else:
    st.dataframe(large[["type","expiration","strike","volume","lastPrice","bid","ask"]])
    fig, ax = plt.subplots(figsize=(10,4))
    sizes = (large["volume"] / large["volume"].max()) * 300
    ax.scatter(
        large["strike"], large["type"],
        s=sizes,
        c=large["type"].map({"call":"green","put":"red"}),
        alpha=0.6
    )
    ax.set_xlabel("Strike", color="white")
    ax.set_ylabel("Type", color="white")
    ax.set_title(f"{options_underlying} Large Option Flow", color="white")
    ax.set_facecolor("black"); fig.patch.set_facecolor("black")
    ax.tick_params(colors="white")
    st.pyplot(fig)

# — Option Gamma Exposure (GEX) Strategy —
def bs_gamma(S, K, T, sigma, r=0, q=0):
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    return np.exp(-q*T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))

st.subheader("⚙️ Option Gamma Exposure (GEX) Strategy")
spot = data["Close"].iloc[-1]
exp_date = datetime.date.fromisoformat(selected_expiry)
now = datetime.datetime.now()
exp_dt = datetime.datetime.combine(exp_date, datetime.time(15,30))
T = max((exp_dt - now).total_seconds(), 0.0) / (365*24*3600)

rows = []
for _, r in pd.concat([calls, puts], ignore_index=True).dropna(subset=["impliedVolatility","openInterest"]).iterrows():
    σ = r.impliedVolatility or 1e-6
    oi = r.openInterest or 0
    γ = bs_gamma(spot, r.strike, T, σ)
    gex = γ * oi * 100
    rows.append((r.type, r.strike, gex))

gex_df = pd.DataFrame(rows, columns=["type","strike","gex"])
call_gex = gex_df.query("type=='call'")["gex"].sum()
put_gex  = gex_df.query("type=='put'")["gex"].sum()
net_gex  = call_gex - put_gex  

with st.sidebar.expander("🔍 GEX Metrics", expanded=True):
    st.metric("Call GEX", f"{call_gex:,.0f}")
    st.metric("Put  GEX", f"{put_gex:,.0f}")
    st.metric("Net  GEX", f"{net_gex:,.0f}")

fig, ax = plt.subplots(figsize=(12,4))
gex_plot = gex_df.assign(net=lambda df: df["gex"].where(df["type"]=="call", -df["gex"]))
gex_plot = gex_plot.groupby("strike")["net"].sum().reset_index()
ax.bar(
    gex_plot["strike"], gex_plot["net"],
    width=np.diff(gex_plot["strike"]).mean() * 0.8 or 1,
    color=np.where(gex_plot["net"]>=0, "lime", "tomato")
)
ax.axhline(0, color="white")
ax.set_title(f"{options_underlying} Net Gamma Exposure by Strike (exp {selected_expiry})", color="white")
ax.set_xlabel("Strike", color="white"); ax.set_ylabel("Net GEX", color="white")
ax.set_facecolor("black"); fig.patch.set_facecolor("black")
ax.tick_params(colors="white")
st.pyplot(fig)

# — News Summary —
st.subheader(f"📰 Latest News about {stock.upper()}")
def fetch_news(tkr):
    url = f"https://news.google.com/search?q={tkr}+stock&hl=en-US&gl=US&ceid=US%3Aen"
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    return ["https://news.google.com"+a["href"][1:] for a in soup.select("a.DY5T1d")][:3]

for link in fetch_news(stock):
    try:
        art = Article(link); art.download(); art.parse(); art.nlp()
        st.markdown(f"### [{art.title}]({link})")
        st.write(art.summary)
        st.markdown("---")
    except:
        continue


