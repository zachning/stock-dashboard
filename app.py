import os
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from yfinance.exceptions import YFRateLimitError
import requests
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from streamlit_autorefresh import st_autorefresh
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from bs4 import BeautifulSoup
from scipy.stats import norm

# — Page Config —
st.set_page_config(
    page_title="Stock & Options Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(
    "<h1 style='text-align:center; color:white;'>📈 Real-Time Stock & Options Dashboard</h1>",
    unsafe_allow_html=True
)

# — Sidebar Controls —
with st.sidebar:
    st.header("⚙️ Controls")
    stock = st.text_input("Stock/Crypto Ticker:", value="AAPL")
    period = st.selectbox("History Period:",
        ["1d","5d","7d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"])
    interval = st.selectbox("History Interval:",
        ["2m","5m","15m","30m","60m","90m","1d","1wk","1mo"])
    refresh_time = st.slider("Refresh Interval (sec):", 300, 3600, 600)
    threshold_mode = st.radio("Signal Sensitivity:",
        ["High Volatility (0.5%)","Low Volatility (0.15%)"])
    threshold = 0.005 if "High" in threshold_mode else 0.0015

    # Options underlying + expiry
    options_underlying = st.selectbox("Options Underlying:", ["SPY","QQQ"])
    opt = yf.Ticker(options_underlying)

    expirations = []
    try:
        expirations = opt.options
    except YFRateLimitError:
        st.warning("⚠️ Rate limit on yfinance.options – falling back to REST API")
    except Exception as e:
        st.warning(f"⚠️ yfinance.options error: {e}")

    if not expirations:
        try:
            url = f"https://query1.finance.yahoo.com/v7/finance/options/{options_underlying}"
            resp = requests.get(url, timeout=5).json()
            dates = resp["optionChain"]["result"][0]["expirationDates"]
            expirations = [datetime.date.fromtimestamp(ts).isoformat() for ts in dates]
            st.info("🔄 Fetched expirations via REST API")
        except Exception as e:
            st.warning(f"⚠️ REST API expiry fetch failed: {e}")

    if not expirations:
        expirations = st.session_state.get("expirations_cache", [])

    if not expirations:
        st.error("❌ Could not fetch expirations automatically.")
        manual_dt = st.date_input("Enter Expiration Date manually:", min_value=datetime.date.today())
        expirations = [manual_dt.isoformat()]

    st.session_state["expirations_cache"] = expirations
    selected_expiry = st.selectbox("Select Expiry Date:", expirations)

# — Auto-refresh —
st_autorefresh(interval=refresh_time * 1000, key="refresh")

# — Data Loader with CSV Fallback & TTL 600s —
@st.cache_data(ttl=600)
def load_data(tkr, per, intr):
    cache_file = "cache.csv"
    try:
        ticker = yf.Ticker(tkr)
        hist   = ticker.history(period=per, interval=intr)
        info   = ticker.info
        if not hist.empty:
            hist.to_csv(cache_file)
            return hist, info
        raise Exception("empty DataFrame")
    except YFRateLimitError:
        st.warning("⚠️ Rate limit on price history – using cached CSV")
    except Exception as e:
        st.warning(f"⚠️ Price fetch error: {e} – using cached CSV")
    if os.path.exists(cache_file):
        try:
            hist = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return hist, {"shortName": tkr}
        except Exception as e2:
            st.error(f"❌ Failed to read cache.csv: {e2}")
    return pd.DataFrame(), {"shortName": tkr}

# — Validate period/interval —
def validate_combo(p, i):
    minute_limits = {"2m":60,"5m":60,"15m":60,"30m":60,"60m":730,"90m":60}
    if i in minute_limits and (p.endswith("y") or p in ["max","ytd","10y","5y"]):
        return False
    return True

if not validate_combo(period, interval):
    st.warning("⚠️ This period+interval combo is not supported.")
    st.stop()

# — Load Price & Info & Sanity Check —
data, info = load_data(stock.upper(), period, interval)

st.markdown("### 📋 Data Sanity Check")
if data.empty:
    st.warning("⚠️ `data` came back empty! No price points to plot.")
else:
    st.write(f"• data.shape = {data.shape}")
    st.dataframe(data.head(5))

# — Company Overview —
st.subheader("🏢 Company Overview")
domain = info.get("website","").replace("https://","").replace("http://","").split("/")[0]
if domain:
    st.image(f"https://logo.clearbit.com/{domain}", width=100)
st.markdown(f"### {info.get('shortName', stock.upper())}")
if info.get("website"):
    st.markdown(f"[Visit Website]({info['website']})")

# — Price / MA10 / VWAP Chart —
if not data.empty:
    st.subheader(f"📈 {stock.upper()} Price Indicators")
    data["MA10"] = data["Close"].rolling(10).mean()
    vwap = (data["Close"] * data["Volume"]).cumsum() / data["Volume"].cumsum()
    fig1, ax1 = plt.subplots(figsize=(14,6))
    ax1.plot(data.index, data["Close"], color="cyan", label="Close")
    ax1.plot(data.index, data["MA10"], color="orange", label="MA10")
    ax1.plot(data.index, vwap,      color="green",  label="VWAP")
    ax1.set_title(f"{stock.upper()} Close │ MA10 │ VWAP", color="white")
    ax1.set_facecolor("black"); fig1.patch.set_facecolor("black")
    ax1.tick_params(colors="white"); ax1.legend()
    st.pyplot(fig1)
else:
    st.info("ℹ️ Skipping price‐indicator chart because no data is available.")

# — Volume Chart —
if not data.empty:
    st.subheader(f"📊 {stock.upper()} Volume")
    fig2, ax2 = plt.subplots(figsize=(14,3))
    ax2.bar(data.index, data["Volume"], color="purple")
    ax2.set_title("Volume", color="white")
    ax2.set_facecolor("black"); fig2.patch.set_facecolor("black")
    ax2.tick_params(colors="white")
    st.pyplot(fig2)
else:
    st.info("ℹ️ Skipping volume chart because no data is available.")

# — Prediction Module —
if not data.empty:
    st.subheader(f"🔮 {stock.upper()} 30-Min Prediction")
    model_choice = st.radio("Model:", ["Auto-ARIMA","LSTM","Prophet"], horizontal=True)
    close_data = data["Close"].dropna()

    if len(close_data) >= 200:
        recent = close_data.tail(300)

        if model_choice == "Auto-ARIMA":
            m   = ARIMA(recent, order=(3,1,2)).fit()
            fc  = m.forecast(30)
            idx = pd.date_range(start=data.index[-1], periods=30, freq="T")
            change = (fc.iloc[-1] - recent.iloc[-1]) / recent.iloc[-1]
            fig, ax = plt.subplots(figsize=(14,6))
            ax.plot(data.index[-300:], recent, color="cyan", label="Hist Close")
            ax.plot(idx, fc,                color="red",   label="Forecast")
            if change > threshold:
                ax.annotate("📈 Buy", xy=(idx[-1],fc.iloc[-1]),
                            xytext=(idx[-1],fc.iloc[-1]+0.5),
                            arrowprops=dict(facecolor="green"))
                st.success("📈 Buy Signal")
            elif change < -threshold:
                ax.annotate("📉 Sell", xy=(idx[-1],fc.iloc[-1]),
                            xytext=(idx[-1],fc.iloc[-1]-0.5),
                            arrowprops=dict(facecolor="red"))
                st.error("📉 Sell Signal")
            else:
                st.info("⏸️ Hold")
            ax.legend(); st.pyplot(fig)
            rmse = np.sqrt(mean_squared_error(recent[-30:], fc))
            mape = mean_absolute_percentage_error(recent[-30:], fc)
            st.markdown(f"**RMSE:** {rmse:.4f}   **MAPE:** {mape*100:.2f}%")

        elif model_choice == "LSTM":
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(recent.values.reshape(-1,1))
            X, y   = [], []
            lb     = 60
            for i in range(lb, len(scaled)):
                X.append(scaled[i-lb:i,0]); y.append(scaled[i,0])
            X, y = np.array(X), np.array(y)
            X    = X.reshape(X.shape[0], X.shape[1], 1)
            net  = Sequential([
                LSTM(50, return_sequences=True, input_shape=(lb,1)),
                LSTM(50),
                Dense(1)
            ])
            net.compile("adam","mean_squared_error")
            net.fit(X, y, epochs=10, batch_size=8, verbose=0)
            seq, preds = scaled[-lb:], []
            for _ in range(30):
                p = net.predict(seq.reshape(1,lb,1), verbose=0)[0,0]
                preds.append(p); seq = np.append(seq, p)[-lb:]
            preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
            idx   = pd.date_range(start=data.index[-1], periods=30, freq="T")
            change = (preds[-1] - recent.iloc[-1]) / recent.iloc[-1]
            fig, ax = plt.subplots(figsize=(14,6))
            ax.plot(data.index[-300:], recent, color="cyan", label="Hist Close")
            ax.plot(idx, preds,             color="red",   label="Forecast")
            if change > threshold:
                ax.annotate("📈 Buy", xy=(idx[-1],preds[-1]),
                            xytext=(idx[-1],preds[-1]+0.5),
                            arrowprops=dict(facecolor="green"))
                st.success("📈 Buy Signal")
            elif change < -threshold:
                ax.annotate("📉 Sell", xy=(idx[-1],preds[-1]),
                            xytext=(idx[-1],preds[-1]-0.5),
                            arrowprops=dict(facecolor="red"))
                st.error("📉 Sell Signal")
            else:
                st.info("⏸️ Hold")
            ax.legend(); st.pyplot(fig)
            rmse = np.sqrt(mean_squared_error(recent[-30:], preds))
            mape = mean_absolute_percentage_error(recent[-30:], preds)
            st.markdown(f"**RMSE:** {rmse:.4f}   **MAPE:** {mape*100:.2f}%")

        else:  # Prophet
            dfp = pd.DataFrame({
                "ds": data.index[-300:].tz_localize(None),
                "y" : recent
            })
            m   = Prophet(daily_seasonality=True).fit(dfp)
            fut = m.make_future_dataframe(periods=30, freq="min")
            fcst= m.predict(fut)
            change = (fcst["yhat"].iloc[-1] - recent.iloc[-1]) / recent.iloc[-1]
            fig, ax = plt.subplots(figsize=(14,6))
            ax.plot(dfp["ds"],    dfp["y"],      color="cyan", label="Hist Close")
            ax.plot(fcst["ds"][-30:], fcst["yhat"][-30:], color="red", label="Forecast")
            if change > threshold:
                ax.annotate("📈 Buy",
                            xy=(fcst["ds"].iloc[-1],fcst["yhat"].iloc[-1]),
                            xytext=(fcst["ds"].iloc[-1],fcst["yhat"].iloc[-1]+0.5),
                            arrowprops=dict(facecolor="green"))
                st.success("📈 Buy Signal")
            elif change < -threshold:
                ax.annotate("📉 Sell",
                            xy=(fcst["ds"].iloc[-1],fcst["yhat"].iloc[-1]),
                            xytext=(fcst["ds"].iloc[-1],fcst["yhat"].iloc[-1]-0.5),
                            arrowprops=dict(facecolor="red"))
                st.error("📉 Sell Signal")
            else:
                st.info("⏸️ Hold")
            ax.legend(); st.pyplot(fig)
            rmse = np.sqrt(mean_squared_error(dfp["y"][-30:], fcst["yhat"][-30:]))
            mape = mean_absolute_percentage_error(dfp["y"][-30:], fcst["yhat"][-30:])
            st.markdown(f"**RMSE:** {rmse:.4f}   **MAPE:** {mape*100:.2f}%")
    else:
        st.info(f"⌛ Waiting for 200+ data points (current: {len(close_data)})...")
else:
    st.info("ℹ️ Skipping prediction module because no data is available.")

# — Intraday Options & Large Trades —
st.subheader("⚡ Intraday Options & Large Trades")
chain = opt.option_chain(selected_expiry)
calls = chain.calls.assign(type="call", expiration=selected_expiry)
puts  = chain.puts.assign(type="put",  expiration=selected_expiry)
df_opts = pd.concat([calls, puts], ignore_index=True).fillna({"volume":0})
thresh  = df_opts["volume"].quantile(0.95)
large   = df_opts[df_opts["volume"]>=thresh]

st.markdown(
    f"### Large Option Trades (Vol ≥ {thresh:.0f}, 95th pct) "
    f"for {options_underlying} exp {selected_expiry}"
)
if large.empty:
    st.info("No large option trades today.")
else:
    st.dataframe(large[["type","expiration","strike","volume","lastPrice","bid","ask"]])
    fig, ax = plt.subplots(figsize=(10,4))
    sizes = (large["volume"]/large["volume"].max())*300
    ax.scatter(large["strike"], large["type"], s=sizes,
               c=large["type"].map({"call":"green","put":"red"}),
               alpha=0.6)
    ax.set_xlabel("Strike", color="white"); ax.set_ylabel("Type", color="white")
    ax.set_title(f"{options_underlying} Large Option Flow", color="white")
    ax.set_facecolor("black"); fig.patch.set_facecolor("black")
    ax.tick_params(colors="white"); st.pyplot(fig)

# — Option Gamma Exposure (GEX) Strategy —
st.subheader("⚙️ Option Gamma Exposure (GEX) Strategy")
if not data.empty:
    spot    = data["Close"].iloc[-1]
    exp_dt  = datetime.datetime.combine(
        datetime.date.fromisoformat(selected_expiry),
        datetime.time(15,30)
    )
    T       = max((exp_dt - datetime.datetime.now()).total_seconds(), 0) / (365*24*3600)

    def bs_gamma(S,K,T,sigma,r=0,q=0):
        d1 = (np.log(S/K)+(r-q+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        return np.exp(-q*T)*norm.pdf(d1)/(S*sigma*np.sqrt(T))

    rows=[]
    for _, r in df_opts.dropna(subset=["impliedVolatility","openInterest"]).iterrows():
        σ  = r.impliedVolatility or 1e-6
        oi = r.openInterest or 0
        γ  = bs_gamma(spot, r.strike, T, σ)
        rows.append((r.type, r.strike, γ*oi*100))

    gex_df   = pd.DataFrame(rows, columns=["type","strike","gex"])
    call_gex = gex_df.query("type=='call'")["gex"].sum()
    put_gex  = gex_df.query("type=='put'")["gex"].sum()
    net_gex  = call_gex - put_gex

    with st.sidebar.expander("🔍 GEX Metrics", expanded=True):
        st.metric("Call GEX", f"{call_gex:,.0f}")
        st.metric("Put  GEX", f"{put_gex:,.0f}")
        st.metric("Net  GEX", f"{net_gex:,.0f}")

    fig, ax = plt.subplots(figsize=(12,4))
    gex_plot = gex_df.assign(net=lambda d: d["gex"].where(d["type"]=="call", -d["gex"]))
    gex_plot = gex_plot.groupby("strike")["net"].sum().reset_index()
    ax.bar(
        gex_plot["strike"], gex_plot["net"],
        width=np.diff(gex_plot["strike"]).mean()*0.8 or 1,
        color=np.where(gex_plot["net"]>=0, "lime", "tomato")
    )
    ax.axhline(0, color="white")
    ax.set_title(f"{options_underlying} Net Gamma Exposure by Strike (exp {selected_expiry})", color="white")
    ax.set_xlabel("Strike", color="white"); ax.set_ylabel("Net GEX", color="white")
    ax.set_facecolor("black"); fig.patch.set_facecolor("black")
    ax.tick_params(colors="white"); st.pyplot(fig)
else:
    st.info("ℹ️ Skipping GEX plot because no data is available.")

# — News Summary —
st.subheader(f"📰 Latest News about {stock.upper()}")
def fetch_news(tkr):
    url  = f"https://news.google.com/search?q={tkr}+stock&hl=en-US&gl=US&ceid=US%3Aen"
    r    = requests.get(url, timeout=5)
    soup = BeautifulSoup(r.text, "html.parser")
    return ["https://news.google.com"+a["href"][1:] for a in soup.select("a.DY5T1d")][:3]

for link in fetch_news(stock):
    try:
        # using newspaper may still fail server-side, guard it
        from newspaper import Article
        art = Article(link)
        art.download(); art.parse(); art.nlp()
        st.markdown(f"### [{art.title}]({link})")
        st.write(art.summary)
        st.markdown("---")
    except Exception:
        st.info("⚠️ Could not parse this article preview.")
        continue

