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

    options_underlying = st.selectbox("Options Underlying:", ["SPY","QQQ"])
    opt = yf.Ticker(options_underlying)

    # — Expiration dates with fallback & cache —
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

    # session_state cache
    if not expirations:
        expirations = st.session_state.get("expirations_cache", [])

    if not expirations:
        st.error("❌ Could not fetch expirations automatically.")
        dt = st.date_input("Enter Expiration Date manually:", min_value=datetime.date.today())
        expirations = [dt.isoformat()]

    st.session_state["expirations_cache"] = expirations
    selected_expiry = st.selectbox("Select Expiry Date:", expirations)

# — Auto-refresh —
st_autorefresh(interval=refresh_time * 1000, key="refresh")

# — Fetcher & Loader using session_state as fallback —
@st.cache_data(ttl=600)
def fetch_data(tkr, per, intr):
    """Always hits Yahoo; returns (hist_df, info_dict)."""
    ticker = yf.Ticker(tkr)
    hist   = ticker.history(period=per, interval=intr)
    info   = ticker.info
    return hist, info

def load_data(tkr, per, intr):
    """Attempts fetch_data(); on rate-limit or empty, falls back to session_state."""
    try:
        hist, info = fetch_data(tkr, per, intr)
        if hist.empty:
            raise YFRateLimitError("empty result")
        # save into session_state
        st.session_state["last_data"] = hist
        st.session_state["last_info"] = info
        return hist, info

    except YFRateLimitError:
        st.warning("⚠️ Rate limit or empty data – using in-memory cache")
    except Exception as e:
        st.warning(f"⚠️ Price fetch error: {e} – using in-memory cache")

    # fallback
    if "last_data" in st.session_state:
        return st.session_state["last_data"], st.session_state.get("last_info", {})
    else:
        # no cache yet
        return pd.DataFrame(), {}

# — Validate period/interval combos —
def validate_combo(p, i):
    minute_limits = {"2m":60,"5m":60,"15m":60,"30m":60,"60m":730,"90m":60}
    if i in minute_limits and (p.endswith("y") or p in ["max","ytd","10y","5y"]):
        return False
    return True

if not validate_combo(period, interval):
    st.warning("⚠️ This period+interval combo is not supported.")
    st.stop()

# — Load & Sanity Check —
data, info = load_data(stock.upper(), period, interval)
st.markdown("### 📋 Data Sanity Check")
if data.empty:
    st.warning("⚠️ No price data available (and no cache). Charts will be skipped.")
else:
    st.write(f"• data.shape = {data.shape}")
    st.dataframe(data.head(3))

# — Company Overview —
st.subheader("🏢 Company Overview")
domain = info.get("website","").replace("https://","").replace("http://","").split("/")[0]
if domain:
    st.image(f"https://logo.clearbit.com/{domain}", width=100)
st.markdown(f"### {info.get('shortName', stock.upper())}")
if info.get("website"):
    st.markdown(f"[Visit Website]({info['website']})")

# — Price Indicators Chart —
if not data.empty:
    st.subheader(f"📈 {stock.upper()} Price │ MA10 │ VWAP")
    data["MA10"] = data["Close"].rolling(10).mean()
    vwap = (data["Close"] * data["Volume"]).cumsum() / data["Volume"].cumsum()
    fig, ax = plt.subplots(figsize=(14,6))
    ax.plot(data.index, data["Close"], color="cyan", label="Close")
    ax.plot(data.index, data["MA10"], color="orange", label="MA10")
    ax.plot(data.index, vwap,      color="green",  label="VWAP")
    ax.set_facecolor("black"); fig.patch.set_facecolor("black")
    ax.tick_params(colors="white"); ax.legend(); ax.set_title("Price Indicators", color="white")
    st.pyplot(fig)
else:
    st.info("ℹ️ Skipping Price Indicators (no data).")

# — Volume Chart —
if not data.empty:
    st.subheader("📊 Volume")
    fig, ax = plt.subplots(figsize=(14,3))
    ax.bar(data.index, data["Volume"], color="purple")
    ax.set_facecolor("black"); fig.patch.set_facecolor("black")
    ax.tick_params(colors="white"); ax.set_title("Volume", color="white")
    st.pyplot(fig)
else:
    st.info("ℹ️ Skipping Volume (no data).")

# — 30-Min Prediction Module —
if not data.empty:
    st.subheader(f"🔮 {stock.upper()} 30-Min Prediction")
    model_choice = st.radio("Model:", ["Auto-ARIMA","LSTM","Prophet"], horizontal=True)
    close = data["Close"].dropna()
    if len(close) < 200:
        st.info(f"⌛ Need ≥200 points, got {len(close)}. Waiting…")
    else:
        recent = close.tail(300)

        def plot_and_metrics(idx, pred, label):
            change = (pred[-1] - recent.iloc[-1]) / recent.iloc[-1]
            fig, ax = plt.subplots(figsize=(14,6))
            ax.plot(data.index[-300:], recent, color="cyan", label="Hist")
            ax.plot(idx, pred,              color="red",   label="Forecast")
            if change > threshold:
                ax.annotate("📈 Buy", xy=(idx[-1],pred[-1]),
                            xytext=(idx[-1],pred[-1]+0.5),
                            arrowprops=dict(facecolor="green"))
                st.success("📈 Buy Signal")
            elif change < -threshold:
                ax.annotate("📉 Sell", xy=(idx[-1],pred[-1]),
                            xytext=(idx[-1],pred[-1]-0.5),
                            arrowprops=dict(facecolor="red"))
                st.error("📉 Sell Signal")
            else:
                st.info("⏸️ Hold")
            rmse = np.sqrt(mean_squared_error(recent[-30:], pred))
            mape = mean_absolute_percentage_error(recent[-30:], pred)
            ax.set_facecolor("black"); fig.patch.set_facecolor("black")
            ax.tick_params(colors="white"); ax.legend(); ax.set_title(label, color="white")
            st.pyplot(fig)
            st.markdown(f"**RMSE:** {rmse:.4f}    **MAPE:** {mape*100:.2f}%")

        idx = pd.date_range(start=data.index[-1], periods=30, freq="T")
        if model_choice == "Auto-ARIMA":
            m   = ARIMA(recent, order=(3,1,2)).fit()
            fc  = m.forecast(30)
            plot_and_metrics(idx, fc.values, "Auto-ARIMA Forecast")

        elif model_choice == "LSTM":
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(recent.values.reshape(-1,1))
            X, y   = [], []
            lb     = 60
            for i in range(lb, len(scaled)):
                X.append(scaled[i-lb:i,0]); y.append(scaled[i,0])
            X, y = np.array(X), np.array(y)
            X    = X.reshape(X.shape[0], X.shape[1],1)
            net  = Sequential([
                LSTM(50, return_sequences=True, input_shape=(lb,1)),
                LSTM(50),
                Dense(1)
            ])
            net.compile("adam","mean_squared_error")
            net.fit(X,y,epochs=10,batch_size=8,verbose=0)
            seq, preds = scaled[-lb:], []
            for _ in range(30):
                p = net.predict(seq.reshape(1,lb,1),verbose=0)[0,0]
                preds.append(p); seq = np.append(seq,p)[-lb:]
            preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
            plot_and_metrics(idx, preds, "LSTM Forecast")

        else:  # Prophet
            dfp = pd.DataFrame({
                "ds": data.index[-300:].tz_localize(None),
                "y" : recent
            })
            m   = Prophet(daily_seasonality=True).fit(dfp)
            fut = m.make_future_dataframe(periods=30, freq="min")
            fcst= m.predict(fut)
            plot_and_metrics(fcst["ds"][-30:], fcst["yhat"][-30:].values, "Prophet Forecast")

else:
    st.info("ℹ️ Skipping Prediction Module (no data).")

# — Intraday Options & Large Trades —
st.subheader("⚡ Intraday Options & Large Trades")
chain = opt.option_chain(selected_expiry)
calls = chain.calls.assign(type="call", expiration=selected_expiry)
puts  = chain.puts.assign(type="put",  expiration=selected_expiry)
df_o  = pd.concat([calls, puts], ignore_index=True).fillna({"volume":0})
th    = df_o["volume"].quantile(0.95)
large = df_o[df_o["volume"]>=th]

st.markdown(f"### Large Trades (Vol ≥ {th:.0f}, 95th pct) for {options_underlying} exp {selected_expiry}")
if large.empty:
    st.info("No large option trades.")
else:
    st.dataframe(large[["type","expiration","strike","volume","lastPrice","bid","ask"]])
    fig, ax = plt.subplots(figsize=(10,4))
    sizes = (large["volume"]/large["volume"].max())*300
    ax.scatter(large["strike"], large["type"], s=sizes,
               c=large["type"].map({"call":"lime","put":"tomato"}),
               alpha=0.6)
    ax.set_xlabel("Strike", color="white"); ax.set_ylabel("Type", color="white")
    ax.set_title(f"{options_underlying} Large Flow", color="white")
    ax.set_facecolor("black"); fig.patch.set_facecolor("black")
    ax.tick_params(colors="white"); st.pyplot(fig)

# — Option Gamma Exposure (GEX) —
st.subheader("⚙️ Option Gamma Exposure (GEX) Strategy")
if not data.empty:
    spot   = data["Close"].iloc[-1]
    exp_dt = datetime.datetime.combine(
        datetime.date.fromisoformat(selected_expiry),
        datetime.time(15,30)
    )
    T      = max((exp_dt - datetime.datetime.now()).total_seconds(), 0) / (365*24*3600)

    def bs_gamma(S,K,T,sigma,r=0,q=0):
        d1 = (np.log(S/K)+(r-q+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        return np.exp(-q*T)*norm.pdf(d1)/(S*sigma*np.sqrt(T))

    rows=[]
    for _, r in df_o.dropna(subset=["impliedVolatility","openInterest"]).iterrows():
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
    plot_df = gex_df.assign(net=lambda d: d["gex"].where(d["type"]=="call", -d["gex"]))
    plot_df = plot_df.groupby("strike")["net"].sum().reset_index()
    ax.bar(
        plot_df["strike"], plot_df["net"],
        width=np.diff(plot_df["strike"]).mean()*0.8 or 1,
        color=np.where(plot_df["net"]>=0, "lime", "tomato")
    )
    ax.axhline(0, color="white")
    ax.set_xlabel("Strike", color="white"); ax.set_ylabel("Net GEX", color="white")
    ax.set_title(f"{options_underlying} Net GEX by Strike (exp {selected_expiry})", color="white")
    ax.set_facecolor("black"); fig.patch.set_facecolor("black")
    ax.tick_params(colors="white"); st.pyplot(fig)
else:
    st.info("ℹ️ Skipping GEX (no data).")

# — News Summary —
st.subheader(f"📰 Latest News about {stock.upper()}")
def fetch_news(tkr):
    url  = f"https://news.google.com/search?q={tkr}+stock&hl=en-US&gl=US&ceid=US%3Aen"
    r    = requests.get(url, timeout=5)
    soup = BeautifulSoup(r.text, "html.parser")
    return ["https://news.google.com"+a["href"][1:] for a in soup.select("a.DY5T1d")][:3]

for link in fetch_news(stock):
    try:
        from newspaper import Article
        art = Article(link); art.download(); art.parse(); art.nlp()
        st.markdown(f"### [{art.title}]({link})"); st.write(art.summary); st.markdown("---")
    except:
        st.info("⚠️ Unable to parse article preview.")

