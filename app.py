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
import pmdarima as pm
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# 页面配置
st.set_page_config(page_title="Stock Dashboard", layout="wide", initial_sidebar_state="expanded")
st.markdown("<h1 style='text-align: center; color: white;'>📈 Real-Time Stock Monitoring Dashboard (Pro Version)</h1>", unsafe_allow_html=True)

# Sidebar 控制区
with st.sidebar:
    st.header("⚙️ Controls")
    stock = st.text_input("Enter Stock/Crypto Ticker:", value="AAPL")
    period = st.selectbox("Select Period:", ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"])
    interval = st.selectbox("Select Interval:", ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1d", "1wk", "1mo"])
    refresh_time = st.slider("Refresh Interval (Seconds):", 10, 300, 60)
    threshold_mode = st.radio("Select Signal Sensitivity:", ["High Volatility (0.5%)", "Low Volatility (0.15%)"])
    threshold = 0.005 if threshold_mode == "High Volatility (0.5%)" else 0.0015

st_autorefresh(interval=refresh_time * 1000, key="refresh")

# 数据加载函数
@st.cache_data(ttl=refresh_time)
def load_data(ticker, period, interval):
    data = yf.Ticker(ticker).history(period=period, interval=interval)
    info = yf.Ticker(ticker).info
    return data, info

# 校验合法组合
def validate_combo(period, interval):
    minute_intervals = {"1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60, "60m": 730, "90m": 60}
    if interval in minute_intervals:
        if period.endswith("y") or period in ["max", "ytd", "10y", "5y"]:
            return False
        if interval == "1m" and period not in ["1d", "5d", "7d"]:
            return False
    return True

if not validate_combo(period, interval):
    st.warning("⚠️ Unsupported period + interval combination.")
    st.stop()

# 获取数据
data, info = load_data(stock.upper(), period, interval)

# 公司信息展示
st.subheader("🏢 Company Overview")
try:
    domain = info.get('website', '').split('//')[-1].split('/')[0]
    logo_url = f"https://logo.clearbit.com/{domain}"
    st.image(logo_url, width=120)
    st.markdown(f"### {info.get('shortName', stock.upper())}")
    if info.get('website'):
        st.markdown(f"[Official Site]({info['website']})")
except:
    st.warning("⚠️ Unable to load logo or website.")

# 价格图 + MA10 + VWAP
st.subheader(f"📈 {stock.upper()} Price with MA10 & VWAP")
data['MA10'] = data['Close'].rolling(10).mean()
vwap = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
fig1, ax1 = plt.subplots(figsize=(14,6))
ax1.plot(data.index, data['Close'], label='Close', color='cyan')
ax1.plot(data.index, data['MA10'], label='MA10', color='orange')
ax1.plot(data.index, vwap, label='VWAP', color='green')
ax1.set_facecolor('black'); fig1.patch.set_facecolor('black')
ax1.tick_params(colors='white'); ax1.legend(); st.pyplot(fig1)

# 成交量图
st.subheader(f"📊 {stock.upper()} Volume")
fig2, ax2 = plt.subplots(figsize=(14,3))
ax2.bar(data.index, data['Volume'], color='purple')
ax2.set_facecolor('black'); fig2.patch.set_facecolor('black')
ax2.tick_params(colors='white'); st.pyplot(fig2)

# 预测与信号 + 导出CSV
st.subheader(f"🔮 {stock.upper()} 30-Minute Price Prediction")
model_choice = st.radio("Model:", ["Auto-ARIMA", "LSTM", "Prophet"], horizontal=True)
close_series = data['Close'].dropna()
# 最少需要的数据点数（1分钟间隔约需200分钟，视需求可调整）
min_data_points = 60  # 60分钟数据即可开始预测，缩短等待时间
if len(close_series) >= min_data_points:
    window = 300
    series = close_series.tail(window)
    future_index = pd.date_range(start=series.index[-1], periods=30, freq='T')

    if model_choice == "Auto-ARIMA":
        st.info("Training Auto-ARIMA...")
        arima = pm.auto_arima(series, start_p=1, start_q=1, max_p=5, max_q=5,
                              seasonal=False, stepwise=True, suppress_warnings=True)
        forecast_vals = arima.predict(n_periods=30)
    elif model_choice == "LSTM":
        st.info("Training LSTM...")
        scaler = MinMaxScaler(); scaled = scaler.fit_transform(series.values.reshape(-1,1))
        X, y = [], []
        for i in range(60, len(scaled)):
            X.append(scaled[i-60:i,0]); y.append(scaled[i,0])
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1],1))
        m = Sequential(); m.add(LSTM(50, return_sequences=True, input_shape=(60,1)))
        m.add(LSTM(50)); m.add(Dense(1)); m.compile('adam','mse'); m.fit(X,y,epochs=10,batch_size=8,verbose=0)
        inputs = scaled[-60:]
        preds=[]
        for _ in range(30):
            p = m.predict(inputs.reshape(1,60,1),verbose=0)
            preds.append(p[0,0]); inputs = np.append(inputs, p)[-60:]
        forecast_vals = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    else:
        st.info("Training Prophet...")
        dfp = pd.DataFrame({'ds': series.index.tz_localize(None), 'y': series.values})
        mp = Prophet(daily_seasonality=True); mp.fit(dfp)
        fut = mp.make_future_dataframe(periods=30, freq='min')
        res = mp.predict(fut); forecast_vals = res['yhat'].iloc[-30:].values

    # 画图和信号，并显示误差指标
    change = (forecast_vals[-1] - series.iloc[-1]) / series.iloc[-1]
    fig3, ax3 = plt.subplots(figsize=(14,6))
    ax3.plot(series.index, series.values, color='cyan', label='Hist')
    ax3.plot(future_index, forecast_vals, color='red', label='Forecast')
    if change > threshold:
        ax3.annotate('📈Buy', xy=(future_index[-1], forecast_vals[-1]),
                     xytext=(future_index[-1], forecast_vals[-1]*1.005),
                     arrowprops=dict(facecolor='green'))
        st.success('📈 Buy Signal')
    elif change < -threshold:
        ax3.annotate('📉Sell', xy=(future_index[-1], forecast_vals[-1]),
                     xytext=(future_index[-1], forecast_vals[-1]*0.995),
                     arrowprops=dict(facecolor='red'))
        st.error('📉 Sell Signal')
    else:
        st.info('⏸ Hold')
    ax3.set_facecolor('black'); fig3.patch.set_facecolor('black')
    ax3.tick_params(colors='white'); ax3.legend(); st.pyplot(fig3)

    # 显示 RMSE 和 MAPE
    y_true = series.tail(30).values
    y_pred = forecast_vals
    st.markdown(f"**RMSE:** {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
    st.markdown(f"**MAPE:** {mean_absolute_percentage_error(y_true, y_pred)*100:.2f}%")

    # 导出CSV
    df_out = pd.DataFrame({'Datetime': future_index, 'Predicted_Close': forecast_vals})
    st.download_button('📥 Download Forecast CSV',
                       df_out.to_csv(index=False).encode('utf-8'),
                       file_name=f"{stock.upper()}_forecast.csv",
                       mime='text/csv')
else:
    st.info(f"⌛ Waiting for at least {min_data_points} data points (current: {len(close_series)})...")

# 新闻摘要
st.subheader(f"📰 News about {stock.upper()}")
def fetch_news(t):
    url = f"https://news.google.com/search?q={t}+stock&hl=en-US&gl=US&ceid=US%3Aen"
    s = requests.get(url); sp = BeautifulSoup(s.text,'html.parser')
    links = sp.select('a.DY5T1d')
    return ["https://news.google.com"+l['href'][1:] for l in links][:3]
for link in fetch_news(stock):
    art = Article(link); art.download(); art.parse(); art.nlp()
    st.markdown(f"### [{art.title}]({link})")
    st.write(art.summary)
    st.markdown('---')

# ====================
# Dockerization Instructions
# ====================
# 1. 创建 requirements.txt，内容如下：
# streamlit
# yfinance
# pandas
# numpy
# matplotlib
# scikit-learn
# streamlit-autorefresh
# requests
# beautifulsoup4
# newspaper3k
# lxml_html_clean
# prophet
# pmdarima
# tensorflow

# 2. Dockerfile 示例：
# ---------------------------------
# FROM python:3.10-slim
# ENV PYTHONUNBUFFERED=1
# WORKDIR /app
# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt
# COPY app.py ./
# EXPOSE 8501
# CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
# ---------------------------------

# 3. 构建并运行 Docker 镜像：
#   docker build -t stock-dashboard .
#   docker run -d -p 8501:8501 --name stock-dashboard stock-dashboard

# 完成！在浏览器访问 http://localhost:8501 即可看到仪表盘。


