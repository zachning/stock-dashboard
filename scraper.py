import yfinance as yf
import pandas as pd

# Pull data
ticker = yf.Ticker("AAPL")  # You can change "AAPL" to "TSLA" or "BTC-USD"
hist = ticker.history(period="1d", interval="1m")  # 1-day, 1-minute intervals

# Save to CSV
hist.to_csv("aapl_data.csv")
print("✅ Stock data saved successfully!")
