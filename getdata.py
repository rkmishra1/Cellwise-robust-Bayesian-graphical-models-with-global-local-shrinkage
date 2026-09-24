"""Download NIFTY 50 + NIFTY Next 50 constituents + index from Yahoo Finance,
build a daily log-return matrix for the application section."""
import json, time
import numpy as np
import pandas as pd
import yfinance as yf

NIFTY50 = """ADANIENT ADANIPORTS APOLLOHOSP ASIANPAINT AXISBANK BAJAJ-AUTO BAJFINANCE BAJAJFINSV
BEL BHARTIARTL CIPLA COALINDIA DRREDDY EICHERMOT ETERNAL GRASIM HCLTECH HDFCBANK HDFCLIFE HINDALCO
HINDUNILVR ICICIBANK INDIGO INFY ITC JIOFIN JSWSTEEL KOTAKBANK LT M&M MARUTI MAXHEALTH NESTLEIND
NTPC ONGC POWERGRID RELIANCE SBILIFE SHRIRAMFIN SBIN SUNPHARMA TCS TATACONSUM TMPV TATASTEEL TECHM
TITAN TRENT ULTRACEMCO WIPRO""".split()

NEXT50 = """ABB ADANIENSOL ADANIGREEN ADANIPOWER AMBUJACEM BAJAJHLDNG BANKBARODA BPCL BRITANNIA
BOSCHLTD CANBK CGPOWER CHOLAFIN CUMMINSIND DIVISLAB DLF DMART GAIL GODREJCP HDFCAMC HAL HINDZINC
HYUNDAI INDHOTEL IOC IRFC JINDALSTEL LODHA LTM MAZDOCK MOTHERSON MUTHOOTFIN PIDILITIND PFC PNB
RECLTD SHREECEM SIEMENS ENRIN SOLARINDS TATACAP TMCV TATAPOWER TORNTPHARM TVSMOTOR UNIONBANK
UNITDSPR VBL VEDL ZYDUSLIFE""".split()

TICKERS = [t + ".NS" for t in NIFTY50 + NEXT50]

def main():
    print(f"downloading {len(TICKERS)+1} symbols", flush=True)
    px = yf.download(TICKERS + ["^NSEI"], start="2023-06-01", auto_adjust=True,
                     progress=False)["Close"]
    px = px.dropna(how="all")
    # keep ^NSEI aside
    idx = px["^NSEI"]
    stocks = px.drop(columns="^NSEI")
    # coverage filter: at most 5 missing days in the sample
    good = [c for c in stocks.columns if stocks[c].notna().sum() >= 0.95 * len(stocks)]
    print(f"{len(good)}/{len(stocks.columns)} stocks pass coverage filter", flush=True)
    stocks = stocks[good].ffill().dropna()
    idx = idx.loc[stocks.index].ffill()
    rets = np.log(stocks).diff().dropna()
    mkt = np.log(idx).diff().dropna()
    rets = rets.loc[mkt.index]
    print("returns matrix:", rets.shape, flush=True)
    rets.to_csv("returns.csv")
    mkt.to_csv("market.csv", header=True)
    with open("universe.json", "w") as f:
        json.dump(list(rets.columns), f)
    print("date range:", rets.index[0], "->", rets.index[-1], flush=True)

if __name__ == "__main__":
    main()
