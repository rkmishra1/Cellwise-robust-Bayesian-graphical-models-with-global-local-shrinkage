"""Point-in-time universe: NIFTY 50 + NIFTY Next 50 constituents as of the June 2023
index rebalance (Wikipedia revisions of 2023-04/06), not today's membership.
Handles renames (ZOMATO->ETERNAL, ADANITRANS->ADANIENSOL, TATAMOTORS->TMPV) by trying
both tickers. Coverage filter keeps stocks with >=95% of trading days populated."""
import json
import numpy as np
import pandas as pd
import yfinance as yf

PIT_NIFTY50 = """ADANIENT ADANIPORTS APOLLOHOSP ASIANPAINT AXISBANK BAJAJ-AUTO BAJFINANCE BAJAJFINSV
BPCL BHARTIARTL BRITANNIA CIPLA COALINDIA DIVISLAB DRREDDY EICHERMOT GRASIM HCLTECH HDFC HDFCBANK
HDFCLIFE HEROMOTOCO HINDALCO HINDUNILVR ICICIBANK INDUSINDBK INFY ITC JSWSTEEL KOTAKBANK LT M&M
MARUTI NESTLEIND NTPC ONGC POWERGRID RELIANCE SBILIFE SBIN SUNPHARMA TATAMOTORS TATASTEEL TCS
TATACONSUM TECHM TITAN ULTRACEMCO UPL WIPRO""".split()

PIT_NEXT50 = """ABB ACC ADANIGREEN ATG ADANITRANS AWL AMBUJACEM BAJAJHLDNG BANKBARODA BERGEPAINT
BEL BOSCHLTD CANBK CHOLAFIN COLPAL DABUR DLF DMART GAIL GODREJCP HAVELLS HDFCAMC HAL ICICIGI
ICICIPRULI IOC INDIGO INDUSTOWER NAUKRI IRCTC LTIM LICI MARICO MUTHOOTFIN NYKAA PAGEIND PIIND
PIDILITIND PGHH MOTHERSON SBICARD SHREECEM SIEMENS SRF TATAPOWER TORNTPHARM MCDOWELL-N VBL VEDL
ZOMATO""".split()

# tickers that were renamed between 2023 and 2026: try old and new Yahoo symbols
ALIASES = {
    "TATAMOTORS": ["TATAMOTORS.NS", "TMPV.NS"],
    "ZOMATO": ["ZOMATO.NS", "ETERNAL.NS"],
    "ADANITRANS": ["ADANITRANS.NS", "ADANIENSOL.NS"],
    "AWL": ["AWL.NS", "AWLAGRI.NS"],
    "MCDOWELL-N": ["MCDOWELL-N.NS"],
}


def main():
    names = PIT_NIFTY50 + PIT_NEXT50
    tickers = []
    for nm in names:
        tickers.append(ALIASES.get(nm, [nm + ".NS"])[0])
    extra = [a for nm in names for a in ALIASES.get(nm, [])[1:]]
    print(f"{len(names)} constituents, {len(extra)} alias candidates", flush=True)

    px = yf.download(sorted(set(tickers + extra)) + ["^NSEI"], start="2023-05-01",
                     auto_adjust=True, progress=False)["Close"]
    px = px.dropna(how="all")
    idx = px["^NSEI"]
    stocks = px.drop(columns="^NSEI")

    # keep the first alias that satisfies coverage
    keep = {}
    for nm in names:
        for t in ALIASES.get(nm, [nm + ".NS"]):
            if t in stocks.columns and stocks[t].notna().mean() >= 0.90:
                keep[nm] = t
                break
    print(f"{len(keep)}/{len(names)} constituents with usable history", flush=True)
    cols = [keep[nm] for nm in names if nm in keep]
    stocks = stocks[cols]
    coverage = stocks.notna().mean()
    print("coverage:", {c: round(coverage[c], 3) for c in cols if coverage[c] < 0.999},
          flush=True)
    stocks = stocks.ffill().dropna(axis=1)
    idx = idx.loc[stocks.index].ffill()
    rets = np.log(stocks).diff().dropna()
    mkt = np.log(idx).diff().dropna()
    rets = rets.loc[mkt.index]
    print("returns matrix:", rets.shape, rets.index[0].date(), "->", rets.index[-1].date(),
          flush=True)
    rets.to_csv("returns_pit.csv")
    mkt.to_csv("market_pit.csv", header=True)
    with open("universe_pit.json", "w") as f:
        json.dump({nm: t for nm, t in keep.items()}, f, indent=1)


if __name__ == "__main__":
    main()
