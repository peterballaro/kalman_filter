import yfinance as yf
import pandas as pd

tickers = [
    "FPE", "PFF", "PGX", "PSK", "PFFA", "PFXF", "VRP", "HYG", "ICVT", "LQD", 'IEF', "TLT",
]
raw = yf.download(tickers, start="2012-01-01", auto_adjust=False)
adj_close = raw["Adj Close"]
returns = adj_close.pct_change()
returns.to_csv("data/adj_close_returns_etfs.csv")
print(returns.head())

etf_names = {
etf_names = {
    "FPE": "First Trust Preferred Securities and Income ETF",
    "PFF": "iShares Preferred and Income Securities ETF",
    "PGX": "Invesco Preferred ETF",
    "PSK": "SPDR Wells Fargo Preferred Stock ETF",
    "PFFA": "Virtus InfraCap U.S. Preferred Stock ETF",
    "PFXF": "VanEck Preferred Securities ex Financials ETF",
    "VRP": "Invesco Variable Rate Preferred ETF",
    "HYG": "iShares iBoxx $ High Yield Corporate Bond ETF",
    "ICVT": "iShares Convertible Bond ETF",
    "LQD": "iShares iBoxx $ Investment Grade Corporate Bond ETF",
    "IEF": "iShares 7-10 Year Treasury Bond ETF",
    "TLT": "iShares 20+ Year Treasury Bond ETF"
}
}
