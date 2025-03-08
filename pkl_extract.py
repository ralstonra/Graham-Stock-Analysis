import pickle
import os

# Load NYSE tickers
if os.path.exists('nyse_tickers.pkl'):
    with open('nyse_tickers.pkl', 'rb') as f:
        nyse_tickers = pickle.load(f)
    print("NYSE Tickers:", sorted(list(nyse_tickers))[:10], "...")  # Show first 10 for brevity
    print(f"Total NYSE Tickers: {len(nyse_tickers)}")

# Load NASDAQ tickers
if os.path.exists('nasdaq_tickers.pkl'):
    with open('nasdaq_tickers.pkl', 'rb') as f:
        nasdaq_tickers = pickle.load(f)
    print("NASDAQ Tickers:", sorted(list(nasdaq_tickers))[:10], "...")  # Show first 10 for brevity
    print(f"Total NASDAQ Tickers: {len(nasdaq_tickers)}")