import sqlite3
import pickle
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)

def extract_cached_data(tickers, db_path="api_cache.db"):
    if not os.path.exists(db_path):
        logging.error(f"Cache database {db_path} not found.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for ticker in tickers:
        ticker = ticker.upper()
        logging.info(f"Extracting cached data for {ticker}")

        # Query all cached data for this ticker
        cursor.execute("SELECT endpoint, service, data, timestamp FROM api_cache WHERE ticker=?", (ticker,))
        results = cursor.fetchall()

        for endpoint, service, data_blob, timestamp in results:
            try:
                data = pickle.loads(data_blob)
                logging.info(f"Ticker: {ticker}, Endpoint: {endpoint}, Service: {service}, Timestamp: {timestamp}")
                if isinstance(data, (pd.DataFrame, pd.Series)):
                    print(f"\nData for {ticker} ({endpoint}, {service}):\n{data.head().to_dict() if not data.empty else 'Empty DataFrame'}\n")
                elif isinstance(data, dict):
                    print(f"\nData for {ticker} ({endpoint}, {service}):\n{data}\n")
                elif isinstance(data, list):
                    print(f"\nData for {ticker} ({endpoint}, {service}):\n{data[:5] if data else 'Empty List'}\n")  # Show first 5 items or empty
            except Exception as e:
                logging.error(f"Error unpickling data for {ticker}, {endpoint}, {service}: {str(e)}")

    conn.close()

if __name__ == "__main__":
    tickers_to_check = ["AAON", "ACLS", "ALNT"]  # Adjust as needed for more tickers
    extract_cached_data(tickers_to_check)