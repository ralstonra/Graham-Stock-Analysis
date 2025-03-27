import os
import requests
import time
from datetime import datetime

# Get API key from environment variable
API_KEY = os.getenv("FMP_API_KEY")
if not API_KEY:
    raise ValueError("API key not found. Please set the FMP_API_KEY environment variable.")

# List of tickers to test (modify this list with your problem stocks)
TICKERS = ["KO", "JPM", "IBM"]

# Base URL for the FMP API endpoint
BASE_URL = "https://financialmodelingprep.com/api/v3/financial-statement-full-as-reported/"

# Directory to save the response files
LOG_DIR = "fmp_responses"
os.makedirs(LOG_DIR, exist_ok=True)

# Function to fetch and save raw API response for a ticker
def fetch_and_log_response(ticker):
    url = f"{BASE_URL}{ticker}?apikey={API_KEY}"
    try:
        response = requests.get(url)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if response.status_code == 200:
            # Save the raw response to a file
            file_path = os.path.join(LOG_DIR, f"{ticker}_response.json")
            with open(file_path, "w") as f:
                f.write(response.text)
            print(f"{timestamp} - Saved response for {ticker} to {file_path}")
        else:
            # Log error details if the request fails
            print(f"{timestamp} - Error for {ticker}: Status code {response.status_code}, Response: {response.text}")
    except Exception as e:
        # Log any exceptions that occur during the request
        print(f"{timestamp} - Exception for {ticker}: {str(e)}")

# Fetch responses for each ticker with a delay to avoid rate limiting
for ticker in TICKERS:
    fetch_and_log_response(ticker)
    time.sleep(1)  # 1-second delay between requests

print("Finished fetching responses for all tickers.")