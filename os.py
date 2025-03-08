import asyncio
import logging
from graham_data import fetch_historical_data  # Adjust import based on your file structure
import os

# Configure logging (similar to os.py)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nyse_graham_screen.log', mode='w'),  # Overwrite log file
        logging.StreamHandler()  # Also print to console
    ]
)

# Global set to prevent duplicate fetches
FETCHED_TICKERS = set()

async def run_test():
    ticker = "F"
    exchange = "Stock"
    roe_10y, rotc_10y, eps_10y, div_10y, years = await fetch_historical_data(ticker, exchange)
    logging.info(f"Test results for {ticker}: ROE={roe_10y[:5]}..., ROIC={rotc_10y[:5]}..., EPS={eps_10y[:5]}..., Dividends={div_10y[:5]}..., Years={years}")
    print(f"Test results for {ticker}: ROE={roe_10y[:5]}..., ROIC={rotc_10y[:5]}..., EPS={eps_10y[:5]}..., Dividends={div_10y[:5]}..., Years={years}")

if __name__ == "__main__":
    asyncio.run(run_test())