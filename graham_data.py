import asyncio
import logging
import pandas as pd
import yfinance as yf
import aiohttp
import time
import sqlite3
import os
import pickle
import hashlib
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from fmp_python.fmp import FMP
from alpha_vantage.fundamentaldata import FundamentalData
from config import (BASE_DIR, CACHE_DB, NYSE_TICKERS_FILE, NASDAQ_TICKERS_FILE, NYSE_LIST_FILE,
                   NASDAQ_LIST_FILE, FMP_API_KEYS, ALPHA_VANTAGE_API_KEYS, CACHE_EXPIRY)

# Throttling configuration
MAX_CALLS_PER_MINUTE = 300
CALLS_PER_TICKER = 1
MAX_TICKERS_PER_MINUTE = MAX_CALLS_PER_MINUTE // CALLS_PER_TICKER
SECONDS_PER_MINUTE = 60
DELAY_BETWEEN_CALLS = SECONDS_PER_MINUTE / MAX_CALLS_PER_MINUTE
MAX_CONCURRENT_TICKERS = 10

# Database lock
DB_LOCK = asyncio.Lock()

# Logging setup (consistent with config.py)
logging.basicConfig(level=logging.DEBUG, filename=os.path.join(BASE_DIR, 'nyse_graham_screen.log'),
                    format='%(asctime)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logging.getLogger().addHandler(console)

# Global ticker sets (initialized later)
VALID_NYSE_TICKERS = None
VALID_NASDAQ_TICKERS = None

# Cached ticker data
def load_cached_data(cache_file):
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logging.error(f"Error loading {cache_file}: {str(e)}")
    return []

NYSE_CACHED_DATA = load_cached_data(NYSE_TICKERS_FILE)
NASDAQ_CACHED_DATA = load_cached_data(NASDAQ_TICKERS_FILE)

def get_file_hash(file_path):
    if not os.path.exists(file_path):
        logging.warning(f"{file_path} not found. Returning None.")
        return None
    try:
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            while chunk := f.read(4096):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logging.error(f"Failed to compute hash for {file_path}: {str(e)}")
        return None

async def _make_request(url: str, params: dict = None, headers: dict = None, timeout: int = 10) -> tuple[int, str]:
    """Make an asynchronous HTTP request with a timeout."""
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.get(url, params=params, headers=headers) as response:
                return response.status, await response.text()
    except Exception as e:
        logging.error(f"Request failed: {str(e)}")
        return 500, ""

def calculate_graham_value(earnings, expected_return=0.0):
    if not earnings or earnings <= 0:
        return None
    growth_rate = 0.05
    return earnings * (8.5 + 2 * growth_rate) * (4.4 / expected_return) if expected_return > 0 else None

def calculate_graham_score(ticker, price, pe_ratio, debt_to_equity, roe_10y, eps_10y):
    score = 0
    if pe_ratio and 0 < pe_ratio <= 15:
        score += 1
    if debt_to_equity and 0 <= debt_to_equity <= 1.2:
        score += 1
    if roe_10y and len(roe_10y) > 0 and sum(roe_10y) / len(roe_10y) > 10:
        score += 1
    if eps_10y and len(eps_10y) > 1:
        growth = (eps_10y[-1] - eps_10y[0]) / eps_10y[0] * 100 if eps_10y[0] != 0 else 0
        if growth > 0:
            score += 1
    stock = yf.Ticker(ticker)
    balance_sheet = stock.balance_sheet
    if not balance_sheet.empty:
        current_assets = balance_sheet.loc['Total Current Assets'].dropna().iloc[0] if 'Total Current Assets' in balance_sheet.index else 0
        current_liabilities = balance_sheet.loc['Total Current Liabilities'].dropna().iloc[0] if 'Total Current Liabilities' in balance_sheet.index else 0
        current_ratio = current_assets / current_liabilities if current_liabilities > 0 else 0
        if current_ratio > 2:
            score += 1
    return score

def calculate_total_graham_score(graham_score, rotc_10y, div_10y):
    total_score = graham_score
    if rotc_10y and len(rotc_10y) > 0 and sum(rotc_10y) / len(rotc_10y) > 10:
        total_score += 1
    if div_10y and len(div_10y) > 0 and sum(div_10y) / len(div_10y) > 0:
        total_score += 1
    return total_score

def get_stocks_connection():
    conn = sqlite3.connect(CACHE_DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS stocks
                     (ticker TEXT, date TEXT, price REAL, roe REAL, rotc REAL, eps REAL, dividend REAL, graham_score INTEGER, intrinsic_value REAL, buy_price REAL, sell_price REAL)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS graham_qualifiers
                     (ticker TEXT PRIMARY KEY, graham_score INTEGER, date TEXT, sector TEXT, exchange TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS screening_progress
                     (exchange TEXT, ticker TEXT, timestamp TEXT, file_hash TEXT, status TEXT, PRIMARY KEY (exchange, ticker))''')
    cursor.execute("PRAGMA table_info(screening_progress)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'status' not in columns:
        logging.info("Adding 'status' column to screening_progress table")
        cursor.execute("ALTER TABLE screening_progress ADD COLUMN status TEXT")
    conn.commit()
    return conn, cursor

async def load_and_filter_tickers(file_path, exchange_filter=None):
    """Asynchronously load and filter tickers from a file."""
    if not os.path.exists(file_path):
        logging.error(f"Ticker file not found: {file_path}")
        return []

    tickers = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if not lines:
                logging.error(f"Ticker file is empty: {file_path}")
                return []

            header = lines[0].strip().split('|')
            logging.debug(f"File header: {header}")
            ticker_col = header.index('ACT Symbol') if 'ACT Symbol' in header else header.index('Symbol')
            exchange_col = header.index('Exchange') if 'Exchange' in header and exchange_filter else None
            market_col = header.index('Market Category') if 'Market Category' in header else None
            etf_col = header.index('ETF')
            test_col = header.index('Test Issue')

            for line in lines[1:]:
                cols = line.strip().split('|')
                if len(cols) <= max(ticker_col, etf_col, test_col, exchange_col or market_col or 0):
                    logging.warning(f"Malformed line skipped: {line.strip()}")
                    continue
                ticker = cols[ticker_col]
                if exchange_filter and exchange_col is not None:
                    if (cols[exchange_col] == exchange_filter and
                        cols[etf_col] == 'N' and cols[test_col] == 'N'):
                        tickers.append(ticker)
                elif market_col is not None:
                    if (cols[market_col] in ['Q', 'G', 'S'] and
                        cols[etf_col] == 'N' and cols[test_col] == 'N'):
                        tickers.append(ticker)
        
        logging.info(f"Loaded {len(tickers)} tickers from {file_path}")
    except Exception as e:
        logging.error(f"Error parsing {file_path}: {str(e)}")
    return tickers

async def initialize_tickers():
    """Initialize global ticker sets asynchronously."""
    global VALID_NYSE_TICKERS, VALID_NASDAQ_TICKERS
    if VALID_NYSE_TICKERS is None or VALID_NASDAQ_TICKERS is None:
        nyse_tickers = await load_and_filter_tickers(NYSE_LIST_FILE, exchange_filter='N')
        nasdaq_tickers = await load_and_filter_tickers(NASDAQ_LIST_FILE)
        VALID_NYSE_TICKERS = set(nyse_tickers)
        VALID_NASDAQ_TICKERS = set(nasdaq_tickers)
        logging.info(f"Initialized VALID_NYSE_TICKERS with {len(VALID_NYSE_TICKERS)} tickers")
        logging.info(f"Initialized VALID_NASDAQ_TICKERS with {len(VALID_NASDAQ_TICKERS)} tickers")

async def filter_common_stocks(file_path, exchange_filter=None):
    """Asynchronously filter common stocks from a file, returning all available data."""
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return []

    common_stocks = []
    total_lines = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]  # Skip header
            total_lines = len(lines)
            logging.info(f"Reading {file_path}: {total_lines} lines found (excluding header)")
            for i, line in enumerate(tqdm(lines, desc=f"Filtering {os.path.basename(file_path)}")):
                if not line.strip() or line.strip().startswith("File Creation Time:"):
                    logging.debug(f"Skipped line {i} in {file_path}: Non-ticker data - {line.strip()}")
                    continue

                fields = line.strip().split('|')
                if len(fields) < 8:
                    logging.warning(f"Line {i} in {file_path} has {len(fields)} fields, expected 8: {line.strip()}")
                    continue

                ticker, security_name, exchange, is_etf = fields[0].strip(), fields[1].strip(), fields[2].strip(), fields[5].strip()
                logging.debug(f"Processing ticker: {ticker}, exchange: {exchange}, is_etf: {is_etf}, security_name: {security_name}")

                if file_path == NYSE_LIST_FILE and exchange_filter and exchange != exchange_filter:
                    logging.debug(f"Excluded {ticker}: Exchange '{exchange}' does not match filter '{exchange_filter}'")
                    continue

                if (is_etf == 'Y' or 
                    any(keyword in security_name.lower() for keyword in ['preferred', 'warrant', 'units', 'note', 'bond', 'debenture', 'trust', 'right', 'depositary', 'series']) or 
                    ('common stock' not in security_name.lower() and 'ordinary shares' not in security_name.lower() and 'common shares' not in security_name.lower()) or 
                    not ticker or not ticker.replace('+', '').replace('=', '').isalnum()):
                    logging.debug(f"Excluded {ticker}: Non-company stock or invalid format - Security Name: {security_name}, ETF: {is_etf}, Ticker Check: {ticker}")
                    continue

                historical_data = await fetch_historical_data(ticker, file_path == NYSE_LIST_FILE and "NYSE" or "NASDAQ")
                roe_10y, rotc_10y, eps_10y, div_10y, years = historical_data
                logging.debug(f"Historical data for {ticker}: years={len(years)}, roe_10y={len(roe_10y)}, rotc_10y={len(rotc_10y)}, eps_10y={len(eps_10y)}, div_10y={len(div_10y)}")
                common_stocks.append({
                    "ticker": ticker,
                    "historical_data": {
                        "roe_10y": roe_10y,
                        "rotc_10y": rotc_10y,
                        "eps_10y": eps_10y,
                        "div_10y": div_10y,
                        "years": years
                    }
                })
                logging.debug(f"Included {ticker}: Company stock with available historical data")

    except Exception as e:
        logging.error(f"Error reading or parsing {file_path}: {str(e)}")
        return []

    logging.info(f"Processed {total_lines} lines in {file_path}, filtered to {len(common_stocks)} company stocks with available data")
    return common_stocks

# Global execution guard for fetch_historical_data
FETCHED_TICKERS = set()

async def fetch_historical_data(ticker, exchange="Stock"):
    """Fetch historical financial data for a ticker from Yahoo or FMP, using a single source."""
    if ticker in FETCHED_TICKERS:
        logging.warning(f"Skipping duplicate fetch for {ticker} ({exchange})")
        return [0.0] * 10, [0.0] * 10, [0.0] * 10, [0.0] * 10, list(range(2015, 2025))
    FETCHED_TICKERS.add(ticker)
    logging.debug(f"Fetching historical data for {ticker} ({exchange})")
    common_years = []
    try:
        # Yahoo Finance block
        stock = yf.Ticker(ticker)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 10)  # Aim for 10 years
        ohlc = stock.history(start=start_date, end=end_date, interval="1mo", actions=True)

        if ohlc.empty:
            logging.warning(f"No OHLC data for {ticker} from Yahoo Finance. Falling back to FMP.")
            raise ValueError("No OHLC data from Yahoo Finance")

        years_ohlc = sorted(ohlc.index.year.unique())
        logging.debug(f"Years from Yahoo OHLC for {ticker}: {years_ohlc}")

        income_stmt = stock.financials.transpose()
        balance_sheet = stock.balance_sheet.transpose()
        cash_flow = stock.cashflow.transpose()
        logging.debug(f"Financials for {ticker}: income_stmt={income_stmt.shape}, balance_sheet={balance_sheet.columns.tolist()}, cash_flow={cash_flow.shape}")

        total_equity = balance_sheet.get('Total Stockholder Equity', balance_sheet.get('Stockholders Equity', pd.Series())).dropna()
        if total_equity.empty:
            logging.warning(f"No valid equity data found for {ticker} from Yahoo Finance.")
            raise ValueError("No valid equity data from Yahoo Finance")

        net_income = income_stmt.get('Net Income', pd.Series()).dropna()
        invested_capital = balance_sheet.get('Invested Capital', pd.Series()).dropna()
        dividends = ohlc['Dividends'].resample('YE').sum().dropna()

        years_financial = sorted(set(total_equity.index.year) & set(net_income.index.year) & set(invested_capital.index.year))
        if len(years_financial) < 10:
            logging.warning(f"Insufficient financial data from Yahoo Finance for {ticker}: only {len(years_financial)} years. Falling back to FMP.")
            raise ValueError("Insufficient financial data from Yahoo Finance")

        common_years = sorted(set(years_ohlc) & set(years_financial))[-10:]
        roe_10y = []
        rotc_10y = []
        eps_10y = []
        div_10y = []

        for year in common_years:
            year_income = net_income[net_income.index.year == year].iloc[-1] if not net_income.empty else 0.0
            year_equity = total_equity[total_equity.index.year == year].iloc[-1] if not total_equity.empty else 0.0
            year_invested_cap = invested_capital[invested_capital.index.year == year].iloc[-1] if not invested_capital.empty else 0.0
            year_div = dividends[dividends.index.year == year].sum() if not dividends.empty else 0.0

            if year_equity != 0:
                roe_10y.append((year_income / year_equity) * 100 if not pd.isna(year_income) else 0.0)
            else:
                roe_10y.append(0.0)

            if year_invested_cap != 0:
                rotc_10y.append((year_income / year_invested_cap) * 100 if not pd.isna(year_income) else 0.0)
            else:
                rotc_10y.append(0.0)

            shares = stock.info.get('sharesOutstanding', 1e6)
            eps_10y.append((year_income / shares) if not pd.isna(year_income) and shares != 0 else 0.0)
            div_10y.append(year_div if not pd.isna(year_div) else 0.0)

        logging.debug(f"Calculated Yahoo metrics for {ticker}: roe_10y={roe_10y[:5]}..., rotc_10y={rotc_10y[:5]}..., eps_10y={eps_10y[:5]}..., div_10y={div_10y[:5]}...")

    except ValueError as ve:
        logging.warning(f"Yahoo Finance failed for {ticker} ({exchange}): {str(ve)}. Attempting FMP fallback.")
        try:
            for api_key in FMP_API_KEYS:
                try:
                    async with aiohttp.ClientSession() as session:
                        url_income = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?apikey={api_key}"
                        async with session.get(url_income) as response:
                            logging.debug(f"FMP income response status for {ticker}: {response.status}")
                            income_data = await response.json()
                            logging.debug(f"FMP raw income_data for {ticker}: {income_data[:2] if income_data else 'Empty'}...")
                        url_balance = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?apikey={api_key}"
                        async with session.get(url_balance) as response:
                            logging.debug(f"FMP balance response status for {ticker}: {response.status}")
                            balance_data = await response.json()
                            logging.debug(f"FMP raw balance_data for {ticker}: {balance_data[:2] if balance_data else 'Empty'}...")
                        url_cash = f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}?apikey={api_key}"
                        async with session.get(url_cash) as response:
                            logging.debug(f"FMP cash response status for {ticker}: {response.status}")
                            cash_data = await response.json()
                            logging.debug(f"FMP raw cash_data for {ticker}: {cash_data[:2] if cash_data else 'Empty'}...")
                        url_div = f"https://financialmodelingprep.com/api/v3/historical-price-full/stock_dividend/{ticker}?apikey={api_key}"
                        async with session.get(url_div) as response:
                            logging.debug(f"FMP dividend response status for {ticker}: {response.status}")
                            div_data = await response.json()
                            logging.debug(f"FMP raw div_data for {ticker}: {div_data.get('historical', [])[:2] if div_data.get('historical') else 'Empty'}...")

                    # Extract years and financials
                    years = [int(entry['calendarYear']) for entry in income_data if 'calendarYear' in entry and isinstance(entry.get('calendarYear'), (str, int))]
                    logging.debug(f"Extracted FMP years for {ticker}: {years}")
                    if not years:
                        logging.warning(f"No valid years found in FMP income_data for {ticker}")
                        raise ValueError("No valid years in FMP data")
                    net_income = pd.Series({str(int(entry['calendarYear'])): entry.get('netIncome', np.nan) for entry in income_data if 'calendarYear' in entry and isinstance(entry.get('calendarYear'), (str, int))})
                    total_equity = pd.Series({str(int(entry['calendarYear'])): entry.get('totalStockholdersEquity', np.nan) for entry in balance_data if 'calendarYear' in entry and isinstance(entry.get('calendarYear'), (str, int))})
                    invested_capital = pd.Series({str(int(entry['calendarYear'])): entry.get('investedCapital', entry.get('totalAssets', np.nan) - entry.get('totalCurrentLiabilities', np.nan)) for entry in balance_data if 'calendarYear' in entry and isinstance(entry.get('calendarYear'), (str, int))})

                    logging.debug(f"Net income indices: {net_income.index.tolist()}")
                    logging.debug(f"Total equity indices: {total_equity.index.tolist()}")
                    logging.debug(f"Invested capital indices: {invested_capital.index.tolist()}")

                    # Ensure 10-year range from the most recent year
                    years_ohlc = list(range(max(2015, min(years)), min(datetime.now().year + 1, max(years) + 1)))
                    years_financial = [int(year) for year in years if year in years_ohlc and not pd.isna(net_income.get(str(year))) and not pd.isna(total_equity.get(str(year))) and not pd.isna(invested_capital.get(str(year)))]
                    common_years = sorted(set(years_financial) & set(years_ohlc))[-10:] or sorted(years_financial)[-10:] or list(range(max(2015, max(years) - 9), max(years) + 1))

                    # Process dividends: Convert date to datetime
                    roe_10y = []
                    rotc_10y = []
                    eps_10y = []
                    div_10y = []

                    if div_data.get('historical'):
                        div_df = pd.DataFrame(div_data['historical'])
                        div_df['date'] = pd.to_datetime(div_df['date'], errors='coerce')
                        div_df = div_df.set_index('date')
                    else:
                        div_df = pd.DataFrame()

                    for year in common_years:
                        try:
                            year_income = net_income.get(str(year), np.nan)
                            year_equity = total_equity.get(str(year), np.nan)
                            year_invested_cap = invested_capital.get(str(year), np.nan)
                            year_div = div_df[div_df.index.year == year]['dividend'].sum() if not div_df.empty else 0.0

                            if not pd.isna(year_equity) and year_equity != 0:
                                roe_10y.append(year_income / year_equity * 100 if not pd.isna(year_income) else 0.0)
                            else:
                                roe_10y.append(0.0)

                            if not pd.isna(year_invested_cap) and year_invested_cap != 0:
                                rotc_10y.append(year_income / year_invested_cap * 100 if not pd.isna(year_income) else 0.0)
                            else:
                                rotc_10y.append(0.0)

                            shares = stock.info.get('sharesOutstanding', next((entry.get('weightedAverageShsOut') for entry in balance_data if 'weightedAverageShsOut' in entry), 1e6))
                            eps_10y.append(year_income / shares if not pd.isna(year_income) and shares != 0 else 0.0)
                            div_10y.append(year_div if not pd.isna(year_div) else 0.0)
                        except Exception as e:
                            logging.error(f"Error calculating metrics for {ticker} in year {year}: {str(e)}")
                            roe_10y.append(0.0)
                            rotc_10y.append(0.0)
                            eps_10y.append(0.0)
                            div_10y.append(0.0)

                    logging.debug(f"Calculated FMP metrics for {ticker}: roe_10y={roe_10y[:5]}..., rotc_10y={rotc_10y[:5]}..., eps_10y={eps_10y[:5]}..., div_10y={div_10y[:5]}...")
                    break
                except Exception as e:
                    logging.warning(f"FMP API key {api_key} failed for {ticker}: {str(e)}")
                    if api_key == FMP_API_KEYS[-1]:
                        raise
        except Exception as e:
            logging.error(f"FMP failed for {ticker} ({exchange}): {str(e)}")
            common_years = list(range(2015, 2025))  # Default to 10 years
            roe_10y = [0.0] * len(common_years)
            rotc_10y = [0.0] * len(common_years)
            eps_10y = [0.0] * len(common_years)
            div_10y = [0.0] * len(common_years)

    if not common_years:
        logging.warning(f"No years available for {ticker}. Using default range.")
        common_years = list(range(2015, 2025))
        roe_10y = [0.0] * len(common_years)
        rotc_10y = [0.0] * len(common_years)
        eps_10y = [0.0] * len(common_years)
        div_10y = [0.0] * len(common_years)

    logging.info(f"Fetched historical data for {ticker}: ROE={len(roe_10y)}, ROIC={len(rotc_10y)}, EPS={len(eps_10y)}, Dividends={len(div_10y)}, Years={len(common_years)}")
    return roe_10y, rotc_10y, eps_10y, div_10y, common_years

async def fetch_batch_data(tickers, expected_return=0.0, margin_of_safety=0.33, exchange="Stock", valid_tickers=None):
    if valid_tickers is None:
        valid_tickers = set(tickers) if exchange == "Stock" else (VALID_NYSE_TICKERS if exchange == "NYSE" else VALID_NASDAQ_TICKERS)

    results = []
    tasks = []
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TICKERS)

    async def fetch_data(ticker):
        async with semaphore:
            try:
                if ticker not in valid_tickers:
                    logging.warning(f"Ticker {ticker} not in valid {exchange} tickers. Skipping.")
                    return {"ticker": ticker, "exchange": exchange, "error": "Invalid ticker"}

                await asyncio.sleep(DELAY_BETWEEN_CALLS)

                stock = yf.Ticker(ticker)
                info = stock.info
                price = info.get('regularMarketPrice', info.get('previousClose', None))

                if price is None:
                    logging.error(f"No price data for {ticker} from yfinance")
                    return {"ticker": ticker, "exchange": exchange, "error": "No price data"}

                pe_ratio = info.get('trailingPE', None)
                debt_to_equity = info.get('debtToEquity', None)
                if debt_to_equity:
                    debt_to_equity = float(debt_to_equity) / 100 if isinstance(debt_to_equity, (int, float)) else debt_to_equity

                if pe_ratio is None or debt_to_equity is None:
                    fmp_data = await fetch_with_multiple_keys_async(ticker, "quote", FMP_API_KEYS, service="FMP", retries=3, backoff=2)
                    if fmp_data and isinstance(fmp_data, list) and len(fmp_data) > 0:
                        if pe_ratio is None:
                            pe_ratio = fmp_data[0].get('pe', None)
                        if debt_to_equity is None:
                            debt_to_equity = fmp_data[0].get('debtToEquity', None)

                if price is None or (pe_ratio is None and debt_to_equity is None):
                    logging.error(f"Insufficient data for {ticker} from both Yahoo and FMP after retries")
                    return {"ticker": ticker, "exchange": exchange, "error": "Insufficient data"}

                historical_data = await fetch_historical_data(ticker, exchange)
                roe_10y, rotc_10y, eps_10y, div_10y, years = historical_data

                earnings = price / pe_ratio if pe_ratio else None
                intrinsic_value = calculate_graham_value(earnings, expected_return) if earnings else None
                buy_price = intrinsic_value * (1 - margin_of_safety) if intrinsic_value else None
                sell_price = intrinsic_value * 1.5 if intrinsic_value else None

                graham_score = calculate_graham_score(ticker, price, pe_ratio, debt_to_equity, roe_10y, eps_10y)
                total_graham_score = calculate_total_graham_score(graham_score, rotc_10y, div_10y) if graham_score else 0

                return {
                    "ticker": ticker,
                    "exchange": exchange,
                    "price": price,
                    "pe_ratio": pe_ratio,
                    "debt_to_equity": debt_to_equity,
                    "intrinsic_value": intrinsic_value,
                    "buy_price": buy_price,
                    "sell_price": sell_price,
                    "graham_score": graham_score,
                    "total_graham_score": total_graham_score,
                    "years": years,
                    "roe_10y": roe_10y,
                    "rotc_10y": rotc_10y,
                    "eps_10y": eps_10y,
                    "div_10y": div_10y
                }
            except Exception as e:
                logging.error(f"Error processing {ticker} ({exchange}): {str(e)}")
                return {"ticker": ticker, "exchange": exchange, "error": str(e)}

    for ticker in tickers:
        tasks.append(fetch_data(ticker))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if isinstance(r, dict)]

async def fetch_stock_data(ticker, exchange="Stock", expected_return=0.0, margin_of_safety=0.33):
    try:
        await asyncio.sleep(DELAY_BETWEEN_CALLS)

        stock = yf.Ticker(ticker)
        info = stock.info
        price = info.get('regularMarketPrice', info.get('previousClose', None))

        if price is None:
            logging.error(f"No price data for {ticker} from yfinance")
            return {"ticker": ticker, "exchange": exchange, "error": "No price data"}

        pe_ratio = info.get('trailingPE', None)
        debt_to_equity = info.get('debtToEquity', None)
        if debt_to_equity:
            debt_to_equity = float(debt_to_equity) / 100 if isinstance(debt_to_equity, (int, float)) else debt_to_equity

        if pe_ratio is None or debt_to_equity is None:
            fmp_data = await fetch_with_multiple_keys_async(ticker, "quote", FMP_API_KEYS, service="FMP")
            if fmp_data and isinstance(fmp_data, list) and len(fmp_data) > 0:
                if pe_ratio is None:
                    pe_ratio = fmp_data[0].get('pe', None)
                if debt_to_equity is None:
                    debt_to_equity = fmp_data[0].get('debtToEquity', None)

        if price is None or (pe_ratio is None and debt_to_equity is None):
            logging.error(f"Insufficient data for {ticker} from both Yahoo and FMP")
            return {"ticker": ticker, "exchange": exchange, "error": "Insufficient data"}

        historical_data = await fetch_historical_data(ticker, exchange)
        roe_10y, rotc_10y, eps_10y, div_10y, years = historical_data

        earnings = price / pe_ratio if pe_ratio else None
        intrinsic_value = calculate_graham_value(earnings, expected_return) if earnings else None
        buy_price = intrinsic_value * (1 - margin_of_safety) if intrinsic_value else None
        sell_price = intrinsic_value * 1.5 if intrinsic_value else None

        graham_score = calculate_graham_score(ticker, price, pe_ratio, debt_to_equity, roe_10y, eps_10y)
        total_graham_score = calculate_total_graham_score(graham_score, rotc_10y, div_10y) if graham_score else 0

        return {
            "ticker": ticker,
            "exchange": exchange,
            "price": price,
            "pe_ratio": pe_ratio,
            "debt_to_equity": debt_to_equity,
            "intrinsic_value": intrinsic_value,
            "buy_price": buy_price,
            "sell_price": sell_price,
            "graham_score": graham_score,
            "total_graham_score": total_graham_score,
            "years": years,
            "roe_10y": roe_10y,
            "rotc_10y": rotc_10y,
            "eps_10y": eps_10y,
            "div_10y": div_10y
        }
    except Exception as e:
        logging.error(f"Error processing {ticker} ({exchange}): {str(e)}")
        return {"ticker": ticker, "exchange": exchange, "error": str(e)}

async def screen_nyse_graham_stocks(batch_size=25, cancel_event=None, tickers=None, root=None, update_progress_animated=None, refresh_favorites_dropdown=None, conn=None, cursor=None):
    exchange = "NYSE"
    file_path = NYSE_LIST_FILE
    filtered_tickers = await load_and_filter_tickers(file_path, exchange_filter='N')
    tickers = filtered_tickers if tickers is None else tickers
    logging.info(f"Initial NYSE tickers after filtering: {len(tickers)}")

    current_file_hash = get_file_hash(file_path)
    with DB_LOCK:
        cursor.execute("SELECT DISTINCT file_hash FROM screening_progress WHERE exchange=?", (exchange,))
        stored_hash = cursor.fetchone()
        stored_hash = stored_hash[0] if stored_hash else None

        if stored_hash != current_file_hash:
            logging.info(f"Detected new version of {file_path} (hash: {current_file_hash}). Resetting NYSE screening data.")
            cursor.execute("DELETE FROM screening_progress WHERE exchange=?", (exchange,))
            cursor.execute("DELETE FROM graham_qualifiers WHERE exchange=?", (exchange,))
            conn.commit()
            remaining_tickers = tickers
        else:
            cursor.execute("SELECT ticker FROM screening_progress WHERE exchange=?", (exchange,))
            completed_tickers = set(row[0] for row in cursor.fetchall())
            remaining_tickers = [t for t in tickers if t not in completed_tickers]
            logging.info(f"Resuming NYSE screening with {len(remaining_tickers)} remaining tickers from previous run.")

    if not remaining_tickers:
        logging.info(f"All NYSE tickers already processed for this file version.")
        return [], [], []

    qualifying_stocks, graham_scores, exchanges = [], [], []
    logging.info(f"Starting NYSE screening for {len(remaining_tickers)} tickers (first 5 criteria, >= 4/5)")
    total_tickers = len(remaining_tickers)
    processed_tickers = 0
    valid_tickers_set = set(tickers)

    for i in range(0, len(remaining_tickers), batch_size):
        if cancel_event and cancel_event.is_set():
            logging.info(f"NYSE screening cancelled by user at batch {i//batch_size + 1}")
            break
        batch = remaining_tickers[i:i + batch_size]
        logging.info(f"Processing NYSE batch {i//batch_size + 1}/{len(remaining_tickers)//batch_size + 1} (tickers: {len(batch)})")
        try:
            results = await fetch_batch_data(batch, exchange=exchange, valid_tickers=valid_tickers_set)
            logging.debug(f"Completed fetch_batch_data for batch {i//batch_size + 1} with {len(results)} results")
            for result in results:
                if cancel_event and cancel_event.is_set():
                    logging.info(f"Cancelling NYSE batch processing due to user request at ticker {result.get('ticker', 'unknown')}")
                    break
                if isinstance(result, dict):
                    ticker = result['ticker']
                    logging.debug(f"Evaluating ticker {ticker}")
                    if 'error' in result:
                        logging.info(f"Completed evaluation for {ticker} ({exchange}): Did not pass due to {result['error']}")
                        with DB_LOCK:
                            cursor.execute("INSERT OR REPLACE INTO screening_progress (exchange, ticker, timestamp, file_hash, status) VALUES (?, ?, ?, ?, ?)",
                                          (exchange, ticker, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), current_file_hash, "failed"))
                            conn.commit()
                        continue
                    if result['graham_score'] >= 4:
                        qualifying_stocks.append(ticker)
                        graham_scores.append(result['graham_score'])
                        exchanges.append(exchange)
                        with DB_LOCK:
                            cursor.execute("INSERT OR REPLACE INTO graham_qualifiers (ticker, graham_score, date, sector, exchange) VALUES (?, ?, ?, ?, ?)",
                                          (ticker, result['graham_score'], datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Unknown', exchange))
                            conn.commit()
                    processed_tickers += 1
                    progress = (processed_tickers / total_tickers) * 100 if total_tickers > 0 else 100
                    if root and update_progress_animated:
                        root.after(0, update_progress_animated, progress, exchange, remaining_tickers)
                    with DB_LOCK:
                        cursor.execute("INSERT OR REPLACE INTO screening_progress (exchange, ticker, timestamp, file_hash, status) VALUES (?, ?, ?, ?, ?)",
                                      (exchange, ticker, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), current_file_hash, "completed"))
                        conn.commit()
                    logging.debug(f"Processed ticker {ticker}, progress {progress:.1f}%")
        except Exception as e:
            logging.error(f"Error processing NYSE batch {batch}: {str(e)}")
            break

    if not cancel_event or not cancel_event.is_set():
        if root and update_progress_animated:
            root.after(0, update_progress_animated, 100, exchange, remaining_tickers)
        if qualifying_stocks:
            list_name = await save_qualifying_stocks_to_favorites(qualifying_stocks, graham_scores, exchanges, exchange)
            if root and refresh_favorites_dropdown:
                root.after(0, refresh_favorites_dropdown, list_name)
            logging.info(f"Completed NYSE screening. Found {len(qualifying_stocks)} qualifying stocks added to favorites as '{list_name}'")
        else:
            logging.info(f"Completed NYSE screening. No qualifying NYSE stocks found.")
    else:
        logging.warning(f"NYSE screening interrupted before completion")
    return qualifying_stocks, graham_scores, exchanges

async def screen_nasdaq_graham_stocks(batch_size=25, cancel_event=None, tickers=None, root=None, update_progress_animated=None, refresh_favorites_dropdown=None, conn=None, cursor=None):
    exchange = "NASDAQ"
    file_path = NASDAQ_LIST_FILE
    filtered_tickers = await load_and_filter_tickers(file_path)
    tickers = filtered_tickers if tickers is None else tickers
    logging.info(f"Initial NASDAQ tickers after filtering: {len(tickers)}")

    current_file_hash = get_file_hash(file_path)
    with DB_LOCK:
        cursor.execute("SELECT DISTINCT file_hash FROM screening_progress WHERE exchange=?", (exchange,))
        stored_hash = cursor.fetchone()
        stored_hash = stored_hash[0] if stored_hash else None

        if stored_hash != current_file_hash:
            logging.info(f"Detected new version of {file_path} (hash: {current_file_hash}). Resetting NASDAQ screening data.")
            cursor.execute("DELETE FROM screening_progress WHERE exchange=?", (exchange,))
            cursor.execute("DELETE FROM graham_qualifiers WHERE exchange=?", (exchange,))
            conn.commit()
            remaining_tickers = tickers
        else:
            cursor.execute("SELECT ticker FROM screening_progress WHERE exchange=?", (exchange,))
            completed_tickers = set(row[0] for row in cursor.fetchall())
            remaining_tickers = [t for t in tickers if t not in completed_tickers]
            logging.info(f"Resuming NASDAQ screening with {len(remaining_tickers)} remaining tickers from previous run.")

    if not remaining_tickers:
        logging.info(f"All NASDAQ tickers already processed for this file version.")
        return [], [], []

    qualifying_stocks, graham_scores, exchanges = [], [], []
    logging.info(f"Starting NASDAQ screening for {len(remaining_tickers)} tickers (first 5 criteria, >= 4/5)")
    total_tickers = len(remaining_tickers)
    processed_tickers = 0
    valid_tickers_set = set(tickers)

    for i in range(0, len(remaining_tickers), batch_size):
        if cancel_event and cancel_event.is_set():
            logging.info(f"NASDAQ screening cancelled by user at batch {i//batch_size + 1}")
            break
        batch = remaining_tickers[i:i + batch_size]
        logging.info(f"Processing NASDAQ batch {i//batch_size + 1}/{len(remaining_tickers)//batch_size + 1} (tickers: {len(batch)})")
        try:
            results = await fetch_batch_data(batch, exchange=exchange, valid_tickers=valid_tickers_set)
            logging.debug(f"Completed fetch_batch_data for batch {i//batch_size + 1} with {len(results)} results")
            for result in results:
                if cancel_event and cancel_event.is_set():
                    logging.info(f"Cancelling NASDAQ batch processing due to user request at ticker {result.get('ticker', 'unknown')}")
                    break
                if isinstance(result, dict):
                    ticker = result['ticker']
                    logging.debug(f"Evaluating ticker {ticker}")
                    if 'error' in result:
                        logging.info(f"Completed evaluation for {ticker} ({exchange}): Did not pass due to {result['error']}")
                        with DB_LOCK:
                            cursor.execute("INSERT OR REPLACE INTO screening_progress (exchange, ticker, timestamp, file_hash, status) VALUES (?, ?, ?, ?, ?)",
                                          (exchange, ticker, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), current_file_hash, "failed"))
                            conn.commit()
                        continue
                    if result['graham_score'] >= 4:
                        qualifying_stocks.append(ticker)
                        graham_scores.append(result['graham_score'])
                        exchanges.append(exchange)
                        with DB_LOCK:
                            cursor.execute("INSERT OR REPLACE INTO graham_qualifiers (ticker, graham_score, date, sector, exchange) VALUES (?, ?, ?, ?, ?)",
                                          (ticker, result['graham_score'], datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Unknown', exchange))
                            conn.commit()
                    processed_tickers += 1
                    progress = (processed_tickers / total_tickers) * 100 if total_tickers > 0 else 100
                    if root and update_progress_animated:
                        root.after(0, update_progress_animated, progress, exchange, remaining_tickers)
                    with DB_LOCK:
                        cursor.execute("INSERT OR REPLACE INTO screening_progress (exchange, ticker, timestamp, file_hash, status) VALUES (?, ?, ?, ?, ?)",
                                      (exchange, ticker, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), current_file_hash, "completed"))
                        conn.commit()
                    logging.debug(f"Processed ticker {ticker}, progress {progress:.1f}%")
        except Exception as e:
            logging.error(f"Error processing NASDAQ batch {batch}: {str(e)}")
            break

    if not cancel_event or not cancel_event.is_set():
        if root and update_progress_animated:
            root.after(0, update_progress_animated, 100, exchange, remaining_tickers)
        if qualifying_stocks:
            list_name = await save_qualifying_stocks_to_favorites(qualifying_stocks, graham_scores, exchanges, exchange)
            if root and refresh_favorites_dropdown:
                root.after(0, refresh_favorites_dropdown, list_name)
            logging.info(f"Completed NASDAQ screening. Found {len(qualifying_stocks)} qualifying stocks added to favorites as '{list_name}'")
        else:
            logging.info(f"Completed NASDAQ screening. No qualifying NASDAQ stocks found.")
    else:
        logging.warning(f"NASDAQ screening interrupted before completion")
    return qualifying_stocks, graham_scores, exchanges

async def save_qualifying_stocks_to_favorites(qualifying_stocks, graham_scores, exchanges, exchange):
    import json
    from config import FAVORITES_FILE

    list_name = f"{exchange}_Qualifiers_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    favorites = {}
    if os.path.exists(FAVORITES_FILE):
        try:
            with open(FAVORITES_FILE, 'r') as f:
                favorites = json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON in {FAVORITES_FILE}: {str(e)}. Starting fresh.")

    favorites[list_name] = {
        "tickers": qualifying_stocks,
        "graham_scores": graham_scores,
        "exchanges": exchanges,
        "date_added": datetime.now().isoformat()
    }

    try:
        with open(FAVORITES_FILE, 'w') as f:
            json.dump(favorites, f, indent=4)
        logging.info(f"Saved {len(qualifying_stocks)} qualifying {exchange} stocks to favorites as '{list_name}'")
        return list_name
    except Exception as e:
        logging.error(f"Error saving favorites to {FAVORITES_FILE}: {str(e)}")
        return None

async def fetch_with_multiple_keys_async(ticker, endpoint, api_keys, service="FMP", retries=3, backoff=2):
    if not api_keys or not isinstance(api_keys, list):
        logging.error(f"No API keys provided for {service}")
        return None

    for attempt in range(retries):
        for api_key in api_keys:
            try:
                url = (f"https://financialmodelingprep.com/api/v3/{endpoint}/{ticker}?apikey={api_key}" 
                       if service == "FMP" 
                       else f"https://www.alphavantage.co/query?function={endpoint}&symbol={ticker}&apikey={api_key}")
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status == 200:
                            data = await response.json()
                            logging.info(f"Successfully fetched {service} data for {ticker} with key {api_key}")
                            return data
                        elif response.status == 429:
                            logging.warning(f"API key {api_key} failed for {ticker} ({service}): 429 - Rate limit reached. Waiting {backoff ** attempt} seconds...")
                            await asyncio.sleep(backoff ** attempt)
                            continue
                        else:
                            logging.error(f"Failed to fetch {service} data for {ticker} with key {api_key}: Status {response.status}")
                            continue
            except Exception as e:
                logging.error(f"Error fetching {service} data for {ticker} with key {api_key}: {str(e)}")
                continue
        if attempt < retries - 1:
            logging.warning(f"Retrying {ticker} after {backoff ** attempt} seconds...")
            await asyncio.sleep(backoff ** attempt)
    logging.error(f"Failed to fetch {service} data for {ticker} after {retries} attempts")
    return None