import sqlite3
import time
import logging
import aiohttp
import asyncio
from typing import Dict, List, Optional, Tuple
import yfinance as yf
from datetime import datetime, timedelta
import os
import json
import pandas as pd
import threading
import hashlib
import concurrent.futures
import requests
import random
from decouple import config
from bs4 import BeautifulSoup
from ftplib import FTP
import pdfplumber  # Added for PDF parsing
from config import (
    FMP_API_KEYS, FRED_API_KEY, CACHE_DB, NYSE_LIST_FILE, NASDAQ_LIST_FILE,
    USER_DATA_DIR, FAVORITES_LOCK, FAVORITES_FILE, CACHE_EXPIRY,
    MAX_CALLS_PER_MINUTE_PAID, graham_logger, get_file_hash, FileHashError, USE_FREE_API_KEY,
    API_DISABLED
)
from graham_utils import paid_rate_limiter, free_rate_limiter

# Constants for batch processing and concurrency control
MAX_CONCURRENT_TICKERS = 10
MAX_TICKERS_PER_MINUTE = 250
DELAY_BETWEEN_CALLS = 0.1
BATCH_SIZE = 50

# Hardcoded known NYSE ADRs (foreign companies listed as ADRs)
KNOWN_NYSE_ADRS = {
    'YPF', 'GGAL', 'SUPV', 'BBAR', 'BIOX', 'TEO', 'TGS', 'BMA', 'LOMA', 'CEPU', 'PAM', 'CRESY', 'IRS', 'EDN', 'IXHL', 'IREN', 'KTRA', 'BHP', 'TEAM', 'ESRX', 'NVX', 'WDS', 'CXO', 'LITM', 'SONI', 'MXCT', 'NOVO', 'NHC', 'FGR', 'AOIL', 'NVA', 'PEN', 'CSL', 'IONR', 'EBKDY', 'RAIFY', 'OMVKY', 'ANDR', 'EBSOY', 'VIE', 'POSTY', 'SBO', 'FLW', 'VIG', 'BUD', 'CMBT', 'GLPG', 'ACG', 'MDXH', 'UCB', 'MTLS', 'UMICY', 'NYXH', 'KBCSY', 'SOLVY', 'AGESY', 'BLAMY', 'BGAOY', 'GALP', 'DIETY', 'AGFAY', 'ACKAY', 'BPOST', 'BEKAY', 'ABEV', 'BBD', 'NU', 'VALE', 'PBR', 'ITUB', 'GGB', 'PBR.A', 'NVNI', 'JBSAY', 'PAGS', 'BAK', 'UGP', 'SID', 'SGML', 'EBR.B', 'CIG', 'SUZ', 'ATLX', 'MRRTY', 'TLRY', 'CGC', 'DNN', 'BTG', 'GOLD', 'BITF', 'LAC', 'CVE', 'FSM', 'KGC', 'BTE', 'AG', 'EQX', 'BB', 'CRON', 'NAK', 'IAG', 'HBM', 'EXK', 'NGD', 'LTM', 'SQM', 'ENIC', 'BCH', 'BSAC', 'CCU', 'AKO.B', 'AKO.A', 'YOUX', 'NIO', 'BQ', 'RAYA', 'WKSP', 'ORIS', 'CJET', 'JD', 'PONY', 'DIDI', 'WED', 'JZ', 'LI', 'VNET', 'LU', 'YMM', 'FTAI', 'IQ', 'XPEV', 'BZ', 'BEKE', 'EC', 'GPRK', 'TGLS', 'CIB', 'AVAL', 'CMTOY', 'INSR', 'CLVR', 'CLVRW', 'FRO', 'ROBO', 'TORO', 'GIFA', 'GDEV', 'CTRM', 'NEUH', 'BOCIF', 'IOBT', 'NVO', 'GMAB', 'EVAX', 'ASND', 'DOGEF', 'COLOF', 'GLTO', 'VWDRY', 'CDLR', 'DSDVY', 'AMKBY', 'CABGY', 'NVZMY', 'PANDY', 'DANSKE', 'LIQT', 'DOGEF', 'BVNRY', 'NOK', 'AS', 'NRDBY', 'SEOAY', 'SAXPY', 'NESTY', 'KNEBY', 'MEOAY', 'NOKTY', 'WRTBY', 'ORINY', 'YITY', 'OUTKY', 'KCRAY', 'KNEBY', 'FOJCY', 'SNY', 'TTE', 'DANOY', 'SAF', 'CGEMY', 'ABVX', 'CSTM', 'SQNS', 'PRNDY', 'CLLS', 'SBGSY', 'AF', 'CRTO', 'LVMUY', 'DG', 'ALO', 'CRRFY', 'SCGLY', 'AMTD', 'JMIA', 'DB', 'SAP', 'IMTX', 'SYIEY', 'BNTX', 'MYTE', 'DB'
}

# Hardcoded known NASDAQ ADRs (foreign companies listed as ADRs)
KNOWN_NASDAQ_ADRS = {
    'ASML', 'LOGI', 'PHG', 'NXPI', 'CNMD', 'CRON', 'GRMN', 'TEVA', 'NICE', 'WIX', 'CYBR', 'TSEM', 'CAMT', 'SEDG', 'INMD', 'ZIM', 'GLBE', 'FROG', 'MNDY', 'ODD', 'PLTK', 'GMAB', 'NNDM', 'SSYS', 'DRTS', 'PRTC', 'URGN', 'CGNT', 'WKME', 'PERI', 'SPNS', 'AUDC', 'MGIC', 'SILC', 'VLN', 'ALLT', 'RDWR', 'ITRN', 'GILT', 'KMDA', 'MDWD', 'RDHL', 'SLGN', 'PYPD', 'EVGN', 'INVZ', 'ENLV', 'NNOX', 'FUSN', 'BWAY', 'ALVR', 'SMWB', 'SVRE', 'CLBT', 'PGY', 'NVMI', 'TARO', 'ELWS', 'SHIP', 'TORO', 'GIFA', 'GDEV', 'CTRM', 'NHIH', 'BOCH', 'IOBT', 'NVO', 'GMAB', 'EVAX', 'ASND', 'CLPBY', 'GLTO', 'VWDRY', 'CDLR', 'DSDVY', 'AMKBY', 'CABGY', 'NVZMY', 'PANDY', 'DNKEY', 'LIQT', 'DOGE', 'BVNRY'
}

# Hardcoded US-only list to prevent false positives
KNOWN_US_ONLY = {
    'AYI', 'MLM', 'TRV', 'FSK', 'NDSN', 'KLAC', 'GNTX', 'NVDA', 'AAON', 'POOL', 'MPWR', 'SEIC',
    'PLPC', 'SCVL', 'FAST', 'CTAS', 'TXN', 'PCAR', 'LRCX', 'COLM', 'GLPI', 'IMKTA', 'ARCC',
    'AMAT', 'OTTR', 'UFPI', 'WDFC', 'IPAR', 'MKTX'
}

class CacheManager:
    def __init__(self):
        self.aaa_yield_cache = None
        self.cache_timestamp = None
        self.cache_duration = timedelta(days=7)
        self.schema_initialized = False
        self.schema_lock = threading.Lock()
        self.fetched_tickers = set()
        self.fetched_tickers_lock = threading.Lock()
        self.db_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.aaa_cache_lock = threading.Lock()  # New lock for AAA yield cache

    def get_aaa_yield(self, api_key: str, default_yield: float = 0.045) -> float:
        """Fetch Moody's Seasoned AAA Corporate Bond Yield from FRED with caching."""
        current_time = datetime.now()
        with self.aaa_cache_lock:  # Protect cache access
            if (self.aaa_yield_cache is not None and
                self.cache_timestamp is not None and
                current_time - self.cache_timestamp < self.cache_duration):
                graham_logger.debug(f"Using cached AAA yield: {self.aaa_yield_cache:.4f}")
                return self.aaa_yield_cache
        if not api_key:
            graham_logger.error("FRED_API_KEY not set. Using default yield: 4.5%")
            return default_yield
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id=AAA&api_key={api_key}&file_type=json&limit=1&sort_order=desc"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if 'observations' in data and len(data['observations']) > 0:
                yield_value = float(data['observations'][0]['value']) / 100
                with self.aaa_cache_lock:  # Update cache under lock
                    self.aaa_yield_cache = yield_value
                    self.cache_timestamp = current_time
                graham_logger.info(f"Fetched AAA yield: {yield_value:.4f}")
                return yield_value
            else:
                raise ValueError("No observations found in FRED response")
        except requests.RequestException as e:
            graham_logger.error(f"Error fetching AAA yield from FRED: {str(e)}")
            with self.aaa_cache_lock:
                if self.aaa_yield_cache is not None:
                    graham_logger.info("Using cached AAA yield due to fetch error")
                    return self.aaa_yield_cache
                else:
                    graham_logger.info(f"Using default AAA yield: {default_yield}")
                    return default_yield

    def get_stocks_connection(self, max_retries=3, retry_delay=1):
        """Establish a connection to the SQLite database with retry logic and increased timeout."""
        for attempt in range(max_retries):
            try:
                conn = sqlite3.connect(CACHE_DB, timeout=60)
                cursor = conn.cursor()
                
                # Check if schema needs to be initialized
                with self.schema_lock:
                    if not self.schema_initialized:
                        # Create or update the 'stocks' table with new raw data columns
                        cursor.execute('''CREATE TABLE IF NOT EXISTS stocks (
                            ticker TEXT PRIMARY KEY,
                            date TEXT,
                            roe TEXT,
                            rotc TEXT,
                            eps TEXT,
                            dividend TEXT,
                            ticker_list_hash TEXT,
                            balance_data TEXT,
                            cash_flow_data TEXT,
                            key_metrics_data TEXT,
                            timestamp REAL,
                            company_name TEXT,
                            debt_to_equity REAL,
                            eps_ttm REAL,
                            book_value_per_share REAL,
                            common_score INTEGER,
                            latest_revenue REAL,
                            available_data_years INTEGER,
                            sector TEXT,
                            years TEXT,
                            latest_total_assets REAL,
                            latest_total_liabilities REAL,
                            latest_shares_outstanding REAL,
                            latest_long_term_debt REAL,
                            latest_short_term_debt REAL,
                            latest_current_assets REAL,
                            latest_current_liabilities REAL,
                            latest_book_value REAL,
                            historic_pe_ratios TEXT,
                            latest_net_income REAL,
                            eps_cagr REAL,
                            latest_free_cash_flow REAL,
                            raw_income_data TEXT,
                            raw_balance_data TEXT,
                            raw_dividend_data TEXT,
                            raw_profile_data TEXT,
                            raw_cash_flow_data TEXT,
                            raw_key_metrics_data TEXT,
                            exchange TEXT,
                            is_foreign BOOLEAN
                        )''')
                        cursor.execute("PRAGMA table_info(stocks)")
                        columns = [col[1] for col in cursor.fetchall()]
                        new_columns = [
                            'latest_total_assets REAL',
                            'latest_total_liabilities REAL',
                            'latest_shares_outstanding REAL',
                            'latest_long_term_debt REAL',
                            'latest_short_term_debt REAL',
                            'latest_current_assets REAL',
                            'latest_current_liabilities REAL',
                            'latest_book_value REAL',
                            'historic_pe_ratios TEXT',
                            'latest_net_income REAL',
                            'eps_cagr REAL',
                            'latest_free_cash_flow REAL',
                            'raw_income_data TEXT',
                            'raw_balance_data TEXT',
                            'raw_dividend_data TEXT',
                            'raw_profile_data TEXT',
                            'raw_cash_flow_data TEXT',
                            'raw_key_metrics_data TEXT',
                            'exchange TEXT',
                            'is_foreign BOOLEAN'
                        ]
                        for col in new_columns:
                            col_name = col.split()[0]
                            if col_name not in columns:
                                cursor.execute(f"ALTER TABLE stocks ADD COLUMN {col}")
                                graham_logger.info(f"Added '{col_name}' column to stocks table")
                        # Create or update the 'graham_qualifiers' table
                        cursor.execute('''CREATE TABLE IF NOT EXISTS graham_qualifiers (
                            ticker TEXT PRIMARY KEY,
                            common_score INTEGER,
                            date TEXT,
                            sector TEXT,
                            exchange TEXT,
                            min_criteria INTEGER
                        )''')
                        
                        # Create the 'screening_progress' table if it doesn't exist
                        cursor.execute('''CREATE TABLE IF NOT EXISTS screening_progress (
                            exchange TEXT,
                            ticker TEXT,
                            timestamp TEXT,
                            file_hash TEXT,
                            status TEXT,
                            PRIMARY KEY (exchange, ticker)
                        )''')
                        
                        conn.commit()
                        self.schema_initialized = True
                        graham_logger.info("Database schema initialized successfully.")
                
                return conn, cursor
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    graham_logger.warning(f"Database locked, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    graham_logger.error(f"Failed to initialize database connection: {str(e)}")
                    raise
            except sqlite3.Error as e:
                graham_logger.error(f"Failed to initialize database connection: {str(e)}")
                raise
        raise sqlite3.OperationalError("Failed to connect to database after retries")

    def add_fetched_ticker(self, ticker: str):
        """Add a ticker to the set of fetched tickers."""
        with self.fetched_tickers_lock:
            self.fetched_tickers.add(ticker)

    def is_ticker_fetched(self, ticker: str) -> bool:
        """Check if a ticker has been fetched."""
        with self.fetched_tickers_lock:
            return ticker in self.fetched_tickers

    def get_db_executor(self):
        """Return the database executor."""
        return self.db_executor

# Initialize CacheManager
cache_manager = CacheManager()

class TickerManager:
    def __init__(self, nyse_file: str, nasdaq_file: str, user_data_dir: str = USER_DATA_DIR):
        self.nyse_tickers = {}
        self.nasdaq_tickers = {}
        self.filtered_nyse = []
        self.filtered_nasdaq = []
        self.nyse_file = nyse_file
        self.nasdaq_file = nasdaq_file
        self.user_data_dir = user_data_dir

    async def fetch_adr_lists(self):
        adrs = set()
        cache_file = os.path.join(self.user_data_dir, "adr_cache.json")
        cache_expiry = 30 * 24 * 60 * 60  # 30 days
        # Check cache
        if os.path.exists(cache_file) and (time.time() - os.path.getmtime(cache_file)) < cache_expiry:
            try:
                with open(cache_file, 'r') as f:
                    adrs = set(json.load(f))
                graham_logger.info(f"Loaded {len(adrs)} ADRs from cache")
                return adrs
            except Exception as e:
                graham_logger.warning(f"Failed to load ADR cache: {str(e)}")
        # Scrape Investing.com (updated URL and selector)
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get('https://www.investing.com/equities/world-adrs') as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        soup = BeautifulSoup(text, 'html.parser')
                        for a in soup.select('#stockDirectoryTable td.text-left a'):  # Updated selector
                            ticker = a.text.strip().upper()
                            if self.is_valid_ticker(ticker):
                                adrs.add(ticker)
                                graham_logger.debug(f"Scraped ADR from Investing.com: {ticker}")
            except Exception as e:
                graham_logger.warning(f"Investing.com scrape failed: {str(e)}")
            # Scrape TopForeignStocks (main page and follow complete list links if possible)
            try:
                async with session.get('https://topforeignstocks.com/foreign-adrs-list/') as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        soup = BeautifulSoup(text, 'html.parser')
                        # Find links to complete lists (e.g., NYSE ADRs Excel)
                        for a in soup.select('a[href*="complete-list-of-adrs"]'):
                            sub_url = a['href']
                            async with session.get(sub_url) as sub_resp:
                                if sub_resp.status == 200:
                                    sub_text = await sub_resp.text()
                                    sub_soup = BeautifulSoup(sub_text, 'html.parser')
                                    for td in sub_soup.select('td.column-1'):  # Ticker column in tables
                                        ticker = td.text.strip().upper()
                                        if self.is_valid_ticker(ticker):
                                            adrs.add(ticker)
                                            graham_logger.debug(f"Scraped ADR from TopForeignStocks subpage: {ticker}")
            except Exception as e:
                graham_logger.warning(f"TopForeignStocks scrape failed: {str(e)}")
            # Scrape NYSE PDF (improved parsing with tables)
            try:
                pdf_url = 'https://www.nyse.com/publicdocs/nyse/data/CurListofallStocks.pdf'
                async with session.get(pdf_url) as resp:
                    if resp.status == 200:
                        pdf_data = await resp.read()
                        with open('temp.pdf', 'wb') as f:
                            f.write(pdf_data)
                        with pdfplumber.open('temp.pdf') as pdf:
                            for page in pdf.pages:
                                tables = page.extract_tables()
                                for table in tables:
                                    for row in table[1:]:  # Skip header
                                        if len(row) > 7 and row[7] == 'A':  # Share Type column (index 7 based on header)
                                            ticker = row[2].strip().upper()  # Symbol column
                                            if self.is_valid_ticker(ticker):
                                                adrs.add(ticker)
                                                graham_logger.debug(f"PDF scraped ADR: {ticker}")
                        os.remove('temp.pdf')
            except Exception as e:
                graham_logger.warning(f"NYSE PDF scrape failed: {str(e)}")
        # Save to cache
        try:
            with open(cache_file, 'w') as f:
                json.dump(list(adrs), f)
            graham_logger.info(f"Saved {len(adrs)} ADRs to cache")
        except Exception as e:
            graham_logger.warning(f"Failed to save ADR cache: {str(e)}")
        return adrs

    async def initialize(self, force_update=False, callback=None):
        if force_update:
            await self.download_ticker_files()
        use_cache = not force_update
        self.filtered_nyse = await load_and_filter_tickers(self.nyse_file, exchange_filter='N', use_cache=use_cache)
        self.filtered_nasdaq = await load_and_filter_tickers(self.nasdaq_file, exchange_filter='Q', use_cache=use_cache)
        self.nyse_tickers = {ticker["ticker"]: ticker["security_name"] for ticker in self.filtered_nyse}
        self.nasdaq_tickers = {ticker["ticker"]: ticker["security_name"] for ticker in self.filtered_nasdaq}
        graham_logger.info(f"Initialized NYSE common stock tickers: {len(self.nyse_tickers)}")
        graham_logger.info(f"Initialized NASDAQ common stock tickers: {len(self.nasdaq_tickers)}")
        # Add dynamic ADR fetch here
        try:
            dynamic_adrs = await self.fetch_adr_lists()
            self.adr_tickers = dynamic_adrs | KNOWN_NYSE_ADRS | KNOWN_NASDAQ_ADRS  # Merge all
            graham_logger.info(f"Fetched {len(dynamic_adrs)} dynamic ADRs; total ADRs: {len(self.adr_tickers)}")
        except Exception as e:
            graham_logger.warning(f"Dynamic ADR fetch failed: {str(e)}. Using hardcoded lists only ({len(KNOWN_NYSE_ADRS | KNOWN_NASDAQ_ADRS)} ADRs).")
            self.adr_tickers = KNOWN_NYSE_ADRS | KNOWN_NASDAQ_ADRS
        if callback:
            callback()

    async def download_ticker_files(self):
        try:
            ftp = FTP('ftp.nasdaqtrader.com')
            ftp.login()
            ftp.cwd('SymbolDirectory')
            files_to_download = ['nasdaqlisted.txt', 'otherlisted.txt']
            for file_name in files_to_download:
                local_path = os.path.join(self.user_data_dir, file_name)
                with open(local_path, 'wb') as f:
                    ftp.retrbinary(f'RETR {file_name}', f.write)
                # Compute and store the hash
                file_hash = get_file_hash(local_path)
                conn, cursor = cache_manager.get_stocks_connection()
                try:
                    exchange = 'NASDAQ' if 'nasdaq' in file_name.lower() else 'NYSE'
                    # Insert or update the hash with a dummy ticker to track file freshness
                    cursor.execute("""
                        INSERT OR REPLACE INTO screening_progress (exchange, ticker, timestamp, file_hash, status)
                        VALUES (?, ?, ?, ?, ?)
                    """, (exchange, 'FILE_HASH_TRACKER', datetime.now().strftime('%Y-%m-%d %H:%M:%S'), file_hash, 'completed'))
                    conn.commit()
                except sqlite3.Error as e:
                    graham_logger.error(f"Error updating hash for {file_name}: {str(e)}")
                finally:
                    conn.close()
            ftp.quit()
            graham_logger.info("Successfully downloaded and updated ticker files from FTP.")
            
            # Clear invalid ticker files
            for invalid_file in [
                os.path.join(self.user_data_dir, "NYSE Invalid Tickers.txt"),
                os.path.join(self.user_data_dir, "NASDAQ Invalid Tickers.txt")
            ]:
                with open(invalid_file, 'w') as f:
                    f.write("")
                graham_logger.info(f"Cleared invalid tickers file: {invalid_file}")
            
        except Exception as e:
            graham_logger.error(f"Failed to download ticker files: {str(e)}")
            raise

    def get_tickers(self, exchange: str) -> set:
        if exchange == "NYSE":
            return set(self.nyse_tickers.keys())
        elif exchange == "NASDAQ":
            return set(self.nasdaq_tickers.keys())
        return set()

    def get_security_name(self, ticker: str) -> str:
        return self.nyse_tickers.get(ticker, self.nasdaq_tickers.get(ticker, 'Unknown'))

    def is_valid_ticker(self, ticker: str) -> bool:
        return ticker in self.nyse_tickers or ticker in self.nasdaq_tickers

async def load_and_filter_tickers(file_path: str, exchange_filter: Optional[str] = None, update_rate_limit=None, use_cache: bool = True) -> List[Dict]:
    try:
        file_hash = get_file_hash(file_path)
    except FileHashError as e:
        graham_logger.error(str(e))
        return []
    graham_logger.debug(f"Computed hash for {file_path}: {file_hash}")
    conn, cursor = cache_manager.get_stocks_connection()
    try:
        if use_cache:
            cursor.execute("SELECT ticker, timestamp, company_name FROM stocks WHERE ticker_list_hash = ?", (file_hash,))
            cached_data = cursor.fetchall()
            cached_entries = {row[0]: (row[1], row[2]) for row in cached_data}
            graham_logger.debug(f"Loaded {len(cached_entries)} entries from stocks table for hash {file_hash}")
        else:
            cached_entries = {}
        ticker_data = []
        fresh_count = 0
        stale_count = 0
        missing_count = 0
        invalid_timestamp_count = 0
        with open(file_path, 'r') as f:
            lines = f.readlines()
            headers = lines[0].strip().split('|')
            if 'otherlisted' in file_path:
                ticker_column = 'ACT Symbol'
            elif 'nasdaqlisted' in file_path:
                ticker_column = 'Symbol'
            else:
                raise ValueError(f"Unknown file format: {file_path}")
            graham_logger.debug(f"Using ticker column '{ticker_column}' for {file_path}")
            try:
                symbol_idx = headers.index(ticker_column)
                security_name_idx = headers.index('Security Name')
            except ValueError as e:
                raise ValueError(f"Column not found in {file_path}: {str(e)}")
            security_type_idx = headers.index('Security Type') if 'Security Type' in headers else None
            exchange_idx = headers.index('Exchange') if 'Exchange' in headers else None
            for line in lines[1:]:
                parts = line.strip().split('|')
                if len(parts) <= symbol_idx:
                    continue
                ticker = parts[symbol_idx].strip()
                if not ticker or any(suffix in ticker for suffix in ['.', '$']):
                    continue
                if len(parts) > security_name_idx:
                    security_name = parts[security_name_idx].strip()
                else:
                    security_name = 'Unknown'
                if security_type_idx is not None and len(parts) > security_type_idx:
                    security_type = parts[security_type_idx].strip()
                    if security_type != "Common Stock":
                        continue
                if exchange_filter and exchange_idx is not None and len(parts) > exchange_idx:
                    exchange = parts[exchange_idx].strip()
                    if exchange != exchange_filter:
                        continue
                if use_cache:
                    timestamp, company_name = cached_entries.get(ticker, (None, None))
                else:
                    timestamp, company_name = None, None
                current_time = time.time()
                if timestamp is not None:
                    if isinstance(timestamp, (int, float)):
                        age = current_time - timestamp
                        if age < CACHE_EXPIRY:
                            # Suppress per-ticker fresh log; just count
                            fresh_count += 1
                            is_fresh = True
                        else:
                            # Keep stale as info (issue indicator), or change to debug if you want full suppression
                            graham_logger.info(f"{ticker} is stale: age={age:.2f} seconds")
                            stale_count += 1
                            is_fresh = False
                    else:
                        graham_logger.info(f"{ticker} has invalid timestamp type: {type(timestamp)}")
                        invalid_timestamp_count += 1
                        is_fresh = False
                else:
                    # Suppress per-ticker not found log; just count
                    missing_count += 1
                    is_fresh = False
                ticker_data.append({
                    "ticker": ticker,
                    "is_fresh": is_fresh,
                    "company_name": company_name if is_fresh else None,
                    "security_name": security_name
                })
        graham_logger.info(f"Loaded {len(ticker_data)} tickers from {file_path} after filtering")
        graham_logger.info(f"Cache hit/miss summary: Hits={fresh_count}, Misses={len(ticker_data) - fresh_count} (Total={len(ticker_data)})")
        return ticker_data
    except sqlite3.Error as e:
        graham_logger.error(f"Database error while loading tickers from {file_path}: {str(e)}")
        return []
    except Exception as e:
        graham_logger.error(f"Error loading tickers from {file_path}: {str(e)}")
        return []
    finally:
        conn.close()

async def fetch_with_multiple_keys_async(ticker, endpoint, api_keys, retries=3, backoff=2, update_rate_limit=None, session=None, cancel_event=None):
    """
    Fetch data from FMP API with multiple API keys, handling retries and rate limits.
    Uses stable endpoints with appropriate parameters.
    """
    if API_DISABLED:
        graham_logger.warning(f"API disabled: Skipping fetch for {ticker} ({endpoint})")
        return None
    if not api_keys or all(not key for key in api_keys):
        graham_logger.error(f"No valid API keys provided for endpoint {endpoint}")
        return None
    for api_key in api_keys:
        if cancel_event and cancel_event.is_set():
            graham_logger.info(f"Cancelling fetch for {ticker} ({endpoint})")
            return None
        key_type = "paid" if api_key == FMP_API_KEYS[0] else "free"
        graham_logger.debug(f"Attempting fetch for {ticker} ({endpoint}) with {key_type} key ending in {api_key[-4:]}")
        limiter = paid_rate_limiter if api_key == FMP_API_KEYS[0] else free_rate_limiter
        for attempt in range(retries):
            graham_logger.debug(f"Attempt {attempt + 1}/{retries} for {ticker} ({endpoint}) with {key_type} key")
            try:
                await limiter.acquire()
                if endpoint in ["income-statement", "balance-sheet-statement", "cash-flow-statement", "key-metrics"]:
                    url = f"https://financialmodelingprep.com/api/v3/{endpoint}/{ticker}?period=annual&limit=10&apikey={api_key}"
                elif endpoint == "historical-price-full/stock_dividend":
                    url = f"https://financialmodelingprep.com/api/v3/{endpoint}/{ticker}?apikey={api_key}"
                elif endpoint == "profile":
                    url = f"https://financialmodelingprep.com/api/v3/{endpoint}/{ticker}?apikey={api_key}"
                else:
                    url = f"https://financialmodelingprep.com/api/v3/{endpoint}/{ticker}?apikey={api_key}"
                graham_logger.info(f"Requesting URL: {url}")
                if session:
                    async with session.get(url) as response:
                        if response.status == 429:
                            if update_rate_limit:
                                update_rate_limit(f"Rate limit hit for {key_type} key ending {api_key[-4:]}")
                            graham_logger.warning(f"Rate limit hit for {ticker} ({endpoint}) with {key_type} key, retrying after 60 seconds")
                            await asyncio.sleep(60)
                            continue
                        elif response.status != 200:
                            raise aiohttp.ClientError(f"API returned status {response.status}")
                        data = await response.json()
                        if not data:
                            raise ValueError("Empty response from API")
                        graham_logger.info(f"Successfully fetched {endpoint} data for {ticker} with {key_type} key")
                        return data
                else:
                    async with aiohttp.ClientSession() as temp_session:
                        async with temp_session.get(url) as response:
                            if response.status == 429:
                                if update_rate_limit:
                                    update_rate_limit(f"Rate limit hit for {key_type} key ending {api_key[-4:]}")
                                graham_logger.warning(f"Rate limit hit for {ticker} ({endpoint}) with {key_type} key, retrying after 60 seconds")
                                await asyncio.sleep(60)
                                continue
                            elif response.status != 200:
                                raise aiohttp.ClientError(f"API returned status {response.status}")
                            data = await response.json()
                            if not data:
                                raise ValueError("Empty response from API")
                            graham_logger.info(f"Successfully fetched {endpoint} data for {ticker} with {key_type} key")
                            return data
            except aiohttp.ClientError as e:
                graham_logger.error(f"Network error for {ticker} ({endpoint}) with {key_type} key: {str(e)}")
                if attempt < retries - 1:
                    await asyncio.sleep(min(60 * (attempt + 1), 300))
                else:
                    break
            except json.JSONDecodeError:
                graham_logger.error(f"JSON decoding error for {ticker} ({endpoint}) with {key_type} key")
                break
            except ValueError as e:
                graham_logger.error(f"Data error for {ticker} ({endpoint}) with {key_type} key: {str(e)}")
                break
    graham_logger.error(f"All attempts and keys exhausted for {ticker} ({endpoint})")
    return None

async def fetch_fmp_data(ticker: str, keys: List[str], update_rate_limit=None, cancel_event=None) -> Tuple[Optional[List[Dict]], Optional[List[Dict]], Optional[Dict], Optional[List[Dict]], Optional[List[Dict]], Optional[List[Dict]]]:
    if API_DISABLED:
        graham_logger.warning(f"API disabled: Attempting cache fallback for {ticker}")
        cached = get_stock_data_from_db(ticker)
        if cached:
            return (
                cached.get('raw_income_data', []),
                cached.get('raw_balance_data', []),
                cached.get('raw_dividend_data', {}),
                cached.get('raw_profile_data', {}),
                cached.get('raw_cash_flow_data', []),
                cached.get('raw_key_metrics_data', [])
            )
        graham_logger.error(f"No cache for {ticker} and API disabled")
        return None, None, None, None, None, None
    
    api_keys = keys if USE_FREE_API_KEY else [keys[0]]
    graham_logger.debug(f"Fetching data for {ticker} using {'all keys' if USE_FREE_API_KEY else 'paid key only'}")
    income_data = await asyncio.wait_for(fetch_with_multiple_keys_async(ticker, "income-statement", api_keys, retries=3, update_rate_limit=update_rate_limit, cancel_event=cancel_event), timeout=30)
    balance_data = await asyncio.wait_for(fetch_with_multiple_keys_async(ticker, "balance-sheet-statement", api_keys, retries=3, update_rate_limit=update_rate_limit, cancel_event=cancel_event), timeout=30)
    dividend_data = await asyncio.wait_for(fetch_with_multiple_keys_async(ticker, "historical-price-full/stock_dividend", api_keys, retries=3, update_rate_limit=update_rate_limit, cancel_event=cancel_event), timeout=30)
    profile_data = await asyncio.wait_for(fetch_with_multiple_keys_async(ticker, "profile", api_keys, retries=3, update_rate_limit=update_rate_limit, cancel_event=cancel_event), timeout=30)
    cash_flow_data = await asyncio.wait_for(fetch_with_multiple_keys_async(ticker, "cash-flow-statement", api_keys, retries=3, update_rate_limit=update_rate_limit, cancel_event=cancel_event), timeout=30)
    key_metrics_data = await asyncio.wait_for(fetch_with_multiple_keys_async(ticker, "key-metrics", api_keys, retries=3, update_rate_limit=update_rate_limit, cancel_event=cancel_event), timeout=30)
    
    data_list = [income_data, balance_data, dividend_data, profile_data, cash_flow_data, key_metrics_data]
    critical_endpoints = ["income", "balance", "cash_flow", "key_metrics"]
    missing_critical = [name for i, name in enumerate(critical_endpoints) if data_list[i] is None or data_list[i] == [] or data_list[i] == {}]
    if missing_critical:
        graham_logger.error(f"Critical data missing for {ticker}: {', '.join(missing_critical)}. Aborting fetch.")
        return None, None, None, None, None, None
    
    empty_endpoints = [name for name, d in zip(["income", "balance", "dividend", "profile", "cash_flow", "key_metrics"], data_list) if d == [] or d == {}]
    if empty_endpoints:
        graham_logger.warning(f"FMP data for {ticker} is empty for: {', '.join(empty_endpoints)}. Proceeding with partial data.")
    
    graham_logger.info(f"Successfully fetched FMP data for {ticker}")
    return income_data, balance_data, dividend_data, profile_data, cash_flow_data, key_metrics_data

async def fetch_historical_data(ticker: str, exchange: str = "Stock", update_rate_limit=None, cancel_event=None, income_data=None, balance_data=None, dividend_data=None, cash_flow_data=None) -> Tuple[List[float], List[float], List[float], List[float], List[int], Dict[str, float], List[Dict], float]:
    """Process FMP data into historical financial metrics for Graham analysis and calculate Free Cash Flow (FCF)."""
    roe_list = []
    rotc_list = []
    eps_list = []
    div_list = []
    revenue = {}
    balance_data_list = []
    free_cash_flow = None
    if not income_data or not balance_data or not dividend_data or not cash_flow_data:
        graham_logger.error(f"No income, balance, dividend, or cash flow data provided for {ticker}")
        return [], [], [], [], [], {}, [], None
    try:
        years_income = [int(entry['date'].split('-')[0]) for entry in income_data if 'date' in entry]
        years_balance = [int(entry['date'].split('-')[0]) for entry in balance_data if 'date' in entry]
    except (KeyError, ValueError) as e:
        graham_logger.error(f"Invalid date format in data for {ticker}: {str(e)}")
        return [], [], [], [], [], {}, [], None
    years_available = sorted(set(years_income) & set(years_balance))[:10]
    if not years_available:
        graham_logger.error(f"No common years found for {ticker} between income and balance data")
        return [], [], [], [], [], {}, [], None
    if len(years_available) < 10:
        graham_logger.warning(f"{ticker}: Only {len(years_available)} years available, may affect scoring")
    dividend_history = dividend_data.get('historical', [])
    div_dict = {year: 0.0 for year in years_available}
    for div_entry in dividend_history:
        if 'date' in div_entry and 'adjDividend' in div_entry:
            try:
                div_year = int(div_entry['date'].split('-')[0])
                adj_dividend = float(div_entry['adjDividend'])
                if div_year in years_available:
                    div_dict[div_year] += adj_dividend
                    graham_logger.debug(f"{ticker}: Added dividend {adj_dividend} for year {div_year} (total now {div_dict[div_year]})")
            except (ValueError, TypeError) as e:
                graham_logger.warning(f"{ticker}: Invalid dividend entry {div_entry}: {str(e)}")
    graham_logger.info(f"{ticker}: Dividend summing complete: {div_dict}")
    if cash_flow_data and cash_flow_data[0]:
        latest_cash_flow = cash_flow_data[0]
        operating_cash_flow = latest_cash_flow.get('netCashProvidedByOperatingActivities')
        capex = latest_cash_flow.get('capitalExpenditure')
        if operating_cash_flow is not None and capex is not None:
            try:
                operating_cash_flow = float(operating_cash_flow)
                capex = float(capex)
                free_cash_flow = operating_cash_flow + capex
                graham_logger.info(f"{ticker}: Calculated FCF = {operating_cash_flow} + {capex} = {free_cash_flow}")
            except (ValueError, TypeError) as e:
                graham_logger.error(f"{ticker}: Error converting cash flow data to float: {str(e)}")
                free_cash_flow = None
        else:
            graham_logger.warning(f"{ticker}: Missing cash flow data - operating_cash_flow: {operating_cash_flow}, capex: {capex}")
            free_cash_flow = None
    else:
        graham_logger.warning(f"{ticker}: No cash flow data available")
        free_cash_flow = None
    for year in years_available:
        year_str = str(year)
        income_entry = next((entry for entry in income_data if entry['date'].startswith(year_str)), None)
        balance_entry = next((entry for entry in balance_data if entry['date'].startswith(year_str)), None)
        if not income_entry or not balance_entry:
            continue
        net_income = float(income_entry.get('netIncome', 0.0))
        equity = float(balance_entry.get('totalStockholdersEquity', 1.0))
        roe = (net_income / equity * 100) if equity != 0 else 0.0
        roe_list.append(roe)
        operating_income = float(income_entry.get('operatingIncome', 0.0))
        total_assets = float(balance_entry.get('totalAssets', 1.0))
        rotc = (operating_income / total_assets * 100) if total_assets != 0 else 0.0
        rotc_list.append(rotc)
        eps = float(income_entry.get('eps', 0.0))
        eps_list.append(eps)
        revenue[year_str] = float(income_entry.get('revenue', 0.0))
        div = div_dict.get(year, 0.0)
        div_list.append(div)
        balance_data_list.append(balance_entry)
    graham_logger.debug(f"Fetched data for {ticker}: ROE={len(roe_list)}, ROTC={len(rotc_list)}, EPS={len(eps_list)}, Div={len(div_list)}, FCF={free_cash_flow}")
    return roe_list, rotc_list, eps_list, div_list, years_available, revenue, balance_data_list, free_cash_flow

def calculate_cagr(start_value, end_value, years):
    if start_value <= 0 or end_value <= 0 or years <= 0:
        return 0
    return (end_value / start_value) ** (1 / years) - 1

def calculate_common_criteria(ticker: str, eps_list: List[float], div_list: List[float], revenue: Dict[str, float], balance_data: List[Dict], debt_to_equity: Optional[float], available_data_years: int, latest_revenue: float) -> Optional[int]:
    """Calculate Graham score with data validation (6 common criteria)."""
    if available_data_years < 10:
        graham_logger.warning(f"{ticker}: Insufficient data years - {available_data_years} < 10")
        return None
    if len(eps_list) < 10 or len(div_list) < 10:
        graham_logger.warning(f"{ticker}: Incomplete lists - EPS len={len(eps_list)}, Div len={len(div_list)}")
        return None
    if not balance_data or 'totalCurrentAssets' not in balance_data[0] or 'totalCurrentLiabilities' not in balance_data[0]:
        graham_logger.warning(f"Missing required balance sheet fields for {ticker}")
        return None
    score = 0
    revenue_passed = latest_revenue >= 500_000_000
    graham_logger.debug(f"{ticker}: Criterion 1 - Revenue >= $500M: {'Yes' if revenue_passed else 'No'} (${latest_revenue / 1e6:.2f}M)")
    if revenue_passed:
        score += 1
    latest_balance = balance_data[-1]
    current_assets = float(latest_balance.get('totalCurrentAssets', 0))
    current_liabilities = float(latest_balance.get('totalCurrentLiabilities', 1))
    current_ratio = current_assets / current_liabilities if current_liabilities > 0 else 0
    current_passed = current_ratio > 2
    graham_logger.debug(f"{ticker}: Criterion 2 - Current Ratio > 2: {'Yes' if current_passed else 'No'} ({current_ratio:.2f})")
    if current_passed:
        score += 1
    negative_eps_count = sum(1 for eps in eps_list if eps <= 0)
    stability_passed = negative_eps_count == 0
    graham_logger.debug(f"{ticker}: Criterion 3 - All Positive EPS: {'Yes' if stability_passed else 'No'} (Negative EPS years: {negative_eps_count}, EPS List: {eps_list})")
    if stability_passed:
        score += 1
    dividend_passed = all(div > 0 for div in div_list) if div_list else False  # False if empty
    graham_logger.debug(f"{ticker}: Criterion 4 - Uninterrupted Dividends: {'Yes' if dividend_passed else 'No'} (Div List: {div_list})")
    if dividend_passed:
        score += 1
    if len(eps_list) >= 2 and eps_list[0] > 0 and eps_list[-1] > 0:
        cagr = calculate_cagr(eps_list[0], eps_list[-1], available_data_years - 1)
        growth_passed = cagr > 0.03
        graham_logger.debug(f"{ticker}: Criterion 5 - EPS CAGR > 3%: {'Yes' if growth_passed else 'No'} ({cagr:.2%}, EPS List: {eps_list})")
        if growth_passed:
            score += 1
    else:
        graham_logger.debug(f"{ticker}: Criterion 5 - EPS CAGR > 3%: No (invalid EPS), EPS List: {eps_list}")
    debt_passed = debt_to_equity is not None and debt_to_equity < 2
    graham_logger.debug(f"{ticker}: Criterion 6 - Debt-to-Equity < 2: {'Yes' if debt_passed else 'No'} ({debt_to_equity if debt_to_equity is not None else 'N/A'})")
    if debt_passed:
        score += 1
    graham_logger.debug(f"{ticker}: Common Score = {score}/6 with {available_data_years} years of data")
    return score

def calculate_financial_common_criteria(ticker: str, eps_list: List[float], div_list: List[float], revenue: Dict[str, float],
                                       balance_data: List[Dict], key_metrics_data: List[Dict], available_data_years: int,
                                       latest_revenue: float) -> Optional[int]:
    if available_data_years < 10:
        graham_logger.warning(f"{ticker}: Insufficient data - {available_data_years} years")
        return None
    if len(eps_list) < 10 or len(div_list) < 10:
        graham_logger.warning(f"{ticker}: Incomplete lists - EPS len={len(eps_list)}, Div len={len(div_list)}")
        return None
    if not balance_data:
        graham_logger.warning(f"{ticker}: No balance data available")
        return None
    score = 0
    revenue_passed = latest_revenue >= 500_000_000
    if revenue_passed:
        score += 1
    latest_balance = balance_data[-1]  # Use latest (most recent) entry
    current_assets = float(latest_balance.get('totalCurrentAssets', 0))
    current_liabilities = float(latest_balance.get('totalCurrentLiabilities', 1))
    current_ratio = current_assets / current_liabilities if current_liabilities > 0 else 0
    if current_ratio > 2:
        score += 1
    if sum(1 for eps in eps_list if eps <= 0) == 0:
        score += 1
    dividend_passed = all(div > 0 for div in div_list) if div_list else False  # False if empty
    if dividend_passed:
        score += 1
    if len(eps_list) >= 2 and eps_list[0] > 0 and eps_list[-1] > 0:
        cagr = calculate_cagr(eps_list[0], eps_list[-1], available_data_years - 1)
        if cagr > 0.03:
            score += 1
    else:
        graham_logger.debug(f"{ticker}: Skipping EPS CAGR (insufficient valid data)")
    total_assets = float(latest_balance.get('totalAssets', 1))
    equity = float(latest_balance.get('totalStockholdersEquity', 0))
    equity_to_assets = (equity / total_assets) * 100 if total_assets > 0 else 0
    if equity_to_assets > 10:
        score += 1
    graham_logger.debug(f"{ticker}: Equity/Assets >10%: {'Yes' if equity_to_assets > 10 else 'No'} ({equity_to_assets:.2f}%)")
    bank_metrics = get_bank_metrics(key_metrics_data)
    if bank_metrics['roa'] is not None and bank_metrics['roa'] > 0.01:
        score += 1
    if bank_metrics['roe'] is not None and bank_metrics['roe'] > 0.10:
        score += 1
    if bank_metrics['netInterestMargin'] is not None and bank_metrics['netInterestMargin'] > 0.03:
        score += 1
    graham_logger.debug(f"{ticker}: Financial Score = {score}/9 with {available_data_years} years")
    return score

def calculate_graham_score_8(ticker: str, price: float, pe_ratio: Optional[float], pb_ratio: Optional[float],
                             debt_to_equity: Optional[float], eps_list: List[float], div_list: List[float],
                             revenue: Dict[str, float], balance_data: List[Dict], key_metrics_data: List[Dict],
                             available_data_years: int, latest_revenue: float, sector: str) -> int:
    if available_data_years < 10:
        graham_logger.warning(f"{ticker}: Insufficient data years for full Graham score - {available_data_years} < 10")
        return 0  # Or None, but return 0 to allow partial scoring
    if len(eps_list) < 10 or len(div_list) < 10:
        graham_logger.warning(f"{ticker}: Incomplete lists for Graham score - EPS len={len(eps_list)}, Div len={len(div_list)}")
        return 0
    is_financial = (sector == "Financials")
    if is_financial:
        common_score = calculate_financial_common_criteria(ticker, eps_list, div_list, revenue, balance_data,
                                                          key_metrics_data, available_data_years, latest_revenue)
    else:
        common_score = calculate_common_criteria(ticker, eps_list, div_list, revenue, balance_data, debt_to_equity,
                                                available_data_years, latest_revenue)
    if common_score is None:
        return 0
    score = common_score
    if pe_ratio is not None and pe_ratio <= 15:
        score += 1
    pb_passed = False
    if is_financial:
        tangible_bvps = get_tangible_book_value_per_share(key_metrics_data)
        ptbv_ratio = price / tangible_bvps if tangible_bvps and tangible_bvps > 0 else None
        pb_passed = ptbv_ratio <= 1.5 if ptbv_ratio is not None else False
    else:
        pb_passed = pb_ratio <= 1.5 if pb_ratio is not None else False
    if pb_passed:
        score += 1
    return score

def get_bank_metrics(key_metrics_data: list) -> dict:
    """Extract latest ROA, ROE, and Net Interest Margin from key_metrics_data for financial stocks."""
    if not key_metrics_data:
        return {'roa': None, 'roe': None, 'netInterestMargin': None}
    
    latest = key_metrics_data[0]
    
    def safe_float(value):
        try:
            return float(value) if value is not None else None
        except (ValueError, TypeError):
            graham_logger.warning(f"Failed to convert value to float: {value} (type: {type(value)})")
            return None
    
    roa = safe_float(latest.get('returnOnAssets'))
    roe = safe_float(latest.get('returnOnEquity'))
    nim = safe_float(latest.get('netInterestMargin'))
    
    return {'roa': roa, 'roe': roe, 'netInterestMargin': nim}

async def calculate_graham_value(earnings: Optional[float], stock_data: dict) -> float:
    if not earnings or earnings <= 0:
        return float('nan')
    aaa_yield = cache_manager.get_aaa_yield(FRED_API_KEY)
    if aaa_yield <= 0:
        return float('nan')
    eps_cagr = stock_data.get('eps_cagr', 0.0)
    g = max(eps_cagr * 100, 0)
    max_multiplier = 15 if stock_data.get('sector') == "Financials" else 20
    earnings_multiplier = min(8.5 + 2 * g, max_multiplier)
    normalization_factor = 4.4
    value = (earnings * earnings_multiplier * normalization_factor) / (100 * aaa_yield)
    return value

def map_fmp_sector_to_app(fmp_sector: str) -> str:
    """Map FMP sector names to app's standard sector categories."""
    sector_mapping = {
        'Technology': 'Information Technology',
        'Financial Services': 'Financials',
        'Consumer Cyclical': 'Consumer Discretionary',
        'Consumer Defensive': 'Consumer Staples',
        'Healthcare': 'Health Care',
        'Communication Services': 'Communication Services',
        'Energy': 'Energy',
        'Industrials': 'Industrials',
        'Basic Materials': 'Materials',
        'Real Estate': 'Real Estate',
        'Utilities': 'Utilities'
    }
    return sector_mapping.get(fmp_sector, 'Unknown')

async def fetch_batch_data(tickers, screening_mode=True, expected_return=0.0, margin_of_safety=0.33, exchange="Stock", ticker_manager=None, update_rate_limit=None, cancel_event=None, adr_tickers=None):
    """Fetch data in batches: FMP during screening, YFinance price + cache during analysis."""
    if API_DISABLED:
        graham_logger.warning("API disabled: Fetching from cache only where possible.")
    if ticker_manager is None:
        ticker_manager = TickerManager(NYSE_LIST_FILE, NASDAQ_LIST_FILE)
        await ticker_manager.initialize()
    results = []
    error_tickers = []
    tasks = []
    cache_hits = 0
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TICKERS)
    nyse_tickers = ticker_manager.get_tickers("NYSE")
    nasdaq_tickers = ticker_manager.get_tickers("NASDAQ")
    def save_to_db(result):
        future = cache_manager.get_db_executor().submit(_save_to_db, result)
        future.result()
    def _save_to_db(result):
        if 'timestamp' not in result or result['timestamp'] is None:
            result['timestamp'] = time.time()
        conn, cursor = cache_manager.get_stocks_connection()
        try:
            ticker_exchange = result['exchange']
            if ticker_exchange == "NYSE":
                ticker_list_hash = get_file_hash(NYSE_LIST_FILE)
            elif ticker_exchange == "NASDAQ":
                ticker_list_hash = get_file_hash(NASDAQ_LIST_FILE)
            else:
                ticker_list_hash = "Unknown"
            cursor.execute(
                """INSERT OR REPLACE INTO stocks 
                   (ticker, date, roe, rotc, eps, dividend, ticker_list_hash, balance_data, 
                    cash_flow_data, key_metrics_data, timestamp, company_name, debt_to_equity, 
                    eps_ttm, book_value_per_share, common_score, latest_revenue, 
                    available_data_years, sector, years,
                    latest_total_assets, latest_total_liabilities, latest_shares_outstanding,
                    latest_long_term_debt, latest_short_term_debt, latest_current_assets,
                    latest_current_liabilities, latest_book_value, historic_pe_ratios,
                    latest_net_income, eps_cagr, latest_free_cash_flow,
                    raw_income_data, raw_balance_data, raw_dividend_data, raw_profile_data,
                    raw_cash_flow_data, raw_key_metrics_data, exchange, is_foreign) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (result['ticker'], datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 ",".join(map(str, result['roe_list'] if 'roe_list' in result else [])),
                 ",".join(map(str, result['rotc_list'] if 'rotc_list' in result else [])),
                 ",".join(map(str, result['eps_list'] if 'eps_list' in result else [])),
                 ",".join(map(str, result['div_list'] if 'div_list' in result else [])),
                 ticker_list_hash, 
                 json.dumps(result['balance_data'] if 'balance_data' in result else []),
                 json.dumps(result['cash_flow_data'] if 'cash_flow_data' in result else []),
                 json.dumps(result['key_metrics_data'] if 'key_metrics_data' in result else []),
                 result['timestamp'], result['company_name'], result['debt_to_equity'],
                 result['eps_ttm'], result['book_value_per_share'], result['common_score'],
                 result['latest_revenue'], result['available_data_years'], result['sector'],
                 ",".join(map(str, result['years'] if 'years' in result else [])),
                 result['latest_total_assets'], result['latest_total_liabilities'],
                 result['latest_shares_outstanding'], result['latest_long_term_debt'],
                 result['latest_short_term_debt'], result['latest_current_assets'],
                 result['latest_current_liabilities'], result['latest_book_value'],
                 result['historic_pe_ratios'], result['latest_net_income'], result['eps_cagr'],
                 result['free_cash_flow'],
                 json.dumps(result['raw_income_data'] if 'raw_income_data' in result else []),
                 json.dumps(result['raw_balance_data'] if 'raw_balance_data' in result else []),
                 json.dumps(result['raw_dividend_data'] if 'raw_dividend_data' in result else {}),
                 json.dumps(result['raw_profile_data'] if 'raw_profile_data' in result else {}),
                 json.dumps(result['raw_cash_flow_data'] if 'raw_cash_flow_data' in result else []),
                 json.dumps(result['raw_key_metrics_data'] if 'raw_key_metrics_data' in result else []),
                 result.get('exchange', 'Unknown'),
                 result.get('is_foreign', False)
                )
            )
            conn.commit()
            graham_logger.info(f"Saved {result['ticker']} to database with hash {ticker_list_hash}")
        except sqlite3.Error as e:
            graham_logger.error(f"Database error saving {result['ticker']}: {str(e)}")
        finally:
            conn.close()

    async def fetch_data(ticker):
        try:
            async with semaphore:
                if cancel_event and cancel_event.is_set():
                    return {"ticker": ticker, "exchange": exchange, "error": "Cancelled by user"}
                ticker_exchange = "NYSE" if ticker in nyse_tickers else "NASDAQ" if ticker in nasdaq_tickers else "Unknown"
                if ticker_exchange == "Unknown":
                    graham_logger.warning(f"Skipping {ticker}: Not found in NYSE or NASDAQ tickers")
                    return {"ticker": ticker, "exchange": ticker_exchange, "error": "Invalid ticker"}
                cached_data = get_stock_data_from_db(ticker)
                current_time = time.time()
                if cached_data and cached_data['timestamp'] and (current_time - cached_data['timestamp'] < CACHE_EXPIRY) and cached_data.get('years'):
                    nonlocal cache_hits
                    cache_hits += 1
                    graham_logger.debug(f"Cache hit for {ticker}: Years={cached_data['years']}")
                    if screening_mode:
                        return {
                            "ticker": ticker,
                            "exchange": ticker_exchange,
                            "company_name": cached_data['company_name'],
                            "common_score": cached_data['common_score'],
                            "available_data_years": cached_data['available_data_years'],
                            "sector": cached_data['sector'],
                            "is_foreign": cached_data.get('is_foreign', False)
                        }
                    else:
                        # Fetch price from Yahoo Finance
                        max_retries = 3
                        delay = 1
                        for attempt in range(max_retries):
                            try:
                                stock = yf.Ticker(ticker)
                                history = stock.history(period="1d")
                                if not history.empty:
                                    price = history['Close'].iloc[-1]
                                else:
                                    raise ValueError(f"No historical price data for {ticker}")
                                break
                            except requests.exceptions.HTTPError as e:
                                if e.response.status_code == 429 and attempt < max_retries - 1:
                                    graham_logger.warning(f"Rate limit hit for {ticker}, retrying in {delay} seconds...")
                                    await asyncio.sleep(delay)
                                    delay *= 2
                                else:
                                    graham_logger.error(f"Error fetching price for {ticker}: {str(e)}")
                                    return {"ticker": ticker, "exchange": ticker_exchange, "error": f"YFinance fetch failed: {str(e)}"}
                            except Exception as e:
                                graham_logger.error(f"Error fetching price for {ticker}: {str(e)}")
                                return {"ticker": ticker, "exchange": ticker_exchange, "error": f"YFinance fetch failed: {str(e)}"}
                        else:
                            graham_logger.error(f"Failed to fetch price for {ticker} after {max_retries} attempts")
                            return {"ticker": ticker, "exchange": ticker_exchange, "error": "Rate limit exceeded"}

                        pe_ratio = price / cached_data['eps_ttm'] if cached_data['eps_ttm'] and cached_data['eps_ttm'] > 0 else None
                        pb_ratio = price / cached_data['book_value_per_share'] if cached_data['book_value_per_share'] and cached_data['book_value_per_share'] > 0 else None
                        intrinsic_value = await calculate_graham_value(cached_data['eps_ttm'], cached_data) if cached_data['eps_ttm'] and cached_data['eps_ttm'] > 0 else float('nan')
                        buy_price = intrinsic_value * (1 - margin_of_safety) if not pd.isna(intrinsic_value) else float('nan')
                        sell_price = intrinsic_value * (1 + expected_return) if not pd.isna(intrinsic_value) else float('nan')
                        revenue = {str(cached_data['years'][-1]): cached_data['latest_revenue']} if cached_data['years'] and cached_data['latest_revenue'] else {}
                        graham_score = calculate_graham_score_8(
                            ticker, price, pe_ratio, pb_ratio, cached_data['debt_to_equity'],
                            cached_data['eps_list'], cached_data['div_list'], revenue,
                            cached_data['balance_data'], cached_data['key_metrics_data'],
                            cached_data['available_data_years'],
                            cached_data['latest_revenue'],
                            cached_data['sector']
                        )
                        result = {
                            "ticker": ticker,
                            "exchange": ticker_exchange,
                            "company_name": cached_data['company_name'],
                            "price": price,
                            "common_score": cached_data['common_score'],
                            "graham_score": graham_score,
                            "roe_list": cached_data['roe_list'],
                            "rotc_list": cached_data['rotc_list'],
                            "eps_list": cached_data['eps_list'],
                            "div_list": cached_data['div_list'],
                            "years": cached_data.get('years', []),
                            "balance_data": cached_data['balance_data'],
                            "cash_flow_data": cached_data['cash_flow_data'],
                            "key_metrics_data": cached_data['key_metrics_data'],
                            "available_data_years": cached_data['available_data_years'],
                            "latest_revenue": cached_data['latest_revenue'],
                            "debt_to_equity": cached_data['debt_to_equity'],
                            "eps_ttm": cached_data['eps_ttm'],
                            "book_value_per_share": cached_data['book_value_per_share'],
                            "pe_ratio": pe_ratio,
                            "pb_ratio": pb_ratio,
                            "intrinsic_value": intrinsic_value,
                            "buy_price": buy_price,
                            "sell_price": sell_price,
                            "sector": cached_data['sector'],
                            "latest_total_assets": cached_data['latest_total_assets'],
                            "latest_total_liabilities": cached_data['latest_total_liabilities'],
                            "latest_shares_outstanding": cached_data['latest_shares_outstanding'],
                            "latest_long_term_debt": cached_data['latest_long_term_debt'],
                            "latest_short_term_debt": cached_data['latest_short_term_debt'],
                            "latest_current_assets": cached_data['latest_current_assets'],
                            "latest_current_liabilities": cached_data['latest_current_liabilities'],
                            "latest_book_value": cached_data['latest_book_value'],
                            "historic_pe_ratios": cached_data['historic_pe_ratios'],
                            "latest_net_income": cached_data['latest_net_income'],
                            "eps_cagr": cached_data.get('eps_cagr', 0.0),
                            "free_cash_flow": cached_data.get('latest_free_cash_flow', 0.0),
                            "is_foreign": cached_data.get('is_foreign', False)
                        }
                        save_to_db(result)
                        return result
                else:
                    if cached_data and not cached_data.get('years'):
                        graham_logger.warning(f"Cache for {ticker} missing 'years', refetching data")
                    # Fetch fresh data
                    await asyncio.sleep(DELAY_BETWEEN_CALLS)
                    security_name = ticker_manager.get_security_name(ticker)
                    income_data, balance_data, dividend_data, profile_data, cash_flow_data, key_metrics_data = await fetch_fmp_data(ticker, FMP_API_KEYS, update_rate_limit, cancel_event)
                    if not all([income_data, balance_data, dividend_data, profile_data, cash_flow_data, key_metrics_data]):
                        missing = [name for name, data in [
                            ("income_data", income_data),
                            ("balance_data", balance_data),
                            ("dividend_data", dividend_data),
                            ("profile_data", profile_data),
                            ("cash_flow_data", cash_flow_data),
                            ("key_metrics_data", key_metrics_data)
                        ] if not data]
                        graham_logger.warning(f"Skipping {ticker} due to missing data: {', '.join(missing)}")
                        return {"ticker": ticker, "exchange": ticker_exchange, "error": f"Missing data: {', '.join(missing)}"}
                    company_name = profile_data[0].get('companyName', security_name) if profile_data else security_name
                    sector = map_fmp_sector_to_app(profile_data[0].get('sector', 'Unknown')) if profile_data else 'Unknown'
                    KNOWN_ADRS = KNOWN_NYSE_ADRS if ticker_exchange == "NYSE" else KNOWN_NASDAQ_ADRS
                    is_adr_keyword = any(keyword in security_name.lower() for keyword in ['american depositary', 'adr', 'ads', 'global depositary'])
                    country = profile_data[0].get('country', '') if profile_data else ''  # Empty fallback
                    ticker_upper = ticker.upper()
                    is_foreign = (
                        (country and country.upper() not in ['US', 'UNITED STATES', 'USA']) or
                        is_adr_keyword or
                        (adr_tickers is not None and ticker_upper in adr_tickers) or
                        ticker_upper in KNOWN_ADRS
                    ) and ticker_upper not in KNOWN_US_ONLY
                    if country == '':
                        graham_logger.warning(f"{ticker}: API missing 'country' - fallback to keyword/known list check: is_foreign={is_foreign}")
                    graham_logger.debug(f"{ticker}: is_foreign={is_foreign} (country={country}, ADR_keyword={is_adr_keyword}, known_ADR={ticker_upper in KNOWN_ADRS}, US_only={ticker_upper in KNOWN_US_ONLY})")

                    latest_balance = balance_data[0] if balance_data else None
                    shareholder_equity = float(latest_balance['totalStockholdersEquity']) if latest_balance and 'totalStockholdersEquity' in latest_balance else None
                    long_term_debt = float(latest_balance['longTermDebt']) if latest_balance and 'longTermDebt' in latest_balance else None
                    debt_to_equity = long_term_debt / shareholder_equity if shareholder_equity and shareholder_equity != 0 and long_term_debt is not None else None
                    shares_outstanding = float(income_data[0]['weightedAverageShsOut']) if income_data and 'weightedAverageShsOut' in income_data[0] else None
                    book_value_per_share = shareholder_equity / shares_outstanding if shares_outstanding and shares_outstanding > 0 and shareholder_equity is not None else None

                    roe_list, rotc_list, eps_list, div_list, years_available, revenue, balance_data_list, free_cash_flow = await fetch_historical_data(
                        ticker, ticker_exchange, update_rate_limit, cancel_event, income_data, balance_data, dividend_data, cash_flow_data
                    )
                    latest_net_income = float(income_data[0]['netIncome']) if income_data and 'netIncome' in income_data[0] else None
                    available_data_years = len(years_available)
                    latest_revenue = revenue.get(str(years_available[-1]) if years_available else '', 0)
                    eps_ttm = eps_list[-1] if eps_list else None
                    common_score = calculate_common_criteria(ticker, eps_list, div_list, revenue, balance_data_list, debt_to_equity, available_data_years, latest_revenue)
                    tangible_bvps = get_tangible_book_value_per_share(key_metrics_data)

                    if len(eps_list) >= 2 and eps_list[0] > 0 and eps_list[-1] > 0:
                        eps_cagr = calculate_cagr(eps_list[0], eps_list[-1], len(eps_list) - 1)
                    else:
                        eps_cagr = 0.0

                    latest_total_assets = float(latest_balance['totalAssets']) if latest_balance and 'totalAssets' in latest_balance else None
                    latest_total_liabilities = float(latest_balance['totalLiabilities']) if latest_balance and 'totalLiabilities' in latest_balance else None
                    latest_shares_outstanding = shares_outstanding
                    latest_long_term_debt = long_term_debt
                    latest_short_term_debt = float(latest_balance['shortTermDebt']) if latest_balance and 'shortTermDebt' in latest_balance else None
                    latest_current_assets = float(latest_balance['totalCurrentAssets']) if latest_balance and 'totalCurrentAssets' in latest_balance else None
                    latest_current_liabilities = float(latest_balance['totalCurrentLiabilities']) if latest_balance and 'totalCurrentLiabilities' in latest_balance else None
                    latest_book_value = shareholder_equity
                    historic_pe_ratios = json.dumps([entry.get('peRatio', 0) for entry in key_metrics_data]) if key_metrics_data else '[]'

                    full_result = {
                        "ticker": ticker,
                        "exchange": ticker_exchange,
                        "company_name": company_name,
                        "roe_list": roe_list,
                        "rotc_list": rotc_list,
                        "eps_list": eps_list,
                        "div_list": div_list,
                        "years": years_available,
                        "balance_data": balance_data_list,
                        "cash_flow_data": cash_flow_data,
                        "key_metrics_data": key_metrics_data,
                        "available_data_years": available_data_years,
                        "latest_revenue": latest_revenue,
                        "debt_to_equity": debt_to_equity,
                        "eps_ttm": eps_ttm,
                        "book_value_per_share": book_value_per_share,
                        "timestamp": time.time(),
                        "common_score": common_score,
                        "sector": sector,
                        "latest_total_assets": latest_total_assets,
                        "latest_total_liabilities": latest_total_liabilities,
                        "latest_shares_outstanding": latest_shares_outstanding,
                        "latest_long_term_debt": long_term_debt,
                        "latest_short_term_debt": latest_short_term_debt,
                        "latest_current_assets": latest_current_assets,
                        "latest_current_liabilities": latest_current_liabilities,
                        "latest_book_value": latest_book_value,
                        "historic_pe_ratios": historic_pe_ratios,
                        "latest_net_income": latest_net_income,
                        "eps_cagr": eps_cagr,
                        "free_cash_flow": free_cash_flow,
                        "raw_income_data": income_data,
                        "raw_balance_data": balance_data,
                        "raw_dividend_data": dividend_data,
                        "raw_profile_data": profile_data,
                        "raw_cash_flow_data": cash_flow_data,
                        "raw_key_metrics_data": key_metrics_data,
                        "is_foreign": is_foreign
                    }

                    if not screening_mode:
                        max_retries = 3
                        delay = 1
                        for attempt in range(max_retries):
                            try:
                                stock = yf.Ticker(ticker)
                                history = stock.history(period="1d")
                                if not history.empty:
                                    price = history['Close'].iloc[-1]
                                else:
                                    raise ValueError(f"No historical price data for {ticker}")
                                break
                            except requests.exceptions.HTTPError as e:
                                if e.response.status_code == 429 and attempt < max_retries - 1:
                                    graham_logger.warning(f"Rate limit hit for {ticker}, retrying in {delay} seconds...")
                                    await asyncio.sleep(delay)
                                    delay *= 2
                                else:
                                    graham_logger.error(f"Error fetching price for {ticker}: {str(e)}")
                                    return {"ticker": ticker, "exchange": ticker_exchange, "error": f"YFinance fetch failed: {str(e)}"}
                            except Exception as e:
                                graham_logger.error(f"Error fetching price for {ticker}: {str(e)}")
                                return {"ticker": ticker, "exchange": ticker_exchange, "error": f"YFinance fetch failed: {str(e)}"}
                        else:
                            graham_logger.error(f"Failed to fetch price for {ticker} after {max_retries} attempts")
                            return {"ticker": ticker, "exchange": ticker_exchange, "error": "Rate limit exceeded"}

                        pe_ratio = price / full_result['eps_ttm'] if full_result['eps_ttm'] and full_result['eps_ttm'] > 0 else None
                        pb_ratio = price / full_result['book_value_per_share'] if full_result['book_value_per_share'] and full_result['book_value_per_share'] > 0 else None
                        intrinsic_value = await calculate_graham_value(full_result['eps_ttm'], full_result) if full_result['eps_ttm'] and full_result['eps_ttm'] > 0 else float('nan')
                        buy_price = intrinsic_value * (1 - margin_of_safety) if not pd.isna(intrinsic_value) else float('nan')
                        sell_price = intrinsic_value * (1 + expected_return) if not pd.isna(intrinsic_value) else float('nan')
                        graham_score = calculate_graham_score_8(
                            ticker, price, pe_ratio, pb_ratio, full_result['debt_to_equity'],
                            full_result['eps_list'], full_result['div_list'], revenue,
                            full_result['balance_data'], full_result['key_metrics_data'],
                            full_result['available_data_years'],
                            full_result['latest_revenue'],
                            full_result['sector']
                        )
                        full_result.update({
                            "price": price,
                            "pe_ratio": pe_ratio,
                            "pb_ratio": pb_ratio,
                            "intrinsic_value": intrinsic_value,
                            "buy_price": buy_price,
                            "sell_price": sell_price,
                            "graham_score": graham_score
                        })

                    save_to_db(full_result)
                    if screening_mode:
                        return {
                            "ticker": ticker,
                            "exchange": ticker_exchange,
                            "company_name": company_name,
                            "common_score": common_score,
                            "available_data_years": available_data_years,
                            "sector": sector,
                            "is_foreign": is_foreign
                        }
                    return full_result
        except Exception as e:
            graham_logger.error(f"Error processing {ticker}: {str(e)}")
            return {"ticker": ticker, "exchange": ticker_exchange, "error": f"Processing failed: {str(e)}"}

    for ticker in tickers:
        tasks.append(fetch_data(ticker))

    try:
        results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=60)
    except asyncio.TimeoutError:
        graham_logger.error("Timeout while fetching batch data")
        results = []

    valid_results = []
    error_tickers = []
    for ticker, result in zip(tickers, results):
        if isinstance(result, Exception):
            graham_logger.error(f"Exception processing {ticker}: {str(result)}")
            error_tickers.append(ticker)
        elif isinstance(result, dict):
            if 'error' in result:
                graham_logger.error(f"Batch error for {result['ticker']}: {result['error']}")
                error_tickers.append(ticker)
                graham_logger.debug(f"Error for {ticker}: {result['error']}")
            else:
                valid_results.append(result)
        else:
            graham_logger.error(f"Unexpected result type for {ticker}: {type(result)}")
            error_tickers.append(ticker)
        
        if error_tickers:
            await asyncio.sleep(1)

    graham_logger.debug(f"Batch fetch complete: {len(valid_results)} valid, {len(error_tickers)} errors, {cache_hits} cache hits")
    graham_logger.debug(f"Error tickers during batch fetch: {error_tickers}")
    return valid_results, error_tickers, cache_hits

async def fetch_stock_data(ticker, expected_return=0.0, margin_of_safety=0.33, exchange="Stock", ticker_manager=None, update_rate_limit=None, cancel_event=None, adr_tickers=None):
    results, error_tickers = await fetch_batch_data(
        [ticker],
        screening_mode=False,
        expected_return=expected_return,
        margin_of_safety=margin_of_safety,
        exchange=exchange,
        ticker_manager=ticker_manager,
        update_rate_limit=update_rate_limit,
        cancel_event=cancel_event,
        adr_tickers=adr_tickers
    )
    if not results or 'error' in results[0]:
        graham_logger.error(f"Failed to fetch data for {ticker}")
        raise ValueError(f"Failed to fetch data for {ticker}")
    return results[0]

async def save_qualifying_stocks_to_favorites(qualifying_stocks, exchange):
    """Save qualifying stocks to favorites."""
    list_name = f"{exchange}_Qualifying_Stocks"
    with FAVORITES_LOCK:
        favorites = load_favorites()
        if not isinstance(favorites, dict):
            favorites = {}
        favorites[list_name] = qualifying_stocks
        with open(FAVORITES_FILE, 'w') as f:
            json.dump(favorites, f, indent=4)
    graham_logger.info(f"Saved qualifying stocks to {list_name}")
    return list_name

def load_favorites():
    if os.path.exists(FAVORITES_FILE):
        try:
            with open(FAVORITES_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            graham_logger.error(f"Error loading favorites: {str(e)}")
    return {}

def get_stock_data_from_db(ticker):
    conn, cursor = cache_manager.get_stocks_connection()
    try:
        cursor.execute("SELECT * FROM stocks WHERE ticker = ?", (ticker,))
        row = cursor.fetchone()
        if row:
            columns = [desc[0] for desc in cursor.description]
            stock_dict = dict(zip(columns, row))
            years = [int(y) for y in stock_dict['years'].split(",")] if stock_dict.get('years') else []
            roe_list = [float(x) if x.strip() else 0.0 for x in stock_dict['roe'].split(",")] if stock_dict['roe'] else []
            rotc_list = [float(x) if x.strip() else 0.0 for x in stock_dict['rotc'].split(",")] if stock_dict['rotc'] else []
            eps_list = [float(x) if x.strip() else 0.0 for x in stock_dict['eps'].split(",")] if stock_dict['eps'] else []
            div_list = [float(x) if x.strip() else 0.0 for x in stock_dict['dividend'].split(",")] if stock_dict['dividend'] else []
            if years:
                data = list(zip(years, roe_list, rotc_list, eps_list, div_list))
                data.sort(key=lambda x: x[0])
                years, roe_list, rotc_list, eps_list, div_list = zip(*data)
                years = list(years)
                roe_list = list(roe_list)
                rotc_list = list(rotc_list)
                eps_list = list(eps_list)
                div_list = list(div_list)
            result = {
                "ticker": stock_dict['ticker'],
                "date": stock_dict['date'],
                "roe_list": roe_list,
                "rotc_list": rotc_list,
                "eps_list": eps_list,
                "div_list": div_list,
                "years": years,
                "balance_data": json.loads(stock_dict['balance_data']) if stock_dict['balance_data'] else [],
                "cash_flow_data": json.loads(stock_dict['cash_flow_data']) if stock_dict.get('cash_flow_data') else [],
                "key_metrics_data": json.loads(stock_dict['key_metrics_data']) if stock_dict.get('key_metrics_data') else [],
                "timestamp": stock_dict['timestamp'],
                "company_name": stock_dict['company_name'],
                "debt_to_equity": stock_dict.get('debt_to_equity'),
                "eps_ttm": stock_dict.get('eps_ttm'),
                "book_value_per_share": stock_dict.get('book_value_per_share'),
                "common_score": stock_dict.get('common_score', 0),
                "latest_revenue": stock_dict.get('latest_revenue', 0.0),
                "available_data_years": stock_dict['available_data_years'],
                "sector": stock_dict.get('sector', 'Unknown'),
                "latest_total_assets": stock_dict.get('latest_total_assets'),
                "latest_total_liabilities": stock_dict.get('latest_total_liabilities'),
                "latest_shares_outstanding": stock_dict.get('latest_shares_outstanding'),
                "latest_long_term_debt": stock_dict.get('latest_long_term_debt'),
                "latest_short_term_debt": stock_dict.get('latest_short_term_debt'),
                "latest_current_assets": stock_dict.get('latest_current_assets'),
                "latest_current_liabilities": stock_dict.get('latest_current_liabilities'),
                "latest_book_value": stock_dict.get('latest_book_value'),
                "historic_pe_ratios": json.loads(stock_dict.get('historic_pe_ratios', '[]')),
                "latest_net_income": stock_dict.get('latest_net_income'),
                "eps_cagr": stock_dict.get('eps_cagr', 0.0),
                "latest_free_cash_flow": stock_dict.get('latest_free_cash_flow'),
                "exchange": stock_dict.get('exchange', 'Unknown'),
                "is_foreign": stock_dict.get('is_foreign', False)
            }
            graham_logger.debug(f"Retrieved from DB for {ticker}: Years={years}, Dividend={div_list}, EPS={eps_list}, FCF={stock_dict.get('latest_free_cash_flow', 0.0)}")
            return result
        return None
    except sqlite3.Error as e:
        graham_logger.error(f"Database error fetching {ticker}: {str(e)}")
        return None
    finally:
        conn.close()

def clear_in_memory_caches():
    graham_logger.info("No in-memory caches to clear (relying solely on SQLite database)")

def get_tangible_book_value_per_share(key_metrics_data: List[Dict]) -> Optional[float]:
    if key_metrics_data and 'tangibleBookValuePerShare' in key_metrics_data[0]:
        return float(key_metrics_data[0]['tangibleBookValuePerShare'])
    return None

async def screen_exchange_graham_stocks(exchange: str, batch_size: int = 18, cancel_event=None, tickers=None, root=None, update_progress_animated=None, refresh_favorites_dropdown=None, ticker_manager=None, update_rate_limit=None, min_criteria: int = 6, sector_filter: Optional[str] = None, separate_financials: bool = False, adr_tickers=None) -> Tuple[List[str], List[int], List[str], List[str], List[str]]:
    """Screen stocks for given exchange with configurable minimum criteria threshold and sector filter."""
    graham_logger.info(f"Starting {exchange} Graham screening with min_criteria={min_criteria}, sector_filter={sector_filter or 'All'}, separate_financials={separate_financials}, {len(tickers) if tickers else 'all'} tickers")
    if ticker_manager is None:
        ticker_manager = TickerManager(NYSE_LIST_FILE, NASDAQ_LIST_FILE)
        await ticker_manager.initialize()
    file_path = NYSE_LIST_FILE if exchange == "NYSE" else NASDAQ_LIST_FILE
    current_file_hash = get_file_hash(file_path)
    conn, cursor = cache_manager.get_stocks_connection()
    try:
        invalid_file = os.path.join(USER_DATA_DIR, f"{exchange} Invalid Tickers.txt")
        # Do not clear invalid_file here; load existing persistent errors
        # Check stored hash
        cursor.execute("SELECT file_hash FROM screening_progress WHERE exchange=? AND ticker='FILE_HASH_TRACKER'", (exchange,))
        last_hash_row = cursor.fetchone()
        last_file_hash = last_hash_row[0] if last_hash_row else None
        files_updated = (current_file_hash != last_file_hash)
        graham_logger.info(f"{exchange} ticker files updated since last screen: {files_updated}")
        if last_file_hash != current_file_hash:
            graham_logger.info(f"New version of {file_path} detected. Resetting data.")
            cursor.execute("DELETE FROM screening_progress WHERE exchange=?", (exchange,))
            cursor.execute("DELETE FROM stocks WHERE exchange=? AND ticker_list_hash != ?", (exchange, current_file_hash))
            cursor.execute("DELETE FROM graham_qualifiers WHERE exchange=?", (exchange,))
            conn.commit()
        ticker_list = list(ticker_manager.get_tickers(exchange))
        # Filter out invalid tickers
        valid_tickers = [t for t in ticker_list if ticker_manager.is_valid_ticker(t)]
        invalid_tickers = set(ticker_list) - set(valid_tickers)
        if invalid_tickers:
            graham_logger.warning(f"Invalid tickers excluded: {invalid_tickers}")
            # Append new invalid to existing file (no clear)
            with open(invalid_file, 'a') as f:
                for ticker in sorted(invalid_tickers):
                    f.write(ticker + '\n')
        filtered_ticker_data = [{"ticker": ticker} for ticker in valid_tickers]
        tickers = filtered_ticker_data if tickers is None else [t for t in tickers if ticker_manager.is_valid_ticker(t["ticker"])]
        valid_tickers = [t["ticker"] for t in tickers]
        graham_logger.info(f"Processing {len(valid_tickers)} {exchange} tickers")
        # Load persistent errors from invalid file
        persistent_errors = set()
        if os.path.exists(invalid_file):
            with open(invalid_file, 'r') as f:
                persistent_errors = {line.strip() for line in f if line.strip()}
        # If files not updated, exclude persistent errors from screening
        if not files_updated:
            valid_tickers = [t for t in valid_tickers if t not in persistent_errors]
            graham_logger.info(f"Skipped {len(persistent_errors)} persistent errors (files not updated)")
        qualifying_stocks, common_scores, exchanges = [], [], []
        total_tickers = len(valid_tickers)
        processed_tickers = 0
        passed_tickers = 0
        error_tickers = []
        financial_qualifying_stocks = []
        non_financial_qualifying_stocks = []
        sample_interval = max(10, total_tickers // 50)
        dynamic_batch_size = min(batch_size, max(10, MAX_CALLS_PER_MINUTE_PAID // 6))
        total_start_time = time.time()
        if root and update_progress_animated:
            root.after(0, lambda p=0, t=valid_tickers, pt=0, e=0: update_progress_animated(p, t, pt, e))
        for i in range(0, len(valid_tickers), dynamic_batch_size):
            if cancel_event and cancel_event.is_set():
                graham_logger.info("Screening cancelled by user")
                break
            batch_start = time.time()
            batch = valid_tickers[i:i + dynamic_batch_size]
            batch_results, batch_error_tickers, cache_hits = await fetch_batch_data(
                batch,
                True,
                exchange=exchange,
                ticker_manager=ticker_manager,
                update_rate_limit=update_rate_limit,
                cancel_event=cancel_event,
                adr_tickers=adr_tickers
            )
            error_tickers.extend(batch_error_tickers)
            for result in batch_results:
                ticker = result['ticker']
                log_full = (processed_tickers < 10) or (processed_tickers >= total_tickers - 10) or (processed_tickers % sample_interval == 0)
                if 'error' in result:
                    if log_full:
                        graham_logger.warning(f"Skipping {ticker} due to error: {result['error']}")
                    continue
                common_score = result.get('common_score')
                available_data_years = result.get('available_data_years', 0)
                sector = result.get('sector', 'Unknown')
                if separate_financials and sector != "Financials":
                    if log_full:
                        graham_logger.info(f"{ticker}: Skipped - Not financial (sector: {sector})")
                    continue
                if sector_filter and sector != sector_filter:
                    if log_full:
                        graham_logger.info(f"{ticker}: Skipped - Sector {sector} != {sector_filter}")
                    continue
                if available_data_years >= 10 and common_score is not None and common_score >= min_criteria:
                    if sector == "Financials":
                        financial_qualifying_stocks.append(ticker)
                    else:
                        non_financial_qualifying_stocks.append(ticker)
                    if log_full:
                        graham_logger.info(f"{ticker}: Qualified with {common_score}/{min_criteria} criteria met (sector: {sector})")
                    qualifying_stocks.append(ticker)
                    common_scores.append(common_score)
                    exchanges.append(result['exchange'])
                    passed_tickers += 1
                    cursor.execute(
                        "INSERT OR REPLACE INTO graham_qualifiers (ticker, common_score, date, sector, exchange, min_criteria) VALUES (?, ?, ?, ?, ?, ?)",
                        (ticker, common_score, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), result['sector'], result['exchange'], min_criteria)
                    )
                else:
                    if log_full:
                        reason = "Insufficient data years" if available_data_years < 10 else f"Score {common_score} below threshold {min_criteria}" if common_score is not None else "No score calculated"
                        graham_logger.info(f"{ticker}: Disqualified - {reason} (sector: {sector})")
                processed_tickers += 1
                progress = (processed_tickers / total_tickers) * 100
                elapsed_total = time.time() - total_start_time
                avg_time_per_ticker = elapsed_total / processed_tickers if processed_tickers > 0 else 0
                remaining_tickers = total_tickers - processed_tickers
                eta = remaining_tickers * avg_time_per_ticker
                if root and update_progress_animated:
                    try:
                        root.after(0, lambda p=progress, t=valid_tickers, pt=passed_tickers, e=eta: update_progress_animated(p, t, pt, e))
                    except Exception as e:
                        graham_logger.error(f"Error scheduling progress update for {ticker}: {str(e)}")
                cursor.execute(
                    "INSERT OR REPLACE INTO screening_progress (exchange, ticker, timestamp, file_hash, status) VALUES (?, ?, ?, ?, ?)",
                    (exchange, ticker, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), current_file_hash, "completed")
                )
                conn.commit()
            graham_logger.info(f"Batch {i // dynamic_batch_size + 1} took {time.time() - batch_start:.2f} seconds")
        # Revalidate error_tickers with retries
        max_retries = 1  # Configurable; set to 0 to disable
        screening_errors = error_tickers[:]  # Separate screening errors from persistent
        error_tickers = []  # Reset to collect final
        for retry in range(max_retries):
            if not screening_errors:
                break
            retry_batch_size = dynamic_batch_size
            retry_qualifying = []
            retry_scores = []
            retry_exchanges = []
            retry_errors = []
            for j in range(0, len(screening_errors), retry_batch_size):
                if cancel_event and cancel_event.is_set():
                    graham_logger.info("Revalidation cancelled by user")
                    break
                retry_batch = screening_errors[j:j + retry_batch_size]
                retry_results, batch_retry_errors, _ = await fetch_batch_data(
                    retry_batch,
                    screening_mode=True,
                    exchange=exchange,
                    ticker_manager=ticker_manager,
                    update_rate_limit=update_rate_limit,
                    cancel_event=cancel_event,
                    adr_tickers=adr_tickers
                )
                retry_errors.extend(batch_retry_errors)
                for result in retry_results:
                    ticker = result['ticker']
                    if 'error' in result:
                        retry_errors.append(ticker)
                        continue
                    common_score = result.get('common_score')
                    available_data_years = result.get('available_data_years', 0)
                    sector = result.get('sector', 'Unknown')
                    if separate_financials and sector != "Financials":
                        continue
                    if sector_filter and sector != sector_filter:
                        continue
                    if available_data_years >= 10 and common_score is not None and common_score >= min_criteria:
                        if sector == "Financials":
                            financial_qualifying_stocks.append(ticker)
                        else:
                            non_financial_qualifying_stocks.append(ticker)
                        retry_qualifying.append(ticker)
                        retry_scores.append(common_score)
                        retry_exchanges.append(result['exchange'])
                        passed_tickers += 1
                        cursor.execute(
                            "INSERT OR REPLACE INTO graham_qualifiers (ticker, common_score, date, sector, exchange, min_criteria) VALUES (?, ?, ?, ?, ?, ?)",
                            (ticker, common_score, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), result['sector'], result['exchange'], min_criteria)
                        )
            # Update main lists with successful retries
            qualifying_stocks.extend(retry_qualifying)
            common_scores.extend(retry_scores)
            exchanges.extend(retry_exchanges)
            screening_errors = retry_errors[:]  # Update for next retry if any
            graham_logger.info(f"Revalidation retry {retry + 1}: Recovered {len(retry_qualifying)} tickers, {len(screening_errors)} remaining screening errors")
        error_tickers.extend(screening_errors)  # Add remaining screening errors to final
        # If files updated, retry persistent errors
        if files_updated:
            persistent_retry_qualifying = []
            persistent_retry_scores = []
            persistent_retry_exchanges = []
            persistent_retry_errors = []
            for j in range(0, len(persistent_errors), dynamic_batch_size):
                if cancel_event and cancel_event.is_set():
                    graham_logger.info("Persistent errors retry cancelled by user")
                    break
                retry_batch = list(persistent_errors)[j:j + dynamic_batch_size]  # Since set, convert to list
                retry_results, batch_retry_errors, _ = await fetch_batch_data(
                    retry_batch,
                    screening_mode=True,
                    exchange=exchange,
                    ticker_manager=ticker_manager,
                    update_rate_limit=update_rate_limit,
                    cancel_event=cancel_event,
                    adr_tickers=adr_tickers
                )
                persistent_retry_errors.extend(batch_retry_errors)
                for result in retry_results:
                    ticker = result['ticker']
                    if 'error' in result:
                        persistent_retry_errors.append(ticker)
                        continue
                    common_score = result.get('common_score')
                    available_data_years = result.get('available_data_years', 0)
                    sector = result.get('sector', 'Unknown')
                    if separate_financials and sector != "Financials":
                        continue
                    if sector_filter and sector != sector_filter:
                        continue
                    if available_data_years >= 10 and common_score is not None and common_score >= min_criteria:
                        if sector == "Financials":
                            financial_qualifying_stocks.append(ticker)
                        else:
                            non_financial_qualifying_stocks.append(ticker)
                        persistent_retry_qualifying.append(ticker)
                        persistent_retry_scores.append(common_score)
                        persistent_retry_exchanges.append(result['exchange'])
                        passed_tickers += 1
                        cursor.execute(
                            "INSERT OR REPLACE INTO graham_qualifiers (ticker, common_score, date, sector, exchange, min_criteria) VALUES (?, ?, ?, ?, ?, ?)",
                            (ticker, common_score, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), result['sector'], result['exchange'], min_criteria)
                        )
            # Update main lists with successful persistent retries
            qualifying_stocks.extend(persistent_retry_qualifying)
            common_scores.extend(persistent_retry_scores)
            exchanges.extend(persistent_retry_exchanges)
            # Remove recovered from persistent_errors
            persistent_errors -= set(persistent_retry_qualifying)
            # Add remaining retry errors back to persistent
            persistent_errors.update(persistent_retry_errors)
            graham_logger.info(f"Persistent errors retry (files updated): Recovered {len(persistent_retry_qualifying)} tickers, {len(persistent_errors)} remaining persistent errors")
        else:
            graham_logger.info(f"No retry for {len(persistent_errors)} persistent errors (files unchanged)")
        # Final error_tickers includes remaining screening errors and persistent (but persistent not retried unless updated)
        error_tickers = list(set(error_tickers) | persistent_errors)
        # Write final persistent errors to file (overwrite with current set)
        with open(invalid_file, 'w') as f:
            for ticker in sorted(error_tickers):
                f.write(ticker + '\n')
        graham_logger.info(f"Updated {len(error_tickers)} invalid tickers in {invalid_file}")
        graham_logger.info(f"Completed {exchange} screening with min_criteria={min_criteria}: {processed_tickers}/{total_tickers} processed, {passed_tickers} passed, {len(error_tickers)} errors")
        if error_tickers and total_tickers <= 20:
            graham_logger.info(f"Error tickers: {error_tickers}")
        elif error_tickers:
            sample_size = min(5, len(error_tickers))
            error_sample = random.sample(error_tickers, sample_size)
            graham_logger.info(f"Error tickers (random sample): {error_sample} (and {len(error_tickers) - sample_size} more)")

        # Update FILE_HASH_TRACKER after screening (even on cancel/error) to stabilize future runs
        try:
            cursor.execute(
                "INSERT OR REPLACE INTO screening_progress (exchange, ticker, timestamp, file_hash, status) VALUES (?, ?, ?, ?, ?)",
                (exchange, 'FILE_HASH_TRACKER', datetime.now().strftime('%Y-%m-%d %H:%M:%S'), current_file_hash, 'completed')
            )
            conn.commit()
            graham_logger.info(f"Updated FILE_HASH_TRACKER for {exchange} with hash {current_file_hash}")
        except sqlite3.Error as e:
            graham_logger.error(f"Failed to update FILE_HASH_TRACKER: {str(e)}")

        return qualifying_stocks, common_scores, exchanges, error_tickers, financial_qualifying_stocks
    except Exception as e:
        graham_logger.error(f"Screening error for {exchange}: {str(e)}")
        return [], [], [], [], []
    finally:
        conn.close()

def get_aaa_yield(api_key: str = FRED_API_KEY, default_yield: float = 0.045) -> float:
    """Wrapper for cache_manager.get_aaa_yield."""
    return cache_manager.get_aaa_yield(api_key, default_yield)

if __name__ == "__main__":
    test_tickers = [{"ticker": "IBM"}, {"ticker": "JPM"}, {"ticker": "KO"}]
    asyncio.run(screen_exchange_graham_stocks(exchange="NYSE", tickers=test_tickers))