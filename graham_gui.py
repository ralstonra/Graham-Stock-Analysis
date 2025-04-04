import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog, simpledialog
import asyncio
import logging
import pandas as pd
import sqlite3
import sys
import threading
import time
import os
import json
import yfinance as yf
from datetime import datetime
from graham_data import (screen_nasdaq_graham_stocks, screen_nyse_graham_stocks, fetch_batch_data,
                         fetch_stock_data, get_stocks_connection, fetch_with_multiple_keys_async,
                         NYSE_LIST_FILE, NASDAQ_LIST_FILE, TickerManager, get_file_hash,
                         calculate_graham_value, calculate_graham_score_8, calculate_common_criteria, 
                         clear_in_memory_caches, save_qualifying_stocks_to_favorites, get_stock_data_from_db)
from config import BASE_DIR, FMP_API_KEYS, FAVORITES_FILE, paid_rate_limiter, free_rate_limiter, CACHE_EXPIRY
import queue
import shutil
import requests
import ftplib

# Logging setup
logger = logging.getLogger()
if not logger.handlers:
    logging.basicConfig(filename=os.path.join(BASE_DIR, 'nyse_graham_screen.log'), level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logger.addHandler(console)

FAVORITES_LOCK = threading.Lock()
DATA_DIR = BASE_DIR  # Assuming DATA_DIR is the same as BASE_DIR; adjust if necessary

class GrahamScreeningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Analysis (Graham Defensive)")
        self.root.geometry("1200x900")

        self.nyse_screen_var = tk.BooleanVar(value=False)
        self.nasdaq_screen_var = tk.BooleanVar(value=False)
        self.cancel_event = threading.Event()
        self.margin_of_safety_var = tk.DoubleVar(value=33.0)
        self.expected_return_var = tk.DoubleVar(value=0.0)
        self.ticker_manager = TickerManager(NYSE_LIST_FILE, NASDAQ_LIST_FILE)
        self.screening_active = False
        self.analysis_lock = threading.Lock()
        self.ticker_cache = {}
        self.ticker_cache_lock = threading.Lock()

        self.task_queue = queue.Queue()
        self.asyncio_thread = threading.Thread(target=self.run_asyncio_loop, args=(self.task_queue,), daemon=True)
        self.asyncio_thread.start()

        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(expand=True, fill="both", padx=5, pady=5)
        self.main_frame.grid_rowconfigure((0, 1), weight=1)
        self.main_frame.grid_columnconfigure((0, 1, 2), weight=1, minsize=400)

        self.left_frame = ttk.Frame(self.main_frame, width=400)
        self.middle_frame = ttk.Frame(self.main_frame, width=400)
        self.right_frame = ttk.Frame(self.main_frame, width=400)

        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=2)
        self.middle_frame.grid(row=0, column=1, sticky="nsew", padx=2)
        self.right_frame.grid(row=0, column=2, sticky="nsew", padx=2)

        self.left_frame.grid_columnconfigure(0, weight=1)
        self.left_frame.grid_rowconfigure(tuple(range(19)), weight=1)
        self.middle_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", font=("Helvetica", 9), background="#f0f0f0")
        style.configure("TButton", font=("Helvetica", 9), padding=2)
        style.configure("TCheckbutton", font=("Helvetica", 9))
        style.configure("TEntry", font=("Helvetica", 9))
        style.configure("TCombobox", font=("Helvetica", 9))
        style.configure("TProgressbar", thickness=15)

        self.create_widgets()

        paid_rate_limiter.on_sleep = self.on_rate_limit_sleep

    def on_rate_limit_sleep(self, sleep_time):
        message = f"Rate limit reached, pausing for {sleep_time / 60:.1f} minutes"
        self.root.after(0, lambda: self.rate_limit_label.config(text=message))

    def run_asyncio_loop(self, task_queue):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        while True:
            coro = task_queue.get()
            if coro is None:
                break
            loop.run_until_complete(coro)
        loop.close()

    def create_widgets(self):
        ttk.Label(self.left_frame, text="Enter Stock Tickers (comma-separated, e.g., AOS, AAPL):").grid(row=0, column=0, pady=1, sticky="ew")
        self.entry = ttk.Combobox(self.left_frame, width=50, values=list(self.ticker_manager.get_tickers("NYSE") | self.ticker_manager.get_tickers("NASDAQ")))
        self.entry.grid(row=1, column=0, pady=1, sticky="ew")

        def validate_tickers(*args):
            tickers = self.parse_tickers(self.entry.get())
            return bool(tickers)

        self.entry.bind('<FocusOut>', lambda e: validate_tickers())

        ttk.Label(self.left_frame, text="Search Results:").grid(row=2, column=0, pady=1, sticky="ew")
        self.search_entry = ttk.Entry(self.left_frame, width=50)
        self.search_entry.grid(row=3, column=0, pady=1, sticky="ew")
        self.search_entry.bind('<KeyRelease>', self.filter_tree)

        self.favorites = self.load_favorites()
        ttk.Label(self.left_frame, text="Favorite Lists:").grid(row=4, column=0, pady=1, sticky="ew")
        self.favorite_var = tk.StringVar(value="Select Favorite")
        self.favorite_menu = ttk.Combobox(self.left_frame, textvariable=self.favorite_var, values=list(self.favorites.keys()))
        self.favorite_menu.grid(row=5, column=0, pady=1, sticky="ew")

        def load_favorite(event=None):
            selected = self.favorite_var.get()
            if selected and selected != "Select Favorite":
                self.entry.delete(0, tk.END)
                tickers = self.favorites[selected]
                self.entry.insert(0, ",".join(tickers))

        self.favorite_menu.bind('<<ComboboxSelected>>', load_favorite)

        def save_favorite():
            if not validate_tickers():
                messagebox.showwarning("Invalid Tickers", "No valid tickers entered.")
                return
            name = simpledialog.askstring("Save Favorite", "Enter list name:")
            if name and self.entry.get().strip():
                tickers = self.parse_tickers(self.entry.get())
                valid_tickers = [t for t in tickers if self.ticker_manager.is_valid_ticker(t)]
                if not valid_tickers:
                    messagebox.showwarning("Invalid Tickers", "No valid NYSE or NASDAQ tickers.")
                    return
                self.favorites[name] = valid_tickers
                self.save_favorites()
                self.favorite_menu['values'] = list(self.favorites.keys())
                self.favorite_var.set(name)

        ttk.Button(self.left_frame, text="Save Favorite", command=save_favorite).grid(row=6, column=0, pady=1, sticky="ew")
        ttk.Button(self.left_frame, text="Manage Favorites", command=self.manage_favorites).grid(row=7, column=0, pady=1, sticky="ew")

        ttk.Label(self.left_frame, text="Margin of Safety (%):").grid(row=8, column=0, pady=1, sticky="ew")
        self.margin_of_safety_label = ttk.Label(self.left_frame, text="33%")
        self.margin_of_safety_label.grid(row=9, column=0, pady=1, sticky="ew")
        ttk.Scale(self.left_frame, from_=0, to=50, orient=tk.HORIZONTAL, variable=self.margin_of_safety_var,
                  command=lambda value: self.margin_of_safety_label.config(text=f"{int(float(value))}%")).grid(row=10, column=0, pady=1, sticky="ew")

        ttk.Label(self.left_frame, text="Expected Rate of Return (%):").grid(row=11, column=0, pady=1, sticky="ew")
        self.expected_return_label = ttk.Label(self.left_frame, text="0%")
        self.expected_return_label.grid(row=12, column=0, pady=1, sticky="ew")
        ttk.Scale(self.left_frame, from_=0, to=20, orient=tk.HORIZONTAL, variable=self.expected_return_var,
                  command=lambda value: self.expected_return_label.config(text=f"{int(float(value))}%")).grid(row=13, column=0, pady=1, sticky="ew")

        self.analyze_button = ttk.Button(self.left_frame, text="Analyze Stocks", command=self.analyze_multiple_stocks)
        self.analyze_button.grid(row=15, column=0, pady=1, sticky="ew")
        ttk.Button(self.left_frame, text="Refresh Data", command=self.refresh_multiple_stocks).grid(row=16, column=0, pady=1, sticky="ew")

        ttk.Checkbutton(self.middle_frame, text="Run NYSE Graham Screening", variable=self.nyse_screen_var).grid(row=0, column=0, pady=2, padx=2, sticky="w")
        ttk.Button(self.middle_frame, text="Run NYSE Screening", command=self.run_nyse_screening).grid(row=1, column=0, pady=2, sticky="ew")
        ttk.Button(self.middle_frame, text="Cancel Screening", command=self.cancel_screening).grid(row=2, column=0, pady=2, sticky="ew")
        ttk.Button(self.middle_frame, text="Show NYSE Qualifying Stocks", command=self.display_nyse_qualifying_stocks).grid(row=3, column=0, pady=2, sticky="ew")
        ttk.Button(self.middle_frame, text="Export NYSE Qualifying Stocks", command=self.export_nyse_qualifying_stocks).grid(row=4, column=0, pady=2, sticky="ew")
        ttk.Button(self.middle_frame, text="Check Ticker File Ages", command=self.check_file_age).grid(row=5, column=0, pady=1, sticky="ew")
        ttk.Button(self.middle_frame, text="Update Ticker Files", command=self.update_ticker_files).grid(row=6, column=0, pady=1, sticky="ew")

        ttk.Checkbutton(self.right_frame, text="Run NASDAQ Graham Screening", variable=self.nasdaq_screen_var).grid(row=0, column=0, pady=2, padx=2, sticky="w")
        ttk.Button(self.right_frame, text="Run NASDAQ Screening", command=self.run_nasdaq_screening).grid(row=1, column=0, pady=2, sticky="ew")
        ttk.Button(self.right_frame, text="Cancel Screening", command=self.cancel_screening).grid(row=2, column=0, pady=2, sticky="ew")
        ttk.Button(self.right_frame, text="Show NASDAQ Qualifying Stocks", command=self.display_nasdaq_qualifying_stocks).grid(row=3, column=0, pady=2, sticky="ew")
        ttk.Button(self.right_frame, text="Export NASDAQ Qualifying Stocks", command=self.export_nasdaq_qualifying_stocks).grid(row=4, column=0, pady=2, sticky="ew")

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.middle_frame, variable=self.progress_var, maximum=100, length=300, mode='determinate')
        self.progress_bar.grid(row=9, column=0, pady=2, sticky="ew")
        self.progress_label = ttk.Label(self.middle_frame, text="Progress: 0% (0/0 stocks processed, 0 passed)")
        self.progress_label.grid(row=10, column=0, pady=1, sticky="ew")

        self.cache_var = tk.DoubleVar()
        self.cache_bar = ttk.Progressbar(self.middle_frame, variable=self.cache_var, maximum=100, length=300, mode='determinate')
        self.cache_bar.grid(row=11, column=0, pady=2, sticky="ew")
        self.cache_label = ttk.Label(self.middle_frame, text="Cache Usage: 0% (0 cached, 0 fresh)")
        self.cache_label.grid(row=12, column=0, pady=1, sticky="ew")

        self.rate_limit_label = ttk.Label(self.middle_frame, text="")
        self.rate_limit_label.grid(row=13, column=0, pady=1, sticky="ew")
        self.status_label = ttk.Label(self.middle_frame, text="")
        self.status_label.grid(row=14, column=0, pady=1, sticky="ew")

        ttk.Button(self.middle_frame, text="Clear Cache", command=self.clear_cache).grid(row=15, column=0, pady=2, sticky="ew")
        ttk.Button(self.middle_frame, text="Help", command=self.show_help).grid(row=16, column=0, pady=2, sticky="ew")

        self.full_tree_frame = ttk.Frame(self.main_frame)
        self.full_tree_frame.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)
        self.full_tree_frame.grid_columnconfigure(0, weight=1)
        self.full_tree_frame.grid_rowconfigure(0, weight=3)
        self.full_tree_frame.grid_rowconfigure(1, weight=0)
        self.full_tree_frame.grid_rowconfigure(2, weight=1)

        self.tree = ttk.Treeview(self.full_tree_frame, columns=(1, 2, 3, 4, 5, 6, 7), show="tree headings", height=15)
        self.v_scrollbar = ttk.Scrollbar(self.full_tree_frame, orient="vertical", command=self.tree.yview)
        self.h_scrollbar = ttk.Scrollbar(self.full_tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.h_scrollbar.grid(row=1, column=0, sticky="ew")
        self.tree.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)

        self.tree.heading("#0", text="Ticker")
        self.tree.heading(1, text="Company Name")
        self.tree.heading(2, text="Exchange")
        self.tree.heading(3, text="Score")
        self.tree.heading(4, text="Price ($)")
        self.tree.heading(5, text="Intrinsic Value ($)")
        self.tree.heading(6, text="Buy Price ($)")
        self.tree.heading(7, text="Sell Price ($)")
        self.tree.column("#0", width=80, anchor="center")
        self.tree.column(1, width=150, anchor="center")
        self.tree.column(2, width=80, anchor="center")
        self.tree.column(3, width=80, anchor="center")
        self.tree.column(4, width=80, anchor="center")
        self.tree.column(5, width=100, anchor="center")
        self.tree.column(6, width=80, anchor="center")
        self.tree.column(7, width=80, anchor="center")

        self.tree.tag_configure('highlight', background='lightgreen')

        for col in self.tree["columns"]:
            self.tree.heading(col, command=lambda c=col: self.sort_tree(self.tree["columns"].index(c)))

        self.data_frame = ttk.Frame(self.full_tree_frame)
        self.notebook = ttk.Notebook(self.data_frame)
        self.historical_frame = ttk.Frame(self.notebook)
        self.metrics_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.historical_frame, text="Historical Data")
        self.notebook.add(self.metrics_frame, text="Metrics")
        self.data_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")
        self.notebook.grid(row=0, column=0, sticky="nsew")
        self.data_frame.grid_columnconfigure(0, weight=1)
        self.data_frame.grid_rowconfigure(0, weight=1)

        self.tree.bind("<<TreeviewSelect>>", self.update_tabs)

    def parse_tickers(self, tickers_input):
        if not tickers_input or not tickers_input.strip():
            return []
        tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip() and t.strip().isalnum() and len(t.strip()) <= 5]
        return list(set(tickers))

    def load_favorites(self):
        with FAVORITES_LOCK:
            if os.path.exists(FAVORITES_FILE):
                try:
                    with open(FAVORITES_FILE, 'r') as f:
                        favorites = json.load(f)
                        return {k: v if isinstance(v, list) else [] for k, v in favorites.items()}
                except json.JSONDecodeError:
                    logging.error(f"Corrupted favorites file {FAVORITES_FILE}")
                    return {}
            return {}

    def save_favorites(self):
        with FAVORITES_LOCK:
            try:
                if os.path.exists(FAVORITES_FILE):
                    shutil.copy(FAVORITES_FILE, FAVORITES_FILE + ".bak")
                with open(FAVORITES_FILE, 'w') as f:
                    json.dump(self.favorites, f, indent=4)
            except Exception as e:
                logging.error(f"Failed to save favorites: {str(e)}")
                messagebox.showerror("Error", f"Failed to save favorites: {str(e)}")

    def update_progress_animated(self, progress, tickers=None, passed_tickers=0, eta=None):
        self.progress_var.set(progress)
        if tickers is not None and isinstance(tickers, (list, tuple)):
            total_tickers = len(tickers)
            processed = int(progress / 100 * total_tickers)
            eta_text = f", ETA: {eta:.2f} seconds" if eta is not None else ""
            self.progress_label.config(text=f"Progress: {progress:.1f}% ({processed}/{total_tickers} stocks processed, {passed_tickers} passed{eta_text})")
        else:
            eta_text = f", ETA: {eta:.2f} seconds" if eta is not None else ""
            self.progress_label.config(text=f"Progress: {progress:.1f}% (Screening, {passed_tickers} passed{eta_text})")
        self.root.update_idletasks()

    def update_cache_usage(self, cache_hits, total):
        if total > 0:
            cache_percent = (cache_hits / total) * 100
            self.cache_var.set(cache_percent)
            self.cache_label.config(text=f"Cache Usage: {cache_percent:.1f}% ({cache_hits} cached, {total - cache_hits} fresh)")
        else:
            self.cache_var.set(0)
            self.cache_label.config(text="Cache Usage: 0% (0 cached, 0 fresh)")
        self.root.update_idletasks()

    def update_rate_limit(self, message):
        self.rate_limit_label.config(text=message)
        self.root.update_idletasks()

    def refresh_favorites_dropdown(self, selected_list=None):
        self.favorites = self.load_favorites()
        logging.debug(f"Refreshed favorites dropdown with keys: {list(self.favorites.keys())}")
        self.favorite_menu['values'] = list(self.favorites.keys())
        if selected_list and selected_list in self.favorites:
            self.favorite_var.set(selected_list)
            logging.debug(f"Set dropdown to: {selected_list}")

    def cancel_screening(self):
        if messagebox.askyesno("Confirm Cancel", "Are you sure you want to cancel the screening?"):
            self.cancel_event.set()
            self.status_label.config(text="Cancelling screening...")

    def run_screening(self, exchange, screen_func):
        if self.screening_active:
            messagebox.showwarning("Warning", f"A {exchange} screening is already in progress.")
            return
        if (exchange == "NYSE" and not self.nyse_screen_var.get()) or (exchange == "NASDAQ" and not self.nasdaq_screen_var.get()):
            return
        if self.nyse_screen_var.get() and self.nasdaq_screen_var.get():
            messagebox.showwarning("Warning", "Cannot run NYSE and NASDAQ screening simultaneously.")
            self.nyse_screen_var.set(False)
            self.nasdaq_screen_var.set(False)
            return

        file_path = NYSE_LIST_FILE if exchange == "NYSE" else NASDAQ_LIST_FILE
        if not os.path.exists(file_path):
            messagebox.showerror("Error", f"{exchange} list file missing: {file_path}")
            return

        file_age = time.time() - os.path.getmtime(file_path)
        if file_age > 365 * 24 * 60 * 60:
            messagebox.showwarning("Old Data", f"{exchange} list file is over 365 days old.")

        conn, cursor = get_stocks_connection()
        try:
            cursor.execute("SELECT MAX(timestamp) FROM stocks")
            max_timestamp = cursor.fetchone()[0]
            if max_timestamp and time.time() - max_timestamp > 365 * 24 * 60 * 60:
                messagebox.showwarning("Old Data", "The cached data is over a year old. Consider refreshing the data.")
        except sqlite3.Error as e:
            logging.error(f"Error checking data age: {str(e)}")
        finally:
            conn.close()

        self.progress_var.set(0)
        self.cache_var.set(0)
        self.progress_label.config(text=f"Progress: 0% (0/0 stocks processed, 0 passed)")
        self.cache_label.config(text="Cache Usage: 0% (0 cached, 0 fresh)")
        self.rate_limit_label.config(text="")
        self.status_label.config(text=f"Running {exchange} screening...")
        self.root.update()
        self.cancel_event.clear()
        self.screening_active = True

        async def screening_task():
            try:
                await self.ticker_manager.initialize()
                qualifying_stocks, common_scores, exchanges, error_tickers = await screen_func(
                    batch_size=50,
                    cancel_event=self.cancel_event,
                    root=self.root,
                    update_progress_animated=self.update_progress_animated,
                    refresh_favorites_dropdown=self.refresh_favorites_dropdown,
                    ticker_manager=self.ticker_manager,
                    update_rate_limit=self.update_rate_limit
                )
                logging.info(f"Qualifying stocks for {exchange}: {qualifying_stocks}")
                if not self.cancel_event.is_set():
                    if qualifying_stocks:
                        list_name = await save_qualifying_stocks_to_favorites(qualifying_stocks, exchange)
                        if list_name:
                            self.root.after(0, lambda: self.refresh_favorites_dropdown(list_name))
                        else:
                            logging.error(f"Failed to save qualifying stocks for {exchange}")
                    else:
                        logging.info(f"No qualifying stocks found for {exchange}")
                    conn, cursor = get_stocks_connection()
                    try:
                        processed_tickers = cursor.execute("SELECT COUNT(*) FROM screening_progress WHERE exchange=? AND status='completed'", (exchange,)).fetchone()[0]
                        total_tickers = processed_tickers + len(error_tickers)
                        cache_hits = sum(1 for t in ticker_list if get_stock_data_from_db(t) and time.time() - get_stock_data_from_db(t)['timestamp'] < CACHE_EXPIRY)
                        self.root.after(0, lambda: self.update_cache_usage(cache_hits, total_tickers))
                        summary = f"Completed {exchange} screening.\nProcessed {total_tickers} stocks,\nFound {len(qualifying_stocks)} qualifiers,\n{len(error_tickers)} errors"
                        if error_tickers:
                            summary += f"\nError tickers: {', '.join(error_tickers)}"
                        self.root.after(0, lambda: messagebox.showinfo("Screening Complete", summary))
                    except sqlite3.Error as e:
                        logging.error(f"Database error in screening summary: {str(e)}")
                    finally:
                        conn.close()
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Screening failed: {str(e)}"))
                logging.error(f"Screening task failed for {exchange}: {str(e)}", exc_info=True)
            finally:
                self.root.after(0, lambda: self.progress_label.config(text=f"Progress: 100% (Screening Complete - {exchange})"))
                self.root.after(0, lambda: self.status_label.config(text=""))
                setattr(self, f"{exchange.lower()}_screen_var", tk.BooleanVar(value=False))
                self.screening_active = False

        ticker_list = list(self.ticker_manager.get_tickers(exchange))
        self.task_queue.put(screening_task())

    def run_nyse_screening(self):
        self.run_screening("NYSE", screen_nyse_graham_stocks)

    def run_nasdaq_screening(self):
        self.run_screening("NASDAQ", screen_nasdaq_graham_stocks)

    async def fetch_company_name(self, ticker):
        """Fetch company name from YFinance only."""
        try:
            stock = yf.Ticker(ticker)
            return stock.info.get('longName', 'Unknown')
        except Exception as e:
            logging.error(f"Error fetching company name for {ticker}: {str(e)}")
            return 'Unknown'

    def format_float(self, value, precision=2):
        if pd.isna(value) or not isinstance(value, (int, float)):
            return "N/A"
        return f"{value:.{precision}f}"

    async def fetch_cached_data(self, ticker, exchange="Unknown"):
        nyse_file_hash = get_file_hash(NYSE_LIST_FILE)
        nasdaq_file_hash = get_file_hash(NASDAQ_LIST_FILE)

        conn, cursor = get_stocks_connection()
        try:
            cursor.execute("SELECT file_hash, exchange, timestamp FROM screening_progress WHERE ticker=?", (ticker,))
            row = cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                progress_dict = dict(zip(columns, row))
                stored_hash = progress_dict['file_hash']
                stored_exchange = progress_dict['exchange']
                timestamp_str = progress_dict['timestamp']
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S').timestamp() if timestamp_str else 0
                if ((stored_exchange == "NYSE" and stored_hash == nyse_file_hash) or 
                    (stored_exchange == "NASDAQ" and stored_hash == nasdaq_file_hash)) and \
                   (time.time() - timestamp < 365 * 24 * 60 * 60):
                    cursor.execute("SELECT * FROM stocks WHERE ticker=?", (ticker,))
                    stock_row = cursor.fetchone()
                    if stock_row:
                        columns = [desc[0] for desc in cursor.description]
                        stock_dict = dict(zip(columns, stock_row))
                        logging.info(f"Database cache hit for {ticker}")
                        stock = yf.Ticker(ticker)
                        info = stock.info
                        price = info.get('regularMarketPrice', info.get('previousClose', None))
                        if price is None:
                            logging.error(f"No price data for {ticker} from YFinance")
                            return None
                        roe_10y = [float(x) if x else 0.0 for x in stock_dict['roe'].split(",")] if stock_dict['roe'] else [0.0] * stock_dict['available_data_years']
                        rotc_10y = [float(x) if x else 0.0 for x in stock_dict['rotc'].split(",")] if stock_dict['rotc'] else [0.0] * stock_dict['available_data_years']
                        eps_10y = [float(x) if x else 0.0 for x in stock_dict['eps'].split(",")] if stock_dict['eps'] else [0.0] * stock_dict['available_data_years']
                        div_10y = [float(x) if x else 0.0 for x in stock_dict['dividend'].split(",")] if stock_dict['dividend'] else [0.0] * stock_dict['available_data_years']
                        balance_data = json.loads(stock_dict['balance_data']) if stock_dict['balance_data'] else []
                        years = list(range(int(datetime.now().year) - stock_dict['available_data_years'] + 1, int(datetime.now().year) + 1))

                        company_name = stock_dict['company_name']
                        debt_to_equity = stock_dict['debt_to_equity']
                        eps_ttm = stock_dict['eps_ttm']
                        book_value_per_share = stock_dict['book_value_per_share']

                        intrinsic_value = calculate_graham_value(eps_ttm, eps_10y) if eps_ttm is not None and eps_ttm > 0 else float('nan')
                        margin_of_safety = self.margin_of_safety_var.get() / 100
                        expected_return = self.expected_return_var.get() / 100
                        buy_price = intrinsic_value * (1 - margin_of_safety) if not pd.isna(intrinsic_value) else float('nan')
                        sell_price = intrinsic_value * (1 + expected_return) if not pd.isna(intrinsic_value) else float('nan')
                        latest_revenue = stock_dict['latest_revenue']
                        graham_score = calculate_graham_score_8(ticker, price, None, None, debt_to_equity, eps_10y, div_10y, {}, balance_data, stock_dict['available_data_years'], latest_revenue)

                        result = {
                            "ticker": stock_dict['ticker'],
                            "exchange": stored_exchange,
                            "company_name": company_name,
                            "price": price,
                            "intrinsic_value": intrinsic_value,
                            "buy_price": buy_price,
                            "sell_price": sell_price,
                            "graham_score": graham_score,
                            "years": years,
                            "roe_10y": roe_10y,
                            "rotc_10y": rotc_10y,
                            "eps_10y": eps_10y,
                            "div_10y": div_10y,
                            "balance_data": balance_data,
                            "latest_revenue": latest_revenue,
                            "available_data_years": stock_dict['available_data_years'],
                            "eps_ttm": eps_ttm,
                            "book_value_per_share": book_value_per_share,
                            "debt_to_equity": debt_to_equity
                        }
                        with self.ticker_cache_lock:
                            self.ticker_cache[ticker] = result
                        logging.debug(f"Cached data retrieved for {ticker}: Score={graham_score}/8 with {stock_dict['available_data_years']} years")
                        return result
            logging.info(f"Cache miss for {ticker}: Data stale or missing")
            return None
        except sqlite3.Error as e:
            logging.error(f"Database error in fetch_cached_data for {ticker}: {str(e)}")
            return None
        finally:
            conn.close()

    async def determine_exchange(self, ticker):
        await self.ticker_manager.initialize()
        nyse_tickers = self.ticker_manager.get_tickers("NYSE")
        nasdaq_tickers = self.ticker_manager.get_tickers("NASDAQ")
        if ticker in nyse_tickers:
            return "NYSE"
        elif ticker in nasdaq_tickers:
            return "NASDAQ"
        return "Unknown"

    async def analyze_multiple_stocks_async(self, tickers):
        # Exclude invalid tickers from analysis
        nyse_invalid_file = os.path.join(DATA_DIR, "NYSE Invalid Tickers.txt")
        nasdaq_invalid_file = os.path.join(DATA_DIR, "NASDAQ Invalid Tickers.txt")
        invalid_tickers = set()
        for invalid_file in [nyse_invalid_file, nasdaq_invalid_file]:
            if os.path.exists(invalid_file):
                with open(invalid_file, 'r') as f:
                    invalid_tickers.update(f.read().splitlines())
        valid_tickers = [t for t in tickers if t not in invalid_tickers]
        if len(valid_tickers) < len(tickers):
            excluded = set(tickers) - set(valid_tickers)
            logging.info(f"Excluded {len(excluded)} invalid tickers: {excluded}")
            self.root.after(0, lambda: messagebox.showinfo("Excluded Tickers", f"Excluded {len(excluded)} invalid tickers: {', '.join(excluded)}"))
        tickers = valid_tickers

        if not tickers:
            messagebox.showwarning("No Tickers", "No valid tickers to analyze.")
            return

        conn, cursor = get_stocks_connection()
        try:
            cursor.execute("SELECT MAX(timestamp) FROM stocks")
            max_timestamp = cursor.fetchone()[0]
            if max_timestamp and time.time() - max_timestamp > 365 * 24 * 60 * 60:
                messagebox.showwarning("Old Data", "The cached data is over a year old. Consider refreshing the data.")
        except sqlite3.Error as e:
            logging.error(f"Error checking data age: {str(e)}")
        finally:
            conn.close()

        self.progress_var.set(0)
        self.cache_var.set(0)
        self.progress_label.config(text=f"Progress: 0% (0/{len(tickers)} stocks processed, 0 passed)")
        self.cache_label.config(text="Cache Usage: 0% (0 cached, 0 fresh)")
        self.rate_limit_label.config(text="")
        self.status_label.config(text="Analyzing stocks...")
        self.root.update()

        results, error_tickers = await fetch_batch_data(
            tickers,
            screening_mode=False,
            expected_return=self.expected_return_var.get() / 100,
            margin_of_safety=self.margin_of_safety_var.get() / 100,
            ticker_manager=self.ticker_manager,
            update_rate_limit=self.update_rate_limit
        )

        valid_results = [r for r in results if 'error' not in r]
        passed_tickers = sum(1 for r in valid_results if r.get('graham_score', 0) >= 5 and r.get('available_data_years', 0) >= 10)

        if error_tickers:
            error_summary = "\n".join(error_tickers)
            self.root.after(0, lambda: messagebox.showwarning("Analysis Errors", f"Errors occurred for the following tickers:\n{error_summary}"))

        for item in self.tree.get_children():
            self.tree.delete(item)
        for widget in self.historical_frame.winfo_children():
            widget.destroy()
        for widget in self.metrics_frame.winfo_children():
            widget.destroy()

        for result in valid_results:
            years_used = result.get('available_data_years', 0)
            if years_used < 10:
                self.root.after(0, lambda t=result['ticker'], y=years_used: messagebox.showwarning("Insufficient Data", f"{t} has only {y} years of data. Results may be incomplete."))
            warning = f" (based on {years_used} years)" if years_used < 10 else ""
            price = result['price']
            buy_price = result['buy_price']
            tags = ('highlight',) if price <= buy_price and not pd.isna(price) and not pd.isna(buy_price) else ()
            self.tree.insert("", "end", text=result['ticker'], values=(
                result['company_name'],
                result['exchange'],
                f"{result['graham_score']}/8{warning}",
                f"${self.format_float(result['price'])}",
                f"${self.format_float(result['intrinsic_value'])}",
                f"${self.format_float(result['buy_price'])}",
                f"${self.format_float(result['sell_price'])}"
            ), tags=tags)

        self.root.after(0, lambda: self.update_progress_animated(100, tickers, passed_tickers))
        self.root.after(0, lambda: self.status_label.config(text=""))

    def analyze_multiple_stocks(self, tickers_input=None):
        if self.analysis_lock.acquire(timeout=5):
            try:
                if tickers_input is None:
                    tickers_input = self.entry.get()
                tickers = self.parse_tickers(tickers_input)
                self.task_queue.put(self.analyze_multiple_stocks_async(tickers))
            finally:
                self.analysis_lock.release()
        else:
            messagebox.showerror("Error", "Unable to start analysis: Lock timeout.")

    def refresh_multiple_stocks(self, tickers_input=None):
        if tickers_input is None:
            tickers_input = self.entry.get()
        tickers = self.parse_tickers(tickers_input)
        if not tickers:
            messagebox.showwarning("No Tickers", "No valid tickers to refresh.")
            return

        conn, cursor = get_stocks_connection()
        try:
            for ticker in tickers:
                cursor.execute("DELETE FROM stocks WHERE ticker = ?", (ticker,))
                with self.ticker_cache_lock:
                    if ticker in self.ticker_cache:
                        del self.ticker_cache[ticker]
            conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Database error in refresh_multiple_stocks: {str(e)}")
        finally:
            conn.close()
        self.task_queue.put(self.analyze_multiple_stocks_async(tickers))

    async def display_results_in_tree(self, tickers, scores, exchanges, source):
        for item in self.tree.get_children():
            self.tree.delete(item)

        async def fetch_company_names(tickers_list):
            tasks = [self.fetch_company_name(ticker) for ticker in tickers_list]
            return await asyncio.gather(*tasks)

        results = []
        for ticker in tickers:
            with self.ticker_cache_lock:
                if ticker in self.ticker_cache:
                    results.append(self.ticker_cache[ticker])
                    continue

            cached_result = await self.fetch_cached_data(ticker, source)
            if cached_result:
                results.append(cached_result)
                with self.ticker_cache_lock:
                    self.ticker_cache[ticker] = cached_result
                continue

            result = await fetch_batch_data(
                [ticker], 
                screening_mode=False, 
                expected_return=self.expected_return_var.get() / 100,
                margin_of_safety=self.margin_of_safety_var.get() / 100,
                exchange=source, 
                ticker_manager=self.ticker_manager,
                update_rate_limit=self.update_rate_limit
            )
            if result and isinstance(result[0], dict) and "error" not in result[0]:
                result = result[0]
                with self.ticker_cache_lock:
                    self.ticker_cache[ticker] = result
                results.append(result)
            else:
                results.append({"ticker": ticker, "exchange": source, "error": "Failed to fetch data"})

        valid_results = [r for r in results if 'error' not in r]
        company_names = await fetch_company_names(tickers)

        display_list = []
        for i, ticker in enumerate(tickers):
            if i < len(valid_results):
                result = valid_results[i]
                company_name = company_names[i] if i < len(company_names) else 'Unknown'
                display_list.append((ticker, company_name, result['exchange'], scores[i], result))
            else:
                display_list.append((ticker, 'Unknown', exchanges[i] if i < len(exchanges) else source, scores[i], {}))

        display_list.sort(key=lambda x: x[0])

        for ticker, company_name, exchange, common_score, result in display_list:
            price = result.get('price', 0)
            buy_price = result.get('buy_price', float('nan'))
            tags = ('highlight',) if price <= buy_price and not pd.isna(price) and not pd.isna(buy_price) else ()
            self.tree.insert("", "end", text=ticker, values=(
                company_name,
                exchange,
                f"{common_score}/6",
                f"${self.format_float(result.get('price', 0))}",
                f"${self.format_float(result.get('intrinsic_value', float('nan')))}",
                f"${self.format_float(result.get('buy_price', float('nan')))}",
                f"${self.format_float(result.get('sell_price', float('nan')))}"
            ), tags=tags)

        self.sort_tree(0)

    async def fetch_metrics_data(self, ticker):
        stock = yf.Ticker(ticker)
        info = stock.info
        balance_data = await fetch_with_multiple_keys_async(ticker, "balance-sheet-statement", FMP_API_KEYS)
        income_data = await fetch_with_multiple_keys_async(ticker, "income-statement", FMP_API_KEYS)
        revenue = {str(entry.get('calendarYear', 0)): float(entry.get('revenue', 0)) for entry in income_data if 'calendarYear' in entry} if income_data else {}
        latest_balance = balance_data[0] if balance_data and len(balance_data) > 0 else {}
        latest_income = income_data[0] if income_data and len(income_data) > 0 else {}
        return info, revenue, latest_balance, latest_income

    def update_tabs(self, event):
        selected = self.tree.selection()
        if not selected:
            return
        ticker = self.tree.item(selected[0], "text")

        with self.ticker_cache_lock:
            result = self.ticker_cache.get(ticker, None)

        if not result:
            return

        for widget in self.historical_frame.winfo_children():
            widget.destroy()
        for widget in self.metrics_frame.winfo_children():
            widget.destroy()

        historical_text = scrolledtext.ScrolledText(self.historical_frame, width=70, height=10)
        historical_text.grid(row=0, column=0, sticky="nsew")
        metrics_text = scrolledtext.ScrolledText(self.metrics_frame, width=70, height=10)
        metrics_text.grid(row=0, column=0, sticky="nsew")

        years = result.get('years', [])
        roe_available = result.get('roe_10y', [])
        rotc_available = result.get('rotc_10y', [])
        eps_available = result.get('eps_10y', [])
        div_available = result.get('div_10y', [])

        if years and len(years) == len(roe_available) == len(rotc_available) == len(eps_available) == len(div_available):
            historical_text.insert(tk.END, f"{len(years)}-Year Historical Data for {ticker}:\nYear\tROE (%)\tROTC (%)\tEPS ($)\tDividend ($)\n")
            for j in range(len(years)):
                historical_text.insert(tk.END, f"{years[j]}\t{roe_available[j]:.2f}\t{rotc_available[j]:.2f}\t{eps_available[j]:.2f}\t{div_available[j]:.2f}\n")
        else:
            historical_text.insert(tk.END, f"Incomplete Historical Data for {ticker}:\n")

        metrics_text.insert(tk.END, f"Graham Criteria Results for {ticker} (Score: {result['graham_score']}/8 with {result['available_data_years']} years):\n")

        tooltip_label = ttk.Label(self.metrics_frame, text="Hover over criteria for details", font=("Helvetica", 8), foreground="gray")
        tooltip_label.grid(row=1, column=0, pady=2)

        async def update_metrics():
            try:
                info, revenue, latest_balance, latest_income = await self.fetch_metrics_data(ticker)
                latest_revenue = max((float(v) for k, v in revenue.items() if k.isdigit()), default=0)
                revenue_pass = latest_revenue >= 500_000_000
                metrics_text.insert(tk.END, f"1. Revenue >= $500M: {'Yes' if revenue_pass else 'No'} (${latest_revenue / 1e6:.2f}M)\n")

                current_assets = float(latest_balance.get('totalCurrentAssets', 0))
                current_liabilities = float(latest_balance.get('totalCurrentLiabilities', 1))
                current_ratio = current_assets / current_liabilities if current_liabilities > 0 else 0
                current_pass = current_ratio > 2
                metrics_text.insert(tk.END, f"2. Current Ratio > 2: {'Yes' if current_pass else 'No'} ({current_ratio:.2f})\n")

                eps_10y = result.get('eps_10y', [])
                expected_years = result['available_data_years']
                negative_eps_count = sum(1 for eps in eps_10y if eps <= 0)
                max_negative_years = min(2, expected_years // 5)
                stability_pass = negative_eps_count <= max_negative_years
                metrics_text.insert(tk.END, f"3. Earnings Stability (<= {max_negative_years} negative years): {'Yes' if stability_pass else 'No'} ({negative_eps_count} negatives)\n")

                div_10y = result.get('div_10y', [])
                dividend_pass = len(div_10y) >= expected_years and all(div > 0 for div in div_10y)
                metrics_text.insert(tk.END, f"4. Uninterrupted Dividends ({expected_years} yrs): {'Yes' if dividend_pass else 'No'}\n")

                eps_growth_pass = False
                if expected_years >= 2:
                    first_eps = eps_10y[0]
                    last_eps = eps_10y[-1]
                    if first_eps > 0 and last_eps > 0:
                        cagr = (last_eps / first_eps) ** (1 / (expected_years - 1)) - 1
                        eps_growth_pass = cagr > 0.03
                        metrics_text.insert(tk.END, f"5. EPS CAGR > 3%: {'Yes' if eps_growth_pass else 'No'} ({cagr:.2%})\n")
                    else:
                        metrics_text.insert(tk.END, f"5. EPS CAGR > 3%: No (Invalid EPS data)\n")
                else:
                    metrics_text.insert(tk.END, f"5. EPS CAGR > 3%: No (Insufficient data)\n")

                debt_to_equity = result.get('debt_to_equity', None)
                debt_pass = debt_to_equity is not None and debt_to_equity < 2
                metrics_text.insert(tk.END, f"6. Debt-to-Equity < 2: {'Yes' if debt_pass else 'No'} ({self.format_float(debt_to_equity)})\n")

                price = result.get('price', None)
                eps_ttm = result.get('eps_ttm', None)
                book_value_per_share = result.get('book_value_per_share', None)
                pe_ratio = price / eps_ttm if price and eps_ttm and eps_ttm > 0 else None
                pe_pass = pe_ratio is not None and pe_ratio <= 15
                metrics_text.insert(tk.END, f"7. P/E Ratio <= 15: {'Yes' if pe_pass else 'No'} ({self.format_float(pe_ratio)})\n")

                pb_ratio = price / book_value_per_share if price and book_value_per_share and book_value_per_share > 0 else None
                pb_pass = pb_ratio is not None and pb_ratio <= 1.5
                metrics_text.insert(tk.END, f"8. P/B Ratio <= 1.5: {'Yes' if pb_pass else 'No'} ({self.format_float(pb_ratio)})\n")
            except Exception as e:
                logging.error(f"Error updating metrics for {ticker}: {str(e)}")
                metrics_text.insert(tk.END, f"Error fetching metrics: {str(e)}\n")

        self.task_queue.put(update_metrics())

    def manage_favorites(self):
        manage_window = tk.Toplevel(self.root)
        manage_window.title("Manage Favorites")
        manage_window.geometry("400x300")

        favorites_listbox = tk.Listbox(manage_window, height=15)
        favorites_listbox.pack(fill="both", expand=True, padx=5, pady=5)
        for name in self.favorites.keys():
            favorites_listbox.insert(tk.END, name)

        def delete_favorite():
            selected = favorites_listbox.curselection()
            if selected:
                name = favorites_listbox.get(selected[0])
                del self.favorites[name]
                self.save_favorites()
                favorites_listbox.delete(selected[0])
                self.refresh_favorites_dropdown()

        ttk.Button(manage_window, text="Delete Selected", command=delete_favorite).pack(pady=5)
        ttk.Button(manage_window, text="Close", command=manage_window.destroy).pack(pady=5)

    def sort_tree(self, col):
        items = [(self.tree.set(k, col), k) for k in self.tree.get_children()]
        items.sort()
        for index, (val, k) in enumerate(items):
            self.tree.move(k, '', index)

    def display_nyse_qualifying_stocks(self):
        conn, cursor = get_stocks_connection()
        try:
            cursor.execute("""
                SELECT g.ticker, s.company_name, g.common_score 
                FROM graham_qualifiers g 
                JOIN stocks s ON g.ticker = s.ticker 
                WHERE g.exchange='NYSE' AND g.common_score IS NOT NULL
                ORDER BY g.common_score DESC, g.ticker ASC
            """)
            results = cursor.fetchall()

            if not results:
                messagebox.showinfo("No Results", "No NYSE stocks meet the Graham criteria.")
                return

            for item in self.tree.get_children():
                self.tree.delete(item)

            for ticker, company_name, common_score in results:
                self.tree.insert("", "end", text=ticker, values=(company_name, "NYSE", f"{common_score}/6", "", "", "", ""))
        except sqlite3.Error as e:
            logging.error(f"Database error in display_nyse_qualifying_stocks: {str(e)}")
        finally:
            conn.close()

    def display_nasdaq_qualifying_stocks(self):
        conn, cursor = get_stocks_connection()
        try:
            cursor.execute("""
                SELECT g.ticker, s.company_name, g.common_score 
                FROM graham_qualifiers g 
                JOIN stocks s ON g.ticker = s.ticker 
                WHERE g.exchange='NASDAQ' AND g.common_score IS NOT NULL
                ORDER BY g.common_score DESC, g.ticker ASC
            """)
            results = cursor.fetchall()

            if not results:
                messagebox.showinfo("No Results", "No NASDAQ stocks meet the Graham criteria.")
                return

            for item in self.tree.get_children():
                self.tree.delete(item)

            for ticker, company_name, common_score in results:
                self.tree.insert("", "end", text=ticker, values=(company_name, "NASDAQ", f"{common_score}/6", "", "", "", ""))
        except sqlite3.Error as e:
            logging.error(f"Database error in display_nasdaq_qualifying_stocks: {str(e)}")
        finally:
            conn.close()

    def export_nyse_qualifying_stocks(self):
        conn, cursor = get_stocks_connection()
        try:
            cursor.execute("SELECT ticker, company_name, common_score FROM stocks WHERE ticker IN (SELECT ticker FROM graham_qualifiers WHERE exchange='NYSE')")
            results = cursor.fetchall()
            if not results:
                messagebox.showinfo("No Data", "No NYSE qualifying stocks to export.")
                return

            df = pd.DataFrame(results, columns=["Ticker", "Company Name", "Common Score"])
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if file_path:
                df.to_csv(file_path, index=False)
                messagebox.showinfo("Export Successful", f"NYSE qualifying stocks exported to {file_path}")
        except sqlite3.Error as e:
            logging.error(f"Database error in export_nyse_qualifying_stocks: {str(e)}")
        finally:
            conn.close()

    def export_nasdaq_qualifying_stocks(self):
        conn, cursor = get_stocks_connection()
        try:
            cursor.execute("SELECT ticker, company_name, common_score FROM stocks WHERE ticker IN (SELECT ticker FROM graham_qualifiers WHERE exchange='NASDAQ')")
            results = cursor.fetchall()
            if not results:
                messagebox.showinfo("No Data", "No NASDAQ qualifying stocks to export.")
                return

            df = pd.DataFrame(results, columns=["Ticker", "Company Name", "Common Score"])
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if file_path:
                df.to_csv(file_path, index=False)
                messagebox.showinfo("Export Successful", f"NASDAQ qualifying stocks exported to {file_path}")
        except sqlite3.Error as e:
            logging.error(f"Database error in export_nasdaq_qualifying_stocks: {str(e)}")
        finally:
            conn.close()

    def clear_cache(self):
        conn, cursor = get_stocks_connection()
        try:
            cursor.execute("DELETE FROM stocks")
            cursor.execute("DELETE FROM screening_progress")
            cursor.execute("DELETE FROM graham_qualifiers")
            conn.commit()
            with self.ticker_cache_lock:
                self.ticker_cache.clear()
            clear_in_memory_caches()
            messagebox.showinfo("Cache Cleared", "All caches have been cleared.")
        except sqlite3.Error as e:
            logging.error(f"Error clearing cache: {str(e)}")
            messagebox.showerror("Error", f"Failed to clear cache: {str(e)}")
        finally:
            conn.close()

    def show_help(self):
        help_text = (
            "Graham Defensive Stock Screener\n\n"
            "1. Enter tickers in the left column (e.g., AOS, AAPL) and click 'Analyze Stocks'.\n"
            "2. Use 'Save Favorite' to store ticker lists and 'Manage Favorites' to edit them.\n"
            "3. Adjust Margin of Safety and Expected Return using sliders.\n"
            "4. Run NYSE or NASDAQ screenings via checkboxes and buttons in the middle/right columns.\n"
            "5. View results in the treeview below; select a stock to see historical data and metrics.\n"
            "6. Export qualifying stocks to CSV using the export buttons.\n"
            "7. Clear cache to refresh all data.\n"
            "8. Check ticker file ages and update them with new buttons.\n\n"
            "Note: Requires internet connection and valid FMP API keys in config.py."
        )
        messagebox.showinfo("Help", help_text)

    def filter_tree(self, event):
        search_term = self.search_entry.get().strip().upper()
        for item in self.tree.get_children():
            self.tree.delete(item)

        conn, cursor = get_stocks_connection()
        try:
            if search_term:
                cursor.execute("SELECT ticker, company_name, common_score FROM stocks WHERE ticker LIKE ?", (f"%{search_term}%",))
            else:
                cursor.execute("SELECT ticker, company_name, common_score FROM stocks")
            results = cursor.fetchall()

            for ticker, company_name, common_score in results:
                self.tree.insert("", "end", text=ticker, values=(company_name, "Unknown", f"{common_score}/6", "", "", "", ""))
        except sqlite3.Error as e:
            logging.error(f"Database error in filter_tree: {str(e)}")
        finally:
            conn.close()

    def check_file_age(self):
        nyse_age = (time.time() - os.path.getmtime(NYSE_LIST_FILE)) / (24 * 3600) if os.path.exists(NYSE_LIST_FILE) else float('inf')
        nasdaq_age = (time.time() - os.path.getmtime(NASDAQ_LIST_FILE)) / (24 * 3600) if os.path.exists(NASDAQ_LIST_FILE) else float('inf')
        messagebox.showinfo("File Ages", f"NYSE file (otherlisted.txt) is {nyse_age:.1f} days old.\nNASDAQ file (nasdaqlisted.txt) is {nasdaq_age:.1f} days old.")

    def update_ticker_files(self):
        async def async_update():
            ftp_host = "ftp.nasdaqtrader.com"
            files_to_update = {
                "otherlisted.txt": NYSE_LIST_FILE,
                "nasdaqlisted.txt": NASDAQ_LIST_FILE
            }
            for remote_file, local_file in files_to_update.items():
                try:
                    ftp = ftplib.FTP(ftp_host)
                    ftp.login()
                    with open(local_file, 'wb') as f:
                        ftp.retrbinary(f'RETR /SymbolDirectory/{remote_file}', f.write)
                    ftp.quit()
                    logging.info(f"Updated {local_file} from FTP")
                except Exception as e:
                    logging.error(f"Failed to update {local_file}: {str(e)}")
                    self.root.after(0, lambda: messagebox.showerror("Update Error", f"Failed to update {local_file}: {str(e)}"))
                    return
            await self.ticker_manager.initialize()
            self.root.after(0, lambda: messagebox.showinfo("Update Complete", "Ticker files updated successfully."))
            self.root.after(0, lambda: self.entry.config(values=list(self.ticker_manager.get_tickers("NYSE") | self.ticker_manager.get_tickers("NASDAQ"))))
        self.task_queue.put(async_update())

    def on_closing(self):
        self.cancel_event.set()
        self.task_queue.put(None)
        self.asyncio_thread.join(timeout=2)
        with self.ticker_cache_lock:
            self.ticker_cache.clear()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = GrahamScreeningApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()