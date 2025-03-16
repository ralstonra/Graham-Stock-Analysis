# graham_gui.py: Graphical user interface for the stock analysis application

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog, simpledialog
import asyncio
import logging
import pandas as pd
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
                         calculate_graham_value, calculate_graham_score_8, clear_in_memory_caches)
from config import BASE_DIR, FMP_API_KEYS, FAVORITES_FILE, rate_limiter
import queue

# Logging setup
logger = logging.getLogger()
if not logger.handlers:
    logging.basicConfig(filename=os.path.join(BASE_DIR, 'nyse_graham_screen.log'), level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logger.addHandler(console)

FAVORITES_LOCK = threading.Lock()

class GrahamScreeningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Analysis (Graham Defensive)")
        self.root.geometry("1200x900")

        # Variables
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

        # Asyncio setup
        self.task_queue = queue.Queue()
        self.asyncio_thread = threading.Thread(target=self.run_asyncio_loop, args=(self.task_queue,), daemon=True)
        self.asyncio_thread.start()

        # GUI setup
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
        self.left_frame.grid_rowconfigure(tuple(range(17)), weight=1)
        self.middle_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)

        # Style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", font=("Helvetica", 9), background="#f0f0f0")
        style.configure("TButton", font=("Helvetica", 9), padding=2)
        style.configure("TCheckbutton", font=("Helvetica", 9))
        style.configure("TEntry", font=("Helvetica", 9))
        style.configure("TCombobox", font=("Helvetica", 9))
        style.configure("TProgressbar", thickness=15)

        # Create widgets
        self.create_widgets()

    def run_asyncio_loop(self, task_queue):
        """Run asyncio event loop in a separate thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        while True:
            coro = task_queue.get()
            if coro is None:
                break
            loop.run_until_complete(coro)
        loop.close()

    def create_widgets(self):
        """Initialize GUI widgets."""
        # Left Column: Stock Ticker Input and Analysis
        ttk.Label(self.left_frame, text="Enter Stock Tickers (comma-separated, e.g., AOS, AAPL):").grid(row=0, column=0, pady=1, sticky="ew")
        self.entry = ttk.Entry(self.left_frame, width=50)
        self.entry.grid(row=1, column=0, pady=1, sticky="ew")

        def validate_tickers(*args):
            tickers = self.parse_tickers(self.entry.get())
            return bool(tickers)

        self.entry.bind('<FocusOut>', lambda e: validate_tickers())

        # Search Bar
        ttk.Label(self.left_frame, text="Search Results:").grid(row=2, column=0, pady=1, sticky="ew")
        self.search_entry = ttk.Entry(self.left_frame, width=50)
        self.search_entry.grid(row=3, column=0, pady=1, sticky="ew")
        self.search_entry.bind('<KeyRelease>', self.filter_tree)

        # Favorites
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
                self.favorites[name] = tickers
                self.save_favorites()
                self.favorite_menu['values'] = list(self.favorites.keys())
                self.favorite_var.set(name)

        ttk.Button(self.left_frame, text="Save Favorite", command=save_favorite).grid(row=6, column=0, pady=1, sticky="ew")
        ttk.Button(self.left_frame, text="Manage Favorites", command=self.manage_favorites).grid(row=7, column=0, pady=1, sticky="ew")

        # Margin of Safety and Expected Return
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

        # Analyze and Refresh Buttons
        self.analyze_button = ttk.Button(self.left_frame, text="Analyze Stocks", command=self.analyze_multiple_stocks)
        self.analyze_button.grid(row=15, column=0, pady=1, sticky="ew")
        ttk.Button(self.left_frame, text="Refresh Data", command=self.refresh_multiple_stocks).grid(row=16, column=0, pady=1, sticky="ew")

        # Middle Column: NYSE Screening
        ttk.Checkbutton(self.middle_frame, text="Run NYSE Graham Screening", variable=self.nyse_screen_var).grid(row=0, column=0, pady=2, padx=2, sticky="w")
        ttk.Button(self.middle_frame, text="Run NYSE Screening", command=self.run_nyse_screening).grid(row=1, column=0, pady=2, sticky="ew")
        ttk.Button(self.middle_frame, text="Cancel Screening", command=self.cancel_screening).grid(row=2, column=0, pady=2, sticky="ew")
        ttk.Button(self.middle_frame, text="Show NYSE Qualifying Stocks", command=self.display_nyse_qualifying_stocks).grid(row=3, column=0, pady=2, sticky="ew")
        ttk.Button(self.middle_frame, text="Export NYSE Qualifying Stocks", command=self.export_nyse_qualifying_stocks).grid(row=4, column=0, pady=2, sticky="ew")

        # Right Column: NASDAQ Screening
        ttk.Checkbutton(self.right_frame, text="Run NASDAQ Graham Screening", variable=self.nasdaq_screen_var).grid(row=0, column=0, pady=2, padx=2, sticky="w")
        ttk.Button(self.right_frame, text="Run NASDAQ Screening", command=self.run_nasdaq_screening).grid(row=1, column=0, pady=2, sticky="ew")
        ttk.Button(self.right_frame, text="Cancel Screening", command=self.cancel_screening).grid(row=2, column=0, pady=2, sticky="ew")
        ttk.Button(self.right_frame, text="Show NASDAQ Qualifying Stocks", command=self.display_nasdaq_qualifying_stocks).grid(row=3, column=0, pady=2, sticky="ew")
        ttk.Button(self.right_frame, text="Export NASDAQ Qualifying Stocks", command=self.export_nasdaq_qualifying_stocks).grid(row=4, column=0, pady=2, sticky="ew")

        # Progress Bar and Feedback
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.middle_frame, variable=self.progress_var, maximum=100, length=300, mode='determinate')
        self.progress_bar.grid(row=9, column=0, pady=2, sticky="ew")
        self.progress_label = ttk.Label(self.middle_frame, text="Progress: 0% (0/0 stocks processed, 0 passed)")
        self.progress_label.grid(row=10, column=0, pady=1, sticky="ew")
        self.rate_limit_label = ttk.Label(self.middle_frame, text="")
        self.rate_limit_label.grid(row=11, column=0, pady=1, sticky="ew")

        ttk.Button(self.middle_frame, text="Clear Cache", command=self.clear_cache).grid(row=12, column=0, pady=2, sticky="ew")
        ttk.Button(self.middle_frame, text="Help", command=self.show_help).grid(row=13, column=0, pady=2, sticky="ew")

        # Treeview Setup
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
        self.tree.heading(3, text="Graham Score (8)")
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
        """Parse and validate ticker input."""
        if not tickers_input.strip():
            return []
        return [t.strip().upper() for t in tickers_input.split(',') if t.strip() and t.strip().isalnum() and len(t.strip()) <= 5]

    def load_favorites(self):
        """Load favorite ticker lists."""
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
        """Save favorite ticker lists."""
        with FAVORITES_LOCK:
            try:
                with open(FAVORITES_FILE, 'w') as f:
                    json.dump(self.favorites, f, indent=4)
            except Exception as e:
                logging.error(f"Failed to save favorites: {str(e)}")
                messagebox.showerror("Error", f"Failed to save favorites: {str(e)}")

    def update_progress_animated(self, progress, tickers=None, passed_tickers=0):
        """Update progress bar and label with stocks processed and passed."""
        self.progress_var.set(progress)
        if tickers is not None and isinstance(tickers, (list, tuple)):
            total_tickers = len(tickers)
            processed = int(progress / 100 * total_tickers)
            self.progress_label.config(text=f"Progress: {progress:.1f}% ({processed}/{total_tickers} stocks processed, {passed_tickers} passed)")
        else:
            self.progress_label.config(text=f"Progress: {progress:.1f}% (Screening, {passed_tickers} passed)")
        self.root.update_idletasks()

    def update_rate_limit(self, message):
        """Update rate limit feedback label."""
        self.rate_limit_label.config(text=message)
        self.root.update_idletasks()

    def refresh_favorites_dropdown(self, selected_list=None):
        """Refresh favorites dropdown menu."""
        self.favorites = self.load_favorites()
        self.favorite_menu['values'] = list(self.favorites.keys())
        if selected_list and selected_list in self.favorites:
            self.favorite_var.set(selected_list)

    def cancel_screening(self):
        """Cancel ongoing screening process."""
        self.cancel_event.set()

    def run_screening(self, exchange, screen_func):
        """Run screening for NYSE or NASDAQ."""
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

        self.progress_var.set(0)
        self.progress_label.config(text=f"Progress: 0% (0/0 stocks processed, 0 passed)")
        self.rate_limit_label.config(text="")
        self.root.update()
        self.cancel_event.clear()
        self.screening_active = True

        async def screening_task():
            try:
                await self.ticker_manager.initialize()
                qualifying_stocks, graham_scores, exchanges = await screen_func(
                    batch_size=50,
                    cancel_event=self.cancel_event,
                    root=self.root,
                    update_progress_animated=self.update_progress_animated,
                    refresh_favorites_dropdown=self.refresh_favorites_dropdown,
                    ticker_manager=self.ticker_manager,
                    update_rate_limit=self.update_rate_limit
                )
                if not self.cancel_event.is_set():
                    conn, cursor = get_stocks_connection()
                    try:
                        processed_tickers = cursor.execute("SELECT COUNT(*) FROM screening_progress WHERE exchange=? AND status='completed'", (exchange,)).fetchone()[0]
                        error_tickers = cursor.execute("SELECT COUNT(*) FROM screening_progress WHERE exchange=? AND status='failed'", (exchange,)).fetchone()[0]
                        total_tickers = processed_tickers + error_tickers
                        summary = f"Completed {exchange} screening.\nProcessed {total_tickers} stocks,\nFound {len(qualifying_stocks)} qualifiers,\n{error_tickers} errors"
                        self.root.after(0, lambda: messagebox.showinfo("Screening Complete", summary))
                    finally:
                        conn.close()
            except Exception as e:
                self.root.after(0, lambda e=e: messagebox.showerror("Error", f"Screening failed: {str(e)}"))
            finally:
                self.root.after(0, lambda: self.progress_label.config(text=f"Progress: 100% (Screening Complete - {exchange})"))
                setattr(self, f"{exchange.lower()}_screen_var", tk.BooleanVar(value=False))
                self.screening_active = False

        self.task_queue.put(screening_task())

    def run_nyse_screening(self):
        self.run_screening("NYSE", screen_nyse_graham_stocks)

    def run_nasdaq_screening(self):
        self.run_screening("NASDAQ", screen_nasdaq_graham_stocks)

    async def fetch_company_name(self, ticker):
        """Fetch company name from yfinance or FMP."""
        try:
            stock = yf.Ticker(ticker)
            company_name = stock.info.get('longName', 'Unknown')
            if company_name != 'Unknown':
                return company_name

            fmp_data = await fetch_with_multiple_keys_async(ticker, "profile", FMP_API_KEYS, update_rate_limit=self.update_rate_limit)
            if fmp_data and isinstance(fmp_data, list) and len(fmp_data) > 0:
                return fmp_data[0].get('companyName', 'Unknown')
            return 'Unknown'
        except Exception as e:
            logging.error(f"Error fetching company name for {ticker}: {str(e)}")
            return 'Unknown'

    def format_float(self, value, precision=2):
        """Format float values for display."""
        if value is None or not isinstance(value, (int, float)) or value != value:
            return "N/A"
        return f"{value:.{precision}f}"

    async def fetch_cached_data(self, ticker, exchange="Unknown"):
        """Fetch cached data from database if fresh."""
        nyse_file_hash = get_file_hash(NYSE_LIST_FILE)
        nasdaq_file_hash = get_file_hash(NASDAQ_LIST_FILE)

        conn, cursor = get_stocks_connection()
        try:
            cursor.execute("SELECT file_hash, exchange, timestamp FROM screening_progress WHERE ticker=?", (ticker,))
            row = cursor.fetchone()
            if row:
                stored_hash, stored_exchange, timestamp = row
                if (stored_exchange == "NYSE" and stored_hash == nyse_file_hash) or (stored_exchange == "NASDAQ" and stored_hash == nasdaq_file_hash):
                    if time.time() - float(timestamp) < 365 * 24 * 60 * 60:  # 1 year
                        cursor.execute("SELECT * FROM stocks WHERE ticker=?", (ticker,))
                        stock_row = cursor.fetchone()
                        if stock_row and len(stock_row) == 11:  # Check schema
                            stock = yf.Ticker(ticker)  # Fetch fresh price
                            info = stock.info
                            price = info.get('regularMarketPrice', info.get('previousClose', stock_row[2]))
                            roe_10y = [float(x) for x in stock_row[3].split(",")] if stock_row[3] else [0.0] * 10
                            rotc_10y = [float(x) for x in stock_row[4].split(",")] if stock_row[4] else [0.0] * 10
                            eps_10y = [float(x) for x in stock_row[5].split(",")] if stock_row[5] else [0.0] * 10
                            div_10y = [float(x) for x in stock_row[6].split(",")] if stock_row[6] else [0.0] * 10
                            balance_data = json.loads(stock_row[8]) if stock_row[8] else []
                            years = list(range(int(datetime.now().year) - 9, int(datetime.now().year) + 1))

                            if not balance_data:
                                return None

                            earnings = eps_10y[-1] if eps_10y and eps_10y[-1] > 0 else None
                            intrinsic_value = calculate_graham_value(earnings, self.expected_return_var.get() / 100, aaa_yield=4.5, eps_10y=eps_10y) if earnings else "NQ"
                            margin_of_safety = self.margin_of_safety_var.get() / 100
                            expected_return = self.expected_return_var.get() / 100
                            buy_price = intrinsic_value * (1 - margin_of_safety) if isinstance(intrinsic_value, (int, float)) else "NQ"
                            sell_price = intrinsic_value * (1 + expected_return) if isinstance(intrinsic_value, (int, float)) else "NQ"
                            latest_revenue = 0  # Placeholder; actual value would need to be fetched or stored
                            graham_score = calculate_graham_score_8(ticker, price, None, None, None, eps_10y, div_10y, {}, balance_data, 10, latest_revenue)

                            result = {
                                "ticker": stock_row[0],
                                "exchange": stored_exchange,
                                "price": price,
                                "pe_ratio": None,
                                "pb_ratio": None,
                                "debt_to_equity": None,
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
                                "latest_revenue": latest_revenue
                            }
                            with self.ticker_cache_lock:
                                self.ticker_cache[ticker] = result
                            return result
            return None
        finally:
            conn.close()

    async def determine_exchange(self, ticker):
        """Determine the exchange for a ticker."""
        await self.ticker_manager.initialize()
        nyse_tickers = self.ticker_manager.get_tickers("NYSE")
        nasdaq_tickers = self.ticker_manager.get_tickers("NASDAQ")
        if ticker in nyse_tickers:
            return "NYSE"
        elif ticker in nasdaq_tickers:
            return "NASDAQ"
        return "Unknown"

    async def analyze_multiple_stocks_async(self, tickers_input=None):
        """Analyze multiple stocks with fresh prices."""
        if tickers_input is None:
            tickers_input = self.entry.get()
        tickers = self.parse_tickers(tickers_input)
        if not tickers:
            messagebox.showwarning("No Tickers", "No valid tickers to analyze.")
            return

        self.progress_var.set(0)
        self.progress_label.config(text=f"Progress: 0% (0/{len(tickers)} stocks processed, 0 passed)")
        self.rate_limit_label.config(text="")
        self.root.update()

        await self.ticker_manager.initialize()
        results = []
        passed_tickers = 0
        for i, ticker in enumerate(tickers):
            with self.ticker_cache_lock:
                if ticker in self.ticker_cache:
                    cached_result = self.ticker_cache[ticker]
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    cached_result['price'] = info.get('regularMarketPrice', info.get('previousClose', cached_result['price']))
                    results.append(cached_result)
                    if cached_result['graham_score'] >= 5:
                        passed_tickers += 1
                    continue

            cached_result = await self.fetch_cached_data(ticker)
            if cached_result:
                results.append(cached_result)
                with self.ticker_cache_lock:
                    self.ticker_cache[ticker] = cached_result
                if cached_result['graham_score'] >= 5:
                    passed_tickers += 1
                continue

            exchange = await self.determine_exchange(ticker)
            result = await fetch_stock_data(ticker, exchange=exchange, expected_return=self.expected_return_var.get() / 100,
                                            margin_of_safety=self.margin_of_safety_var.get() / 100,
                                            update_rate_limit=self.update_rate_limit)
            if result and "error" not in result:
                conn, cursor = get_stocks_connection()
                try:
                    cursor.execute(
                        "INSERT OR REPLACE INTO stocks (ticker, date, price, roe, rotc, eps, dividend, ticker_list_hash, balance_data, ipo_date, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (ticker, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), result['price'],
                         ",".join(map(str, result['roe_10y'])), ",".join(map(str, result['rotc_10y'])),
                         ",".join(map(str, result['eps_10y'])), ",".join(map(str, result['div_10y'])),
                         get_file_hash(NYSE_LIST_FILE) if exchange == "NYSE" else get_file_hash(NASDAQ_LIST_FILE),
                         json.dumps(result['balance_data']), result['ipo_date'], time.time())
                    )
                    conn.commit()
                finally:
                    conn.close()
                with self.ticker_cache_lock:
                    self.ticker_cache[ticker] = result
                results.append(result)
                if result['graham_score'] >= 5:
                    passed_tickers += 1
            else:
                results.append({"ticker": ticker, "exchange": exchange, "error": "Failed to fetch data"})

            progress = ((i + 1) / len(tickers)) * 100
            self.root.after(0, lambda p=progress, pt=passed_tickers: self.update_progress_animated(p, tickers, pt))

        valid_results = [r for r in results if 'error' not in r]
        error_results = [r for r in results if 'error' in r]
        if error_results:
            error_summary = "\n".join([f"{err['ticker']} ({err['exchange']}): {err['error']}" for err in error_results])
            self.root.after(0, lambda: messagebox.showwarning("Analysis Errors", f"Errors occurred:\n{error_summary}"))

        for item in self.tree.get_children():
            self.tree.delete(item)
        for widget in self.historical_frame.winfo_children():
            widget.destroy()
        for widget in self.metrics_frame.winfo_children():
            widget.destroy()

        async def fetch_company_names(results_list):
            tasks = [self.fetch_company_name(result['ticker']) for result in results_list if result['ticker']]
            return await asyncio.gather(*tasks)

        company_names = await fetch_company_names(valid_results)
        total_results = len(valid_results)

        for i, result in enumerate(valid_results):
            company_name = company_names[i] if i < len(company_names) else 'Unknown'
            self.tree.insert("", "end", text=result['ticker'], values=(
                company_name,
                result['exchange'],
                f"{result['graham_score']}/8",
                f"${self.format_float(result['price'])}",
                f"${self.format_float(result['intrinsic_value'])}" if result['intrinsic_value'] != "NQ" else "N/A",
                f"${self.format_float(result['buy_price'])}" if result['buy_price'] != "NQ" else "N/A",
                f"${self.format_float(result['sell_price'])}" if result['sell_price'] != "NQ" else "N/A"
            ))

        self.root.after(0, lambda: self.update_progress_animated(100, tickers, passed_tickers))

    def analyze_multiple_stocks(self, tickers_input=None):
        """Initiate stock analysis."""
        if self.analysis_lock.acquire(timeout=5):
            try:
                tickers = self.parse_tickers(tickers_input or self.entry.get())
                self.task_queue.put(self.analyze_multiple_stocks_async(tickers_input))
            finally:
                self.analysis_lock.release()
        else:
            messagebox.showerror("Error", "Unable to start analysis: Lock timeout.")

    def refresh_multiple_stocks(self, tickers_input=None):
        """Refresh data for specified tickers."""
        tickers = self.parse_tickers(tickers_input if tickers_input is not None else self.entry.get())
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
        finally:
            conn.close()
        self.task_queue.put(self.analyze_multiple_stocks_async(tickers_input))

    async def display_results_in_tree(self, tickers, scores, exchanges, source):
        """Display screening results in treeview."""
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

            result = await fetch_batch_data([ticker], expected_return=self.expected_return_var.get() / 100,
                                            margin_of_safety=self.margin_of_safety_var.get() / 100,
                                            exchange=source, ticker_manager=self.ticker_manager,
                                            update_rate_limit=self.update_rate_limit)
            if result and isinstance(result[0], dict) and "error" not in result[0]:
                result = result[0]
                conn, cursor = get_stocks_connection()
                try:
                    cursor.execute(
                        "INSERT OR REPLACE INTO stocks (ticker, date, price, roe, rotc, eps, dividend, ticker_list_hash, balance_data, ipo_date, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (ticker, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), result['price'],
                         ",".join(map(str, result['roe_10y'])), ",".join(map(str, result['rotc_10y'])),
                         ",".join(map(str, result['eps_10y'])), ",".join(map(str, result['div_10y'])),
                         get_file_hash(NYSE_LIST_FILE) if source == "NYSE" else get_file_hash(NASDAQ_LIST_FILE),
                         json.dumps(result['balance_data']), result['ipo_date'], time.time())
                    )
                    conn.commit()
                finally:
                    conn.close()
                with self.ticker_cache_lock:
                    self.ticker_cache[ticker] = result
                results.append(result)
            else:
                results.append({"ticker": ticker, "exchange": source, "error": "Failed to fetch data"})

        valid_results = [r for r in results if 'error' not in r]
        company_names = await fetch_company_names(tickers)
        for i, ticker in enumerate(tickers):
            company_name = company_names[i] if i < len(company_names) else 'Unknown'
            result = next((r for r in valid_results if r['ticker'] == ticker), {
                "exchange": exchanges[i] if i < len(exchanges) else source,
                "graham_score": scores[i] if i < len(scores) else 0,
                "price": 0,
                "intrinsic_value": "NQ",
                "buy_price": "NQ",
                "sell_price": "NQ"
            })
            self.tree.insert("", "end", text=ticker, values=(
                company_name,
                result['exchange'],
                f"{result['graham_score']}/8",
                f"${self.format_float(result['price'])}",
                f"${self.format_float(result['intrinsic_value'])}" if result['intrinsic_value'] != "NQ" else "N/A",
                f"${self.format_float(result['buy_price'])}" if result['buy_price'] != "NQ" else "N/A",
                f"${self.format_float(result['sell_price'])}" if result['sell_price'] != "NQ" else "N/A"
            ))

    async def fetch_metrics_data(self, ticker):
        """Fetch metrics data for display."""
        stock = yf.Ticker(ticker)
        info = stock.info
        balance_data = await fetch_with_multiple_keys_async(ticker, "balance-sheet-statement", FMP_API_KEYS)
        income_data = await fetch_with_multiple_keys_async(ticker, "income-statement", FMP_API_KEYS)
        revenue = {str(entry.get('calendarYear', 0)): float(entry.get('revenue', 0)) for entry in income_data if 'calendarYear' in entry} if income_data else {}
        latest_balance = balance_data[0] if balance_data and len(balance_data) > 0 else {}
        latest_income = income_data[0] if income_data and len(income_data) > 0 else {}
        return info, revenue, latest_balance, latest_income

    def update_tabs(self, event):
        """Update historical and metrics tabs based on selection."""
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
        roe_10y = result.get('roe_10y', [])
        rotc_10y = result.get('rotc_10y', [])
        eps_10y = result.get('eps_10y', [])
        div_10y = result.get('div_10y', [])

        if len(years) == 10 and len(roe_10y) == 10 and len(rotc_10y) == 10 and len(eps_10y) == 10 and len(div_10y) == 10:
            historical_text.insert(tk.END, f"10-Year Historical Data for {ticker}:\nYear\tROE (%)\tROTC (%)\tEPS ($)\tDividend ($)\n")
            for j in range(10):
                historical_text.insert(tk.END, f"{years[j]}\t{roe_10y[j]:.2f}\t{rotc_10y[j]:.2f}\t{eps_10y[j]:.2f}\t{div_10y[j]:.2f}\n")
        else:
            historical_text.insert(tk.END, f"10-Year Historical Data for {ticker} (Incomplete):\nYear\tROE (%)\tROTC (%)\tEPS ($)\tDividend ($)\n")
            for j in range(min(len(years), 10)):
                historical_text.insert(tk.END, f"{years[j]}\t{roe_10y[j]:.2f}\t{rotc_10y[j]:.2f}\t{eps_10y[j]:.2f}\t{div_10y[j]:.2f}\n")

        metrics_text.insert(tk.END, f"Graham Criteria Results for {ticker} (Score: {result['graham_score']}/8):\n")

        async def update_metrics():
            try:
                info, revenue, latest_balance, latest_income = await self.fetch_metrics_data(ticker)
                latest_revenue = max((float(v) for k, v in revenue.items() if k.isdigit()), default=0)
                revenue_pass = latest_revenue >= 500_000_000
                metrics_text.insert(tk.END, f"1. Revenue >= $500M: {'Yes' if revenue_pass else 'No'} (${latest_revenue/1e6:.2f}M)\n")

                current_assets = float(latest_balance.get('totalCurrentAssets', 0))
                current_liabilities = float(latest_balance.get('totalCurrentLiabilities', 1))
                current_ratio = current_assets / current_liabilities if current_liabilities > 0 else 0
                current_pass = current_ratio > 2
                metrics_text.insert(tk.END, f"2. Current Ratio > 2: {'Yes' if current_pass else 'No'} ({current_ratio:.2f})\n")

                eps_10y = result.get('eps_10y', [])
                expected_years = min(10, len(eps_10y))
                negative_eps_count = sum(1 for eps in eps_10y[-expected_years:] if eps <= 0)
                max_negative_years = min(2, expected_years // 5)
                stability_pass = negative_eps_count <= max_negative_years
                metrics_text.insert(tk.END, f"3. Earnings Stability (<= {max_negative_years} negative years): {'Yes' if stability_pass else 'No'} ({negative_eps_count} negatives)\n")

                div_10y = result.get('div_10y', [])
                dividend_pass = len(div_10y) >= expected_years and all(div > 0 for div in div_10y[-expected_years:])
                metrics_text.insert(tk.END, f"4. Uninterrupted Dividends ({expected_years} yrs): {'Yes' if dividend_pass else 'No'}\n")

                eps_growth_pass = False
                if expected_years >= 2:
                    first_eps = eps_10y[-expected_years]
                    last_eps = eps_10y[-1]
                    if first_eps > 0 and last_eps > 0:
                        cagr = (last_eps / first_eps) ** (1 / (expected_years - 1)) - 1
                        eps_growth_pass = cagr > 0.03
                        metrics_text.insert(tk.END, f"5. EPS CAGR > 3%: {'Yes' if eps_growth_pass else 'No'} ({cagr:.2%})\n")
                    else:
                        metrics_text.insert(tk.END, f"5. EPS CAGR > 3%: No (Invalid EPS data)\n")
                else:
                    metrics_text.insert(tk.END, f"5. EPS CAGR > 3%: No (Insufficient data)\n")

                debt_to_equity = float(info.get('debtToEquity', 0)) / 100 if info.get('debtToEquity') else None
                debt_pass = debt_to_equity is not None and debt_to_equity < 2
                metrics_text.insert(tk.END, f"6. Debt-to-Equity < 2: {'Yes' if debt_pass else 'No'} ({self.format_float(debt_to_equity)})\n")

                pe_ratio = info.get('trailingPE', None)
                pe_pass = pe_ratio is not None and pe_ratio <= 15
                metrics_text.insert(tk.END, f"7. P/E Ratio <= 15: {'Yes' if pe_pass else 'No'} ({self.format_float(pe_ratio)})\n")

                pb_ratio = info.get('priceToBook', None)
                pb_pass = pb_ratio is not None and pb_ratio <= 1.5
                metrics_text.insert(tk.END, f"8. P/B Ratio <= 1.5: {'Yes' if pb_pass else 'No'} ({self.format_float(pb_ratio)})\n")

            except Exception as e:
                logging.error(f"Error updating metrics for {ticker}: {str(e)}")
                metrics_text.insert(tk.END, f"Error fetching metrics: {str(e)}\n")

        self.task_queue.put(update_metrics())

    def manage_favorites(self):
        """Open a dialog to manage favorite lists."""
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
        """Sort treeview by a specified column."""
        items = [(self.tree.set(k, col), k) for k in self.tree.get_children()]
        items.sort()
        for index, (val, k) in enumerate(items):
            self.tree.move(k, '', index)

    def display_nyse_qualifying_stocks(self):
        """Display NYSE qualifying stocks in the treeview."""
        conn, cursor = get_stocks_connection()
        try:
            cursor.execute("SELECT ticker, graham_score, exchange FROM graham_qualifiers WHERE exchange='NYSE' AND graham_score >= 5")
            results = cursor.fetchall()
            if results:
                tickers, scores, exchanges = zip(*results)
                self.task_queue.put(self.display_results_in_tree(tickers, scores, exchanges, "NYSE"))
            else:
                messagebox.showinfo("No Results", "No NYSE stocks meet the Graham criteria (score >= 5).")
        finally:
            conn.close()

    def display_nasdaq_qualifying_stocks(self):
        """Display NASDAQ qualifying stocks in the treeview."""
        conn, cursor = get_stocks_connection()
        try:
            cursor.execute("SELECT ticker, graham_score, exchange FROM graham_qualifiers WHERE exchange='NASDAQ' AND graham_score >= 5")
            results = cursor.fetchall()
            if results:
                tickers, scores, exchanges = zip(*results)
                self.task_queue.put(self.display_results_in_tree(tickers, scores, exchanges, "NASDAQ"))
            else:
                messagebox.showinfo("No Results", "No NASDAQ stocks meet the Graham criteria (score >= 5).")
        finally:
            conn.close()

    def export_nyse_qualifying_stocks(self):
        """Export NYSE qualifying stocks to a CSV file."""
        conn, cursor = get_stocks_connection()
        try:
            cursor.execute("SELECT ticker, exchange, graham_score, price, intrinsic_value, buy_price, sell_price FROM stocks WHERE exchange='NYSE' AND graham_score >= 5")
            results = cursor.fetchall()
            if not results:
                messagebox.showinfo("No Data", "No NYSE qualifying stocks to export.")
                return

            df = pd.DataFrame(results, columns=["Ticker", "Exchange", "Graham Score", "Price", "Intrinsic Value", "Buy Price", "Sell Price"])
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if file_path:
                df.to_csv(file_path, index=False)
                messagebox.showinfo("Export Successful", f"NYSE qualifying stocks exported to {file_path}")
        finally:
            conn.close()

    def export_nasdaq_qualifying_stocks(self):
        """Export NASDAQ qualifying stocks to a CSV file."""
        conn, cursor = get_stocks_connection()
        try:
            cursor.execute("SELECT ticker, exchange, graham_score, price, intrinsic_value, buy_price, sell_price FROM stocks WHERE exchange='NASDAQ' AND graham_score >= 5")
            results = cursor.fetchall()
            if not results:
                messagebox.showinfo("No Data", "No NASDAQ qualifying stocks to export.")
                return

            df = pd.DataFrame(results, columns=["Ticker", "Exchange", "Graham Score", "Price", "Intrinsic Value", "Buy Price", "Sell Price"])
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if file_path:
                df.to_csv(file_path, index=False)
                messagebox.showinfo("Export Successful", f"NASDAQ qualifying stocks exported to {file_path}")
        finally:
            conn.close()

    def clear_cache(self):
        """Clear the in-memory and database caches."""
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
        except Exception as e:
            logging.error(f"Error clearing cache: {str(e)}")
            messagebox.showerror("Error", f"Failed to clear cache: {str(e)}")
        finally:
            conn.close()

    def show_help(self):
        """Display help information."""
        help_text = (
            "Graham Defensive Stock Screener\n\n"
            "1. Enter tickers in the left column (e.g., AOS, AAPL) and click 'Analyze Stocks'.\n"
            "2. Use 'Save Favorite' to store ticker lists and 'Manage Favorites' to edit them.\n"
            "3. Adjust Margin of Safety and Expected Return using sliders.\n"
            "4. Run NYSE or NASDAQ screenings via checkboxes and buttons in the middle/right columns.\n"
            "5. View results in the treeview below; select a stock to see historical data and metrics.\n"
            "6. Export qualifying stocks to CSV using the export buttons.\n"
            "7. Clear cache to refresh all data.\n\n"
            "Note: Requires internet connection and valid FMP API keys in config.py."
        )
        messagebox.showinfo("Help", help_text)

    def filter_tree(self, event):
        """Filter treeview based on search input."""
        search_term = self.search_entry.get().strip().upper()
        for item in self.tree.get_children():
            self.tree.delete(item)

        conn, cursor = get_stocks_connection()
        try:
            if search_term:
                cursor.execute("SELECT ticker, exchange, graham_score, price, intrinsic_value, buy_price, sell_price FROM stocks WHERE ticker LIKE ?", (f"%{search_term}%",))
            else:
                cursor.execute("SELECT ticker, exchange, graham_score, price, intrinsic_value, buy_price, sell_price FROM stocks")
            results = cursor.fetchall()

            for result in results:
                ticker, exchange, graham_score, price, intrinsic_value, buy_price, sell_price = result
                company_name = self.fetch_company_name(ticker)  # Synchronous call for simplicity
                self.tree.insert("", "end", text=ticker, values=(
                    company_name,
                    exchange,
                    f"{graham_score}/8",
                    f"${self.format_float(price)}",
                    f"${self.format_float(intrinsic_value)}" if intrinsic_value != "NQ" else "N/A",
                    f"${self.format_float(buy_price)}" if buy_price != "NQ" else "N/A",
                    f"${self.format_float(sell_price)}" if sell_price != "NQ" else "N/A"
                ))
        finally:
            conn.close()

    def on_closing(self):
        """Handle application closing."""
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