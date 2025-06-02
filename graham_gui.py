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
from datetime import datetime, timedelta
from graham_data import (screen_nasdaq_graham_stocks, screen_nyse_graham_stocks, fetch_batch_data,
                         fetch_stock_data, get_stocks_connection, fetch_with_multiple_keys_async,
                         NYSE_LIST_FILE, NASDAQ_LIST_FILE, TickerManager, get_file_hash,
                         calculate_graham_value, calculate_graham_score_8, calculate_common_criteria, 
                         clear_in_memory_caches, save_qualifying_stocks_to_favorites, get_stock_data_from_db,
                         get_sector_growth_rate, calculate_cagr, get_aaa_yield)
from config import (
    BASE_DIR, FMP_API_KEYS, FAVORITES_FILE, paid_rate_limiter, free_rate_limiter, 
    CACHE_EXPIRY, screening_logger, analyze_logger, USER_DATA_DIR, FRED_API_KEY
)
import queue
import shutil
import requests
import ftplib
import openpyxl
from openpyxl.chart import LineChart, Reference
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, Alignment, Border, Side

FAVORITES_LOCK = threading.Lock()
DATA_DIR = BASE_DIR

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
        self.ticker_manager = TickerManager(NYSE_LIST_FILE, NASDAQ_LIST_FILE, USER_DATA_DIR)
        self.screening_active = False
        self.analysis_lock = threading.Lock()
        self.ticker_cache = {}
        self.ticker_cache_lock = threading.Lock()

        # Asyncio Thread Setup
        self.task_queue = queue.Queue()
        self.asyncio_thread = threading.Thread(target=self.run_asyncio_loop, args=(self.task_queue,), daemon=True)
        self.asyncio_thread.start()

        # Main Frame Layout
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(expand=True, fill="both", padx=5, pady=5)
        self.main_frame.grid_rowconfigure(0, weight=3)
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure((0, 1, 2), weight=1, minsize=400)

        # Sub-Frames
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

        # Styling
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
        # Left Frame Widgets (Analyze Stocks)
        self.left_frame.grid_rowconfigure(tuple(range(15)), weight=1)

        ttk.Label(self.left_frame, text="Enter Stock Tickers (comma-separated, e.g., AOS, AAPL):").grid(row=0, column=0, pady=2, sticky="ew")
        self.entry = ttk.Entry(self.left_frame, width=50)
        self.entry.grid(row=1, column=0, pady=2, sticky="ew")
        self.entry.bind('<FocusOut>', lambda e: self.validate_tickers())

        ttk.Label(self.left_frame, text="Search Results:").grid(row=2, column=0, pady=2, sticky="ew")
        self.search_entry = ttk.Entry(self.left_frame, width=50)
        self.search_entry.grid(row=3, column=0, pady=2, sticky="ew")
        self.search_entry.bind('<KeyRelease>', self.filter_tree)

        self.favorites = self.load_favorites()
        ttk.Label(self.left_frame, text="Favorite Lists:").grid(row=4, column=0, pady=2, sticky="ew")
        self.favorite_var = tk.StringVar(value="Select Favorite")
        self.favorite_menu = ttk.Combobox(self.left_frame, textvariable=self.favorite_var, values=list(self.favorites.keys()))
        self.favorite_menu.grid(row=5, column=0, pady=2, sticky="ew")

        def load_favorite(event=None):
            selected = self.favorite_var.get()
            if selected and selected != "Select Favorite":
                self.entry.delete(0, tk.END)
                tickers = self.favorites[selected]
                self.entry.insert(0, ",".join(tickers))

        self.favorite_menu.bind('<<ComboboxSelected>>', load_favorite)

        def save_favorite():
            print("Save Favorite button clicked")  # Debug
            self.task_queue.put(self.ticker_manager.initialize())
            time.sleep(1)  # Wait for initialization (adjust if needed)
            print(f"NYSE tickers: {len(self.ticker_manager.nyse_tickers)}, NASDAQ tickers: {len(self.ticker_manager.nasdaq_tickers)}")  # Debug
            tickers_input = self.entry.get()
            print(f"Input: {tickers_input}")  # Debug
            if not self.validate_tickers():
                messagebox.showwarning("Invalid Tickers", "No valid tickers entered.")
                return
            name = simpledialog.askstring("Save Favorite", "Enter list name:")
            print(f"List name: {name}")  # Debug
            if not name or not name.strip():
                messagebox.showwarning("Invalid Name", "Please enter a valid list name.")
                return
            if name and tickers_input.strip():
                tickers = self.parse_tickers(tickers_input)
                print(f"Parsed tickers: {tickers}")  # Debug
                valid_tickers = [t for t in tickers if self.ticker_manager.is_valid_ticker(t)]
                print(f"Valid tickers: {valid_tickers}")  # Debug
                if not valid_tickers:
                    messagebox.showwarning("Invalid Tickers", "No valid NYSE or NASDAQ tickers.")
                    return
                self.favorites[name] = valid_tickers
                self.save_favorites()
                self.refresh_favorites_dropdown(name)
                print(f"Saved favorites: {self.favorites}")  # Debug

        ttk.Button(self.left_frame, text="Save Favorite", command=save_favorite).grid(row=6, column=0, pady=2, sticky="ew")
        ttk.Button(self.left_frame, text="Manage Favorites", command=self.manage_favorites).grid(row=7, column=0, pady=2, sticky="ew")

        # Margin of Safety
        self.margin_of_safety_label = ttk.Label(self.left_frame, text="Margin of Safety: 33%")
        self.margin_of_safety_label.grid(row=8, column=0, pady=2, sticky="ew")
        ttk.Scale(self.left_frame, from_=0, to=50, orient=tk.HORIZONTAL, variable=self.margin_of_safety_var,
                  command=lambda value: self.margin_of_safety_label.config(text=f"Margin of Safety: {int(float(value))}%")).grid(row=9, column=0, pady=2, sticky="ew")

        # Expected Rate of Return
        self.expected_return_label = ttk.Label(self.left_frame, text="Expected Rate of Return: 0%")
        self.expected_return_label.grid(row=10, column=0, pady=2, sticky="ew")
        ttk.Scale(self.left_frame, from_=0, to=20, orient=tk.HORIZONTAL, variable=self.expected_return_var,
                  command=lambda value: self.expected_return_label.config(text=f"Expected Rate of Return: {int(float(value))}%")).grid(row=11, column=0, pady=2, sticky="ew")

        self.analyze_button = ttk.Button(self.left_frame, text="Analyze Stocks", command=self.analyze_multiple_stocks)
        self.analyze_button.grid(row=14, column=0, pady=2, sticky="ew")

        # Middle Frame Widgets (Utility Buttons and Progress/Status)
        ttk.Button(self.middle_frame, text="Check Ticker File Ages", command=self.check_file_age).grid(row=0, column=0, pady=2, sticky="ew")
        ttk.Button(self.middle_frame, text="Update Ticker Files", command=self.update_ticker_files).grid(row=1, column=0, pady=2, sticky="ew")
        ttk.Button(self.middle_frame, text="Clear Cache", command=self.clear_cache).grid(row=2, column=0, pady=2, sticky="ew")
        ttk.Button(self.middle_frame, text="Help", command=self.show_help).grid(row=3, column=0, pady=2, sticky="ew")
        ttk.Button(self.middle_frame, text="Cancel Screening", command=self.cancel_screening).grid(row=4, column=0, pady=2, sticky="ew")

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.middle_frame, variable=self.progress_var, maximum=100, length=300, mode='determinate')
        self.progress_bar.grid(row=5, column=0, pady=2, sticky="ew")
        self.progress_label = ttk.Label(self.middle_frame, text="Progress: 0% (0/0 stocks processed, 0 passed)")
        self.progress_label.grid(row=6, column=0, pady=2, sticky="ew")

        self.cache_var = tk.DoubleVar()
        self.cache_bar = ttk.Progressbar(self.middle_frame, variable=self.cache_var, maximum=100, length=300, mode='determinate')
        self.cache_bar.grid(row=7, column=0, pady=2, sticky="ew")
        self.cache_label = ttk.Label(self.middle_frame, text="Cache Usage: 0% (0 cached, 0 fresh)")
        self.cache_label.grid(row=8, column=0, pady=2, sticky="ew")

        self.rate_limit_label = ttk.Label(self.middle_frame, text="")
        self.rate_limit_label.grid(row=9, column=0, pady=2, sticky="ew")
        self.status_label = ttk.Label(self.middle_frame, text="")
        self.status_label.grid(row=10, column=0, pady=2, sticky="ew")

        # Right Frame Widgets (Screenings)
        ttk.Checkbutton(self.right_frame, text="Run NYSE Graham Screening", variable=self.nyse_screen_var).grid(row=0, column=0, pady=2, padx=2, sticky="w")
        ttk.Button(self.right_frame, text="Run NYSE Screening", command=self.run_nyse_screening).grid(row=1, column=0, pady=2, sticky="ew")
        ttk.Button(self.right_frame, text="Show NYSE Qualifying Stocks", command=self.display_nyse_qualifying_stocks).grid(row=2, column=0, pady=2, sticky="ew")
        ttk.Button(self.right_frame, text="Export NYSE Qualifying Stocks", command=self.export_nyse_qualifying_stocks).grid(row=3, column=0, pady=2, sticky="ew")

        ttk.Separator(self.right_frame, orient="horizontal").grid(row=4, column=0, sticky="ew", pady=5)

        ttk.Checkbutton(self.right_frame, text="Run NASDAQ Graham Screening", variable=self.nasdaq_screen_var).grid(row=5, column=0, pady=2, padx=2, sticky="w")
        ttk.Button(self.right_frame, text="Run NASDAQ Screening", command=self.run_nasdaq_screening).grid(row=6, column=0, pady=2, sticky="ew")
        ttk.Button(self.right_frame, text="Show NASDAQ Qualifying Stocks", command=self.display_nasdaq_qualifying_stocks).grid(row=7, column=0, pady=2, sticky="ew")
        ttk.Button(self.right_frame, text="Export NASDAQ Qualifying Stocks", command=self.export_nasdaq_qualifying_stocks).grid(row=8, column=0, pady=2, sticky="ew")

        # Treeview and Tabs
        self.full_tree_frame = ttk.Frame(self.main_frame)
        self.full_tree_frame.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)
        self.full_tree_frame.grid_columnconfigure(0, weight=1)
        self.full_tree_frame.grid_rowconfigure(0, weight=3)
        self.full_tree_frame.grid_rowconfigure(1, weight=0)
        self.full_tree_frame.grid_rowconfigure(2, weight=1)

        self.tree = ttk.Treeview(self.full_tree_frame, columns=(1, 2, 3, 4, 5, 6, 7), show="tree headings", height=10)
        self.v_scrollbar = ttk.Scrollbar(self.full_tree_frame, orient="vertical", command=self.tree.yview)
        self.h_scrollbar = ttk.Scrollbar(self.full_tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.h_scrollbar.grid(row=1, column=0, sticky="ew")
        self.tree.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)

        self.tree.heading("#0", text="Ticker")
        self.tree.heading(1, text="Company Name")
        self.tree.heading(2, text="Sector")
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

        self.historical_frame.grid_rowconfigure(0, weight=1)
        self.historical_frame.grid_columnconfigure(0, weight=1)
        self.metrics_frame.grid_rowconfigure(0, weight=1)
        self.metrics_frame.grid_columnconfigure(0, weight=1)

        self.historical_text = scrolledtext.ScrolledText(self.historical_frame, width=70, height=12)
        self.historical_text.grid(row=0, column=0, sticky="nsew")
        self.metrics_text = scrolledtext.ScrolledText(self.metrics_frame, width=70, height=12)
        self.metrics_text.grid(row=0, column=0, sticky="nsew")

        self.tree.bind("<<TreeviewSelect>>", self.update_tabs)

    def validate_tickers(self):
        tickers = self.parse_tickers(self.entry.get())
        return bool(tickers)

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
                    analyze_logger.error(f"Corrupted favorites file {FAVORITES_FILE}")
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
                analyze_logger.error(f"Failed to save favorites: {str(e)}")
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
        analyze_logger.debug(f"Refreshed favorites dropdown with keys: {list(self.favorites.keys())}")
        self.favorite_menu['values'] = list(self.favorites.keys())
        if selected_list and selected_list in self.favorites:
            self.favorite_var.set(selected_list)
            analyze_logger.debug(f"Set dropdown to: {selected_list}")

    def cancel_screening(self):
        if messagebox.askyesno("Confirm Cancel", "Are you sure you want to cancel the screening?"):
            self.cancel_event.set()
            self.status_label.config(text="Cancelling screening...")

    # Screening Logic
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
            analyze_logger.error(f"Error checking data age: {str(e)}")
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
                screening_logger.info(f"Qualifying stocks for {exchange}: {qualifying_stocks}")
                if not self.cancel_event.is_set():
                    if qualifying_stocks:
                        list_name = await save_qualifying_stocks_to_favorites(qualifying_stocks, exchange)
                        if list_name:
                            self.root.after(0, lambda: self.refresh_favorites_dropdown(list_name))
                        else:
                            screening_logger.error(f"Failed to save qualifying stocks for {exchange}")
                    else:
                        screening_logger.info(f"No qualifying stocks found for {exchange}")
                    conn, cursor = get_stocks_connection()
                    try:
                        processed_tickers = cursor.execute(
                            "SELECT COUNT(*) FROM screening_progress WHERE exchange=? AND status='completed'",
                            (exchange,)
                        ).fetchone()[0]
                        total_tickers = processed_tickers + len(error_tickers)
                        ticker_list = list(self.ticker_manager.get_tickers(exchange))
                        cache_hits = sum(
                            1 for t in ticker_list
                            if get_stock_data_from_db(t) and time.time() - get_stock_data_from_db(t)['timestamp'] < CACHE_EXPIRY
                        )
                        self.root.after(0, lambda: self.update_cache_usage(cache_hits, total_tickers))
                        summary = f"Completed {exchange} screening.\nProcessed {total_tickers} stocks,\nFound {len(qualifying_stocks)} qualifiers,\n{len(error_tickers)} errors"
                        if error_tickers:
                            summary += f"\nError tickers: {', '.join(error_tickers)}"
                        self.root.after(0, lambda: messagebox.showinfo("Screening Complete", summary))
                    except sqlite3.Error as e:
                        screening_logger.error(f"Database error in screening summary: {str(e)}")
                    finally:
                        conn.close()
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Screening failed: {str(e)}"))
                screening_logger.error(f"Screening task failed for {exchange}: {str(e)}", exc_info=True)
            finally:
                self.root.after(0, lambda: self.progress_label.config(text=f"Progress: 100% (Screening Complete - {exchange})"))
                self.root.after(0, lambda: self.status_label.config(text=""))
                setattr(self, f"{exchange.lower()}_screen_var", tk.BooleanVar(value=False))
                self.screening_active = False

        self.task_queue.put(screening_task())

    def run_nyse_screening(self):
        self.run_screening("NYSE", screen_nyse_graham_stocks)

    def run_nasdaq_screening(self):
        self.run_screening("NASDAQ", screen_nasdaq_graham_stocks)

    # Data Fetching and Analysis
    async def fetch_company_name(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            return stock.info.get('longName', 'Unknown')
        except Exception as e:
            analyze_logger.error(f"Error fetching company name for {ticker}: {str(e)}")
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
                   (time.time() - timestamp < CACHE_EXPIRY):
                    cursor.execute("SELECT * FROM stocks WHERE ticker=?", (ticker,))
                    stock_row = cursor.fetchone()
                    if stock_row:
                        columns = [desc[0] for desc in cursor.description]
                        stock_dict = dict(zip(columns, stock_row))
                        analyze_logger.info(f"Database cache hit for {ticker}")
                        stock = yf.Ticker(ticker)
                        info = stock.info
                        price = info.get('regularMarketPrice', info.get('previousClose', None))
                        if price is None:
                            analyze_logger.error(f"No price data for {ticker} from YFinance")
                            return None
                        roe_list = [float(x) if x.strip() else 0.0 for x in stock_dict['roe'].split(",")] if stock_dict['roe'] else []
                        rotc_list = [float(x) if x.strip() else 0.0 for x in stock_dict['rotc'].split(",")] if stock_dict['rotc'] else []
                        eps_list = [float(x) if x.strip() else 0.0 for x in stock_dict['eps'].split(",")] if stock_dict['eps'] else []
                        div_list = [float(x) if x.strip() else 0.0 for x in stock_dict['dividend'].split(",")] if stock_dict['dividend'] else []
                        balance_data = json.loads(stock_dict['balance_data']) if stock_dict['balance_data'] else []
                        years = [int(y) for y in stock_dict['years'].split(",")] if stock_dict.get('years') else []

                        company_name = stock_dict['company_name']
                        debt_to_equity = stock_dict['debt_to_equity']
                        eps_ttm = stock_dict['eps_ttm']
                        book_value_per_share = stock_dict['book_value_per_share']
                        sector = stock_dict.get('sector', 'Unknown')

                        cached_stock_data = {
                            "ticker": stock_dict['ticker'],
                            "sector": sector,
                            "eps_ttm": eps_ttm,
                            "eps_list": eps_list,
                            "eps_cagr": stock_dict.get('eps_cagr', 0.0)
                        }
                        intrinsic_value = await calculate_graham_value(eps_ttm, cached_stock_data) if eps_ttm is not None and eps_ttm > 0 else float('nan')
                        margin_of_safety = self.margin_of_safety_var.get() / 100
                        expected_return = self.expected_return_var.get() / 100
                        buy_price = intrinsic_value * (1 - margin_of_safety) if not pd.isna(intrinsic_value) else float('nan')
                        sell_price = intrinsic_value * (1 + expected_return) if not pd.isna(intrinsic_value) else float('nan')
                        latest_revenue = stock_dict['latest_revenue']
                        graham_score = calculate_graham_score_8(ticker, price, None, None, debt_to_equity, eps_list, div_list, {}, balance_data, stock_dict['available_data_years'], latest_revenue)

                        result = {
                            "ticker": stock_dict['ticker'],
                            "sector": sector,
                            "company_name": company_name,
                            "price": price,
                            "intrinsic_value": intrinsic_value,
                            "buy_price": buy_price,
                            "sell_price": sell_price,
                            "graham_score": graham_score,
                            "years": years,
                            "roe_list": roe_list,
                            "rotc_list": rotc_list,
                            "eps_list": eps_list,
                            "div_list": div_list,
                            "balance_data": balance_data,
                            "latest_revenue": latest_revenue,
                            "available_data_years": stock_dict['available_data_years'],
                            "eps_ttm": eps_ttm,
                            "book_value_per_share": book_value_per_share,
                            "debt_to_equity": debt_to_equity,
                            "eps_cagr": stock_dict.get('eps_cagr', 0.0)
                        }
                        with self.ticker_cache_lock:
                            self.ticker_cache[ticker] = result
                        analyze_logger.debug(f"Cached data retrieved for {ticker}: Score={graham_score}/8 with {stock_dict['available_data_years']} years")
                        return result
            analyze_logger.info(f"Cache miss for {ticker}: Data stale or missing")
            return None
        except sqlite3.Error as e:
            analyze_logger.error(f"Database error in fetch_cached_data for {ticker}: {str(e)}")
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
        analyze_logger.info(f"Starting analysis for tickers: {tickers}")

        nyse_invalid_file = os.path.join(USER_DATA_DIR, "NYSE Invalid Tickers.txt")
        nasdaq_invalid_file = os.path.join(USER_DATA_DIR, "NASDAQ Invalid Tickers.txt")
        invalid_tickers = set()
        for invalid_file in [nyse_invalid_file, nasdaq_invalid_file]:
            if os.path.exists(invalid_file):
                with open(invalid_file, 'r') as f:
                    invalid_tickers.update(f.read().splitlines())
        valid_tickers = [t for t in tickers if t not in invalid_tickers]
        if len(valid_tickers) < len(tickers):
            excluded = set(tickers) - set(valid_tickers)
            analyze_logger.info(f"Excluded {len(excluded)} invalid tickers: {excluded}")
            self.root.after(0, lambda: messagebox.showinfo("Excluded Tickers", f"Excluded {len(excluded)} invalid tickers: {', '.join(excluded)}"))
        tickers = valid_tickers

        if not tickers:
            analyze_logger.warning("No valid tickers provided for analysis")
            messagebox.showwarning("No Tickers", "No valid tickers to analyze.")
            return

        conn, cursor = get_stocks_connection()
        try:
            cursor.execute("SELECT MAX(timestamp) FROM stocks")
            max_timestamp = cursor.fetchone()[0]
            if max_timestamp and time.time() - max_timestamp > 365 * 24 * 60 * 60:
                analyze_logger.warning("Cached data is over a year old")
                messagebox.showwarning("Old Data", "The cached data is over a year old. Consider refreshing the data.")
            else:
                analyze_logger.info(f"Most recent data timestamp: {max_timestamp}")
        except sqlite3.Error as e:
            analyze_logger.error(f"Error checking data age: {str(e)}")
        finally:
            conn.close()

        self.progress_var.set(0)
        self.cache_var.set(0)
        self.progress_label.config(text=f"Progress: 0% (0/{len(tickers)} stocks processed, 0 passed)")
        self.cache_label.config(text="Cache Usage: 0% (0 cached, 0 fresh)")
        self.rate_limit_label.config(text="")
        self.status_label.config(text="Analyzing stocks...")
        self.root.update()

        analyze_logger.info(f"Fetching data for {len(tickers)} tickers with margin_of_safety={self.margin_of_safety_var.get()/100}, expected_return={self.expected_return_var.get()/100}")
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

        analyze_logger.info(f"Analysis complete: {len(valid_results)} valid results, {len(error_tickers)} erros")
        if error_tickers:
            analyze_logger.warning(f"Error tickers: {error_tickers}")
            error_summary = "\n".join(error_tickers)
            self.root.after(0, lambda: messagebox.showwarning("Analysis Errors", f"Errors occurred for the following tickers:\n{error_summary}"))

        analyze_logger.debug("Populating treeview with analysis results")
        for item in self.tree.get_children():
            self.tree.delete(item)

        for result in valid_results:
            years_used = result.get('available_data_years', 0)
            analyze_logger.debug(f"Processing result for {result['ticker']}: Graham Score={result['graham_score']}, Years={years_used}")
            if years_used < 10:
                analyze_logger.warning(f"{result['ticker']} has only {years_used} years of data")
                self.root.after(0, lambda t=result['ticker'], y=years_used: messagebox.showwarning("Insufficient Data", f"{t} has only {y} years of data. Results may be incomplete."))
            warning = f" (based on {years_used} years)" if years_used < 10 else ""

            price = result['price']
            buy_price = result['buy_price']
            tags = ('highlight',) if price <= buy_price and not pd.isna(price) and not pd.isna(buy_price) else ()
            self.tree.insert("", "end", text=result['ticker'], values=(
                result['company_name'],
                result['sector'],
                f"{result['graham_score']}/8{warning}",
                f"${self.format_float(result['price'])}",
                f"${self.format_float(result['intrinsic_value'])}",
                f"${self.format_float(result['buy_price'])}",
                f"${self.format_float(result['sell_price'])}"
            ), tags=tags)

        cache_hits = sum(1 for t in tickers if t in self.ticker_cache)
        analyze_logger.info(f"Cache usage: {cache_hits} hits out of {len(tickers)} total tickers")
        self.root.after(0, lambda: self.update_cache_usage(cache_hits, len(tickers)))
        self.root.after(0, lambda: self.update_progress_animated(100, tickers, passed_tickers))
        self.root.after(0, lambda: self.status_label.config(text=""))
        analyze_logger.info("Analysis fully completed and UI updated")

    def analyze_multiple_stocks(self, tickers_input=None):
        analyze_logger.info("Analyze Stocks button clicked")
        if self.analysis_lock.acquire(timeout=5):
            try:
                if tickers_input is None:
                    tickers_input = self.entry.get()
                tickers = self.parse_tickers(tickers_input)
                analyze_logger.info(f"Parsed tickers from input '{tickers_input}': {tickers}")
                if not tickers:
                    analyze_logger.warning("No tickers parsed from input")
                self.task_queue.put(self.analyze_multiple_stocks_async(tickers))
            finally:
                self.analysis_lock.release()
        else:
            analyze_logger.error("Failed to acquire analysis lock within 5 seconds")
            messagebox.showerror("Error", "Unable to start analysis: Lock timeout.")

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

    def update_tabs(self, event):
        selected = self.tree.selection()
        if not selected:
            return
        ticker = self.tree.item(selected[0], "text")

        with self.ticker_cache_lock:
            result = self.ticker_cache.get(ticker, None)

        if not result:
            analyze_logger.warning(f"No cached data found for {ticker}, fetching fresh data")
            async def fetch_fresh():
                fresh_result = await fetch_stock_data(
                    ticker,
                    expected_return=self.expected_return_var.get() / 100,
                    margin_of_safety=self.margin_of_safety_var.get() / 100,
                    ticker_manager=self.ticker_manager
                )
                with self.ticker_cache_lock:
                    self.ticker_cache[ticker] = fresh_result
                return fresh_result
            result = asyncio.run(fetch_fresh())

        analyze_logger.info(f"Updating tabs for {ticker} with data keys: {list(result.keys())}")

        self.root.after(0, lambda: self.historical_text.delete('1.0', tk.END))
        self.root.after(0, lambda: self.metrics_text.delete('1.0', tk.END))

        years = result.get('years', [])
        roe_list = result.get('roe_list', [])
        rotc_list = result.get('rotc_list', [])
        eps_list = result.get('eps_list', [])
        div_list = result.get('div_list', [])

        analyze_logger.debug(f"Historical data for {ticker}: years={len(years)}, roe={len(roe_list)}, rotc={len(rotc_list)}, eps={len(eps_list)}, dividend={len(div_list)}")
        analyze_logger.debug(f"Years: {years}")
        analyze_logger.debug(f"ROE: {roe_list}")
        analyze_logger.debug(f"ROTC: {rotc_list}")
        analyze_logger.debug(f"EPS: {eps_list}")
        analyze_logger.debug(f"Dividend: {div_list}")

        if not years:
            self.root.after(0, lambda: self.historical_text.insert(tk.END, f"No historical data available for {ticker}\n"))
        elif len(years) == len(roe_list) == len(rotc_list) == len(eps_list) == len(div_list):
            header = f"{len(years)}-Year Historical Data for {ticker}:\nYear\tROE (%)\tROTC (%)\tEPS ($)\tDividend ($)\n"
            self.root.after(0, lambda: self.historical_text.insert(tk.END, header))
            for j in range(len(years)):
                line = f"{years[j]}\t{roe_list[j]:.2f}\t{rotc_list[j]:.2f}\t{eps_list[j]:.2f}\t{div_list[j]:.2f}\n"
                self.root.after(0, lambda l=line: self.historical_text.insert(tk.END, l))
        else:
            message = f"Incomplete Historical Data for {ticker} (Years: {len(years)}, ROE: {len(roe_list)}, ROTC: {len(rotc_list)}, EPS: {len(eps_list)}, Dividend: {len(div_list)})\n"
            self.root.after(0, lambda m=message: self.historical_text.insert(tk.END, m))

        self.root.after(0, lambda: self.metrics_text.insert(tk.END, f"Graham Criteria Results for {ticker} (Score: {result['graham_score']}/8 with {result['available_data_years']} years):\n"))
        self.root.after(0, lambda: self.metrics_text.insert(tk.END, f"Sector: {result['sector']} (Growth Rate: {get_sector_growth_rate(result['sector'])}%)\n\n"))

        async def update_metrics():
            stock_data = get_stock_data_from_db(ticker)
            if not stock_data:
                self.safe_insert(self.metrics_text, f"No data available for {ticker}\n")
                return

            eps_list = stock_data['eps_list']
            div_list = stock_data['div_list']
            balance_data = stock_data['balance_data']
            latest_revenue = stock_data['latest_revenue']
            debt_to_equity = stock_data['debt_to_equity']
            available_data_years = stock_data['available_data_years']
            sector = stock_data['sector']
            eps_ttm = stock_data['eps_ttm']
            book_value_per_share = stock_data['book_value_per_share']
            price = result.get('price', None)

            revenue_passed = latest_revenue >= 500_000_000
            self.safe_insert(self.metrics_text, f"1. Revenue >= $500M: {'Yes' if revenue_passed else 'No'} (${latest_revenue / 1e6:.2f}M)\n")

            if balance_data and 'totalCurrentAssets' in balance_data[0] and 'totalCurrentLiabilities' in balance_data[0]:
                current_assets = float(balance_data[0].get('totalCurrentAssets', 0))
                current_liabilities = float(balance_data[0].get('totalCurrentLiabilities', 1))
                current_ratio = current_assets / current_liabilities if current_liabilities > 0 else 0
                current_passed = current_ratio > 2
                self.safe_insert(self.metrics_text, f"2. Current Ratio > 2: {'Yes' if current_passed else 'No'} ({current_ratio:.2f})\n")
            else:
                self.safe_insert(self.metrics_text, "2. Current Ratio > 2: No (Missing data)\n")

            negative_eps_count = sum(1 for eps in eps_list if eps <= 0)
            stability_passed = negative_eps_count == 0
            self.safe_insert(self.metrics_text, f"3. All Positive EPS: {'Yes' if stability_passed else 'No'} (Negative EPS years: {negative_eps_count})\n")

            dividend_passed = all(div > 0 for div in div_list) if div_list else False
            div_display = f" (Dividends: {', '.join(f'{d:.2f}' for d in div_list)})" if div_list else ""
            self.safe_insert(self.metrics_text, f"4. Uninterrupted Dividends: {'Yes' if dividend_passed else 'No'}{div_display}\n")

            if len(eps_list) >= 2 and eps_list[0] > 0 and eps_list[-1] > 0:
                cagr = calculate_cagr(eps_list[0], eps_list[-1], available_data_years - 1)
                growth_passed = cagr > 0.03
                self.safe_insert(self.metrics_text, f"5. EPS CAGR > 3%: {'Yes' if growth_passed else 'No'} ({cagr:.2%})\n")
            else:
                eps_display = f" (EPS: {', '.join(f'{e:.2f}' for e in eps_list)})" if eps_list else ""
                self.safe_insert(self.metrics_text, f"5. EPS CAGR > 3%: No (Insufficient or invalid data{eps_display})\n")

            debt_passed = debt_to_equity is not None and debt_to_equity < 2
            self.safe_insert(self.metrics_text, f"6. Debt-to-Equity < 2: {'Yes' if debt_passed else 'No'} ({self.format_float(debt_to_equity)})\n")

            pe_ratio = price / eps_ttm if price and eps_ttm and eps_ttm > 0 else None
            pe_pass = pe_ratio is not None and pe_ratio <= 15
            self.safe_insert(self.metrics_text, f"7. P/E Ratio <= 15: {'Yes' if pe_pass else 'No'} ({self.format_float(pe_ratio)})\n")

            pb_ratio = price / book_value_per_share if price and book_value_per_share and book_value_per_share > 0 else None
            pb_pass = pb_ratio is not None and pb_ratio <= 1.5
            self.safe_insert(self.metrics_text, f"8. P/B Ratio <= 1.5: {'Yes' if pb_pass else 'No'} ({self.format_float(pb_ratio)})\n")

            analyze_logger.info(f"Metrics updated for {ticker} using cached data")

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
                SELECT ticker, common_score, sector 
                FROM graham_qualifiers 
                WHERE exchange='NYSE' AND common_score IS NOT NULL
                ORDER BY common_score DESC, ticker ASC
            """)
            qualifier_results = cursor.fetchall()

            if not qualifier_results:
                messagebox.showinfo("No Results", "No NYSE stocks meet the Graham criteria.")
                analyze_logger.info("No qualifying NYSE stocks found in graham_qualifiers.")
                return

            cursor.execute("SELECT ticker, company_name FROM stocks WHERE ticker IN (SELECT ticker FROM graham_qualifiers WHERE exchange='NYSE')")
            ticker_to_company = {row[0]: row[1] for row in cursor.fetchall()}

            for item in self.tree.get_children():
                self.tree.delete(item)

            for ticker, common_score, sector in qualifier_results:
                company_name = ticker_to_company.get(ticker, "Unknown")
                self.tree.insert("", "end", text=ticker, values=(company_name, sector, f"{common_score}/6", "", "", "", ""))
                analyze_logger.debug(f"Added {ticker} to treeview: {company_name}, Sector: {sector}, Score: {common_score}/6")

            analyze_logger.info(f"Displayed {len(qualifier_results)} NYSE qualifying stocks in treeview.")
        except sqlite3.Error as e:
            analyze_logger.error(f"Database error in display_nyse_qualifying_stocks: {str(e)}")
            messagebox.showerror("Error", f"Database error: {str(e)}")
        finally:
            conn.close()

    def display_nasdaq_qualifying_stocks(self):
        conn, cursor = get_stocks_connection()
        try:
            cursor.execute(""" 
                SELECT ticker, common_score, sector 
                FROM graham_qualifiers 
                WHERE exchange='NASDAQ' AND common_score IS NOT NULL
                ORDER BY common_score DESC, ticker ASC
            """)
            qualifier_results = cursor.fetchall()

            if not qualifier_results:
                messagebox.showinfo("No Results", "No NASDAQ stocks meet the Graham criteria.")
                analyze_logger.info("No qualifying NASDAQ stocks found in graham_qualifiers.")
                return

            cursor.execute("SELECT ticker, company_name FROM stocks WHERE ticker IN (SELECT ticker FROM graham_qualifiers WHERE exchange='NASDAQ')")
            ticker_to_company = {row[0]: row[1] for row in cursor.fetchall()}

            for item in self.tree.get_children():
                self.tree.delete(item)

            for ticker, common_score, sector in qualifier_results:
                company_name = ticker_to_company.get(ticker, "Unknown")
                self.tree.insert("", "end", text=ticker, values=(company_name, sector, f"{common_score}/6", "", "", "", ""))
                analyze_logger.debug(f"Added {ticker} to treeview: {company_name}, Sector: {sector}, Score: {common_score}/6")

            analyze_logger.info(f"Displayed {len(qualifier_results)} NASDAQ qualifying stocks in treeview.")
        except sqlite3.Error as e:
            analyze_logger.error(f"Database error in display_nasdaq_qualifying_stocks: {str(e)}")
            messagebox.showerror("Error", f"Database error: {str(e)}")
        finally:
            conn.close()

    def fetch_current_prices(self, tickers):
        """Fetch current prices for a list of tickers using yfinance."""
        try:
            data = yf.download(tickers, period="1d")['Close']
            if len(tickers) == 1:
                return {tickers[0]: data.iloc[-1]}
            else:
                return data.iloc[-1].to_dict()
        except Exception as e:
            analyze_logger.error(f"Error fetching prices: {e}")
            return {ticker: None for ticker in tickers}

    def calculate_intrinsic_value(self, stock_dict):
        """Calculate Graham intrinsic value using EPS and EPS CAGR."""
        eps = stock_dict.get('eps_ttm')
        eps_cagr = stock_dict.get('eps_cagr', 0.0)
        if not eps or eps <= 0:
            return float('nan')
        aaa_yield = get_aaa_yield(FRED_API_KEY)
        if aaa_yield <= 0:
            return float('nan')
        g = eps_cagr * 100  # Convert decimal to percentage
        earnings_multiplier = min(8.5 + 2 * g, 20)  # Cap at 20
        normalization_factor = 4.4
        value = (eps * earnings_multiplier * normalization_factor) / (100 * aaa_yield)
        return value

    def setup_start_here_sheet(self, start_sheet):
        # Set tab color to yellow
        start_sheet.sheet_properties.tabColor = "FFFF00"

        # Set column widths to 10 for columns A to H
        for col in range(1, 9):
            start_sheet.column_dimensions[get_column_letter(col)].width = 10

        # Row 1: Title
        start_sheet['A1'] = "DIRECTIONS, GENERAL NOTES, CALCULATIONS AND CONSTANTS FOR FOLLOWING SPREADSHEETS"
        start_sheet['A1'].font = Font(bold=True, size=16)
        start_sheet['A1'].alignment = Alignment(horizontal='left', vertical='center')
        start_sheet.row_dimensions[1].height = 30

        # Rows 3 to 8: Instructions
        instructions = [
            "Go to NYSE Index below and select an alphabet Letter. Click on a company name hyperlink you'd like to analyze, then click on the company name hyperlink again. This will open the Morningstar quote page for the respective company.",
            "On the Morningstar quote page scroll down to 'financials' and click it. At the bottom of the table click 'All Financials Data.' Click the 'balance sheet' link at the top. Record 'Total Assets' and 'Total Liabilities.'",
            "At the top of the 'All Financials Data' page, select the 'Key Ratios' tab. From the 'Financials' section, record the 'Net Income,' 'Earnings Per Share,' 'Dividends,' 'Shares,' and 'Working Capital' for the most recent year.",
            "Staying on the 'Key Ratios' tab in the 'Key Ratios' section, record 'Return on Equity' and 'Return on Invested Capital' for the last 10 years.",
            "Finally, remaining on the 'Key Ratios' tab, in the 'Key Ratios' section, select the 'Financial Health' tab. Scroll down to 'Long Term Debt' and enter the most recent annual number in the spreadsheet.",
            "The spreadsheet will calculate the rest. Click save. You're done! Good job and may we get rich together!!"
        ]
        for i, text in enumerate(instructions, start=3):
            start_sheet[f'A{i}'] = text
            start_sheet[f'A{i}'].alignment = Alignment(horizontal='left', vertical='center')

        # Rows 11 to 20: Tips
        tips = [
            "Consistently high Return on Equity shows durability; ROE = Net Income / Book Value (should be better than 12%).",
            "Consistently high Return on Total Capital shows durability: ROTC = Net Earnings / Total Capital (should be better than 12% except for banks. Bank Return on Total Assets should be better than 1-1.5%).",
            "Consistently high Earnings Per Share (EPS) shows durable competitive advantage: Earnings Per Share = Total Net Earnings / Number of Shares Outstanding. Should be strong and show an upward trend. Also look for an upward trend then a sharp drop. Sometimes this is a sign of a one-time oops that the market overreacts to.",
            "Durable companies should have Long Term Debt of no more than 5 times current net earnings.",
            "The product the company sells should be something that people use every day but wears out quickly causing repurchase.",
            "Durable companies are not controlled by labor unionsâ€¦ try to avoid them.",
            "Avoid companies that cannot sustain the price of their product in a down economy or a rise in inflation (i.e., airlines), good example - Coca-Cola.",
            "Avoid companies that have to reinvest their earnings in operational costs (i.e., General Motors, Bethlehem Steel), good example - H&R Block, Wrigleys (same products forever with very little change).",
            "Look for companies that buy back shares of their own stock. This shows they have excess cash and want to pay down debt.",
            "Look for a company that uses its earnings to increase overall market value (i.e., Berkshire Hathaway). Look at a ten-year spread on the company's share price and its book value. If these values do not show a significant increase over the last 10 years, it is not using its earnings to increase its worth."
        ]
        for i, tip in enumerate(tips, start=11):
            start_sheet[f'A{i}'] = tip
            start_sheet[f'A{i}'].alignment = Alignment(horizontal='left', vertical='center')

        # Row 22: When to sell
        start_sheet['A22'] = "When to sell? When a company has exceeded its Intrinsic Value, it is in sell territory. If you want to hold, look at the 10-year trend on EPS. Stretch out those earnings over the next 10 years and compare them to the gain one would receive if you sold a share of the stock at the current price and invested it at the AAA Corporate bond rate for the next ten years. If you make more with the bond, it's a good time to sell the stock and look for another investment."
        start_sheet['A22'].alignment = Alignment(horizontal='left', vertical='center')

        # Row 24: AAA Corporate Bond
        start_sheet.merge_cells('A24:B24')
        start_sheet['A24'] = "AAA Corporate Bond"
        start_sheet['A24'].font = Font(bold=True)
        start_sheet['A24'].alignment = Alignment(horizontal='center', vertical='center')
        aaa_yield = get_aaa_yield(FRED_API_KEY)
        start_sheet['C24'] = aaa_yield
        start_sheet['C24'].number_format = '0.00%'
        start_sheet['C24'].alignment = Alignment(horizontal='center', vertical='center')

        # Row 26: Graham Criteria for Investment Grade Stocks
        start_sheet.merge_cells('A26:D26')
        start_sheet['A26'] = "Graham Criteria for Investment Grade Stocks"
        start_sheet['A26'].font = Font(bold=True)
        start_sheet['A26'].alignment = Alignment(horizontal='center', vertical='center')

        # Rows 28 to 34: Investment Grade Criteria
        investment_criteria = [
            "1) Adequate Size = 500 million in annual sales and 100 million in assets.",
            "2) Strong Financially = 2:1 Current Yield.",
            "3) 20 Years of Dividend Payment (I use 10 years for simplicity).",
            "4) No negative Earnings Per Share over the last 10 years.",
            "5) Earnings growth of at least 33% over last 10 years. (I use a CAGR of at least 3%).",
            "6) Share Price <= 1.5x Net Asset Value.",
            "7) Share Price <= 15x Average Earnings for past 3 years."
        ]
        for i, crit in enumerate(investment_criteria, start=28):
            start_sheet[f'A{i}'] = crit
            start_sheet[f'A{i}'].alignment = Alignment(horizontal='left', vertical='center')

        # Row 36: Graham Criteria for Good Value Companies
        start_sheet.merge_cells('A36:E36')
        start_sheet['A36'] = "Graham criteria for good value companies. 7/10 is goal."
        start_sheet['A36'].font = Font(bold=True)
        start_sheet['A36'].alignment = Alignment(horizontal='center', vertical='center')

        # Rows 38 to 47: Value Company Criteria
        value_criteria = [
            "1) Earnings/Price ratio = 2x AAA Corporate Bond Yield.",
            "2) Price/Earnings ratio = .4 of highest P/E average of the last 10 years.",
            "3) Dividend Yield >= 2/3 of AAA Bond Yield.",
            "4) Share Price = 2/3 Tangible Book Value per Share.",
            "5) 2/3 Net Current Asset Value.",
            "6) Total Debt Lower than Tangible Book Value.",
            "7) Current Ratio >= 2.",
            "8) Total Debt <= Net Quick Liquidation Value.",
            "9) Earnings have doubled in the last 10 years.",
            "10) Earnings have declined no more than 5% in the last 10 years."
        ]
        for i, crit in enumerate(value_criteria, start=38):
            start_sheet[f'A{i}'] = crit
            start_sheet[f'A{i}'].alignment = Alignment(horizontal='left', vertical='center')

        # Disable text wrapping for all rows
        for row in start_sheet.iter_rows():
            for cell in row:
                if cell.row in [24, 26, 36]:
                    cell.alignment = Alignment(wrap_text=False, horizontal='center', vertical='center')
                else:
                    cell.alignment = Alignment(wrap_text=False, horizontal='left', vertical='center')

    def export_qualifying_stocks(self, exchange):
        """Export qualifying stocks to an Excel workbook with specified formatting."""
        conn, cursor = get_stocks_connection()
        try:
            # Fetch qualifying tickers and data including eps_cagr
            cursor.execute(""" 
                SELECT g.ticker, g.sector, g.common_score, s.company_name, s.years, s.roe, s.rotc, s.eps, s.dividend,
                       s.debt_to_equity, s.eps_ttm, s.book_value_per_share, s.latest_revenue, s.available_data_years, s.balance_data,
                       s.latest_net_income, s.latest_long_term_debt, s.eps_cagr
                FROM graham_qualifiers g
                LEFT JOIN stocks s ON g.ticker = s.ticker
                WHERE g.exchange=?
            """, (exchange,))
            qualifiers = cursor.fetchall()
            if not qualifiers:
                messagebox.showinfo("No Data", f"No {exchange} qualifying stocks to export.")
                return

            tickers = [row[0] for row in qualifiers]
            prices = self.fetch_current_prices(tickers)

            # Create workbook
            wb = openpyxl.Workbook()

            # "Start Here" sheet
            start_sheet = wb.active
            start_sheet.title = "Start Here"
            self.setup_start_here_sheet(start_sheet)

            # Process stock data for summary tabs
            stock_data_list = []
            for row in qualifiers:
                ticker, sector, common_score, company_name, years, roe, rotc, eps, dividend, debt_to_equity, eps_ttm, book_value_per_share, latest_revenue, available_data_years, balance_data, latest_net_income, latest_long_term_debt, eps_cagr = row
                price = prices.get(ticker, "N/A")
                if price == "N/A" or not isinstance(price, (int, float)):
                    continue  # Skip stocks without valid price data
                try:
                    eps_list = [float(x) for x in eps.split(",")] if eps else []
                    div_list = [float(x) for x in dividend.split(",")] if dividend else []
                    balance_data_dict = json.loads(balance_data) if balance_data else []
                except Exception as e:
                    analyze_logger.error(f"Error parsing data for {ticker}: {e}")
                    continue

                intrinsic_value = self.calculate_intrinsic_value({'eps_ttm': eps_ttm, 'eps_cagr': eps_cagr})
                if pd.isna(intrinsic_value) or intrinsic_value == 0:
                    margin_of_safety = "N/A"
                else:
                    margin_of_safety = (intrinsic_value - price) / intrinsic_value * 100  # Percent current price is below/above intrinsic value
                expected_return = self.expected_return_var.get() / 100
                buy_price = intrinsic_value * (1 - (self.margin_of_safety_var.get() / 100)) if not pd.isna(intrinsic_value) else "N/A"
                sell_price = intrinsic_value * (1 + expected_return) if not pd.isna(intrinsic_value) else "N/A"
                graham_score = calculate_graham_score_8(ticker, price, None, None, debt_to_equity, eps_list, div_list, {}, balance_data_dict, available_data_years, latest_revenue)

                stock_data = {
                    "company_name": company_name,
                    "ticker": ticker,
                    "sector": sector,
                    "mos": margin_of_safety if margin_of_safety != "N/A" else "N/A",
                    "graham_score": graham_score if graham_score is not None else "N/A",
                    "current_price": price,
                    "intrinsic_value": intrinsic_value if not pd.isna(intrinsic_value) else "N/A",
                    "buy_price": buy_price if buy_price != "N/A" else "N/A",
                    "sell_price": sell_price if sell_price != "N/A" else "N/A"
                }
                stock_data_list.append(stock_data)

            # Sort stock data list by company name
            stock_data_list.sort(key=lambda x: x["company_name"])

            # Define financial sectors (adjust based on actual sector names in your data)
            financial_sectors = ['Financial Services', 'Finance', 'Banking', 'financials']

            # Make case-insensitive comparison
            financial_sectors_lower = [s.lower() for s in financial_sectors]
            financial_stocks = [stock for stock in stock_data_list if stock['sector'].lower() in financial_sectors_lower]
            other_stocks = [stock for stock in stock_data_list if stock['sector'].lower() not in financial_sectors_lower]

            # Headers for summary sheets
            headers = ["Company Name", "Ticker", "Sector", "MOS", "Graham Score", "Current Price", "Intrinsic Value", "Buy Price", "Sell Price"]

            # Function to create a summary sheet
            def create_summary_sheet(sheet_name, stocks):
                sheet = wb.create_sheet(sheet_name)
                for col, header in enumerate(headers, start=1):
                    cell = sheet.cell(row=1, column=col, value=header)
                    cell.font = Font(size=12, bold=True)
                    cell.alignment = Alignment(horizontal='center', vertical='center')
                for row_idx, stock in enumerate(stocks, start=2):
                    company_cell = sheet.cell(row=row_idx, column=1, value=stock["company_name"])
                    company_cell.hyperlink = f"#'{stock['ticker']}'!A1"
                    company_cell.style = "Hyperlink"
                    sheet.cell(row=row_idx, column=2, value=stock["ticker"])
                    sheet.cell(row=row_idx, column=3, value=stock["sector"])
                    sheet.cell(row=row_idx, column=4, value=stock["mos"] / 100 if stock["mos"] != "N/A" else "N/A")
                    sheet.cell(row=row_idx, column=5, value=stock["graham_score"] if stock["graham_score"] != "N/A" else "N/A")
                    sheet.cell(row=row_idx, column=6, value=stock["current_price"] if stock["current_price"] != "N/A" else "N/A")
                    sheet.cell(row=row_idx, column=7, value=stock["intrinsic_value"] if stock["intrinsic_value"] != "N/A" else "N/A")
                    sheet.cell(row=row_idx, column=8, value=stock["buy_price"] if stock["buy_price"] != "N/A" else "N/A")
                    sheet.cell(row=row_idx, column=9, value=stock["sell_price"] if stock["sell_price"] != "N/A" else "N/A")

                # Set number formats
                for row in range(2, len(stocks) + 2):
                    sheet.cell(row=row, column=4).number_format = '0.00%'  # MOS as percentage
                    sheet.cell(row=row, column=5).number_format = '0'  # Graham Score as integer
                    for col in range(6, 10):  # Currency columns
                        if sheet.cell(row=row, column=col).value != "N/A":
                            sheet.cell(row=row, column=col).number_format = '$#,##0.00'

                # Set column widths
                column_widths = [55, 12, 25, 12, 15, 18, 20, 15, 15]
                for col, width in enumerate(column_widths, start=1):
                    sheet.column_dimensions[get_column_letter(col)].width = width

                # Set alignment
                for row in sheet.iter_rows():
                    for cell in row:
                        cell.alignment = Alignment(horizontal='center', vertical='center')

                # Add auto_filter with sort on column A
                sheet.auto_filter.ref = f"A1:I{len(stocks) + 1}"
                sheet.auto_filter.add_sort_condition(f"A2:A{len(stocks) + 1}")

            # Create 'Winning Stocks' sheet for non-financial stocks
            if other_stocks:
                create_summary_sheet("Winning Stocks", other_stocks)
                print(f"Created 'Winning Stocks' sheet with {len(other_stocks)} stocks")

            # Create 'Financial Winners' sheet for financial stocks
            if financial_stocks:
                create_summary_sheet("Financial Winners", financial_stocks)
                print(f"Created 'Financial Winners' sheet with {len(financial_stocks)} stocks")
            else:
                print("No financial stocks to create 'Financial Winners' sheet")

            # Sort qualifiers by ticker for individual sheets
            qualifiers.sort(key=lambda x: x[0])  # Assuming ticker is the first element

            # Create individual stock sheets
            for row in qualifiers:
                ticker, sector, common_score, company_name, years, roe, rotc, eps, dividend, debt_to_equity, eps_ttm, book_value_per_share, latest_revenue, available_data_years, balance_data, latest_net_income, latest_long_term_debt, eps_cagr = row
                price = prices.get(ticker, "N/A")
                stock_sheet = wb.create_sheet(ticker)

                # Set row heights to 15 for rows 1 to 50
                for row_num in range(1, 51):
                    stock_sheet.row_dimensions[row_num].height = 15

                # Merge cells A2:K3 for company name and ticker
                stock_sheet.merge_cells('A2:K3')
                company_ticker_cell = stock_sheet['A2']
                company_ticker_cell.value = f"{company_name.upper()} ({ticker})"
                company_ticker_cell.font = Font(bold=True, size=18)
                company_ticker_cell.alignment = Alignment(horizontal='center', vertical='center')
                company_ticker_cell.hyperlink = f"https://www.morningstar.com/stocks/xnys/{ticker}/quote"

                # Set column widths
                stock_sheet.column_dimensions['A'].width = 25
                for col in range(2, 13):  # B to L
                    stock_sheet.column_dimensions[get_column_letter(col)].width = 10
                stock_sheet.column_dimensions['M'].width = 15
                stock_sheet.column_dimensions['N'].width = 10
                stock_sheet.column_dimensions['O'].width = 15

                # Determine last updated date and format as "31-May-25"
                ticker_file = NYSE_LIST_FILE if exchange == "NYSE" else NASDAQ_LIST_FILE
                if os.path.exists(ticker_file):
                    last_updated = datetime.fromtimestamp(os.path.getmtime(ticker_file)).strftime('%d-%b-%y')
                else:
                    last_updated = "Unknown"

                # Merge L1:N1 for "Last Updated:"
                stock_sheet.merge_cells('L1:N1')
                last_updated_label = stock_sheet['L1']
                last_updated_label.value = "Last Updated:"
                last_updated_label.alignment = Alignment(horizontal='center', vertical='center')

                # Set O1 to last updated date
                stock_sheet['O1'].value = last_updated
                stock_sheet['O1'].alignment = Alignment(horizontal='center', vertical='center')

                # Fetch AAA yield
                aaa_yield = get_aaa_yield(FRED_API_KEY)

                # Calculate P/E ratio
                if isinstance(price, (int, float)) and eps_ttm and eps_ttm > 0:
                    pe_ratio = price / eps_ttm
                else:
                    pe_ratio = "N/A"

                # Set L2: current price
                cell_l2 = stock_sheet['L2']
                if isinstance(price, (int, float)):
                    cell_l2.value = price
                    cell_l2.number_format = '$#,##0.00'
                else:
                    cell_l2.value = "N/A"
                cell_l2.font = Font(bold=True)
                cell_l2.alignment = Alignment(horizontal='center', vertical='center')

                # Set M2: "Current Price"
                cell_m2 = stock_sheet['M2']
                cell_m2.value = "Current Price"
                cell_m2.font = Font(bold=True)
                cell_m2.alignment = Alignment(horizontal='center', vertical='center')

                # Set N2: P/E ratio
                cell_n2 = stock_sheet['N2']
                if isinstance(pe_ratio, (int, float)):
                    cell_n2.value = pe_ratio
                    cell_n2.number_format = '0.00'
                else:
                    cell_n2.value = "N/A"
                cell_n2.font = Font(bold=True)
                cell_n2.alignment = Alignment(horizontal='center', vertical='center')

                # Set O2: "P/E Ratio"
                cell_o2 = stock_sheet['O2']
                cell_o2.value = "P/E Ratio"
                cell_o2.font = Font(bold=True)
                cell_o2.alignment = Alignment(horizontal='center', vertical='center')

                # Set L3: AAA yield
                cell_l3 = stock_sheet['L3']
                if isinstance(aaa_yield, (int, float)):
                    cell_l3.value = aaa_yield
                    cell_l3.number_format = '0.00%'
                else:
                    cell_l3.value = "N/A"
                cell_l3.font = Font(bold=True)
                cell_l3.alignment = Alignment(horizontal='center', vertical='center')

                # Merge M3:O3 and set "AAA Corporate Bond Rate"
                stock_sheet.merge_cells('M3:O3')
                cell_m3 = stock_sheet['M3']
                cell_m3.value = "AAA Corporate Bond Rate"
                cell_m3.font = Font(bold=True)
                cell_m3.alignment = Alignment(horizontal='center', vertical='center')

                # Set bold borders
                bold_side = Side(style='thick')
                thin_side = Side(style='thin')

                cell_l2.border = Border(top=bold_side, left=bold_side, bottom=bold_side, right=thin_side)
                cell_m2.border = Border(top=bold_side, right=bold_side, bottom=bold_side, left=thin_side)
                cell_n2.border = Border(top=bold_side, left=bold_side, bottom=bold_side, right=thin_side)
                cell_o2.border = Border(top=bold_side, right=bold_side, bottom=bold_side, left=thin_side)
                cell_l3.border = Border(top=bold_side, left=bold_side, bottom=bold_side, right=bold_side)
                cell_m3.border = Border(top=bold_side, left=bold_side, bottom=bold_side, right=bold_side)

                # Define labels with Unicode subscripts
                sub_1 = '\u2081'
                sub_0 = '\u2080'
                sub_10 = sub_1 + sub_0
                labels = [
                    "Year",
                    f"ROE{sub_10}",
                    f"ROTC{sub_10}",
                    f"EPS{sub_10}",
                    f"EPS{sub_10} % Change",
                    f"EPS{sub_10} Proj",
                    f"DIV{sub_10}",
                    f"DIV{sub_10} % Change",
                    f"DIV{sub_10} Proj"
                ]

                # Set labels in A4:A12 and M4:M12
                for i, label in enumerate(labels):
                    row_num = 4 + i
                    cell_a = stock_sheet[f'A{row_num}']
                    cell_m = stock_sheet[f'M{row_num}']
                    cell_a.value = label
                    cell_m.value = label
                    cell_a.font = Font(size=10, bold=True)
                    cell_m.font = Font(size=10, bold=True)
                    cell_a.alignment = Alignment(horizontal='center', vertical='center')
                    cell_m.alignment = Alignment(horizontal='center', vertical='center')

                # Parse historical data
                years_list = [int(y) for y in years.split(",")] if years else []
                roe_list = [float(x) for x in roe.split(",")] if roe else []
                rotc_list = [float(x) for x in rotc.split(",")] if rotc else []
                eps_list = [float(x) for x in eps.split(",")] if eps else []
                div_list = [float(x) for x in dividend.split(",")] if dividend else []

                # Take the last 10 years
                years_list = years_list[-10:]
                roe_list = roe_list[-10:]
                rotc_list = rotc_list[-10:]
                eps_list = eps_list[-10:]
                div_list = div_list[-10:]

                # Populate B4:K4 with years (bold)
                for col, year in enumerate(years_list, start=2):  # B to K
                    cell = stock_sheet.cell(row=4, column=col)
                    cell.value = year
                    cell.font = Font(bold=True)

                # Populate B5:K5 with ROE (as decimal)
                for col, roe_value in enumerate(roe_list, start=2):
                    cell = stock_sheet.cell(row=5, column=col)
                    if col-2 < len(roe_list):
                        cell.value = roe_value / 100 if isinstance(roe_value, (int, float)) else "N/A"
                    else:
                        cell.value = "N/A"
                    if isinstance(cell.value, (int, float)):
                        cell.number_format = '0.00%'

                # Populate B6:K6 with ROTC (as decimal)
                for col, rotc_value in enumerate(rotc_list, start=2):
                    cell = stock_sheet.cell(row=6, column=col)
                    if col-2 < len(rotc_list):
                        cell.value = rotc_value / 100 if isinstance(rotc_value, (int, float)) else "N/A"
                    else:
                        cell.value = "N/A"
                    if isinstance(cell.value, (int, float)):
                        cell.number_format = '0.00%'

                # Populate B7:K7 with EPS
                for col, eps_value in enumerate(eps_list, start=2):
                    cell = stock_sheet.cell(row=7, column=col)
                    cell.value = eps_value if col-2 < len(eps_list) else "N/A"
                    if isinstance(cell.value, (int, float)):
                        cell.number_format = '$#,##0.00'

                # Calculate EPS % Change in C8:K8
                for col in range(3, 12):  # C to K
                    prev_eps = stock_sheet.cell(row=7, column=col-1).value
                    current_eps = stock_sheet.cell(row=7, column=col).value
                    if isinstance(prev_eps, (int, float)) and isinstance(current_eps, (int, float)) and prev_eps != 0:
                        change = (current_eps - prev_eps) / prev_eps
                        stock_sheet.cell(row=8, column=col).value = change
                        stock_sheet.cell(row=8, column=col).number_format = '0.00%'
                    else:
                        stock_sheet.cell(row=8, column=col).value = "N/A"

                # Populate B10:K10 with Dividends
                for col, div_value in enumerate(div_list, start=2):
                    cell = stock_sheet.cell(row=10, column=col)
                    cell.value = div_value if col-2 < len(div_list) else "N/A"
                    if isinstance(cell.value, (int, float)):
                        cell.number_format = '$#,##0.00'

                # Calculate Dividend % Change in C11:K11
                for col in range(3, 12):  # C to K
                    prev_div = stock_sheet.cell(row=10, column=col-1).value
                    current_div = stock_sheet.cell(row=10, column=col).value
                    if isinstance(prev_div, (int, float)) and isinstance(current_div, (int, float)) and prev_div != 0:
                        change = (current_div - prev_div) / prev_div
                        stock_sheet.cell(row=11, column=col).value = change
                        stock_sheet.cell(row=11, column=col).number_format = '0.00%'
                    else:
                        stock_sheet.cell(row=11, column=col).value = "N/A"

                # Set L4 to "Avgâ‚â‚€"
                stock_sheet['L4'].value = f"Avg{sub_10}"
                stock_sheet['L4'].font = Font(bold=True)
                stock_sheet['L4'].alignment = Alignment(horizontal='center', vertical='center')

                # Set averages using Excel formulas for L5:L12
                stock_sheet['L5'].value = "=AVERAGE(B5:K5)"
                stock_sheet['L5'].number_format = '0.00%'
                stock_sheet['L6'].value = "=AVERAGE(B6:K6)"
                stock_sheet['L6'].number_format = '0.00%'
                stock_sheet['L7'].value = "=AVERAGE(B7:K7)"
                stock_sheet['L7'].number_format = '$#,##0.00'
                stock_sheet['L8'].value = "=AVERAGE(B8:K8)"
                stock_sheet['L8'].number_format = '0.00%'
                stock_sheet['L9'].value = "=AVERAGE(B9:K9)"
                stock_sheet['L9'].number_format = '$#,##0.00'
                stock_sheet['L10'].value = "=AVERAGE(B10:K10)"
                stock_sheet['L10'].number_format = '$#,##0.00'
                stock_sheet['L11'].value = "=AVERAGE(B11:K11)"
                stock_sheet['L11'].number_format = '0.00%'
                stock_sheet['L12'].value = "=AVERAGE(B12:K12)"
                stock_sheet['L12'].number_format = '$#,##0.00'

                # Set EPS projections
                stock_sheet['B9'].value = f"={get_column_letter(11)}7 * (1 + {get_column_letter(12)}8)"
                stock_sheet['B9'].number_format = '$#,##0.00'
                for col in range(3, 12):  # C to K
                    prev_col_letter = get_column_letter(col - 1)
                    stock_sheet.cell(row=9, column=col).value = f"={prev_col_letter}9 * (1 + ${get_column_letter(12)}$8)"
                    stock_sheet.cell(row=9, column=col).number_format = '$#,##0.00'

                # Set Dividend projections
                stock_sheet['B12'].value = f"={get_column_letter(11)}10 * (1 + {get_column_letter(12)}11)"
                stock_sheet['B12'].number_format = '$#,##0.00'
                for col in range(3, 12):  # C to K
                    prev_col_letter = get_column_letter(col - 1)
                    stock_sheet.cell(row=12, column=col).value = f"={prev_col_letter}12 * (1 + ${get_column_letter(12)}$11)"
                    stock_sheet.cell(row=12, column=col).number_format = '$#,##0.00'

                # Row 13
                stock_sheet['A13'].value = "Long Term Debt ($M)"
                stock_sheet['A13'].font = Font(bold=True)
                stock_sheet['A13'].alignment = Alignment(horizontal='center', vertical='center')

                if latest_long_term_debt is not None:
                    stock_sheet['B13'].value = latest_long_term_debt / 1_000_000
                    stock_sheet['B13'].number_format = '$#,##0'
                else:
                    stock_sheet['B13'].value = "N/A"

                stock_sheet.merge_cells('C13:D13')
                stock_sheet['C13'].value = "Net Income ($M)"
                stock_sheet['C13'].font = Font(bold=True)
                stock_sheet['C13'].alignment = Alignment(horizontal='center', vertical='center')

                if latest_net_income is not None:
                    stock_sheet['E13'].value = latest_net_income / 1_000_000
                    stock_sheet['E13'].number_format = '$#,##0'
                else:
                    stock_sheet['E13'].value = "N/A"

                stock_sheet['F13'].value = "LTD/NI"
                stock_sheet['F13'].font = Font(bold=True)
                stock_sheet['F13'].alignment = Alignment(horizontal='center', vertical='center')

                stock_sheet['G13'].value = "=IF(E13=0, \"N/A\", B13/E13)"
                stock_sheet['G13'].number_format = '0.00'

                # Set alignment for row 13
                for col in ['A', 'B', 'C', 'E', 'F', 'G']:
                    stock_sheet[f'{col}13'].alignment = Alignment(horizontal='center', vertical='center')

                # Center B4:L12 horizontally and vertically
                for row in range(4, 13):
                    for col in range(2, 13):  # B to L
                        cell = stock_sheet.cell(row=row, column=col)
                        cell.alignment = Alignment(horizontal='center', vertical='center')

            # Save the workbook
            file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")], initialfile=f"{exchange}_Qualifying_Stocks.xlsx")
            if file_path:
                wb.save(file_path)
                messagebox.showinfo("Export Successful", f"Qualifying stocks exported to {file_path}")
        except Exception as e:
            analyze_logger.error(f"Error exporting qualifying stocks: {e}")
            messagebox.showerror("Export Error", f"An error occurred while exporting: {str(e)}")
        finally:
            conn.close()

    def export_nyse_qualifying_stocks(self):
        self.export_qualifying_stocks("NYSE")

    def export_nasdaq_qualifying_stocks(self):
        self.export_qualifying_stocks("NASDAQ")

    def check_file_age(self):
        files = [
            ("NYSE List", NYSE_LIST_FILE),
            ("NASDAQ List", NASDAQ_LIST_FILE),
            ("Favorites", FAVORITES_FILE)
        ]
        messages = []
        for name, file_path in files:
            if os.path.exists(file_path):
                age_days = (time.time() - os.path.getmtime(file_path)) / (24 * 3600)
                messages.append(f"{name}: {age_days:.1f} days old")
            else:
                messages.append(f"{name}: File not found")
        messagebox.showinfo("File Ages", "\n".join(messages))

    def update_ticker_files(self):
        if messagebox.askyesno("Confirm Update", "This will download the latest NYSE and NASDAQ ticker files from FTP. Continue?"):
            def on_complete():
                self.root.after(0, lambda: messagebox.showinfo("Update Complete", "Ticker files have been updated from FTP."))
            def on_error():
                self.root.after(0, lambda: messagebox.showerror("Update Failed", "Failed to download ticker files. Check logs for details."))
            async def update_task():
                try:
                    await self.ticker_manager.download_ticker_files()
                    on_complete()
                except Exception:
                    on_error()
            self.task_queue.put(update_task())
            messagebox.showinfo("Update Started", "Ticker files update has been queued.")

    def clear_cache(self):
        if messagebox.askyesno("Confirm Clear Cache", "This will clear all cached data. Continue?"):
            clear_in_memory_caches()
            conn, cursor = get_stocks_connection()
            try:
                cursor.execute("DELETE FROM stocks")
                cursor.execute("DELETE FROM screening_progress")
                conn.commit()
                messagebox.showinfo("Cache Cleared", "All cached data has been cleared.")
            except sqlite3.Error as e:
                messagebox.showerror("Error", f"Failed to clear database cache: {str(e)}")
            finally:
                conn.close()

    def show_help(self):
        help_text = """
        Graham Screening App Help:

        - **Analyze Stocks**: Enter tickers (e.g., AOS, AAPL) and click "Analyze Stocks" to evaluate them against Graham's criteria.
        - **Favorites**: Save and load ticker lists using the "Save Favorite" and "Favorite Lists" dropdown.
        - **Screening**: Check NYSE or NASDAQ screening boxes and run to find qualifying stocks.
        - **Export**: Export qualifying stocks to Excel after screening.
        - **Progress**: Monitor screening progress and cache usage in the middle panel.
        - **Tabs**: Select a stock in the treeview to see historical data and metrics below.

        For detailed instructions, see the exported Excel's "Start Here" sheet.
        """
        messagebox.showinfo("Help", help_text)

    def filter_tree(self, event):
        """Filter the treeview based on the search term entered in the search entry widget."""
        search_term = self.search_entry.get().strip().upper()
        
        # Clear the current treeview contents
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Connect to the database
        conn, cursor = get_stocks_connection()
        try:
            # If there's a search term, filter stocks by ticker; otherwise, show all stocks
            if search_term:
                cursor.execute(
                    "SELECT ticker, company_name, common_score FROM stocks WHERE ticker LIKE ?",
                    (f"%{search_term}%",)
                )
            else:
                cursor.execute("SELECT ticker, company_name, common_score FROM stocks")
            
            results = cursor.fetchall()
            
            # Populate the treeview with filtered results
            for ticker, company_name, common_score in results:
                self.tree.insert(
                    "", "end", text=ticker,
                    values=(company_name, "Unknown", f"{common_score}/6", "", "", "", "")
                )
        except sqlite3.Error as e:
            analyze_logger.error(f"Database error in filter_tree: {str(e)}")
        finally:
            conn.close()

    def safe_insert(self, text_widget, content):
        self.root.after(0, lambda: text_widget.insert(tk.END, content))

if __name__ == "__main__":
    root = tk.Tk()
    app = GrahamScreeningApp(root)
    root.mainloop()