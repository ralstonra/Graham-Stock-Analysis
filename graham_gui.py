# graham_gui.py (updated to use graham_logger)
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
import math
import yfinance as yf
import queue
import shutil
import requests
import ftplib
import re
import subprocess
import platform
import aiohttp

from decouple import config
from datetime import datetime, timedelta

from config import (
    BASE_DIR, FMP_API_KEYS, FAVORITES_FILE, CACHE_EXPIRY,
    graham_logger, USER_DATA_DIR, FRED_API_KEY
)
from graham_utils import (paid_rate_limiter, free_rate_limiter)
from export_utils import export_qualifying_stocks
from graham_data import (
    screen_exchange_graham_stocks, fetch_batch_data, fetch_stock_data,
    cache_manager, fetch_with_multiple_keys_async, NYSE_LIST_FILE,
    NASDAQ_LIST_FILE, TickerManager, get_file_hash, calculate_graham_value,
    calculate_graham_score_8, calculate_common_criteria, clear_in_memory_caches,
    save_qualifying_stocks_to_favorites, get_stock_data_from_db, calculate_cagr, get_aaa_yield, get_bank_metrics,
    get_tangible_book_value_per_share
)

FAVORITES_LOCK = threading.Lock()
DATA_DIR = BASE_DIR

class GrahamScreeningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Analysis (Graham Defensive)")
        self.root.geometry("1200x900")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)  # Handle window close
        self.task_queue = queue.Queue(maxsize=200)

        # Variables
        self.nyse_screen_var = tk.BooleanVar(value=False)
        self.nasdaq_screen_var = tk.BooleanVar(value=False)
        self.financial_screen_var = tk.BooleanVar(value=False)
        self.cancel_event = threading.Event()
        self.margin_of_safety_var = tk.DoubleVar(value=33.0)
        self.expected_return_var = tk.DoubleVar(value=0.0)
        self.ticker_manager = TickerManager(NYSE_LIST_FILE, NASDAQ_LIST_FILE, USER_DATA_DIR)
        self.screening_active = False
        self.analysis_lock = threading.Lock()
        self.ticker_cache = {}
        self.ticker_cache_lock = threading.Lock()
        self.ticker_init_complete = threading.Event()  # New event for TickerManager initialization
        self.shutdown_event = threading.Event()

        try:
            # Asyncio Thread Setup
            self.asyncio_loop_started = threading.Event()
            self.asyncio_thread = threading.Thread(target=self.run_asyncio_loop, args=(self.task_queue,), daemon=True)
            self.asyncio_thread.start()
            # Wait for the asyncio loop to start
            if not self.asyncio_loop_started.wait(timeout=2.0):
                raise RuntimeError("Asyncio loop failed to start within 2 seconds")
            # Initialize TickerManager and wait for completion
            self.task_queue.put(self.initialize_ticker_manager())
            if not self.ticker_init_complete.wait(timeout=5.0):
                raise RuntimeError("TickerManager initialization timed out")

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
            style.configure("Invalid.TEntry", fieldbackground="pink")  # For invalid ticker validation

            self.create_widgets()
            self.root.update()  # Force GUI update after widget creation

            paid_rate_limiter.on_sleep = self.on_rate_limit_sleep

            # Tooltips (ensure widgets are created before setting tooltips)
            if hasattr(self, 'entry'):
                self.create_tooltip(self.entry, "Enter comma-separated tickers, e.g., AOS, AAPL")
            if hasattr(self, 'search_entry'):
                self.create_tooltip(self.search_entry, "Search by ticker or company name")
            if hasattr(self, 'favorite_menu'):
                self.create_tooltip(self.favorite_menu, "Select a saved favorites list")
            if hasattr(self, 'margin_of_safety_label'):
                self.create_tooltip(self.margin_of_safety_label, "Discount to intrinsic value for safety margin")
            if hasattr(self, 'expected_return_label'):
                self.create_tooltip(self.expected_return_label, "Premium above intrinsic value for sell target")
            if hasattr(self, 'analyze_button'):
                self.create_tooltip(self.analyze_button, "Analyze the entered tickers")
            if hasattr(self, 'progress_bar'):
                self.create_tooltip(self.progress_bar, "Shows progress of screening or analysis")

            self.check_for_updates()
        except Exception as e:
            graham_logger.error(f"Error initializing GUI: {str(e)}", exc_info=True)
            messagebox.showerror("Initialization Error", f"Failed to initialize the GUI: {str(e)}")
            self.root.quit()

    async def initialize_ticker_manager(self):
        """Initialize TickerManager asynchronously and signal completion."""
        try:
            await self.ticker_manager.initialize()
            self.ticker_init_complete.set()
        except Exception as e:
            graham_logger.error(f"Error initializing TickerManager: {str(e)}", exc_info=True)
            self.ticker_init_complete.set()  # Set anyway to avoid deadlock
        # ADR identification using local otherlisted.txt and nasdaqlisted.txt
        self.adr_tickers = set()
        try:
            import csv
            # Define ADR keywords for Security Name
            adr_keywords = [
                'American Depositary Shares',
                'American Depositary Receipt',
                'ADS',
                'ADR',
                'each representing'
            ]
            # Parse otherlisted.txt (NYSE)
            nyse_adrs = set()
            with open(NYSE_LIST_FILE, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='|')
                for row in reader:
                    symbol = row.get('ACT Symbol', '').strip().upper()
                    security_name = row.get('Security Name', '').strip()
                    exchange = row.get('Exchange', '').strip().upper()
                    if symbol and exchange == 'N' and any(keyword in security_name for keyword in adr_keywords):
                        nyse_adrs.add(symbol)
            self.adr_tickers.update(nyse_adrs)
            graham_logger.info(f"Loaded {len(nyse_adrs)} NYSE ADR tickers from otherlisted.txt: {sorted(nyse_adrs)}")
            # Parse nasdaqlisted.txt (NASDAQ)
            nasdaq_adrs = set()
            with open(NASDAQ_LIST_FILE, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='|')
                for row in reader:
                    symbol = row.get('Symbol', '').strip().upper()
                    security_name = row.get('Security Name', '').strip()
                    if symbol and any(keyword in security_name for keyword in adr_keywords):
                        nasdaq_adrs.add(symbol)
            self.adr_tickers.update(nasdaq_adrs)
            graham_logger.info(f"Loaded {len(nasdaq_adrs)} NASDAQ ADR tickers from nasdaqlisted.txt: {sorted(nasdaq_adrs)}")
            graham_logger.info(f"Total ADR tickers (NYSE + NASDAQ): {len(self.adr_tickers)}")
        except Exception as e:
            graham_logger.warning(f"Failed to load ADR tickers from local files: {e}. Using empty set.")
            self.adr_tickers = set()  # Fallback

    def on_closing(self):
        """Handle window close event to stop asyncio thread."""
        self.cancel_event.set()  # Signal cancel to ongoing tasks
        self.shutdown_event.set()  # Signal shutdown
        self.task_queue.put_nowait(None)  # Signal stop (non-blocking)
        self.asyncio_thread.join(timeout=5)  # Longer timeout for cleanup
        if self.asyncio_thread.is_alive():
            graham_logger.warning("Asyncio thread did not stop cleanly.")
        self.root.destroy()

    def throttled_update_progress(self, progress, tickers=None, passed_tickers=0, eta=None):
        if hasattr(self, '_progress_after_id'):
            self.root.after_cancel(self._progress_after_id)
        self._progress_after_id = self.root.after(500, lambda: self._do_update_progress(progress, tickers, passed_tickers, eta))

    def _do_update_progress(self, progress, tickers, passed_tickers, eta):
        self.progress_var.set(progress)
        if tickers is not None and isinstance(tickers, (list, tuple)):
            total_tickers = len(tickers)
            processed = int(progress / 100 * total_tickers)
            eta_text = f", ETA: {str(timedelta(seconds=int(eta)))[:-3]}" if eta is not None else ""
            self.progress_label.config(text=f"Progress: {progress:.1f}% ({processed}/{total_tickers} stocks processed, {passed_tickers} passed{eta_text})")
        else:
            eta_text = f", ETA: {str(timedelta(seconds=int(eta)))[:-3]}" if eta is not None else ""
            self.progress_label.config(text=f"Progress: {progress:.1f}% (Screening, {passed_tickers} passed{eta_text})")
        self.root.update_idletasks()

    def on_rate_limit_sleep(self, sleep_time):
        message = f"Rate limit reached, using {'free key' if sleep_time > 60 else 'paid key'}, pausing for {sleep_time / 60:.1f} minutes"
        self.root.after(0, lambda: self.rate_limit_label.config(text=message))

    def run_asyncio_loop(self, task_queue):
        """Run the asyncio event loop in a separate thread."""
        self.asyncio_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.asyncio_loop)
        self.asyncio_loop_started.set()  # Signal that the loop has started
        while not self.shutdown_event.is_set():
            try:
                coro = task_queue.get(timeout=1)  # Timeout to check for shutdown
                if coro is None:
                    break
                self.asyncio_loop.run_until_complete(coro)
            except queue.Empty:
                continue  # Loop to check shutdown_event
            except Exception as e:
                graham_logger.error(f"Error in asyncio task: {str(e)}", exc_info=True)
        self.asyncio_loop.close()

    def show_tooltip(self, event):
        """Show tooltip for foreign stocks on hover."""
        item = self.tree.identify_row(event.y)
        if not item:
            self.hide_tooltip(event)
            return
        tags = self.tree.item(item, 'tags')
        if 'foreign' in tags:
            x, y = event.x_root + 10, event.y_root + 10
            if self.tooltip:
                self.tooltip.destroy()
            self.tooltip = tk.Toplevel(self.root)
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.wm_geometry(f"+{x}+{y}")
            label = tk.Label(
                self.tooltip,
                text="Foreign/ADR company: Consider currency, regulatory, tax factors, and verify detection (based on API and keywords).",
                background="lightyellow",
                relief="solid",
                borderwidth=1,
                font=("Helvetica", 8)
            )
            label.pack()

    def hide_tooltip(self, event):
        """Hide tooltip when mouse leaves."""
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

    def create_tooltip(self, widget, text):
        """Create a tooltip for a Tkinter widget."""
        tooltip = None
        def show_tooltip(event):
            nonlocal tooltip
            if tooltip:
                tooltip.destroy()
            tooltip = tk.Toplevel(widget)
            tooltip.wm_overrideredirect(True)  # Remove window decorations
            tooltip.geometry(f"+{event.x_root}+{event.y_root+10}")
            label = tk.Label(tooltip, text=text, background="#ffffe0", relief="solid", borderwidth=1)
            label.pack()
        def hide_tooltip(event):
            nonlocal tooltip
            if tooltip:
                tooltip.destroy()
                tooltip = None
        widget.bind("<Enter>", show_tooltip)
        widget.bind("<Leave>", hide_tooltip)

    def create_widgets(self):
        """Create all GUI widgets with error handling."""
        try:
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
                try:
                    graham_logger.info("Save Favorite button clicked")
                    self.task_queue.put(self.ticker_manager.initialize())
                    time.sleep(1)  # Wait for initialization
                    tickers_input = self.entry.get()
                    if not self.validate_tickers():
                        messagebox.showwarning("Invalid Tickers", "No valid tickers entered.")
                        return
                    name = simpledialog.askstring("Save Favorite", "Enter list name:")
                    if not name or not name.strip():
                        messagebox.showwarning("Invalid Name", "Please enter a valid list name.")
                        return
                    if name and tickers_input.strip():
                        tickers = self.parse_tickers(tickers_input)
                        valid_tickers = [t for t in tickers if self.ticker_manager.is_valid_ticker(t)]
                        if not valid_tickers:
                            messagebox.showwarning("Invalid Tickers", "No valid NYSE or NASDAQ tickers.")
                            return
                        self.favorites[name] = valid_tickers
                        self.save_favorites()
                        self.refresh_favorites_dropdown(name)
                except Exception as e:
                    graham_logger.error(f"Error in save_favorite: {str(e)}", exc_info=True)
                    messagebox.showerror("Error", f"Failed to save favorite: {str(e)}")
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
            # Add minimum criteria selector
            self.min_criteria_var = tk.IntVar(value=6)
            ttk.Label(self.right_frame, text="Min Graham Criteria:").grid(row=9, column=0, pady=2, sticky="w")
            self.min_criteria_menu = ttk.Combobox(self.right_frame, textvariable=self.min_criteria_var, values=[4, 5, 6], state="readonly")
            self.min_criteria_menu.grid(row=10, column=0, pady=2, sticky="ew")
            # Add sector filter selector below min criteria
            self.sector_filter_var = tk.StringVar(value="All")
            ttk.Label(self.right_frame, text="Sector Filter:").grid(row=11, column=0, pady=2, sticky="w")
            self.sector_menu = ttk.Combobox(
                self.right_frame,
                textvariable=self.sector_filter_var,
                values=[
                    "All", "Energy", "Materials", "Industrials", "Consumer Discretionary",
                    "Consumer Staples", "Health Care", "Financials", "Information Technology",
                    "Communication Services", "Utilities", "Real Estate", "Unknown"
                ],
                state="readonly"
            )
            self.sector_menu.grid(row=12, column=0, pady=2, sticky="ew")
            ttk.Checkbutton(self.right_frame, text="Screen Financials Separately", variable=self.financial_screen_var).grid(row=13, column=0, pady=2, padx=2, sticky="w")
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
            # Treeview column configurations
            self.tree.tag_configure('foreign', foreground='purple')
            self.tree.tag_configure('highlight', background='lightgreen')
            # Bind tooltip to foreign-tagged rows
            self.tree.bind('<Motion>', self.show_tooltip)
            self.tree.bind('<Leave>', self.hide_tooltip)
            self.tooltip = None
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
        except Exception as e:
            graham_logger.error(f"Error creating widgets: {str(e)}", exc_info=True)
            messagebox.showerror("Widget Creation Error", f"Failed to create GUI widgets: {str(e)}")

    def safe_update_tree(self, ticker, values, tags=None):
        def update():
            try:
                self.tree.insert("", "end", text=ticker, values=values, tags=tags)
            except tk.TclError as e:
                if "invalid command name" not in str(e):  # Ignore if widget destroyed
                    graham_logger.warning(f"Tk error during tree update for {ticker}: {str(e)}")
        self.root.after(0, update)

    def validate_tickers(self):
        tickers = self.parse_tickers(self.entry.get())
        valid = all(self.ticker_manager.is_valid_ticker(t) for t in tickers)
        self.entry.config(style="Invalid.TEntry" if not valid else "TEntry")
        return valid

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
                    graham_logger.error(f"Corrupted favorites file {FAVORITES_FILE}")
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
                graham_logger.error(f"Failed to save favorites: {str(e)}")
                messagebox.showerror("Error", f"Failed to save favorites: {str(e)}")

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
        graham_logger.debug(f"Refreshed favorites dropdown with keys: {list(self.favorites.keys())}")
        self.favorite_menu['values'] = list(self.favorites.keys())
        if selected_list and selected_list in self.favorites:
            self.favorite_var.set(selected_list)
            graham_logger.debug(f"Set dropdown to: {selected_list}")

    def cancel_screening(self):
        if messagebox.askyesno("Confirm Cancel", "Are you sure you want to cancel the screening?"):
            self.cancel_event.set()
            self.status_label.config(text="Cancelling screening...")

    def run_screening(self, exchange: str, screen_func):
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
        conn, cursor = cache_manager.get_stocks_connection()
        try:
            cursor.execute("SELECT MAX(timestamp) FROM stocks")
            max_timestamp = cursor.fetchone()[0]
            if max_timestamp and time.time() - max_timestamp > 365 * 24 * 60 * 60:
                messagebox.showwarning("Old Data", "The cached data is over a year old. Consider refreshing the data.")
        except sqlite3.Error as e:
            graham_logger.error(f"Error checking data age: {str(e)}")
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
        min_criteria = self.min_criteria_var.get()
        sector_filter = self.sector_filter_var.get() if self.sector_filter_var.get() != "All" else None
        separate_financials = self.financial_screen_var.get()
        async def screening_task():
            try:
                await self.ticker_manager.initialize()
                qualifying_stocks, common_scores, exchanges, error_tickers, financial_qualifying_stocks = await screen_func(
                    exchange=exchange,
                    batch_size=50,
                    cancel_event=self.cancel_event,
                    root=self.root,
                    update_progress_animated=self.throttled_update_progress,
                    refresh_favorites_dropdown=self.refresh_favorites_dropdown,
                    ticker_manager=self.ticker_manager,
                    update_rate_limit=self.update_rate_limit,
                    min_criteria=min_criteria,
                    sector_filter=sector_filter,
                    separate_financials=separate_financials,
                    adr_tickers=self.adr_tickers
                )
                await asyncio.sleep(0)  # Yield to allow Tkinter event processing
                graham_logger.info(f"Qualifying stocks for {exchange}: {qualifying_stocks}")
                if not self.cancel_event.is_set():
                    if qualifying_stocks:
                        list_name = await save_qualifying_stocks_to_favorites(qualifying_stocks, exchange)
                        if list_name:
                            self.root.after(0, lambda: self.refresh_favorites_dropdown(list_name))
                        else:
                            graham_logger.error(f"Failed to save qualifying stocks for {exchange}")
                    else:
                        graham_logger.info(f"No qualifying stocks found for {exchange}")
                    conn, cursor = cache_manager.get_stocks_connection()
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
                        if separate_financials:
                            summary += f"\nFinancial qualifiers: {len(financial_qualifying_stocks)}"
                        if error_tickers:
                            summary += f"\nError tickers: {', '.join(error_tickers)}"
                        self.root.after(0, lambda: messagebox.showinfo("Screening Complete", summary))
                    except sqlite3.Error as e:
                        graham_logger.error(f"Database error in screening summary: {str(e)}")
                    finally:
                        conn.close()
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Screening failed: {str(e)}"))
                graham_logger.error(f"Screening task failed for {exchange}: {str(e)}", exc_info=True)
            finally:
                self.root.after(0, lambda: self.progress_label.config(text=f"Progress: 100% (Screening Complete - {exchange})"))
                self.root.after(0, lambda: self.status_label.config(text=""))
                setattr(self, f"{exchange.lower()}_screen_var", tk.BooleanVar(value=False))
                self.screening_active = False
        self.task_queue.put(screening_task())

    def run_nyse_screening(self):
        self.run_screening("NYSE", screen_exchange_graham_stocks)

    def run_nasdaq_screening(self):
        self.run_screening("NASDAQ", screen_exchange_graham_stocks)

    async def fetch_company_name(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            return stock.info.get('longName', 'Unknown')
        except Exception as e:
            graham_logger.error(f"Error fetching company name for {ticker}: {str(e)}")
            return 'Unknown'

    def format_float(self, value, precision=2):
        if isinstance(value, str):
            return value  # Already "N/A" or similar
        if pd.isna(value) or value is None or not isinstance(value, (int, float)) or math.isinf(value):
            graham_logger.debug(f"format_float: Returning N/A for value {value} (type: {type(value)})")
            return "N/A"
        formatted = f"{value:.{precision}f}"
        graham_logger.debug(f"format_float: Formatted {value} to {formatted}")
        return formatted

    async def fetch_cached_data(self, ticker, exchange="Unknown"):
        nyse_file_hash = get_file_hash(NYSE_LIST_FILE)
        nasdaq_file_hash = get_file_hash(NASDAQ_LIST_FILE)
        for attempt in range(3):  # Retry up to 3 times
            try:
                conn, cursor = cache_manager.get_stocks_connection()
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
                            graham_logger.info(f"Database cache hit for {ticker}")
                            stock = yf.Ticker(ticker)
                            info = stock.info
                            price = info.get('regularMarketPrice', info.get('previousClose', None))
                            if price is None:
                                graham_logger.error(f"No price data for {ticker} from YFinance")
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
                            key_metrics_data = json.loads(stock_dict.get('key_metrics_data', '[]'))
                            graham_score = calculate_graham_score_8(ticker, price, None, None, debt_to_equity, eps_list, div_list, {}, balance_data, key_metrics_data, stock_dict['available_data_years'], stock_dict['latest_revenue'], sector)
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
                                "latest_revenue": stock_dict['latest_revenue'],
                                "available_data_years": stock_dict['available_data_years'],
                                "eps_ttm": eps_ttm,
                                "book_value_per_share": book_value_per_share,
                                "debt_to_equity": debt_to_equity,
                                "eps_cagr": stock_dict.get('eps_cagr', 0.0)
                            }
                            with self.ticker_cache_lock:
                                self.ticker_cache[ticker] = result
                            graham_logger.debug(f"Cached data retrieved for {ticker}: Score={graham_score}/8 with {stock_dict['available_data_years']} years")
                            return result
                graham_logger.info(f"Cache miss for {ticker}: Data stale or missing")
                return None
            except sqlite3.Error as e:
                graham_logger.error(f"Database error in fetch_cached_data for {ticker} (attempt {attempt + 1}/3): {str(e)}")
                if attempt < 2:
                    time.sleep(1)  # Wait before retry
                    continue
                return None
            finally:
                conn.close()
        return None

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
        graham_logger.info(f"Starting analysis for tickers: {tickers}")
        # Validate tickers using TickerManager
        await self.ticker_manager.initialize()
        valid_tickers = [t for t in tickers if self.ticker_manager.is_valid_ticker(t)]
        invalid_tickers = set(tickers) - set(valid_tickers)
        if invalid_tickers:
            graham_logger.info(f"Invalid tickers (not in NYSE/NASDAQ lists): {invalid_tickers}")
            self.root.after(0, lambda: messagebox.showwarning("Invalid Tickers", f"The following tickers are not listed on NYSE or NASDAQ: {', '.join(invalid_tickers)}"))
        if not valid_tickers:
            graham_logger.warning("No valid tickers provided for analysis")
            self.root.after(0, lambda: messagebox.showwarning("No Tickers", "No valid tickers to analyze."))
            return
        # Check invalid tickers files for previously failed tickers
        nyse_invalid_file = os.path.join(USER_DATA_DIR, "NYSE Invalid Tickers.txt")
        nasdaq_invalid_file = os.path.join(USER_DATA_DIR, "NASDAQ Invalid Tickers.txt")
        failed_tickers = set()
        for invalid_file in [nyse_invalid_file, nasdaq_invalid_file]:
            if os.path.exists(invalid_file):
                with open(invalid_file, 'r') as f:
                    failed_tickers.update(f.read().splitlines())
        analysis_tickers = [t for t in valid_tickers if t not in failed_tickers]
        if len(analysis_tickers) < len(valid_tickers):
            excluded = set(valid_tickers) - set(analysis_tickers)
            graham_logger.info(f"Excluded {len(excluded)} previously failed tickers: {excluded}")
            self.root.after(0, lambda: messagebox.showinfo("Excluded Tickers", f"Excluded {len(excluded)} tickers due to previous failures: {', '.join(excluded)}"))
        if not analysis_tickers:
            graham_logger.warning("No valid tickers available for analysis after filtering")
            self.root.after(0, lambda: messagebox.showwarning("No Tickers", "No valid tickers to analyze after filtering."))
            return
        # Check cache age
        conn, cursor = cache_manager.get_stocks_connection()
        try:
            cursor.execute("SELECT MAX(timestamp) FROM stocks")
            max_timestamp = cursor.fetchone()[0]
            if max_timestamp and time.time() - max_timestamp > 365 * 24 * 60 * 60:
                graham_logger.warning("Cached data is over a year old")
                self.root.after(0, lambda: messagebox.showwarning("Old Data", "The cached data is over a year old. Consider refreshing the data."))
        except sqlite3.Error as e:
            graham_logger.error(f"Error checking data age: {str(e)}")
        finally:
            conn.close()
        # Initialize progress UI
        self.progress_var.set(0)
        self.cache_var.set(0)
        self.progress_label.config(text=f"Progress: 0% (0/{len(analysis_tickers)} stocks processed, 0 passed)")
        self.cache_label.config(text="Cache Usage: 0% (0 cached, 0 fresh)")
        self.rate_limit_label.config(text="")
        self.status_label.config(text="Analyzing stocks...")
        self.root.update()
        graham_logger.info(f"Fetching data for {len(analysis_tickers)} tickers with margin_of_safety={self.margin_of_safety_var.get()/100}, expected_return={self.expected_return_var.get()/100}")
        try:
            results, error_tickers, cache_hits = await fetch_batch_data(
                analysis_tickers,
                screening_mode=False,
                expected_return=self.expected_return_var.get() / 100,
                margin_of_safety=self.margin_of_safety_var.get() / 100,
                ticker_manager=self.ticker_manager,
                update_rate_limit=self.update_rate_limit,
                adr_tickers=self.adr_tickers
            )
        except aiohttp.ClientError as e:
                graham_logger.error(f"Network error during analysis: {str(e)}")
                self.root.after(0, lambda: messagebox.showerror("Network Error", "Failed to fetch data due to network issues."))
                return
        except ValueError as e:
            graham_logger.error(f"Data error during analysis: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("Data Error", str(e)))
            return
        except Exception as e:
            graham_logger.error(f"Unexpected error in analysis: {str(e)}", exc_info=True)
            self.root.after(0, lambda: messagebox.showerror("Error", f"Unexpected error: {str(e)}"))
        finally:
            self.root.after(0, lambda: self.progress_var.set(0))
            self.root.after(0, lambda: self.progress_label.config(text="Progress: 0% (0/0 stocks processed, 0 passed)"))
            self.root.after(0, lambda: self.status_label.config(text=""))
            self.root.after(0, lambda: self.root.update_idletasks())
        valid_results = [r for r in results if 'error' not in r]
        passed_tickers = sum(1 for r in valid_results if r.get('graham_score', 0) >= 5 and r.get('available_data_years', 0) >= 10)
        # Collect tickers with insufficient data years
        insufficient_data_tickers = [
            (r['ticker'], r.get('available_data_years', 0))
            for r in valid_results if r.get('available_data_years', 0) < 10
        ]
        if insufficient_data_tickers:
            warning_message = "The following tickers have fewer than 10 years of data, so results may be incomplete:\n" + "\n".join(
                f"{ticker}: {years} years" for ticker, years in insufficient_data_tickers
            )
            graham_logger.warning(f"Tickers with insufficient data: {insufficient_data_tickers}")
            self.root.after(0, lambda: messagebox.showwarning("Insufficient Data", warning_message))
        graham_logger.info(f"Analysis complete: {len(valid_results)} valid results, {len(error_tickers)} errors, {cache_hits} cache hits")
        if error_tickers:
            graham_logger.warning(f"Error tickers: {error_tickers}")
            error_summary = "\n".join(error_tickers)
            self.root.after(0, lambda: messagebox.showwarning("Analysis Errors", f"Errors occurred for the following tickers:\n{error_summary}"))
        graham_logger.debug("Populating treeview with analysis results")
        for item in self.tree.get_children():
            self.tree.delete(item)
        for result in valid_results:
            years_used = result.get('available_data_years', 0)
            warning = f" (based on {years_used} years)" if years_used < 10 else ""
            sector_display = result['sector']
            tags = []
            if result.get('is_foreign', False):
                sector_display += " (Foreign)"
                tags.append('foreign')
            price = result.get('price')
            buy_price = result.get('buy_price')
            if price is not None and buy_price is not None and not pd.isna(price) and not pd.isna(buy_price) and price <= buy_price:
                tags.append('highlight')
            graham_logger.debug(f"Processing result for {result['ticker']}: Graham Score={result['graham_score']}, Years swiss={years_used}, Is Foreign={result.get('is_foreign', False)}, Price={result.get('price')}, Intrinsic={result.get('intrinsic_value')}")
            price_str = f"${self.format_float(result.get('price'))}"
            intrinsic_str = f"${self.format_float(result.get('intrinsic_value'))}"
            buy_str = f"${self.format_float(result.get('buy_price'))}"
            sell_str = f"${self.format_float(result.get('sell_price'))}"
            graham_logger.debug(f"Treeview strings for {result['ticker']}: Price={price_str}, Intrinsic={intrinsic_str}, Buy={buy_str}, Sell={sell_str}")
            self.safe_update_tree(
                result['ticker'],
                (
                    result['company_name'],
                    sector_display,
                    f"{result['graham_score']}/8{warning}",
                    price_str,
                    intrinsic_str,
                    buy_str,
                    sell_str
                ),
                tags
            )
            with self.ticker_cache_lock:
                self.ticker_cache[result['ticker']] = result
        graham_logger.info(f"Cache usage: {cache_hits} hits out of {len(analysis_tickers)} total tickers")
        self.root.after(0, lambda: self.update_cache_usage(cache_hits, len(analysis_tickers)))
        self.root.after(0, lambda: self.throttled_update_progress(100, analysis_tickers, passed_tickers))
        self.root.after(0, lambda: self.status_label.config(text=""))
        graham_logger.info("Analysis fully completed and UI updated")

    def analyze_multiple_stocks(self, tickers_input=None):
        graham_logger.info("Analyze Stocks button clicked")
        if self.analysis_lock.acquire(timeout=5):
            try:
                if tickers_input is None:
                    tickers_input = self.entry.get()
                tickers = self.parse_tickers(tickers_input)
                graham_logger.info(f"Parsed tickers from input '{tickers_input}': {tickers}")
                if not tickers:
                    graham_logger.warning("No tickers parsed from input")
                self.task_queue.put(self.analyze_multiple_stocks_async(tickers))
            finally:
                self.analysis_lock.release()
        else:
            graham_logger.error("Failed to acquire analysis lock within 5 seconds")
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
                update_rate_limit=self.update_rate_limit,
                adr_tickers=self.adr_tickers
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
            self.safe_update_tree(
                ticker,
                (
                    company_name,
                    exchange,
                    f"{common_score}/6",
                    f"${self.format_float(result.get('price', 0))}",
                    f"${self.format_float(result.get('intrinsic_value', float('nan')))}",
                    f"${self.format_float(result.get('buy_price', float('nan')))}",
                    f"${self.format_float(result.get('sell_price', float('nan')))}"
                ),
                tags
            )
        self.sort_tree(0)

    def update_tabs(self, event):
        selected = self.tree.selection()
        if not selected:
            return
        ticker = self.tree.item(selected[0], "text")
        with self.ticker_cache_lock:
            result = self.ticker_cache.get(ticker, None)
        if not result:
            graham_logger.warning(f"No cached data found for {ticker}, fetching fresh data")
            async def fetch_fresh():
                fresh_result = await fetch_stock_data(
                    ticker,
                    expected_return=self.expected_return_var.get() / 100,
                    margin_of_safety=self.margin_of_safety_var.get() / 100,
                    ticker_manager=self.ticker_manager,
                    adr_tickers=self.adr_tickers
                )
                with self.ticker_cache_lock:
                    self.ticker_cache[ticker] = fresh_result
                return fresh_result
            result = asyncio.run(fetch_fresh())
        graham_logger.info(f"Updating tabs for {ticker} with data keys: {list(result.keys())}")
        self.root.after(0, lambda: self.historical_text.delete('1.0', tk.END))
        self.root.after(0, lambda: self.metrics_text.delete('1.0', tk.END))
        years = result.get('years', [])
        roe_list = result.get('roe_list', [])
        rotc_list = result.get('rotc_list', [])
        eps_list = result.get('eps_list', [])
        div_list = result.get('div_list', [])
        graham_logger.debug(f"Historical data for {ticker}: years={len(years)}, roe={len(roe_list)}, rotc={len(rotc_list)}, eps={len(eps_list)}, dividend={len(div_list)}")
        graham_logger.debug(f"Years: {years}")
        graham_logger.debug(f"ROE: {roe_list}")
        graham_logger.debug(f"ROTC: {rotc_list}")
        graham_logger.debug(f"EPS: {eps_list}")
        graham_logger.debug(f"Dividend: {div_list}")
        if not years:
            self.safe_insert(self.historical_text, f"No historical data available for {ticker}\n")
        elif len(years) == len(roe_list) == len(rotc_list) == len(eps_list) == len(div_list):
            header = f"{len(years)}-Year Historical Data for {ticker}:\nYear\tROE (%)\tROTC (%)\tEPS ($)\tDividend ($)\n"
            self.safe_insert(self.historical_text, header)
            for j in range(len(years)):
                line = f"{years[j]}\t{roe_list[j]:.2f}\t{rotc_list[j]:.2f}\t{eps_list[j]:.2f}\t{div_list[j]:.2f}\n"
                self.safe_insert(self.historical_text, line)
        else:
            message = f"Incomplete Historical Data for {ticker} (Years: {len(years)}, ROE: {len(roe_list)}, ROTC: {len(rotc_list)}, EPS: {len(eps_list)}, Dividend: {len(div_list)})\n"
            self.safe_insert(self.historical_text, message)
        self.safe_insert(self.metrics_text, f"Graham Criteria Results for {ticker} (Score: {result['graham_score']}/8 with {result['available_data_years']} years):\n")
        self.safe_insert(self.metrics_text, f"Sector: {result['sector']}\n\n")
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
            graham_logger.info(f"Metrics updated for {ticker} using cached data")
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
        """Display NYSE qualifying stocks in the treeview with company names and scores out of 6."""
        graham_logger.info("Starting display_nyse_qualifying_stocks")
        conn, cursor = cache_manager.get_stocks_connection()
        try:
            # Query to fetch qualifying stocks with a common_score >= min_criteria
            cursor.execute("""
                SELECT g.ticker, s.company_name, g.sector, g.common_score, s.is_foreign
                FROM graham_qualifiers g
                LEFT JOIN stocks s ON g.ticker = s.ticker
                WHERE g.exchange = 'NYSE' AND g.common_score >= ?
                ORDER BY g.common_score DESC, g.ticker ASC
            """, (self.min_criteria_var.get(),))
            results = cursor.fetchall()
            graham_logger.debug(f"Retrieved {len(results)} NYSE qualifying stocks from database")

            # Clear treeview
            for item in self.tree.get_children():
                self.tree.delete(item)
            graham_logger.debug("Treeview cleared")

            if not results:
                graham_logger.info(f"No NYSE qualifiers found for min_criteria={self.min_criteria_var.get()}")
                self.safe_update_tree(
                    "No qualifying stocks found",
                    ("", "", "", "", "", "", "")
                )
                self.root.update_idletasks()
                return

            # Populate treeview
            for ticker, company_name, sector, common_score, is_foreign in results:
                display_name = company_name if company_name else ticker
                sector_display = sector
                tags = []
                if is_foreign:
                    sector_display += " (Foreign)"
                    tags.append('foreign')
                graham_logger.debug(f"Adding to treeview: {ticker}, Name: {display_name}, Sector: {sector_display}, Score: {common_score}/6, Tags: {tags}")
                self.safe_update_tree(
                    ticker,
                    (
                        display_name,
                        sector_display,
                        f"{common_score}/6",
                        "", "", "", ""  # Price, intrinsic value, buy/sell prices are empty
                    ),
                    tags
                )

            # Force treeview sort and UI refresh
            self.sort_tree(0)
            self.root.update_idletasks()
            self.root.update()
            graham_logger.info(f"Displayed {len(results)} NYSE qualifying stocks in treeview")
        except sqlite3.Error as e:
            error_msg = f"Database error displaying NYSE qualifiers: {str(e)}"
            graham_logger.error(error_msg)
            messagebox.showerror("Database Error", error_msg)
        finally:
            conn.close()

    def display_nasdaq_qualifying_stocks(self):
        """Display NASDAQ qualifying stocks in the treeview with company names and scores out of 6."""
        graham_logger.info("Starting display_nasdaq_qualifying_stocks")
        conn, cursor = cache_manager.get_stocks_connection()
        try:
            # Query to fetch qualifying stocks with a common_score >= min_criteria
            cursor.execute("""
                SELECT g.ticker, s.company_name, g.sector, g.common_score, s.is_foreign
                FROM graham_qualifiers g
                LEFT JOIN stocks s ON g.ticker = s.ticker
                WHERE g.exchange = 'NASDAQ' AND g.common_score >= ?
                ORDER BY g.common_score DESC, g.ticker ASC
            """, (self.min_criteria_var.get(),))
            results = cursor.fetchall()
            graham_logger.debug(f"Retrieved {len(results)} NASDAQ qualifying stocks from database")

            # Clear treeview
            for item in self.tree.get_children():
                self.tree.delete(item)
            graham_logger.debug("Treeview cleared")

            if not results:
                graham_logger.info(f"No NASDAQ qualifiers found for min_criteria={self.min_criteria_var.get()}")
                self.safe_update_tree(
                    "No qualifying stocks found",
                    ("", "", "", "", "", "", "")
                )
                self.root.update_idletasks()
                return

            # Populate treeview
            for ticker, company_name, sector, common_score, is_foreign in results:
                display_name = company_name if company_name else ticker
                sector_display = sector
                tags = []
                if is_foreign:
                    sector_display += " (Foreign)"
                    tags.append('foreign')
                graham_logger.debug(f"Adding to treeview: {ticker}, Name: {display_name}, Sector: {sector_display}, Score: {common_score}/6, Tags: {tags}")
                self.safe_update_tree(
                    ticker,
                    (
                        display_name,
                        sector_display,
                        f"{common_score}/6",
                        "", "", "", ""  # Price, intrinsic value, buy/sell prices are empty
                    ),
                    tags
                )

            # Force treeview sort and UI refresh
            self.sort_tree(0)
            self.root.update_idletasks()
            self.root.update()
            graham_logger.info(f"Displayed {len(results)} NASDAQ qualifying stocks in treeview")
        except sqlite3.Error as e:
            error_msg = f"Database error displaying NASDAQ qualifiers: {str(e)}"
            graham_logger.error(error_msg)
            messagebox.showerror("Database Error", error_msg)
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
            graham_logger.error(f"Error fetching prices: {e}")
            return {ticker: None for ticker in tickers}

    def calculate_intrinsic_value(stock_dict):
        eps = stock_dict.get('eps_ttm')
        eps_cagr = stock_dict.get('eps_cagr', 0.0)
        if not eps or eps <= 0:
            return float('nan')
        aaa_yield = cache_manager.get_aaa_yield(FRED_API_KEY)
        if aaa_yield <= 0:
            return float('nan')
        g = max(eps_cagr * 100, 0)
        max_multiplier = 15 if stock_dict.get('sector') == "Financials" else 20
        earnings_multiplier = min(8.5 + 2 * g, max_multiplier)
        normalization_factor = 4.4
        value = (eps * earnings_multiplier * normalization_factor) / (100 * aaa_yield)
        return value if not math.isinf(value) and not pd.isna(value) else float('nan')

    def export_nyse_qualifying_stocks(self):
            threading.Thread(target=self._run_export, args=("NYSE",)).start()
            self.status_label.config(text="Exporting NYSE...")

    def export_nasdaq_qualifying_stocks(self):
        threading.Thread(target=self._run_export, args=("NASDAQ",)).start()
        self.status_label.config(text="Exporting NASDAQ...")

    def _run_export(self, exchange):
        try:
            export_qualifying_stocks(exchange, self.min_criteria_var.get(), self.margin_of_safety_var.get(), self.expected_return_var.get())
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Export Error", str(e)))
        finally:
            self.root.after(0, lambda: self.status_label.config(text=""))

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
            conn, cursor = cache_manager.get_stocks_connection()
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

    def filter_tree(self, event) -> None:
        """Filter the treeview based on the search term entered in the search entry widget."""
        search_term = self.search_entry.get().strip().upper()
        # Clear the current treeview contents
        for item in self.tree.get_children():
            self.tree.delete(item)
        # Connect to the database
        conn, cursor = cache_manager.get_stocks_connection()
        try:
            # If there's a search term, filter stocks by ticker or company_name (case-insensitive); otherwise, show all stocks
            if search_term:
                cursor.execute(
                    "SELECT ticker, company_name, sector, common_score, is_foreign FROM stocks WHERE UPPER(ticker) LIKE ? OR UPPER(company_name) LIKE ?",
                    (f"%{search_term}%", f"%{search_term}%")
                )
            else:
                cursor.execute("SELECT ticker, company_name, sector, common_score, is_foreign FROM stocks")
            results = cursor.fetchall()
            # Populate the treeview with filtered results
            for ticker, company_name, sector, common_score, is_foreign in results:
                tags = []
                sector_display = sector
                if is_foreign:
                    sector_display += " (Foreign)"
                    tags.append('foreign')
                self.safe_update_tree(
                    ticker,
                    (
                        company_name or ticker,
                        sector_display,
                        f"{common_score}/6" if common_score is not None else "N/A",
                        "", "", "", ""
                    ),
                    tags
                )
        except sqlite3.Error as e:
            graham_logger.error(f"Database error in filter_tree: {str(e)}")
        finally:
            conn.close()

    def safe_insert(self, text_widget, content):
        try:
            self.root.after(0, lambda: text_widget.insert(tk.END, content))
        except tk.TclError as e:
            graham_logger.warning(f"Tk error during text insert: {str(e)}")

    def check_for_updates(self):
        """Perform a startup check for Python and library updates."""
        msg = []
        
        # Check Python version
        latest_python = self.get_latest_python_version()
        current_python = sys.version_info[:3]
        if latest_python and current_python < latest_python:
            current_str = '.'.join(map(str, current_python))
            latest_str = '.'.join(map(str, latest_python))
            msg.append(f"Python version {current_str} is outdated. Latest is {latest_str}. "
                       f"Please download and install from https://www.python.org/downloads/ manually.")
        
        # Check libraries
        outdated_packages = self.get_outdated_packages()
        if outdated_packages is None:
            msg.append("Failed to check for library updates (pip error). Please run 'pip list --outdated' manually.")
        elif outdated_packages:
            msg.append("The following libraries have updates available:\n" + "\n".join(outdated_packages))
        
        if msg:
            full_msg = "\n\n".join(msg)
            if outdated_packages and outdated_packages is not None:  # Offer update only if libraries are outdated
                full_msg += "\n\nWould you like to update the libraries now? (Python update is manual.)"
                if messagebox.askyesno("Updates Available", full_msg):
                    self.update_packages(outdated_packages)
                    messagebox.showinfo("Update Complete", "Libraries updated successfully. Please restart the application.")
                    self.root.quit()  # Quit the app to encourage restart
            else:
                messagebox.showwarning("Updates Recommended", full_msg)

    def get_latest_python_version(self):
        """Fetch the latest Python 3 version from python.org."""
        try:
            response = requests.get("https://www.python.org/downloads/", verify=True)
            response.raise_for_status()
            match = re.search(r"Python (\d+\.\d+\.\d+)", response.text)
            if match:
                return tuple(map(int, match.group(1).split('.')))
        except Exception as e:
            graham_logger.warning(f"Failed to fetch latest Python version: {str(e)}")
        return (3, 13, 7)  # Fallback

    def get_outdated_packages(self):
        """Get list of outdated pip packages."""
        try:
            shell = (platform.system() == 'Windows')  # Use shell=True on Windows
            # Replace pythonw.exe with python.exe if necessary
            executable = sys.executable
            if executable.endswith('pythonw.exe'):
                executable = executable[:-len('pythonw.exe')] + 'python.exe'
            cmd = [executable, '-m', 'pip', 'list', '--outdated'] if executable else ['pip', 'list', '--outdated']
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, shell=shell)
            outdated = result.stdout.strip()
            if outdated:
                # Parse table output (skip header lines starting with "Package" or "---")
                lines = outdated.splitlines()
                packages = []
                for line in lines[2:]:  # Skip header and separator line
                    if line.strip():  # Ignore empty lines
                        parts = line.split()
                        if parts:  # Ensure there's at least one part (package name)
                            packages.append(parts[0])
                return packages
            return []
        except subprocess.CalledProcessError as e:
            graham_logger.error(f"Pip subprocess error: Command '{e.cmd}' failed with return code {e.returncode}. Stderr: {e.stderr}")
            return None
        except Exception as e:
            graham_logger.error(f"Error checking pip updates: {str(e)}")
            return None

    def update_packages(self, packages):
        """Update the specified packages via pip."""
        for pkg in packages:
            if not re.match(r'^[a-zA-Z0-9_-]+$', pkg):  # Basic sanitization
                graham_logger.warning(f"Skipping invalid package name: {pkg}")
                continue
            try:
                executable = sys.executable
                if executable.endswith('pythonw.exe'):
                    executable = executable[:-len('pythonw.exe')] + 'python.exe'
                cmd = [executable, '-m', 'pip', 'install', '--upgrade', '--user', pkg]
                subprocess.run(cmd, check=True)  # No shell=True
                graham_logger.info(f"Updated package: {pkg}")
            except Exception as e:
                messagebox.showerror("Update Error", f"Failed to update {pkg}: {str(e)}")
                graham_logger.error(f"Failed to update {pkg}: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = GrahamScreeningApp(root)
    root.mainloop()