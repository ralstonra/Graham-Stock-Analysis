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
                         fetch_stock_data, VALID_NYSE_TICKERS, VALID_NASDAQ_TICKERS, get_stocks_connection,
                         fetch_with_multiple_keys_async, NYSE_LIST_FILE, NASDAQ_LIST_FILE)
from config import BASE_DIR, FMP_API_KEYS, FAVORITES_FILE
import queue

# Set up logging
logging.basicConfig(filename=os.path.join(BASE_DIR, 'nyse_graham_screen.log'), level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logging.getLogger().addHandler(console)

FAVORITES_LOCK = threading.Lock()

class GrahamScreeningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Analysis (Graham Defensive)")
        self.root.geometry("1200x800")

        # Variables
        self.nyse_screen_var = tk.BooleanVar(value=False)
        self.nasdaq_screen_var = tk.BooleanVar(value=False)
        self.cancel_event = threading.Event()
        self.margin_of_safety_var = tk.DoubleVar(value=33.0)
        self.expected_return_var = tk.DoubleVar(value=0.0)
        self.screening_mutex = threading.Lock()

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
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        while True:
            coro = task_queue.get()
            if coro is None:  # Sentinel to stop the loop
                break
            loop.run_until_complete(coro)
        loop.close()

    def create_widgets(self):
        # Left Column: Stock Ticker Input and Analysis
        ttk.Label(self.left_frame, text="Enter Stock Tickers (comma-separated, e.g., AOS, AAPL):").grid(row=0, column=0, pady=1, sticky="ew")
        self.entry = ttk.Entry(self.left_frame, width=50)
        self.entry.grid(row=1, column=0, pady=1, sticky="ew")

        def validate_tickers(*args):
            tickers = self.parse_tickers(self.entry.get())
            if not tickers:
                messagebox.showwarning("Invalid Tickers", "No valid tickers entered.")
                logging.warning("No valid tickers entered in GUI")
                return False
            return True

        self.entry.bind('<FocusOut>', lambda e: validate_tickers())

        # Favorites
        self.favorites = self.load_favorites()
        ttk.Label(self.left_frame, text="Favorite Lists:").grid(row=2, column=0, pady=1, sticky="ew")
        self.favorite_var = tk.StringVar(value="Select Favorite")
        self.favorite_menu = ttk.Combobox(self.left_frame, textvariable=self.favorite_var, values=list(self.favorites.keys()))
        self.favorite_menu.grid(row=3, column=0, pady=1, sticky="ew")

        def load_favorite(event=None):
            selected = self.favorite_var.get()
            if selected and selected != "Select Favorite":
                self.entry.delete(0, tk.END)
                self.entry.insert(0, ",".join(self.favorites[selected]['tickers']))
                logging.info(f"Loaded favorite list '{selected}'")

        self.favorite_menu.bind('<<ComboboxSelected>>', load_favorite)

        def save_favorite():
            if not validate_tickers():
                return
            name = simpledialog.askstring("Save Favorite", "Enter list name:")
            if name and self.entry.get().strip():
                tickers = self.parse_tickers(self.entry.get())
                self.favorites[name] = {
                    "tickers": tickers,
                    "graham_scores": [],
                    "exchanges": ["Stock"] * len(tickers),
                    "date": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                self.save_favorites()
                self.favorite_menu['values'] = list(self.favorites.keys())
                logging.info(f"Saved favorite list '{name}' with tickers: {tickers}")
                messagebox.showinfo("Success", f"Saved favorite list '{name}'")

        ttk.Button(self.left_frame, text="Save Favorite", command=save_favorite).grid(row=4, column=0, pady=1, sticky="ew")
        ttk.Button(self.left_frame, text="Manage Favorites", command=self.manage_favorites).grid(row=5, column=0, pady=1, sticky="ew")

        # Margin of Safety and Expected Return
        ttk.Label(self.left_frame, text="Margin of Safety (%):").grid(row=6, column=0, pady=1, sticky="ew")
        self.margin_of_safety_label = ttk.Label(self.left_frame, text="33%")
        self.margin_of_safety_label.grid(row=7, column=0, pady=1, sticky="ew")
        ttk.Scale(self.left_frame, from_=0, to=50, orient=tk.HORIZONTAL, variable=self.margin_of_safety_var,
                  command=lambda value: self.margin_of_safety_label.config(text=f"{int(float(value))}%")).grid(row=8, column=0, pady=1, sticky="ew")

        ttk.Label(self.left_frame, text="Expected Return (%):").grid(row=9, column=0, pady=1, sticky="ew")
        self.expected_return_label = ttk.Label(self.left_frame, text="0%")
        self.expected_return_label.grid(row=10, column=0, pady=1, sticky="ew")
        ttk.Scale(self.left_frame, from_=0, to=20, orient=tk.HORIZONTAL, variable=self.expected_return_var,
                  command=lambda value: self.expected_return_label.config(text=f"{int(float(value))}%")).grid(row=11, column=0, pady=1, sticky="ew")

        # Analyze Stocks Button
        ttk.Button(self.left_frame, text="Analyze Stocks", command=self.analyze_multiple_stocks).grid(row=14, column=0, pady=1, sticky="ew")

        # Middle Column: NYSE Screening
        ttk.Checkbutton(self.middle_frame, text="Run NYSE Graham Screening", variable=self.nyse_screen_var).grid(row=0, column=0, pady=2, padx=2, sticky="w")
        ttk.Button(self.middle_frame, text="Run NYSE Screening", command=self.run_nyse_screening).grid(row=1, column=0, pady=2, sticky="ew")
        ttk.Button(self.middle_frame, text="Cancel Screening", command=self.cancel_screening).grid(row=2, column=0, pady=2, sticky="ew")

        ttk.Label(self.middle_frame, text="Filter by Exchange:").grid(row=3, column=0, pady=2, sticky="ew")
        self.nyse_sector_var = tk.StringVar(value="All")
        ttk.Combobox(self.middle_frame, textvariable=self.nyse_sector_var, values=["All", "NYSE"]).grid(row=4, column=0, pady=1, sticky="ew")

        ttk.Label(self.middle_frame, text="Filter NYSE Results:").grid(row=5, column=0, pady=2, sticky="ew")
        self.nyse_filter_entry = ttk.Entry(self.middle_frame, width=30)
        self.nyse_filter_entry.grid(row=6, column=0, pady=1, sticky="ew")
        self.nyse_filter_entry.bind('<Return>', lambda e: self.filter_tree(self.nyse_filter_entry.get()))

        ttk.Button(self.middle_frame, text="Show NYSE Qualifying Stocks", command=self.display_nyse_qualifying_stocks).grid(row=7, column=0, pady=2, sticky="ew")
        ttk.Button(self.middle_frame, text="Export NYSE Qualifying Stocks", command=self.export_nyse_qualifying_stocks).grid(row=8, column=0, pady=2, sticky="ew")

        # Right Column: NASDAQ Screening
        ttk.Checkbutton(self.right_frame, text="Run NASDAQ Graham Screening", variable=self.nasdaq_screen_var).grid(row=0, column=0, pady=2, padx=2, sticky="w")
        ttk.Button(self.right_frame, text="Run NASDAQ Screening", command=self.run_nasdaq_screening).grid(row=1, column=0, pady=2, sticky="ew")
        ttk.Button(self.right_frame, text="Cancel Screening", command=self.cancel_screening).grid(row=2, column=0, pady=2, sticky="ew")

        ttk.Label(self.right_frame, text="Filter by Exchange:").grid(row=3, column=0, pady=2, sticky="ew")
        self.nasdaq_sector_var = tk.StringVar(value="All")
        ttk.Combobox(self.right_frame, textvariable=self.nasdaq_sector_var, values=["All", "NASDAQ"]).grid(row=4, column=0, pady=1, sticky="ew")

        ttk.Label(self.right_frame, text="Filter NASDAQ Results:").grid(row=5, column=0, pady=2, sticky="ew")
        self.nasdaq_filter_entry = ttk.Entry(self.right_frame, width=30)
        self.nasdaq_filter_entry.grid(row=6, column=0, pady=1, sticky="ew")
        self.nasdaq_filter_entry.bind('<Return>', lambda e: self.filter_tree(self.nasdaq_filter_entry.get()))

        ttk.Button(self.right_frame, text="Show NASDAQ Qualifying Stocks", command=self.display_nasdaq_qualifying_stocks).grid(row=7, column=0, pady=2, sticky="ew")
        ttk.Button(self.right_frame, text="Export NASDAQ Qualifying Stocks", command=self.export_nasdaq_qualifying_stocks).grid(row=8, column=0, pady=2, sticky="ew")

        # Progress Bar and Label
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.middle_frame, variable=self.progress_var, maximum=100, length=300, mode='determinate')
        self.progress_bar.grid(row=9, column=0, pady=2, sticky="ew")
        self.progress_label = ttk.Label(self.middle_frame, text="Progress: 0% (0/0 tickers processed)")
        self.progress_label.grid(row=10, column=0, pady=1, sticky="ew")

        # Clear Cache and Help Buttons
        ttk.Button(self.middle_frame, text="Clear Cache", command=self.clear_cache).grid(row=11, column=0, pady=2, sticky="ew")
        ttk.Button(self.middle_frame, text="Help", command=self.show_help).grid(row=12, column=0, pady=2, sticky="ew")

        # Treeview Setup
        self.full_tree_frame = ttk.Frame(self.main_frame)
        self.full_tree_frame.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)
        self.full_tree_frame.grid_columnconfigure(0, weight=1)
        self.full_tree_frame.grid_rowconfigure((0, 1, 2), weight=1)

        self.tree = ttk.Treeview(self.full_tree_frame, columns=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), show="tree headings", height=15)
        self.v_scrollbar = ttk.Scrollbar(self.full_tree_frame, orient="vertical", command=self.tree.yview)
        self.h_scrollbar = ttk.Scrollbar(self.full_tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.h_scrollbar.grid(row=1, column=0, sticky="ew")
        self.tree.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)

        for col in self.tree["columns"]:
            self.tree.heading(col, command=lambda c=col: self.sort_tree(self.tree["columns"].index(c)))
            self.tree.column(col, width=100, anchor="center")
        self.tree.column("#0", width=80, anchor="center")

        self.data_frame = ttk.Frame(self.full_tree_frame)
        self.notebook = ttk.Notebook(self.data_frame)
        self.historical_frame = ttk.Frame(self.notebook)
        self.metrics_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.historical_frame, text="Historical Data")
        self.notebook.add(self.metrics_frame, text="Metrics")
        self.data_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")
        self.notebook.grid(row=0, column=0, sticky="nsew")

    def parse_tickers(self, tickers_input):
        if not tickers_input.strip():
            return []
        return [t.strip().upper() for t in tickers_input.split(',') if t.strip() and t.strip().isalnum() and len(t.strip()) <= 5]

    def load_favorites(self):
        with FAVORITES_LOCK:
            if os.path.exists(FAVORITES_FILE):
                try:
                    with open(FAVORITES_FILE, 'r') as f:
                        return json.load(f)
                except json.JSONDecodeError:
                    logging.error(f"Corrupted favorites file {FAVORITES_FILE}. Returning empty dict.")
                    return {}
            return {}

    def save_favorites(self):
        with FAVORITES_LOCK:
            try:
                with open(FAVORITES_FILE, 'w') as f:
                    json.dump(self.favorites, f, indent=4)
                logging.info(f"Saved favorites to {FAVORITES_FILE}")
            except Exception as e:
                logging.error(f"Failed to save favorites: {str(e)}")
                messagebox.showerror("Error", f"Failed to save favorites: {str(e)}")

    def update_progress_animated(self, progress, exchange="", tickers=None):
        self.progress_var.set(progress)
        if tickers is not None:
            total_tickers = len(tickers)
            processed = int(progress / 100 * total_tickers)
            self.progress_label.config(text=f"Progress: {progress:.1f}% ({processed}/{total_tickers} {exchange} tickers processed)")
        else:
            total_tickers = len(VALID_NYSE_TICKERS) if exchange == "NYSE" else len(VALID_NASDAQ_TICKERS) if exchange == "NASDAQ" else 0
            processed = int(progress / 100 * total_tickers)
            self.progress_label.config(text=f"Progress: {progress:.1f}% ({processed}/{total_tickers} {exchange} tickers processed)")
        self.root.update_idletasks()
        logging.debug(f"Progress updated to {progress:.1f}% for {exchange}")

    def refresh_favorites_dropdown(self, selected_list=None):
        self.favorites = self.load_favorites()
        self.favorite_menu['values'] = list(self.favorites.keys())
        if selected_list and selected_list in self.favorites:
            self.favorite_var.set(selected_list)
            self.load_favorite()
        logging.info("Favorites dropdown refreshed")

    def cancel_screening(self):
        self.cancel_event.set()
        logging.info("Screening cancellation requested")

    def run_screening(self, exchange, screen_func):
        with self.screening_mutex:
            if (exchange == "NYSE" and not self.nyse_screen_var.get()) or (exchange == "NASDAQ" and not self.nasdaq_screen_var.get()):
                return
            if self.nyse_screen_var.get() and self.nasdaq_screen_var.get():
                messagebox.showwarning("Warning", "Cannot run NYSE and NASDAQ screening simultaneously.")
                self.nyse_screen_var.set(False)
                self.nasdaq_screen_var.set(False)
                return

            logging.info(f"Starting {exchange} Graham Screening")
            self.progress_var.set(0)
            file_path = NYSE_LIST_FILE if exchange == "NYSE" else NASDAQ_LIST_FILE
            if not os.path.exists(file_path):
                logging.error(f"{exchange} list file missing: {file_path}")
                messagebox.showerror("Error", f"{exchange} list file missing: {file_path}")
                setattr(self, f"{exchange.lower()}_screen_var", False)
                return
            tickers = list(VALID_NYSE_TICKERS if exchange == "NYSE" else VALID_NASDAQ_TICKERS)
            if not tickers:
                logging.error(f"VALID_{exchange}_TICKERS is empty.")
                messagebox.showerror("Error", f"No {exchange} tickers loaded.")
                setattr(self, f"{exchange.lower()}_screen_var", False)
                return
            self.progress_label.config(text=f"Progress: 0% (0/{len(tickers)} {exchange} tickers processed)")
            self.root.update()
            self.cancel_event.clear()

            def target():
                with get_stocks_connection() as (conn, cursor):
                    try:
                        asyncio.run(screen_func(
                            batch_size=50,
                            cancel_event=self.cancel_event,
                            tickers=tickers,
                            root=self.root,
                            update_progress_animated=self.update_progress_animated,
                            refresh_favorites_dropdown=self.refresh_favorites_dropdown,
                            conn=conn,
                            cursor=cursor
                        ))
                    except Exception as e:
                        logging.error(f"Screening error for {exchange}: {str(e)}")
                        self.root.after(0, lambda: messagebox.showerror("Error", f"Screening failed for {exchange}: {str(e)}"))
                    finally:
                        self.root.after(0, lambda: self.progress_label.config(text=f"Progress: 100% (Screening Complete - {len(tickers)} {exchange} tickers processed)"))
                        setattr(self, f"{exchange.lower()}_screen_var", False)
                        logging.info(f"{exchange} screening thread completed or terminated")

            threading.Thread(target=target, daemon=True).start()

    def run_nyse_screening(self):
        self.run_screening("NYSE", screen_nyse_graham_stocks)

    def run_nasdaq_screening(self):
        self.run_screening("NASDAQ", screen_nasdaq_graham_stocks)

    async def fetch_company_name(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            company_name = stock.info.get('longName', 'Unknown')
            if company_name != 'Unknown':
                logging.info(f"Successfully fetched company name for {ticker} from yfinance: {company_name}")
                return company_name

            logging.warning(f"yfinance failed for {ticker}, falling back to FMP")
            fmp_data = await fetch_with_multiple_keys_async(ticker, "profile", FMP_API_KEYS, service="FMP")
            if fmp_data and isinstance(fmp_data, list) and len(fmp_data) > 0:
                logging.info(f"Successfully fetched company name for {ticker} from FMP: {fmp_data[0].get('companyName', 'Unknown')}")
                return fmp_data[0].get('companyName', 'Unknown')

            logging.error(f"No company name data for {ticker} from yfinance or FMP")
            return 'Unknown'
        except Exception as e:
            logging.error(f"Error fetching company name for {ticker}: {str(e)}")
            return 'Unknown'

    def format_float(self, value, precision=2):
        """Helper function to safely format a float or return 'N/A' if None/invalid."""
        if value is None or not isinstance(value, (int, float)) or value != value:  # Check for None, non-numeric, or NaN
            return "N/A"
        return f"{value:.{precision}f}"

    async def analyze_multiple_stocks_async(self, tickers_input=None):
        if tickers_input is None:
            tickers_input = self.entry.get()
        tickers = self.parse_tickers(tickers_input)
        if not tickers:
            messagebox.showwarning("No Tickers", "No valid tickers to analyze.")
            logging.warning("No valid tickers to analyze in GUI")
            return

        logging.info(f"Starting analysis for {len(tickers)} tickers")
        self.progress_var.set(0)
        self.progress_label.config(text=f"Progress: 0% (0/{len(tickers)} tickers processed)")
        self.root.update()

        results = await fetch_batch_data(tickers, expected_return=self.expected_return_var.get() / 100,
                                        margin_of_safety=self.margin_of_safety_var.get() / 100, exchange="Stock")
        valid_results = [r for r in results if 'error' not in r]
        error_results = [r for r in results if 'error' in r]
        for err in error_results:
            self.show_error(f"Error for {err['ticker']} ({err['exchange']}): {err['error']}")

        for item in self.tree.get_children():
            self.tree.delete(item)

        for widget in self.historical_frame.winfo_children():
            widget.destroy()
        for widget in self.metrics_frame.winfo_children():
            widget.destroy()

        self.tree.heading("#0", text="Ticker")
        self.tree.heading(1, text="Company Name")
        self.tree.heading(2, text="Graham Score")
        self.tree.heading(3, text="Exchange")
        self.tree.heading(4, text="Intrinsic Value ($)")
        self.tree.heading(5, text="Price ($)")
        self.tree.heading(6, text="Buy Price ($)")
        self.tree.heading(7, text="Sell Price ($)")
        self.tree.heading(8, text="Total Graham Score")
        self.tree.heading(9, text="Debt/Equity")
        self.tree.heading(10, text="P/E Ratio")
        self.tree.column("#0", width=80, anchor="center")
        self.tree.column(1, width=150, anchor="center")
        self.tree.column(2, width=80, anchor="center")
        self.tree.column(3, width=80, anchor="center")
        self.tree.column(4, width=100, anchor="center")
        self.tree.column(5, width=80, anchor="center")
        self.tree.column(6, width=80, anchor="center")
        self.tree.column(7, width=80, anchor="center")
        self.tree.column(8, width=80, anchor="center")
        self.tree.column(9, width=80, anchor="center")
        self.tree.column(10, width=80, anchor="center")

        async def fetch_company_names(results_list):
            tasks = [self.fetch_company_name(result['ticker']) for result in results_list if result['ticker']]
            return await asyncio.gather(*tasks)

        company_names = await fetch_company_names(valid_results)
        total_results = len(valid_results)

        for i, result in enumerate(valid_results):
            company_name = company_names[i] if i < len(company_names) else 'Unknown'
            if result['intrinsic_value'] is None:
                logging.warning(f"Intrinsic value not calculated for {result['ticker']} due to missing data (EPS: {result['eps_10y'][-1] if result['eps_10y'] else 'N/A'}, Growth: ?, Yield: ?)")
            self.tree.insert("", "end", text=result['ticker'], values=(
                company_name,
                f"{result['graham_score']}/5",
                result['exchange'],
                f"${self.format_float(result['intrinsic_value'])}",
                f"${self.format_float(result['price'])}",
                f"${self.format_float(result['buy_price'])}",
                f"${self.format_float(result['sell_price'])}",
                f"{result['total_graham_score']}/7" if 'total_graham_score' in result and result['total_graham_score'] else "N/A",
                f"{self.format_float(result['debt_to_equity'])}",
                f"{self.format_float(result['pe_ratio'])}"
            ))

            historical_text = scrolledtext.ScrolledText(self.historical_frame, width=70, height=10)
            historical_text.grid(row=0, column=0, sticky="nsew")
            metrics_text = scrolledtext.ScrolledText(self.metrics_frame, width=70, height=10)
            metrics_text.grid(row=0, column=0, sticky="nsew")

            years = result.get('years', [])
            roe_10y = result.get('roe_10y', [])
            rotc_10y = result.get('rotc_10y', [])
            eps_10y = result.get('eps_10y', [])
            div_10y = result.get('div_10y', [])

            if years and len(years) == len(roe_10y) == len(rotc_10y) == len(eps_10y) == len(div_10y) and any(roe_10y) and any(rotc_10y) and any(eps_10y):
                historical_text.insert(tk.END, f"{len(years)}-Year Historical Data for {result['ticker']}:\nYear\tROE (%)\tROTC (%)\tEPS ($)\tDividend ($)\n")
                for j in range(len(years)):
                    historical_text.insert(tk.END, f"{years[j]}\t{roe_10y[j]:.2f}\t{rotc_10y[j]:.2f}\t{eps_10y[j]:.2f}\t{div_10y[j]:.2f}\n")
                metrics_text.insert(tk.END, f"Metrics for {result['ticker']} (informational, all 7 criteria):\n"
                    f"Price: ${self.format_float(result['price'])}\n"
                    f"Graham Score (5): {result['graham_score']}/5\n"
                    f"Total Graham Score (7): {result['total_graham_score']}/7\n"
                    f"Intrinsic Value: ${self.format_float(result['intrinsic_value'])}\n"
                    f"Buy Price: ${self.format_float(result['buy_price'])}\n"
                    f"Sell Price: ${self.format_float(result['sell_price'])}\n"
                    f"Debt/Equity: {self.format_float(result['debt_to_equity'])}\n"
                    f"P/E Ratio: {self.format_float(result['pe_ratio'])}\n")
            else:
                historical_text.insert(tk.END, f"No historical data available for {result['ticker']}.\n")
                metrics_text.insert(tk.END, f"Metrics for {result['ticker']} (informational, all 7 criteria):\n"
                    f"Price: ${self.format_float(result['price'])}\n"
                    f"Graham Score (5): {result['graham_score']}/5\n"
                    f"Total Graham Score (7): {result['total_graham_score']}/7\n"
                    f"Intrinsic Value: ${self.format_float(result['intrinsic_value'])}\n"
                    f"Buy Price: ${self.format_float(result['buy_price'])}\n"
                    f"Sell Price: ${self.format_float(result['sell_price'])}\n"
                    f"Debt/Equity: {self.format_float(result['debt_to_equity'])}\n"
                    f"P/E Ratio: {self.format_float(result['pe_ratio'])}\n")

            progress = (i + 1) / total_results * 100
            self.root.after(0, lambda p=progress: self.update_progress_animated(p, "Stock", tickers))

        self.root.after(0, lambda: self.update_progress_animated(100, "Stock", tickers))
        logging.info(f"Completed analysis for {len(tickers)} tickers")

    def analyze_multiple_stocks(self, tickers_input=None):
        self.task_queue.put(self.analyze_multiple_stocks_async(tickers_input))

    def display_results_in_tree(self, tickers, scores, exchanges, source):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.tree.heading("#0", text="Ticker")
        self.tree.heading(1, text="Company Name")
        self.tree.heading(2, text="Graham Score")
        self.tree.heading(3, text="Exchange")
        self.tree.heading(4, text="Intrinsic Value ($)")
        self.tree.heading(5, text="Price ($)")
        self.tree.heading(6, text="Buy Price ($)")
        self.tree.heading(7, text="Sell Price ($)")
        self.tree.heading(8, text="Total Graham Score")
        self.tree.heading(9, text="Debt/Equity")
        self.tree.heading(10, text="P/E Ratio")
        self.tree.column("#0", width=80, anchor="center")
        self.tree.column(1, width=150, anchor="center")
        self.tree.column(2, width=80, anchor="center")
        self.tree.column(3, width=80, anchor="center")
        self.tree.column(4, width=100, anchor="center")
        self.tree.column(5, width=80, anchor="center")
        self.tree.column(6, width=80, anchor="center")
        self.tree.column(7, width=80, anchor="center")
        self.tree.column(8, width=80, anchor="center")
        self.tree.column(9, width=80, anchor="center")
        self.tree.column(10, width=80, anchor="center")

        async def fetch_company_names(tickers_list):
            tasks = [self.fetch_company_name(ticker) for ticker in tickers_list]
            return await asyncio.gather(*tasks)

        company_names = asyncio.run(fetch_company_names(tickers))

        for i, (ticker, score, exchange) in enumerate(zip(tickers, scores, exchanges)):
            company_name = company_names[i] if i < len(company_names) else 'Unknown'
            result = {"ticker": ticker, "graham_score": score, "exchange": exchange, "price": 0, "intrinsic_value": 0, "buy_price": 0, "sell_price": 0, "years": [], "roe_10y": [], "rotc_10y": [], "eps_10y": [], "div_10y": [], "total_graham_score": 0, "debt_to_equity": None, "pe_ratio": None}
            self.tree.insert("", "end", text=ticker, values=(
                company_name,
                f"{score}/5",
                exchange,
                f"${self.format_float(result['intrinsic_value'])}",
                f"${self.format_float(result['price'])}",
                f"${self.format_float(result['buy_price'])}",
                f"${self.format_float(result['sell_price'])}",
                f"{result['total_graham_score']}/7" if result['total_graham_score'] else "N/A",
                f"{self.format_float(result['debt_to_equity'])}",
                f"{self.format_float(result['pe_ratio'])}"
            ))

    def filter_tree(self, filter_text):
        for item in self.tree.get_children():
            values = self.tree.item(item, "values")
            ticker, company_name, score, exchange = self.tree.item(item, "text"), values[0], values[1], values[2]
            if (filter_text.lower() not in ticker.lower() and
                filter_text.lower() not in company_name.lower() and
                filter_text.lower() not in score.lower() and
                filter_text.lower() not in exchange.lower()):
                self.tree.detach(item)
            else:
                self.tree.reattach(item, '', 0)

    def sort_tree(self, column):
        items = [(self.tree.item(item, "values")[column], item) for item in self.tree.get_children()]
        items.sort(reverse=True if self.tree.heading(column, "descending") else False)
        for i, (_, item) in enumerate(items):
            self.tree.move(item, "", i)
        self.tree.heading(column, descending=not self.tree.heading(column, "descending"))

    def export_qualifying_stocks(self, exchange):
        favorites = self.load_favorites()
        qualifying_lists = [name for name in favorites.keys() if name.startswith(f"{exchange}_Qualifiers_")]
        if qualifying_lists:
            latest_list = max(qualifying_lists, key=lambda x: datetime.strptime(x.split("_")[-1], "%Y%m%d_%H%M%S"))
            tickers = favorites[latest_list]['tickers']
            scores = favorites[latest_list]['graham_scores']
            exchanges = favorites[latest_list]['exchanges']
            df = pd.DataFrame({
                'Ticker': tickers,
                'Graham Score': scores,
                'Exchange': exchanges
            })

            async def fetch_company_names_for_export(tickers_list):
                tasks = [self.fetch_company_name(ticker) for ticker in tickers_list]
                return await asyncio.gather(*tasks)

            company_names = asyncio.run(fetch_company_names_for_export(tickers))
            df['Company Name'] = company_names
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")])
            if file_path:
                if file_path.endswith('.csv'):
                    df.to_csv(file_path, index=False)
                else:
                    df.to_excel(file_path, index=False)
                logging.info(f"Exported {exchange} qualifying stocks to {file_path}")
                messagebox.showinfo("Export Successful", f"Exported {exchange} stocks to {file_path}")
            else:
                logging.warning(f"Export of {exchange} stocks cancelled by user")
        else:
            messagebox.showwarning("No Results", f"No {exchange} qualifying stocks to export.")
            logging.warning(f"No {exchange} qualifying stocks available to export")

    def display_nyse_qualifying_stocks(self):
        favorites = self.load_favorites()
        nyse_lists = [name for name in favorites.keys() if name.startswith("NYSE_Qualifiers_")]
        if nyse_lists:
            latest_list = max(nyse_lists, key=lambda x: datetime.strptime(x.split("_")[-1], "%Y%m%d_%H%M%S"))
            tickers = favorites[latest_list]['tickers']
            scores = favorites[latest_list]['graham_scores']
            exchanges = favorites[latest_list]['exchanges']
            self.display_results_in_tree(tickers, scores, exchanges, "NYSE")
        else:
            messagebox.showinfo("No Results", "No NYSE qualifying stocks found.")
            logging.info("No NYSE qualifying stocks available to display")

    def display_nasdaq_qualifying_stocks(self):
        favorites = self.load_favorites()
        nasdaq_lists = [name for name in favorites.keys() if name.startswith("NASDAQ_Qualifiers_")]
        if nasdaq_lists:
            latest_list = max(nasdaq_lists, key=lambda x: datetime.strptime(x.split("_")[-1], "%Y%m%d_%H%M%S"))
            tickers = favorites[latest_list]['tickers']
            scores = favorites[latest_list]['graham_scores']
            exchanges = favorites[latest_list]['exchanges']
            self.display_results_in_tree(tickers, scores, exchanges, "NASDAQ")
        else:
            messagebox.showinfo("No Results", "No NASDAQ qualifying stocks found.")
            logging.info("No NASDAQ qualifying stocks available to display")

    def export_nyse_qualifying_stocks(self):
        self.export_qualifying_stocks("NYSE")

    def export_nasdaq_qualifying_stocks(self):
        self.export_qualifying_stocks("NASDAQ")

    def manage_favorites(self):
        if not self.favorites:
            messagebox.showinfo("No Favorites", "No favorite lists to manage.")
            logging.info("No favorite lists available to manage")
            return

        def delete_favorite():
            selected = self.favorite_var.get()
            if selected and selected != "Select Favorite":
                del self.favorites[selected]
                self.save_favorites()
                self.favorite_menu['values'] = list(self.favorites.keys())
                self.favorite_var.set("Select Favorite")
                logging.info(f"Deleted favorite list '{selected}'")
                messagebox.showinfo("Success", f"Deleted favorite list '{selected}'")

        def rename_favorite():
            selected = self.favorite_var.get()
            if selected and selected != "Select Favorite":
                new_name = simpledialog.askstring("Rename Favorite", "Enter new name:", initialvalue=selected)
                if new_name and new_name != selected:
                    self.favorites[new_name] = self.favorites.pop(selected)
                    self.save_favorites()
                    self.favorite_menu['values'] = list(self.favorites.keys())
                    self.favorite_var.set(new_name)
                    logging.info(f"Renamed favorite list '{selected}' to '{new_name}'")
                    messagebox.showinfo("Success", f"Renamed favorite list '{selected}' to '{new_name}'")

        manage_window = tk.Toplevel(self.root)
        manage_window.title("Manage Favorites")
        manage_window.geometry("300x120")
        logging.info("Opened Manage Favorites window")
        ttk.Button(manage_window, text="Delete Favorite", command=delete_favorite).grid(row=0, column=0, pady=2, sticky="ew")
        ttk.Button(manage_window, text="Rename Favorite", command=rename_favorite).grid(row=1, column=0, pady=2, sticky="ew")

    def show_error(self, message):
        error_window = tk.Toplevel(self.root)
        error_window.title("Error")
        tk.Label(error_window, text=message, wraplength=300, justify="left").grid(row=0, column=0, pady=5, padx=5, sticky="ew")
        ttk.Button(error_window, text="OK", command=error_window.destroy).grid(row=1, column=0, pady=2, sticky="ew")

    def show_help(self):
        help_window = tk.Toplevel(self.root)
        help_window.title("Help")
        tk.Label(help_window, text="Stock Analysis (Graham Defensive) v1.0\n\nUsage:\n- Use 'Analyze Stocks' to perform informational analysis on any ticker (manual or favorite lists), showing all 7 metrics if data is valid\n- Run NYSE/NASDAQ screening to filter strong stocks (first 5 Graham criteria, >= 4/5) with valid data only, excluding stocks with incomplete, nil, or irregular data\n\nContact: support@example.com", wraplength=300, justify="left").grid(row=0, column=0, pady=5, padx=5, sticky="ew")
        ttk.Button(help_window, text="Close", command=help_window.destroy).grid(row=1, column=0, pady=2, sticky="ew")

    def clear_cache(self):
        try:
            for file in ['nyse_tickers.pkl', 'nasdaq_tickers.pkl']:
                file_path = os.path.join(BASE_DIR, file)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logging.info(f"Removed cached ticker file: {file_path}")
            messagebox.showinfo("Cache Cleared", "API, NYSE, and NASDAQ ticker caches have been cleared.")
        except Exception as e:
            logging.error(f"Error clearing cache: {str(e)}")
            messagebox.showerror("Error", f"Failed to clear cache: {str(e)}")

    def on_closing(self):
        self.cancel_event.set()
        self.task_queue.put(None)  # Signal asyncio thread to stop
        self.asyncio_thread.join(timeout=2)
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = GrahamScreeningApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()