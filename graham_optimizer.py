import tkinter as tk
from tkinter import ttk, messagebox
import asyncio
import logging
import pandas as pd
import sqlite3
import threading
import time
import os
from graham_data import (
    get_stocks_connection, fetch_batch_data, calculate_graham_value, TickerManager,
    NYSE_LIST_FILE, NASDAQ_LIST_FILE, get_stock_data_from_db
)
from config import BASE_DIR, FMP_API_KEYS, CACHE_EXPIRY
import queue
import yfinance as yf

# Define the base directory and log file path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(BASE_DIR, 'graham_optimizer.log')

# Debug the log file path and directory
print(f"Log file path: {log_file}")
if not os.path.exists(BASE_DIR):
    print(f"Base directory does not exist: {BASE_DIR}")
else:
    print(f"Base directory exists: {BASE_DIR}")

# Set up logging with error handling
try:
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
except Exception as e:
    print(f"Error setting up file logging: {e}")
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logging.getLogger().addHandler(console)

class GrahamOptimizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Graham Investment Optimizer ($1000)")
        self.root.geometry("1000x700")

        self.cancel_event = threading.Event()
        self.ticker_manager = TickerManager(NYSE_LIST_FILE, NASDAQ_LIST_FILE)
        self.budget = 1000  # Fixed budget of $1000
        self.analysis_lock = threading.Lock()
        self.ticker_cache = {}
        self.ticker_cache_lock = threading.Lock()

        self.task_queue = queue.Queue()
        self.asyncio_thread = threading.Thread(target=self.run_asyncio_loop, args=(self.task_queue,), daemon=True)
        self.asyncio_thread.start()

        # Main frame layout
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(expand=True, fill="both", padx=5, pady=5)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=3)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Top frame for controls
        self.top_frame = ttk.Frame(self.main_frame)
        self.top_frame.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        self.top_frame.grid_columnconfigure(0, weight=1)

        # Results frame
        self.results_frame = ttk.Frame(self.main_frame)
        self.results_frame.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
        self.results_frame.grid_columnconfigure(0, weight=1)
        self.results_frame.grid_rowconfigure(0, weight=1)

        # Styling
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", font=("Helvetica", 9), background="#f0f0f0")
        style.configure("TButton", font=("Helvetica", 9), padding=2)
        style.configure("TProgressbar", thickness=15)

        self.create_widgets()
        logging.info("Graham Optimizer App initialized")

    def run_asyncio_loop(self, task_queue):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        while True:
            coro = task_queue.get()
            if coro is None:
                break
            try:
                loop.run_until_complete(coro)
            except Exception as e:
                logging.error(f"Error in asyncio loop: {str(e)}", exc_info=True)
        loop.close()

    def create_widgets(self):
        # Top frame controls
        ttk.Label(self.top_frame, text="Graham Investment Optimizer").grid(row=0, column=0, pady=5)
        self.run_button = ttk.Button(self.top_frame, text="Run Optimizer", command=self.run_optimization)
        self.run_button.grid(row=1, column=0, pady=5, sticky="ew")
        ttk.Button(self.top_frame, text="Cancel", command=self.cancel_optimization).grid(row=2, column=0, pady=5, sticky="ew")

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.top_frame, variable=self.progress_var, maximum=100, length=300, mode='determinate')
        self.progress_bar.grid(row=3, column=0, pady=5, sticky="ew")
        self.progress_label = ttk.Label(self.top_frame, text="Progress: 0% (0/0 stocks processed)")
        self.progress_label.grid(row=4, column=0, pady=5)

        self.status_label = ttk.Label(self.top_frame, text="")
        self.status_label.grid(row=5, column=0, pady=5)

        # Results Treeview
        self.tree = ttk.Treeview(self.results_frame, columns=(1, 2, 3, 4, 5, 6, 7, 8), show="tree headings", height=20)
        self.v_scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical", command=self.tree.yview)
        self.h_scrollbar = ttk.Scrollbar(self.results_frame, orient="horizontal", command=self.tree.xview)
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.h_scrollbar.grid(row=1, column=0, sticky="ew")
        self.tree.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)

        self.tree.heading("#0", text="Ticker", command=lambda: self.sort_by_column("#0"))
        self.tree.heading(1, text="Price ($)", command=lambda: self.sort_by_column(1))
        self.tree.heading(2, text="Intrinsic Value ($)", command=lambda: self.sort_by_column(2))
        self.tree.heading(3, text="Discount (%)", command=lambda: self.sort_by_column(3))
        self.tree.heading(4, text="Future Value ($)", command=lambda: self.sort_by_column(4))
        self.tree.heading(5, text="Shares", command=lambda: self.sort_by_column(5))
        self.tree.heading(6, text="Total Future Value ($)", command=lambda: self.sort_by_column(6))
        self.tree.heading(7, text="ROI (%)", command=lambda: self.sort_by_column(7))
        self.tree.heading(8, text="Exchange", command=lambda: self.sort_by_column(8))

        self.tree.column("#0", width=80, anchor="center")
        self.tree.column(1, width=80, anchor="center")
        self.tree.column(2, width=100, anchor="center")
        self.tree.column(3, width=80, anchor="center")
        self.tree.column(4, width=100, anchor="center")
        self.tree.column(5, width=60, anchor="center")
        self.tree.column(6, width=120, anchor="center")
        self.tree.column(7, width=80, anchor="center")
        self.tree.column(8, width=80, anchor="center")

        self.tree.tag_configure('highlight', background='lightgreen')

    def sort_by_column(self, col):
        items = [(self.tree.set(k, col) if col != "#0" else self.tree.item(k, "text"), k) for k in self.tree.get_children('')]
        if not items:
            return
        numeric_columns = [1, 2, 3, 4, 5, 6, 7]
        try:
            if col in numeric_columns or col == "#0":
                cleaned_items = []
                for val, k in items:
                    cleaned_val = val.replace('$', '').replace('%', '')
                    cleaned_items.append((float(cleaned_val) if cleaned_val else float('inf'), k))
                items = cleaned_items
            items.sort(reverse=True)
            for index, (_, k) in enumerate(items):
                self.tree.move(k, '', index)
        except ValueError:
            items.sort()
            for index, (_, k) in enumerate(items):
                self.tree.move(k, '', index)

    def load_qualifying_stocks(self):
        conn, cursor = get_stocks_connection()
        try:
            cursor.execute("""
                SELECT ticker, common_score, exchange 
                FROM graham_qualifiers 
                WHERE common_score >= 5 AND ticker IN (
                    SELECT ticker FROM stocks WHERE available_data_years >= 10
                )
            """)
            stocks = [(row[0], row[1], row[2]) for row in cursor.fetchall()]
            logging.info(f"Loaded {len(stocks)} qualifying stocks from database")
            return stocks
        except sqlite3.Error as e:
            logging.error(f"Database error loading qualifying stocks: {str(e)}")
            return []
        finally:
            conn.close()

    def get_current_price(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            price = info.get('regularMarketPrice', info.get('previousClose', None))
            if price is None:
                logging.warning(f"No price data available for {ticker}")
            return price
        except Exception as e:
            logging.error(f"Error fetching price for {ticker}: {str(e)}")
            return None

    def calculate_future_value(self, eps_ttm, eps_list, years=3):
        if not eps_ttm or eps_ttm <= 0 or not eps_list or len(eps_list) < 2:
            return float('nan')
        if eps_list[0] <= 0 or eps_list[-1] <= 0:
            growth_rate = 0
            logging.debug(f"Non-positive EPS detected; setting growth_rate to 0")
        else:
            growth_rate = min(((eps_list[0] / eps_list[-1]) ** (1 / (len(eps_list) - 1)) - 1), 0.15)
        future_eps = eps_ttm * (1 + growth_rate) ** years
        future_value = calculate_graham_value(future_eps, eps_list)  # Pass eps_list
        logging.debug(f"Future value calc: EPS_TTM={eps_ttm:.2f}, Growth={growth_rate:.2%}, Future_EPS={future_eps:.2f}, Future_Value={future_value:.2f}")
        return future_value

    async def analyze_stocks(self, qualifying_stocks):
        self.progress_var.set(0)
        self.progress_label.config(text=f"Progress: 0% (0/{len(qualifying_stocks)} stocks processed)")
        self.status_label.config(text="Optimizing investment...")
        self.root.update()
        logging.info("Starting stock analysis")

        total_stocks = len(qualifying_stocks)
        processed = 0
        results = []

        for ticker, _, exchange in qualifying_stocks:
            if self.cancel_event.is_set():
                logging.info("Analysis cancelled by user")
                break

            cached_data = get_stock_data_from_db(ticker)
            if not cached_data or not cached_data.get('eps_ttm') or not cached_data.get('eps'):
                logging.warning(f"Skipping {ticker}: Missing cached data or EPS")
                processed += 1
                continue

            current_price = self.get_current_price(ticker)
            if not current_price or not isinstance(current_price, (int, float)):
                logging.warning(f"Skipping {ticker}: No valid current price")
                processed += 1
                continue

            eps_ttm = cached_data['eps_ttm']
            eps_list = cached_data['eps']
            intrinsic_value = calculate_graham_value(eps_ttm, eps_list)
            if not isinstance(intrinsic_value, (int, float)) or pd.isna(intrinsic_value):
                logging.warning(f"Skipping {ticker}: Invalid intrinsic value - {intrinsic_value}")
                processed += 1
                continue

            # Discount calculation: Positive if undervalued, negative if overvalued
            discount = (intrinsic_value - current_price) / intrinsic_value if intrinsic_value > 0 else float('nan')
            future_value = self.calculate_future_value(eps_ttm, eps_list)
            shares = int(self.budget / current_price)
            total_future_value = shares * future_value if not pd.isna(future_value) else float('nan')
            roi = (total_future_value - self.budget) / self.budget if not pd.isna(total_future_value) else float('nan')

            result = {
                'ticker': ticker,
                'exchange': exchange,
                'current_price': current_price,
                'intrinsic_value': intrinsic_value,
                'discount': discount,
                'future_value': future_value,
                'shares': shares,
                'total_future_value': total_future_value,
                'roi': roi
            }
            results.append(result)

            with self.ticker_cache_lock:
                self.ticker_cache[ticker] = result

            processed += 1
            progress = (processed / total_stocks) * 100
            self.root.after(0, lambda p=progress, proc=processed, tot=total_stocks: 
                self.update_progress(p, proc, tot))

        # Populate Treeview
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        valid_results = [r for r in results if not pd.isna(r['roi'])]
        ranked_results = sorted(valid_results, key=lambda x: x['roi'], reverse=True)

        for result in ranked_results:
            # Highlight only if current price <= intrinsic value
            tags = ('highlight',) if result['current_price'] <= result['intrinsic_value'] else ()
            discount_str = f"{result['discount']:.2%}" if not pd.isna(result['discount']) else "N/A"
            self.tree.insert("", "end", text=result['ticker'], values=(
                f"${result['current_price']:.2f}",
                f"${result['intrinsic_value']:.2f}",
                discount_str,
                f"${result['future_value']:.2f}" if not pd.isna(result['future_value']) else "N/A",
                result['shares'],
                f"${result['total_future_value']:.2f}" if not pd.isna(result['total_future_value']) else "N/A",
                f"{result['roi']:.2%}" if not pd.isna(result['roi']) else "N/A",
                result['exchange']
            ), tags=tags)

        self.root.after(0, lambda: self.status_label.config(text="Optimization complete"))
        logging.info(f"Optimization completed: {len(ranked_results)} stocks ranked")

    def update_progress(self, progress, processed, total):
        self.progress_var.set(progress)
        self.progress_label.config(text=f"Progress: {progress:.1f}% ({processed}/{total} stocks processed)")
        self.root.update_idletasks()

    def run_optimization(self):
        if self.analysis_lock.acquire(timeout=5):
            try:
                self.cancel_event.clear()
                qualifying_stocks = self.load_qualifying_stocks()
                if not qualifying_stocks:
                    messagebox.showwarning("No Data", "No qualifying stocks found. Run the Graham Screener first.")
                    logging.warning("No qualifying stocks found")
                    return
                self.task_queue.put(self.analyze_stocks(qualifying_stocks))
            finally:
                self.analysis_lock.release()
        else:
            messagebox.showerror("Error", "Unable to start optimization: Lock timeout.")
            logging.error("Lock timeout during optimization start")

    def cancel_optimization(self):
        if messagebox.askyesno("Confirm Cancel", "Are you sure you want to cancel the optimization?"):
            self.cancel_event.set()
            self.status_label.config(text="Cancelling optimization...")
            logging.info("Optimization cancelled by user")

    def on_closing(self):
        self.cancel_event.set()
        self.task_queue.put(None)
        try:
            self.asyncio_thread.join(timeout=2)
            if self.asyncio_thread.is_alive():
                logging.warning("Asyncio thread did not terminate within timeout.")
        except Exception as e:
            logging.error(f"Error joining asyncio thread: {str(e)}")
        with self.ticker_cache_lock:
            self.ticker_cache.clear()
        self.root.destroy()
        logging.info("Application closed")

if __name__ == "__main__":
    root = tk.Tk()
    app = GrahamOptimizerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()