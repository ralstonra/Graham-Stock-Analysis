import sqlite3
import sys
import os
import json
from config import CACHE_DB  # Assuming config.py is in the same directory or properly imported

def display_stock_data(stock_symbol, db_path=None):
    """
    Display all cached data for a specific stock symbol from the SQLite database.

    Args:
        stock_symbol (str): The stock symbol to retrieve data for (e.g., 'AAPL').
        db_path (str, optional): The path to the SQLite database file.
                                 If not provided, uses CACHE_DB from config.py.
    """
    # Set default database path if not provided
    if db_path is None:
        db_path = CACHE_DB

    # Ensure the database file exists
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        return

    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Query the stocks table for the given stock symbol
        cursor.execute("SELECT * FROM stocks WHERE ticker = ?", (stock_symbol,))
        row = cursor.fetchone()

        if row:
            # Get column names from the cursor description
            column_names = [description[0] for description in cursor.description]

            # Create a dictionary of the row data
            stock_data = dict(zip(column_names, row))

            # Display each field with appropriate formatting
            print(f"\nCached Data for Stock: {stock_symbol}")
            print("-" * 40)
            for col_name, col_value in stock_data.items():
                if col_value is None:
                    print(f"{col_name}: None")
                elif col_name in ['roe', 'rotc', 'eps', 'dividend']:
                    # Handle comma-separated strings by converting to a list of floats
                    values = [float(x) if x.strip() else 0.0 for x in col_value.split(",")] if col_value else []
                    print(f"{col_name}: {values}")
                elif col_name == 'balance_data':
                    # Load and display JSON string
                    balance_data = json.loads(col_value) if col_value else []
                    print(f"{col_name}: {json.dumps(balance_data, indent=2)}")
                else:
                    print(f"{col_name}: {col_value}")
            print("-" * 40)
        else:
            print(f"No data found for stock symbol: {stock_symbol}")

        # Close the database connection
        conn.close()

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON data: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    # Check if the stock symbol is provided as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: python display_stock_data.py <stock_symbol> [db_path]")
        print("  stock_symbol: The stock symbol to retrieve data for (e.g., 'AAPL').")
        print("  db_path: Optional path to the SQLite database file.")
        print("           If not provided, uses CACHE_DB from config.py.")
        sys.exit(1)

    # Convert stock symbol to uppercase (e.g., 'aapl' -> 'AAPL')
    stock_symbol = sys.argv[1].upper()
    # Get database path from second argument if provided
    db_path = sys.argv[2] if len(sys.argv) > 2 else None

    display_stock_data(stock_symbol, db_path)