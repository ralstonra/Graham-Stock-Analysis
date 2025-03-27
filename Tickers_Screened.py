import sqlite3
from config import CACHE_DB  # Replace with your actual database path if not using config.py

def check_screening_progress(exchange=None):
    """
    Check how many tickers were screened and skipped, optionally for a specific exchange.
    
    Args:
        exchange (str): Optional. Filter results by exchange (e.g., 'NYSE', 'NASDAQ').
    """
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(CACHE_DB)
        cursor = conn.cursor()
        
        if exchange:
            # Query for screened tickers for a specific exchange
            cursor.execute(
                "SELECT COUNT(*) FROM screening_progress WHERE exchange = ? AND status = 'completed';",
                (exchange,)
            )
            screened_count = cursor.fetchone()[0]
            
            # Query for skipped tickers for a specific exchange
            cursor.execute(
                "SELECT COUNT(*) FROM screening_progress WHERE exchange = ? AND status = 'failed';",
                (exchange,)
            )
            skipped_count = cursor.fetchone()[0]
            
            print(f"Exchange: {exchange}")
        else:
            # Query for all screened tickers
            cursor.execute("SELECT COUNT(*) FROM screening_progress WHERE status = 'completed';")
            screened_count = cursor.fetchone()[0]
            
            # Query for all skipped tickers
            cursor.execute("SELECT COUNT(*) FROM screening_progress WHERE status = 'failed';")
            skipped_count = cursor.fetchone()[0]

            cursor.execute("SELECT ticker FROM screening_progress WHERE status = 'failed';")
            skipped_tickers = cursor.fetchall()
                    
        # Display the results
        print(f"Screened Tickers: {screened_count}")
        print(f"Skipped Tickers: {skipped_count}")
        print("Skipped Tickers List:", [row[0] for row in skipped_tickers])
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        # Close the connection
        if conn:
            conn.close()

if __name__ == "__main__":
    # Check for all exchanges
    check_screening_progress()
    
    # Optionally, check for a specific exchange (uncomment to use)
    # check_screening_progress(exchange='NYSE')
    # check_screening_progress(exchange='NASDAQ')