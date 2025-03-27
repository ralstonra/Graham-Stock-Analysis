import yfinance as yf

def get_treasury_yield():
    """
    Fetches the most recent 10-year Treasury yield from Yahoo Finance.
    
    Returns:
        float: The 10-year Treasury yield in percentage terms.
    
    Raises:
        ValueError: If no data is available for the ticker '^TNX'.
    """
    tnx = yf.Ticker("^TNX")
    hist = tnx.history(period="1d")
    if hist.empty:
        raise ValueError("No data available for ^TNX")
    yield_value = hist['Close'].iloc[-1]
    return yield_value

if __name__ == "__main__":
    try:
        yield_value = get_treasury_yield()
        print(f"10-Year Treasury Yield: {yield_value:.2f}%")
    except ValueError as e:
        print(e)