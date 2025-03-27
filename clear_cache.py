import sqlite3

conn = sqlite3.connect(r"C:\Users\ralst\OneDrive\Public\stock_analysis\data\api_cache.db")
cursor = conn.cursor()
cursor.execute("DELETE FROM stocks")
conn.commit()
conn.close()
print("Cleared the stocks table.")