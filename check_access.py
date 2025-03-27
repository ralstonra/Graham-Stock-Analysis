import os
from config import DATA_DIR, FAVORITES_FILE

# Ensure the data directory exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Created directory: {DATA_DIR}")

# Check write access to the favorites file
if os.path.exists(FAVORITES_FILE):
    # If the file exists, check if it's writable
    if os.access(FAVORITES_FILE, os.W_OK):
        print(f"Write access to {FAVORITES_FILE}: Yes")
    else:
        print(f"Write access to {FAVORITES_FILE}: No")
else:
    # If the file doesn't exist, check write access to the directory
    if os.access(DATA_DIR, os.W_OK):
        print(f"{FAVORITES_FILE} does not exist yet, but write access to {DATA_DIR}: Yes")
    else:
        print(f"{FAVORITES_FILE} does not exist yet, and write access to {DATA_DIR}: No")