# config.py (patched for #3)
import os
import logging
import threading
import hashlib
import logging.handlers
from decouple import config, UndefinedValueError
from tkinter import filedialog, Tk, messagebox  # Added messagebox for user feedback

# Temporary Tk root for file dialog (hidden)
root = Tk()
root.withdraw()

# Base directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Local user data directory
USER_DATA_DIR = os.path.join(os.path.expanduser("~"), "Graham Stock Analysis")
fallback_used = False
try:
    if not os.path.exists(USER_DATA_DIR):
        if not os.access(os.path.expanduser("~"), os.W_OK):
            raise PermissionError(f"No write permission for {os.path.expanduser('~')}")
        os.makedirs(USER_DATA_DIR)
except (OSError, PermissionError) as e:
    logging.error(f"Failed to create USER_DATA_DIR {USER_DATA_DIR}: {str(e)}")
    USER_DATA_DIR = os.path.join(os.environ.get("TEMP", "/tmp"), "Graham Stock Analysis")
    fallback_used = True
    if not os.path.exists(USER_DATA_DIR):
        try:
            os.makedirs(USER_DATA_DIR)
        except (OSError, PermissionError) as fallback_e:
            logging.critical(f"Failed to create fallback USER_DATA_DIR {USER_DATA_DIR}: {str(fallback_e)}")
            # Graceful degradation: Warn user and suggest manual fix
            root.after(0, lambda: messagebox.showerror(
                "Permissions Error",
                "Cannot create data directory in home or TEMP. App will run in limited mode (no persistent caching). Check OS permissions."
            ))
            # Set to None to indicate no writable dir - handle in file ops below
            USER_DATA_DIR = None

# Validate directory writability first (new check)
def validate_dir_writable(dir_path):
    if dir_path is None:
        return False
    test_file = os.path.join(dir_path, ".writetest")
    try:
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        return True
    except (OSError, PermissionError) as e:
        logging.error(f"Cannot write to directory {dir_path}: {str(e)}")
        return False

# If fallback was used or initial creation failed, re-validate the dir
if fallback_used or USER_DATA_DIR is None or not validate_dir_writable(USER_DATA_DIR):
    if USER_DATA_DIR is not None:
        logging.warning(f"Directory {USER_DATA_DIR} is not writable. Falling back to in-memory mode.")
    USER_DATA_DIR = None  # Disable file-based ops
    # Adjust paths to None or memory-based alternatives if needed (e.g., for logs, use console only)

# Associated file paths (handle if dir is None)
if USER_DATA_DIR is not None:
    CACHE_DB = os.path.join(USER_DATA_DIR, "api_cache.db")
    NYSE_LIST_FILE = os.path.join(USER_DATA_DIR, "otherlisted.txt")
    NASDAQ_LIST_FILE = os.path.join(USER_DATA_DIR, "nasdaqlisted.txt")
    FAVORITES_FILE = os.path.join(USER_DATA_DIR, "stock_favorites.json")
    graham_log_file = os.path.join(USER_DATA_DIR, "graham.log")
else:
    # In-memory mode: Disable file-based features
    CACHE_DB = None
    NYSE_LIST_FILE = None
    NASDAQ_LIST_FILE = None
    FAVORITES_FILE = None
    graham_log_file = None
    logging.warning("Running in in-memory mode: No persistent storage available. Features like caching and favorites are disabled.")

# Validate file writability (skip if in-memory)
def validate_file_writable(file_path):
    if file_path is None:  # In-memory mode
        return True  # Pretend it's ok, but ops will need handling elsewhere
    try:
        with open(file_path, 'a') as f:
            pass
        return True
    except (OSError, PermissionError) as e:
        logging.error(f"Cannot write to {file_path}: {str(e)}")
        return False

# Only validate if not in-memory
if USER_DATA_DIR is not None:
    file_paths = [CACHE_DB, NYSE_LIST_FILE, NASDAQ_LIST_FILE, FAVORITES_FILE, graham_log_file]
    unwritable_files = [fp for fp in file_paths if not validate_file_writable(fp)]
    if unwritable_files:
        error_msg = f"Cannot write to files: {', '.join(unwritable_files)}. Switching to limited mode."
        logging.error(error_msg)
        root.after(0, lambda: messagebox.showwarning("Permissions Warning", error_msg))
        # Don't raise; continue in limited mode (e.g., skip caching)

# API keys and constants (patched for graceful handling)
API_DISABLED = False  # New flag for offline mode
try:
    FMP_API_KEYS = config('FMP_API_KEYS', cast=lambda v: [s.strip() for s in v.split(',')])
    logging.info(f"Loaded FMP_API_KEYS: {FMP_API_KEYS}")
    FRED_API_KEY = config('FRED_API_KEY')
except UndefinedValueError as e:
    logging.error(f"Missing .env file or variables: {str(e)}")
    env_path = filedialog.askopenfilename(title="Locate .env file", filetypes=[("Env files", "*.env")])
    if env_path and os.path.exists(env_path):
        from decouple import Config, RepositoryEnv
        config = Config(RepositoryEnv(env_path))
        FMP_API_KEYS = config('FMP_API_KEYS', cast=lambda v: [s.strip() for s in v.split(',')])
        FRED_API_KEY = config('FRED_API_KEY')
    else:
        FMP_API_KEYS = []  # Fallback to empty list
        FRED_API_KEY = None
        logging.warning("No .env file provided. Running in offline mode (cache-only, no new API fetches).")
        root.after(0, lambda: messagebox.showwarning(
            "API Warning",
            "API keys not configured. App will run in offline mode using cached data only. Some features (e.g., fresh stock fetches) are disabled."
        ))
        API_DISABLED = True  # Set flag to disable API calls

# Additional key validation (ensure non-empty and string-like)
if not API_DISABLED:
    if not FMP_API_KEYS or all(not isinstance(k, str) or not k.strip() for k in FMP_API_KEYS):
        logging.warning("Invalid or empty FMP_API_KEYS. Disabling API.")
        API_DISABLED = True
    if not isinstance(FRED_API_KEY, str) or not FRED_API_KEY.strip():
        logging.warning("Invalid or empty FRED_API_KEY. Using default yields where possible.")
        # Don't fully disable, as FRED is secondary

USE_FREE_API_KEY = config('USE_FREE_API_KEY', default=False, cast=bool)
CACHE_EXPIRY = 365 * 24 * 60 * 60
FAVORITES_LOCK = threading.Lock()
MAX_CALLS_PER_MINUTE_PAID = 750
MAX_CALLS_PER_MINUTE_FREE = 5

class FileHashError(Exception):
    pass

# Single logger setup with cleanup (handle if no log file)
if graham_log_file is not None:
    graham_handler = logging.handlers.RotatingFileHandler(
        graham_log_file, maxBytes=10*1024*1024, backupCount=3
    )
    graham_handler.setLevel(logging.INFO)
    graham_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    graham_handler.setFormatter(graham_formatter)
    graham_logger = logging.getLogger('graham')
    graham_logger.addHandler(graham_handler)
    graham_logger.setLevel(logging.DEBUG)
    graham_logger.propagate = False

    # Cleanup old log backups
    import glob
    for log_file in glob.glob(f"{USER_DATA_DIR}/graham.log.*"):
        try:
            os.remove(log_file)
            graham_logger.info(f"Cleaned up old log backup: {log_file}")
        except OSError as e:
            graham_logger.error(f"Failed to remove {log_file}: {str(e)}")
else:
    # No file logging if in-memory
    graham_logger = logging.getLogger('graham')
    graham_logger.setLevel(logging.DEBUG)
    graham_logger.propagate = False

# Console handler (always add)
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
graham_logger.addHandler(console)

def get_file_hash(file_path: str) -> str:
    try:
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except FileNotFoundError:
        raise FileHashError(f"File not found: {file_path}")
    except Exception as e:
        raise FileHashError(f"Error computing hash for {file_path}: {str(e)}")

root.destroy()  # Clean up Tk root