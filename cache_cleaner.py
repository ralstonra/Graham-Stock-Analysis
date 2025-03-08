import os
import logging

def clear_cache():
    """Clear the API cache database (api_cache.db)."""
    cache_db = "api_cache.db"
    if os.path.exists(cache_db):
        try:
            os.remove(cache_db)
            logging.info(f"Cleared cache database: {cache_db}")
        except Exception as e:
            logging.error(f"Failed to clear cache database {cache_db}: {str(e)}")
    else:
        logging.warning(f"Cache database {cache_db} not found.")