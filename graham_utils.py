# graham_utils.py (unchanged, no logger references)
import asyncio
from collections import deque
import logging
import time
from config import (
    MAX_CALLS_PER_MINUTE_PAID, MAX_CALLS_PER_MINUTE_FREE, get_file_hash, FileHashError, CACHE_EXPIRY
)

class AsyncRateLimiter:
    def __init__(self, max_calls, period, on_sleep=None):
        if not isinstance(max_calls, int) or max_calls <= 0:
            raise ValueError(f"max_calls must be a positive integer, got {max_calls}")
        if not isinstance(period, (int, float)) or period <= 0:
            raise ValueError(f"period must be a positive number, got {period}")
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()
        self.semaphore = asyncio.Semaphore(max_calls)
        self.on_sleep = on_sleep
        self.lock = asyncio.Lock()  # Thread-safe lock

    async def acquire(self, cancel_event=None, timeout=60):
        async with self.semaphore:
            async with self.lock:  # Protect deque access
                current_time = time.time()
                # Clear outdated calls (e.g., after clock reset or long runtime)
                while self.calls and (current_time - self.calls[0] > CACHE_EXPIRY):
                    self.calls.popleft()
                    logging.warning("Cleared outdated call timestamp due to clock skew or long runtime")
                while self.calls and current_time - self.calls[0] >= self.period:
                    self.calls.popleft()
                if len(self.calls) >= self.max_calls:
                    sleep_time = self.period - (current_time - self.calls[0])
                    if sleep_time > 0:
                        if self.on_sleep:
                            self.on_sleep(sleep_time)
                        logging.debug(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                        try:
                            await asyncio.wait_for(asyncio.sleep(sleep_time), timeout=timeout)
                            if cancel_event and cancel_event.is_set():
                                logging.info("Rate limiter cancelled")
                                return False
                        except asyncio.TimeoutError:
                            logging.warning("Rate limiter sleep timed out after {timeout} seconds")
                            return False
                self.calls.append(current_time)
        return True

paid_rate_limiter = AsyncRateLimiter(MAX_CALLS_PER_MINUTE_PAID, 60)
free_rate_limiter = AsyncRateLimiter(MAX_CALLS_PER_MINUTE_FREE, 60)