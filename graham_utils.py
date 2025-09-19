import asyncio
from collections import deque
import logging
import time  # Added import
from config import (
    MAX_CALLS_PER_MINUTE_PAID, MAX_CALLS_PER_MINUTE_FREE, get_file_hash, FileHashError
)

class AsyncRateLimiter:
    def __init__(self, max_calls, period, on_sleep=None):
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()
        self.semaphore = asyncio.Semaphore(max_calls)
        self.on_sleep = on_sleep

    async def acquire(self):
        async with self.semaphore:
            current_time = time.time()
            while self.calls and current_time - self.calls[0] >= self.period:
                self.calls.popleft()
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (current_time - self.calls[0])
                if sleep_time > 0:
                    if self.on_sleep:
                        self.on_sleep(sleep_time)
                    logging.debug(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                    await asyncio.sleep(sleep_time)
            self.calls.append(time.time())

paid_rate_limiter = AsyncRateLimiter(MAX_CALLS_PER_MINUTE_PAID, 60)
free_rate_limiter = AsyncRateLimiter(MAX_CALLS_PER_MINUTE_FREE, 60)