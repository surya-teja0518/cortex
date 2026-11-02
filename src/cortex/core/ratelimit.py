"""Simple per-key token-bucket limiter. Wrap every external call."""
import threading
import time


class NoLimit:
    """For callers whose provider already rate-limits itself (the LLM router does, per provider).
    Stacking a second limiter on top only slows every call down."""

    def wait(self) -> None:
        pass


class RateLimiter:
    def __init__(self, per_minute: int):
        self.interval = 60.0 / per_minute
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.interval
        if delay:
            time.sleep(delay)
