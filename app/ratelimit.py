from fastapi import HTTPException
from time import time
from typing import Dict, Tuple

# Simple in-memory token bucket (per-process). Replace with Redis in prod.
_buckets: Dict[Tuple[str, str], Tuple[float, float]] = {}


def allow(user_id: int, key: str, rps: int, burst: int) -> bool:
    now = time()
    rate = rps
    capacity = burst
    k = (user_id, key)
    tokens, last = _buckets.get(k, (capacity, now))
    tokens = min(capacity, tokens + (now - last) * rate)
    if tokens < 1:
        _buckets[k] = (tokens, now)
        return False
    _buckets[k] = (tokens - 1, now)
    return True


def enforce(user_id: int, key: str, rps: int, burst: int):
    if not allow(user_id, key, rps, burst):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
