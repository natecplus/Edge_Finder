"""One polite way to fetch web data, shared by every adapter.

- Sends a real User-Agent and waits between requests (don't hammer sites).
- Retries a couple of times on network hiccups.
- Optionally caches the raw response on disk, so a backfill that crashes can
  resume without re-downloading anything.
"""
import json
import time
from pathlib import Path

import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (edge-finder personal project; low request rate)",
    "Accept": "application/json, text/html;q=0.9",
}
DEFAULT_DELAY = 3.0   # seconds between live requests

_last_request = 0.0


def _wait(delay: float) -> None:
    global _last_request
    gap = time.monotonic() - _last_request
    if gap < delay:
        time.sleep(delay - gap)
    _last_request = time.monotonic()


def get(url: str, params: dict | None = None, delay: float = DEFAULT_DELAY,
        retries: int = 2, timeout: int = 20) -> requests.Response:
    last_error = None
    for attempt in range(retries + 1):
        _wait(delay)
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code} from {url}")
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last_error = e
            time.sleep(2 * (attempt + 1))
    raise last_error


def get_json(url: str, params: dict | None = None, cache: Path | None = None,
             use_cache: bool = False, delay: float = DEFAULT_DELAY) -> dict:
    """Fetch JSON. Always saves a raw copy to `cache` if given; reads from it
    instead of the network when `use_cache` is True and the file exists."""
    if cache is not None and use_cache and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    data = get(url, params=params, delay=delay).json()
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data), encoding="utf-8")
    return data


def get_text(url: str, params: dict | None = None, cache: Path | None = None,
             use_cache: bool = False, delay: float = DEFAULT_DELAY) -> str:
    if cache is not None and use_cache and cache.exists():
        return cache.read_text(encoding="utf-8")
    text = get(url, params=params, delay=delay).text
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(text, encoding="utf-8")
    return text
