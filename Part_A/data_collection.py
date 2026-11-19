"""data_collection.py - Collect daily weather data from the Open-Meteo Historical Weather API.

Run:
    python data_collection.py

What it does
------------
* Downloads daily weather for 6 cities (London, New York, Sydney, Berlin, Tokyo, Vancouver).
* Splits the period into one request per (location, calendar year), so that every
  request is small and a failed request only needs to repeat one year.
* Saves every response EXACTLY as received to data/raw/<location>_<year>.json.
* Caching: a file that already exists (and is a valid response) is NOT downloaded again.
* Rate limiting: counts requests in sliding windows and waits before exceeding
  5,000 requests/hour or 10,000 requests/24h (plus a soft per-minute limit).
* Errors: HTTP 429 / 5xx / network errors -> wait (exponential backoff) and retry.
          HTTP 400 -> the request itself is wrong; it is NOT retried as-is.
          If the 400 is caused by an optional (extra) variable, the extra
          variables are dropped and the request is rebuilt (= the request is fixed).

No data processing happens here - see sensing.py.
"""

from __future__ import annotations

import json
import time
from collections import deque
from datetime import date
from pathlib import Path

import requests

# ----------------------------------------------------------------------------- config
ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"

# Open-Meteo counts "API calls" with weights: a request with more than 10 variables or
# more than 14 days counts as several calls (weight = max(1, n_vars/10) * max(1, n_days/14)).
# We therefore count BOTH the number of HTTP requests and their weighted cost, and wait
# before either one would exceed the limit (with a small safety margin).
HOURLY_LIMIT = 5_000          # API fair-use limit per hour
DAILY_LIMIT = 10_000          # API fair-use limit per 24 hours
MINUTE_LIMIT = 600            # API fair-use limit per minute
SAFETY = 0.9                  # use only 90% of each limit
MAX_RETRIES = 6
TIMEOUT_SEC = 60

START_YEAR = 2010
END_DATE = date(2026, 10, 31)  # fixed for reproducibility (ERA5 has ~5 days delay)

RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"

LOCATIONS = {
    "London": (51.5074, -0.1278),
    "New York": (40.7128, -74.0060),
    "Sydney": (-33.8688, 151.2093),
    "Berlin": (52.5200, 13.4050),
    "Tokyo": (35.6762, 139.6503),
    "Vancouver": (49.2827, -123.1207),
}

# The 10 variables required by the assignment (the Part B test set will contain these).
REQUIRED_DAILY = [
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_mean",
    "precipitation_sum",
    "rain_sum",
    "precipitation_hours",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
]

# Extra variables, collected only for exploration (they may be missing from the Part B
# test set, so the final model features are built from the 10 required variables only).
EXTRA_DAILY = [
    "relative_humidity_2m_mean",
    "pressure_msl_mean",
    "cloud_cover_mean",
    "dew_point_2m_mean",
]


# ----------------------------------------------------------------------------- rate limiter
def request_weight(params: dict) -> float:
    n_vars = len(params["daily"].split(","))
    n_days = (date.fromisoformat(params["end_date"]) - date.fromisoformat(params["start_date"])).days + 1
    return max(1.0, n_vars / 10) * max(1.0, n_days / 14)


class RateLimiter:
    """Sliding windows (minute / hour / 24h). Blocks before a request would break a limit,
    counting both raw requests and their weighted cost."""

    SPANS = {"minute": (60, MINUTE_LIMIT), "hour": (3600, HOURLY_LIMIT), "day": (86400, DAILY_LIMIT)}

    def __init__(self) -> None:
        self.log: deque[tuple[float, float]] = deque()   # (timestamp, weight)
        self.total = 0
        self.total_weight = 0.0

    def wait(self, weight: float) -> None:
        while True:
            now = time.time()
            while self.log and now - self.log[0][0] >= 86400:
                self.log.popleft()
            waits = []
            for span, limit in self.SPANS.values():
                recent = [(t, w) for t, w in self.log if now - t < span]
                n_req = len(recent) + 1
                cost = sum(w for _, w in recent) + weight
                if n_req > limit * SAFETY or cost > limit * SAFETY:
                    waits.append(span - (now - recent[0][0]) if recent else 1)
            if not waits:
                return
            sleep_for = max(1.0, min(waits)) + 1
            print(f"  [rate-limit] waiting {sleep_for:.0f}s ...")
            time.sleep(sleep_for)

    def record(self, weight: float) -> None:
        self.log.append((time.time(), weight))
        self.total += 1
        self.total_weight += weight


# ----------------------------------------------------------------------------- helpers
def slug(location: str) -> str:
    return location.lower().replace(" ", "_")


def year_ranges():
    for year in range(START_YEAR, END_DATE.year + 1):
        start = date(year, 1, 1)
        end = min(date(year, 12, 31), END_DATE)
        yield year, start, end


def is_valid_cached(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return "daily" in payload and not payload.get("error")
    except (json.JSONDecodeError, OSError):
        return False


def fetch(session: requests.Session, limiter: RateLimiter, params: dict) -> requests.Response:
    """GET with retries for 429 / 5xx / network errors. Returns the final response."""
    weight = request_weight(params)
    for attempt in range(1, MAX_RETRIES + 1):
        limiter.wait(weight)
        try:
            resp = session.get(ENDPOINT, params=params, timeout=TIMEOUT_SEC)
        except requests.RequestException as exc:
            limiter.record(weight)
            backoff = min(2 ** attempt, 120)
            print(f"  network error ({exc.__class__.__name__}), retry {attempt} in {backoff}s")
            time.sleep(backoff)
            continue
        limiter.record(weight)

        if resp.status_code == 200:
            return resp
        if resp.status_code == 429 or 500 <= resp.status_code < 600:
            retry_after = resp.headers.get("Retry-After")
            backoff = int(retry_after) if retry_after and retry_after.isdigit() else min(2 ** attempt * 5, 600)
            print(f"  HTTP {resp.status_code}, retry {attempt} in {backoff}s")
            time.sleep(backoff)
            continue
        return resp  # 400 or other client error: do not retry blindly
    raise RuntimeError(f"Giving up after {MAX_RETRIES} attempts: {params}")


def build_params(lat: float, lon: float, start: date, end: date, variables: list[str]) -> dict:
    return {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(variables),
        "timezone": "auto",
    }


# ----------------------------------------------------------------------------- main loop
def collect() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    limiter = RateLimiter()
    variables = REQUIRED_DAILY + EXTRA_DAILY
    downloaded = skipped = 0

    with requests.Session() as session:
        for location, (lat, lon) in LOCATIONS.items():
            for year, start, end in year_ranges():
                out = RAW_DIR / f"{slug(location)}_{year}.json"
                if is_valid_cached(out):
                    skipped += 1
                    continue

                resp = fetch(session, limiter, build_params(lat, lon, start, end, variables))

                if resp.status_code == 400:
                    reason = resp.json().get("reason", resp.text[:200])
                    print(f"  HTTP 400 for {location} {year}: {reason}")
                    if variables != REQUIRED_DAILY:
                        # Fix the request: drop the optional variables and try again.
                        print("  -> fixing request: removing extra variables, keeping the 10 required")
                        variables = list(REQUIRED_DAILY)
                        resp = fetch(session, limiter, build_params(lat, lon, start, end, variables))
                    if resp.status_code != 200:
                        raise RuntimeError(f"Request still invalid ({resp.status_code}): {resp.text[:300]}")
                elif resp.status_code != 200:
                    raise RuntimeError(f"Unexpected HTTP {resp.status_code}: {resp.text[:300]}")

                payload = resp.json()
                if payload.get("error") or "daily" not in payload:
                    raise RuntimeError(f"Invalid payload for {location} {year}: {payload}")

                # Save the response exactly as received (no processing).
                out.write_text(resp.text, encoding="utf-8")
                downloaded += 1
                print(f"Saved {out.name}  ({len(payload['daily']['time'])} days)")
                time.sleep(0.2)

    print(f"\nDone. downloaded={downloaded}, already cached={skipped}, "
          f"HTTP requests sent={limiter.total} (weighted cost {limiter.total_weight:.0f}), "
          f"raw dir={RAW_DIR}")


if __name__ == "__main__":
    collect()
