"""sensing.py - Turn the local raw JSON files into one table.

Run:
    python sensing.py

* Reads every data/raw/*.json file (NO calls to the API are made here).
* Converts the "daily" block of each response into rows.
* Adds the columns `location` and `date`.
* Removes duplicates by (location + date) and sorts.
* Saves data/raw/sensed_data.csv
* Then creates RainTomorrow (via preprocessing.py) and saves data/processed/final_dataset.csv

The function `json_to_frame` is also used in Part B to turn a raw test JSON
into the same table format.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
SENSED_PATH = RAW_DIR / "sensed_data.csv"

# Same coordinates as in data_collection.py - used to recognise the city of a raw file.
LOCATIONS = {
    "London": (51.5074, -0.1278),
    "New York": (40.7128, -74.0060),
    "Sydney": (-33.8688, 151.2093),
    "Berlin": (52.5200, 13.4050),
    "Tokyo": (35.6762, 139.6503),
    "Vancouver": (49.2827, -123.1207),
}
SLUG_TO_NAME = {name.lower().replace(" ", "_"): name for name in LOCATIONS}

# Common period for all cities (a first collection attempt also downloaded London 2000-2009;
# those raw files are kept untouched but are outside the study period).
START_DATE = "2010-01-01"

REQUIRED_DAILY = [
    "weather_code", "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "apparent_temperature_mean", "precipitation_sum", "rain_sum", "precipitation_hours",
    "wind_speed_10m_max", "wind_direction_10m_dominant",
]
EXTRA_DAILY = ["relative_humidity_2m_mean", "pressure_msl_mean", "cloud_cover_mean", "dew_point_2m_mean"]


def location_from_coords(lat: float, lon: float) -> str:
    """Nearest known city to the grid point returned by the API (API snaps to its grid)."""
    return min(LOCATIONS, key=lambda n: math.dist((lat, lon), LOCATIONS[n]))


def location_from_filename(path: Path) -> str | None:
    stem = path.stem
    for slug, name in SLUG_TO_NAME.items():
        if stem.startswith(slug):
            return name
    return None


def json_to_frame(payload: dict, location: str | None = None) -> pd.DataFrame:
    """One raw API response -> DataFrame with `location`, `date` and the daily variables."""
    daily = payload["daily"]
    df = pd.DataFrame(daily).rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    if location is None:
        location = location_from_coords(payload["latitude"], payload["longitude"])
    df.insert(0, "location", location)
    return df


def load_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    files = sorted(p for p in raw_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files in {raw_dir}. Run data_collection.py first.")
    frames = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "daily" not in payload:
            print(f"skip {path.name}: no 'daily' block")
            continue
        frames.append(json_to_frame(payload, location_from_filename(path)))
    return pd.concat(frames, ignore_index=True)


def build_sensed(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    df = load_raw(raw_dir)
    keep = ["location", "date"] + REQUIRED_DAILY + [c for c in EXTRA_DAILY if c in df.columns]
    df = df[keep]
    df = df[df["date"] >= START_DATE]
    n_before = len(df)
    df = (df.drop_duplicates(subset=["location", "date"], keep="first")
            .sort_values(["location", "date"])
            .reset_index(drop=True))
    print(f"rows read: {n_before:,} | duplicates removed: {n_before - len(df):,} | rows: {len(df):,}")
    return df


if __name__ == "__main__":
    sensed = build_sensed()
    sensed.to_csv(SENSED_PATH, index=False)
    print(f"saved {SENSED_PATH} with columns: {list(sensed.columns)}")

    # Next step of the data contract: RainTomorrow per location (next day's precipitation_sum
    # >= 1.0 mm), no next-day features, last day of every location removed.
    # The logic lives in preprocessing.py so that the exact same code runs on the test set.
    import preprocessing
    final = preprocessing.prepare(sensed)
    preprocessing.FINAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(preprocessing.FINAL_PATH, index=False)
    print(f"saved {preprocessing.FINAL_PATH}: {final.shape[0]:,} rows x {final.shape[1]} columns")
