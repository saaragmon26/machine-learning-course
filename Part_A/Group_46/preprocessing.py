"""preprocessing.py - Data preparation pipeline (cleaning, imputation, features, target).

Run:
    python preprocessing.py          # reads data/raw/sensed_data.csv -> data/processed/final_dataset.csv

The same functions are used in dataset_creation.ipynb (where every step is
explained) and will be reused in Part B to turn raw test JSON files into
model features (`prepare(..., with_target=False)`).

Main rules
----------
* RainTomorrow(day d) = 1 if precipitation_sum(day d+1) >= 1.0 mm, else 0
  (computed inside each location after sorting by date, from the RAW value of
  the next day - never from an imputed value).
* No feature uses information from the next day (only today and past days).
* The last day of every location (no "tomorrow") is removed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SENSED_PATH = ROOT / "data" / "raw" / "sensed_data.csv"
FINAL_PATH = ROOT / "data" / "processed" / "final_dataset.csv"

RAIN_THRESHOLD_MM = 1.0

REQUIRED_DAILY = [
    "weather_code", "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "apparent_temperature_mean", "precipitation_sum", "rain_sum", "precipitation_hours",
    "wind_speed_10m_max", "wind_direction_10m_dominant",
]
NUMERIC_REQUIRED = [c for c in REQUIRED_DAILY if c != "weather_code"]

# WMO weather codes that Open-Meteo can return.
VALID_WMO = {0, 1, 2, 3, 45, 48, 51, 53, 55, 56, 57, 61, 63, 65, 66, 67,
             71, 73, 75, 77, 80, 81, 82, 85, 86, 95, 96, 99}

WMO_GROUP = {
    0: "clear", 1: "clear",
    2: "cloudy", 3: "cloudy",
    45: "fog", 48: "fog",
    51: "drizzle", 53: "drizzle", 55: "drizzle", 56: "drizzle", 57: "drizzle",
    61: "rain", 63: "rain", 65: "rain", 66: "rain", 67: "rain",
    80: "rain", 81: "rain", 82: "rain",
    71: "snow", 73: "snow", 75: "snow", 77: "snow", 85: "snow", 86: "snow",
    95: "thunder", 96: "thunder", 99: "thunder",
}

# Discretisation of daily precipitation (mm) - common meteorological intensity classes.
PRECIP_BINS = [-np.inf, 0.1, 2.5, 10.0, np.inf]
PRECIP_LABELS = [0, 1, 2, 3]  # 0 = dry, 1 = light, 2 = moderate, 3 = heavy (ordinal!)

FEATURES = [
    # kept as is
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "precipitation_sum", "precipitation_hours", "wind_speed_10m_max",
    # derived
    "temp_range", "apparent_diff", "wind_dir_sin", "wind_dir_cos",
    "doy_sin", "doy_cos", "rain_today", "snow_water", "precip_intensity",
    "precip_lag1", "precip_3d", "temp_mean_diff1", "wc_group",
]
TARGET = "RainTomorrow"


# ----------------------------------------------------------------------------- quality checks
def validity_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean masks of physically impossible / inconsistent values."""
    tol = 0.05
    return {
        "min_temp > max_temp": df["temperature_2m_min"] > df["temperature_2m_max"] + tol,
        "mean_temp outside [min,max]": (df["temperature_2m_mean"] < df["temperature_2m_min"] - tol)
                                       | (df["temperature_2m_mean"] > df["temperature_2m_max"] + tol),
        "temperature outside [-60,60]": df[["temperature_2m_max", "temperature_2m_min",
                                            "temperature_2m_mean"]].abs().gt(60).any(axis=1),
        "apparent temp outside [-80,70]": ~df["apparent_temperature_mean"].between(-80, 70)
                                          & df["apparent_temperature_mean"].notna(),
        "precipitation < 0": df["precipitation_sum"] < 0,
        "rain < 0": df["rain_sum"] < 0,
        "rain > precipitation": df["rain_sum"] > df["precipitation_sum"] + tol,
        "precip_hours outside [0,24]": ~df["precipitation_hours"].between(0, 24)
                                       & df["precipitation_hours"].notna(),
        "wind speed outside [0,250]": ~df["wind_speed_10m_max"].between(0, 250)
                                      & df["wind_speed_10m_max"].notna(),
        "wind dir outside [0,360]": ~df["wind_direction_10m_dominant"].between(0, 360)
                                    & df["wind_direction_10m_dominant"].notna(),
        "unknown weather_code": ~df["weather_code"].isin(VALID_WMO) & df["weather_code"].notna(),
    }


# columns to blank when a rule fires
RULE_COLUMNS = {
    "min_temp > max_temp": ["temperature_2m_min", "temperature_2m_max"],
    "mean_temp outside [min,max]": ["temperature_2m_mean"],
    "temperature outside [-60,60]": ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean"],
    "apparent temp outside [-80,70]": ["apparent_temperature_mean"],
    "precipitation < 0": ["precipitation_sum"],
    "rain < 0": ["rain_sum"],
    "rain > precipitation": ["rain_sum"],
    "precip_hours outside [0,24]": ["precipitation_hours"],
    "wind speed outside [0,250]": ["wind_speed_10m_max"],
    "wind dir outside [0,360]": ["wind_direction_10m_dominant"],
    "unknown weather_code": ["weather_code"],
}


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    masks = validity_masks(df)
    return pd.DataFrame({"rule": list(masks), "rows": [int(m.sum()) for m in masks.values()]})


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Replace impossible values with NaN (they are imputed later like any missing value)."""
    df = df.copy()
    for rule, mask in validity_masks(df).items():
        if mask.any():
            df.loc[mask, RULE_COLUMNS[rule]] = np.nan
    return df


# ----------------------------------------------------------------------------- calendar & target
def complete_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Make every location a continuous daily series (missing days become NaN rows)."""
    out = []
    for loc, g in df.groupby("location", sort=True):
        g = g.set_index("date").sort_index()
        full = pd.date_range(g.index.min(), g.index.max(), freq="D")
        g = g.reindex(full)
        g["row_was_missing"] = ~g.index.isin(df.loc[df["location"] == loc, "date"])
        g["location"] = loc
        out.append(g.rename_axis("date").reset_index())
    return pd.concat(out, ignore_index=True)


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """RainTomorrow from the next day's (raw, cleaned) precipitation inside each location."""
    df = df.sort_values(["location", "date"]).copy()
    nxt = df.groupby("location")["precipitation_sum"].shift(-1)
    df[TARGET] = np.where(nxt.isna(), np.nan, (nxt >= RAIN_THRESHOLD_MM).astype(float))
    return df


# ----------------------------------------------------------------------------- imputation
def impute(df: pd.DataFrame) -> pd.DataFrame:
    """Per location: time interpolation for continuous variables (weather is smooth in time),
    then a location x month median as fallback. weather_code: nearest valid day."""
    df = df.sort_values(["location", "date"]).copy()
    df["_month"] = df["date"].dt.month
    cont = [c for c in NUMERIC_REQUIRED if c in df.columns]

    def _interp(g: pd.DataFrame) -> pd.DataFrame:
        g = g.set_index("date")
        g[cont] = g[cont].interpolate(method="time", limit=3, limit_direction="both")
        g["weather_code"] = g["weather_code"].ffill(limit=1).bfill(limit=1)
        return g.reset_index()

    df = df.groupby("location", group_keys=False)[df.columns].apply(_interp)
    for c in cont + ["weather_code"]:
        if df[c].isna().any():
            med = df.groupby(["location", "_month"])[c].transform("median")
            df[c] = df[c].fillna(med).fillna(df[c].median())
    return df.drop(columns="_month")


# ----------------------------------------------------------------------------- features
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["location", "date"]).copy()
    g = df.groupby("location")

    df["temp_range"] = df["temperature_2m_max"] - df["temperature_2m_min"]
    df["apparent_diff"] = df["apparent_temperature_mean"] - df["temperature_2m_mean"]

    rad = np.deg2rad(df["wind_direction_10m_dominant"])
    df["wind_dir_sin"] = np.sin(rad)          # + = wind from the east
    df["wind_dir_cos"] = np.cos(rad)          # + = wind from the north

    doy = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    df["rain_today"] = (df["precipitation_sum"] >= RAIN_THRESHOLD_MM).astype(int)
    df["snow_water"] = (df["precipitation_sum"] - df["rain_sum"]).clip(lower=0)
    df["precip_intensity"] = pd.cut(df["precipitation_sum"], PRECIP_BINS,
                                    labels=PRECIP_LABELS, right=False).astype(int)

    # past-only features (shift(+1) = yesterday). If yesterday is unknown (first row of a
    # raw file), fall back to "no change" values so the row is still usable.
    df["precip_lag1"] = g["precipitation_sum"].shift(1).fillna(df["precipitation_sum"])
    df["precip_3d"] = g["precipitation_sum"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    df["temp_mean_diff1"] = (df["temperature_2m_mean"] - g["temperature_2m_mean"].shift(1)).fillna(0)

    df["wc_group"] = df["weather_code"].round().map(WMO_GROUP).fillna("cloudy")
    return df


# ----------------------------------------------------------------------------- full pipeline
def prepare(sensed: pd.DataFrame, with_target: bool = True) -> pd.DataFrame:
    df = sensed.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["location", "date"])
    df = clean(df)
    df = complete_calendar(df)
    if with_target:
        df = add_target(df)
    df = impute(df)
    df = add_features(df)

    cols = ["location", "date"] + FEATURES + ([TARGET] if with_target else [])
    if with_target:
        df = df[df[TARGET].notna() & ~df["row_was_missing"]]   # drops last day per location
        df[TARGET] = df[TARGET].astype(int)
    else:
        df = df[~df["row_was_missing"]]
    return df[cols].reset_index(drop=True)


if __name__ == "__main__":
    sensed = pd.read_csv(SENSED_PATH, parse_dates=["date"])
    final = prepare(sensed)
    FINAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(FINAL_PATH, index=False)
    print(f"saved {FINAL_PATH}: {final.shape[0]:,} rows x {final.shape[1]} columns")
    print(final[TARGET].value_counts(normalize=True).round(4))
