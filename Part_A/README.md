Group: 46

# Machine Learning Project: Part A (Data Collection, Sensing and Dataset Creation)

**Task:** binary classification of `RainTomorrow`, i.e. will at least **1.0 mm** of precipitation fall tomorrow?
**Data source:** [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) (ERA5 reanalysis).
**Locations:** London, New York, Sydney, Berlin, Tokyo, Vancouver.
**Study period:** 2010-01-01 to 2026-10-31 (daily data, about 6,148 days per city).

---

## Project structure

```
Group_46/
├── README.md                     # this file
├── requirements.txt              # packages + exact versions
├── data_collection.py            # API collection loop -> data/raw/*.json
├── sensing.py                    # raw JSON -> data/raw/sensed_data.csv (no API calls)
├── preprocessing.py              # cleaning, imputation, features, target -> final_dataset.csv
├── dataset_creation.ipynb        # full analysis (EDA, quality, preparation) with saved outputs
├── verify_requirements.py        # automatic self-check of the assignment requirements (47 checks)
├── REQUIREMENTS_CHECKLIST.md     # every requirement of the instructions -> where it is implemented
├── CONCEPTS_GUIDE.md             # course concepts (lectures 1-5) and how each appears in this project
├── report/
│   ├── Group_46_Part_A_report.pdf   # submitted report (max 10 pages)
│   └── Group_46_Part_A_report.docx  # editable source of the report
├── figures/                      # every figure used in the report (created by the notebook)
└── data/
    ├── raw/
    │   ├── <city>_<year>.json    # API responses, exactly as received (one per city per year)
    │   └── sensed_data.csv       # union of all raw files + location + date, de-duplicated
    └── processed/
        └── final_dataset.csv     # model-ready table with the target RainTomorrow
```

## What each file does

| File | Role |
|---|---|
| `data_collection.py` | Sends one GET request per (city, calendar year) to `https://archive-api.open-meteo.com/v1/archive` with `latitude, longitude, start_date, end_date, daily, timezone=auto`. Collects the **10 required daily variables** plus 4 extra ones for exploration (`relative_humidity_2m_mean`, `pressure_msl_mean`, `cloud_cover_mean`, `dew_point_2m_mean`). **Rate limiting:** a sliding-window counter tracks both the number of requests and their *weighted* cost (Open-Meteo counts a request with more than 10 variables or more than 14 days as several calls). It waits before going over the per-minute, 5,000-per-hour or 10,000-per-24h limits. **Errors:** on HTTP 429, 5xx or a network error it waits (Retry-After header or exponential backoff) and retries. On HTTP 400 it does not retry blindly: it fixes the request (drops the optional variables) first. **Caching:** a response that is already saved and valid is never downloaded again. |
| `sensing.py` | Reads every local `data/raw/*.json` file (it makes **no** API calls), turns each `daily` block into rows, adds `location` and `date`, keeps the common study period, removes duplicates by `location + date`, sorts, and saves `data/raw/sensed_data.csv`. `json_to_frame()` will be reused in Part B to read the raw test JSON files. |
| `preprocessing.py` | The data-preparation pipeline (same code for train and the future test set): (1) replaces impossible values with NaN; (2) completes each city to a continuous daily calendar; (3) creates `RainTomorrow` per city from the **raw** `precipitation_sum` of the next day (1 if ≥ 1.0 mm); (4) imputes per city (time interpolation, then a city × month median); (5) derives the features; (6) drops the last day of every city and keeps only features that use today and past days. Writes `data/processed/final_dataset.csv`. |
| `verify_requirements.py` | Runs 47 automatic checks of the requirements: API params, rate limiter, 429/5xx retry, 400 fix, cache (with a fake HTTP session, no API calls), no API calls in sensing, an independent re-computation of `RainTomorrow`, no next-day leakage, reproducibility, files and report format. See `REQUIREMENTS_CHECKLIST.md`. |
| `dataset_creation.ipynb` | Every analysis in the report: variable description, prior probabilities, class balance, relationships, suspected predictors, removable variables, missing and invalid data checks, discretization and feature derivation. Saves all figures to `figures/`. Outputs are stored in the notebook. |

## Final dataset (`data/processed/final_dataset.csv`)

`location`, `date` + 19 features + `RainTomorrow`:

* **Kept:** `temperature_2m_max`, `temperature_2m_min`, `temperature_2m_mean`, `precipitation_sum`, `precipitation_hours`, `wind_speed_10m_max`
* **Derived:** `temp_range`, `apparent_diff`, `wind_dir_sin`, `wind_dir_cos`, `doy_sin`, `doy_cos`, `rain_today`, `snow_water`, `precip_intensity` (ordinal 0–3), `precip_lag1`, `precip_3d`, `temp_mean_diff1`, `wc_group` (7 weather groups)
* **Removed (with justification in the report):** `rain_sum` (duplicate of precipitation), `apparent_temperature_mean` (replaced by `apparent_diff`), raw `wind_direction_10m_dominant` (circular, replaced by sin/cos), raw `weather_code` (replaced by `wc_group`). The 4 extra variables are **not** used as features because the Part B test JSON may contain only the 10 required variables.

## How to run (reproduce everything)

```bash
python -m pip install -r requirements.txt

python data_collection.py        # 1) download raw JSON (skips files that already exist)
python sensing.py                # 2) data/raw/sensed_data.csv + data/processed/final_dataset.csv
python preprocessing.py          # 3) (optional) re-create final_dataset.csv only
jupyter nbconvert --to notebook --execute --inplace dataset_creation.ipynb   # 4) analysis + figures
```

Step 1 is only needed if `data/raw/*.json` is missing: the submitted raw files are enough to reproduce steps 2–4 exactly.
The notebook also re-creates `sensed_data.csv` and `final_dataset.csv` itself, so opening it and choosing *Run All* is enough.

**Google Colab:** upload the whole folder (or clone it), `%cd` into it, run `!pip install -r requirements.txt`, then *Run All*.

## Reproducibility notes

* Fixed collection period (`END_DATE = 2026-10-31`). ERA5 has a delay of about 5 days, so the period is final and will not change.
* No randomness in the pipeline (the only sampling is `random_state=0` for one scatter plot).
* Raw files are stored exactly as returned by the API.
* `data/raw` holds 142 JSON files. An earlier collection run (before the weighted rate limiter was added) also downloaded 2000–2009 for London, New York, Sydney and Berlin, with 10 more optional variables. Those files are kept exactly as received, but `sensing.py` keeps only the common study period (2010+) and the 14 collected variables, so all cities are treated the same way.
* Outputs were produced with the exact versions in `requirements.txt`.
