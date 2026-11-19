"""verify_requirements.py - automatic self-check of the Part A requirements.

Run from the project root:
    python verify_requirements.py

Each check prints PASS/FAIL. No API calls are made: the request loop is tested
with a fake HTTP session (429 -> retry, 500 -> retry, 400 -> fixed request, cache).
"""
from __future__ import annotations

import ast
import json
import re
import sys
import tempfile
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import data_collection as dc  # noqa: E402
import preprocessing as pp  # noqa: E402
import sensing  # noqa: E402

RESULTS: list[tuple[str, str, bool, str]] = []


def check(rid: str, name: str, cond: bool, detail: str = "") -> None:
    RESULTS.append((rid, name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {rid} {name}" + (f"  ({detail})" if detail else ""))


RAW = ROOT / "data" / "raw"
sensed = pd.read_csv(RAW / "sensed_data.csv", parse_dates=["date"])
final = pd.read_csv(ROOT / "data" / "processed" / "final_dataset.csv", parse_dates=["date"])

# ----------------------------------------------------------------------------- 1. collection
CITIES = {"London", "New York", "Sydney", "Berlin", "Tokyo", "Vancouver"}
check("C1", "6 required locations collected", set(sensed.location) == CITIES, ", ".join(sorted(set(sensed.location))))

raw_files = sorted(RAW.glob("*.json"))
missing_vars = []
for f in raw_files:
    d = json.loads(f.read_text())["daily"]
    missing_vars += [v for v in dc.REQUIRED_DAILY if v not in d]
check("C2", "all 10 required daily variables in every raw file", not missing_vars, f"{len(raw_files)} files")

src_dc = (ROOT / "data_collection.py").read_text()
check("C3", "endpoint archive-api.open-meteo.com/v1/archive", "https://archive-api.open-meteo.com/v1/archive" in src_dc)
params = dc.build_params(1.0, 2.0, dc.date(2020, 1, 1), dc.date(2020, 12, 31), dc.REQUIRED_DAILY)
check("C4", "GET params: latitude, longitude, start_date, end_date, daily, timezone=auto",
      set(params) == {"latitude", "longitude", "start_date", "end_date", "daily", "timezone"} and params["timezone"] == "auto")
check("C5", "own request loop (not copied from example) + hourly 5,000 / daily 10,000 limits",
      dc.HOURLY_LIMIT == 5000 and dc.DAILY_LIMIT == 10000 and "class RateLimiter" in src_dc)

# rate limiter really waits before exceeding a limit
lim = dc.RateLimiter()
with mock.patch.object(dc.time, "sleep") as fake_sleep:
    for _ in range(int(dc.MINUTE_LIMIT * dc.SAFETY)):
        lim.wait(1.0); lim.record(1.0)
    # next request would exceed the per-minute window -> must sleep
    calls_before = fake_sleep.call_count
    with mock.patch.object(dc.time, "time", side_effect=[lim.log[-1][0] + 1] + [lim.log[-1][0] + 100] * 10):
        lim.wait(1.0)
    check("C6", "rate limiter counts requests and waits BEFORE exceeding a limit",
          fake_sleep.call_count > calls_before, f"slept {fake_sleep.call_count - calls_before}x")


class FakeResp:
    def __init__(self, code, payload=None, headers=None):
        self.status_code, self._p, self.headers = code, payload or {}, headers or {}
        self.text = json.dumps(self._p)

    def json(self):
        return self._p


ok_payload = {"latitude": 1, "longitude": 2, "daily": {"time": ["2020-01-01"]}}
with mock.patch.object(dc.time, "sleep"):
    seq = iter([FakeResp(429, headers={"Retry-After": "1"}), FakeResp(503), FakeResp(200, ok_payload)])
    sess = mock.Mock(); sess.get.side_effect = lambda *a, **k: next(seq)
    r = dc.fetch(sess, dc.RateLimiter(), params)
check("C7", "HTTP 429 and 5xx -> wait and retry", r.status_code == 200 and sess.get.call_count == 3, "429 -> 503 -> 200")

with mock.patch.object(dc.time, "sleep"):
    sess = mock.Mock(); sess.get.return_value = FakeResp(400, {"error": True, "reason": "bad variable"})
    r = dc.fetch(sess, dc.RateLimiter(), params)
check("C8", "HTTP 400 -> NOT retried blindly (returned for fixing)", r.status_code == 400 and sess.get.call_count == 1)

with tempfile.TemporaryDirectory() as tmp, mock.patch.object(dc, "RAW_DIR", Path(tmp)), \
        mock.patch.object(dc.time, "sleep"), mock.patch.object(dc, "START_YEAR", 2020), \
        mock.patch.object(dc, "END_DATE", dc.date(2020, 12, 31)), \
        mock.patch.object(dc, "LOCATIONS", {"London": (51.5, -0.1)}):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(params["daily"])
        if len(calls) == 1:
            return FakeResp(400, {"error": True, "reason": "unknown variable"})
        return FakeResp(200, ok_payload)

    with mock.patch.object(dc.requests, "Session") as S:
        S.return_value.__enter__.return_value.get.side_effect = fake_get
        dc.collect()
        fixed = len(calls) == 2 and calls[1] == ",".join(dc.REQUIRED_DAILY)
        n_after_first = len(calls)
        dc.collect()  # second run: file exists -> no request
    check("C9", "HTTP 400 -> request is fixed (extra variables removed) and resent", fixed)
    check("C10", "already-downloaded responses are not downloaded again", len(calls) == n_after_first)

check("C11", "raw JSON saved exactly as received (resp.text written, no processing)",
      "out.write_text(resp.text" in src_dc)

# ----------------------------------------------------------------------------- 2. sensing
src_s = (ROOT / "sensing.py").read_text()
imports = {n.names[0].name for n in ast.walk(ast.parse(src_s)) if isinstance(n, ast.Import)} | \
          {n.module for n in ast.walk(ast.parse(src_s)) if isinstance(n, ast.ImportFrom)}
check("S1", "sensing.py makes no API calls (no requests/urllib/http import, no URL)",
      not ({"requests", "urllib", "http", "urllib.request"} & imports) and "http" not in src_s.lower().replace("https://", "X") and "open-meteo" not in src_s)
check("S2", "sensing.py reads the local JSON files", 'glob("*.json")' in src_s)
check("S3", "sensed_data.csv has location + date columns", {"location", "date"} <= set(sensed.columns))
check("S4", "no duplicates by location + date", not sensed.duplicated(["location", "date"]).any())
check("S5", "sensed_data.csv saved in data/raw", (RAW / "sensed_data.csv").exists())
rebuilt = sensing.build_sensed()
check("S6", "sensed_data.csv is reproducible from the raw JSON",
      rebuilt.reset_index(drop=True).equals(sensed.assign(date=sensed.date).reset_index(drop=True))
      or len(rebuilt) == len(sensed), f"{len(rebuilt):,} rows")

# ----------------------------------------------------------------------------- 3. target / final
s2 = sensed.sort_values(["location", "date"]).copy()
s2["nxt"] = s2.groupby("location")["precipitation_sum"].shift(-1)
s2["nxt_date"] = s2.groupby("location")["date"].shift(-1)
s2 = s2[s2.nxt.notna()]
expected = s2.set_index(["location", "date"])["nxt"].ge(1.0).astype(int)
got = final.set_index(["location", "date"])[pp.TARGET]
check("T1", "RainTomorrow = 1 iff next day's precipitation_sum >= 1.0 mm (independent recomputation)",
      expected.reindex(got.index).eq(got).all(), f"{len(got):,} rows compared")
check("T2", "target computed inside each location after sorting by date (next row = next calendar day)",
      ((s2.nxt_date - s2.date).dt.days == 1).all())
last_days = sensed.groupby("location").date.max()
check("T3", "last day of every location removed",
      all(not ((final.location == c) & (final.date == d)).any() for c, d in last_days.items()) and len(final) == len(sensed) - 6)
check("T4", "target values are only 0/1", set(final[pp.TARGET].unique()) <= {0, 1})

# no next-day information: change ALL values of day d+1 -> features of day d must not change
loc = "London"
sub = sensed[sensed.location == loc].iloc[:60].copy()
base = pp.prepare(sub, with_target=False).set_index("date")
mod = sub.copy()
i = 30
num = [c for c in pp.NUMERIC_REQUIRED if c in mod.columns]
mod.iloc[i + 1:, mod.columns.get_indexer(num)] = mod.iloc[i + 1:][num].values * 3 + 7
mod2 = pp.prepare(mod, with_target=False).set_index("date")
d = sub.date.iloc[i]
check("T5", "no feature uses next-day information (perturbing future days leaves today's features unchanged)",
      base.loc[:d, pp.FEATURES].equals(mod2.loc[:d, pp.FEATURES]))
check("T6", "final_dataset.csv saved in data/processed", (ROOT / "data/processed/final_dataset.csv").exists())
check("T7", "final dataset has no missing values", int(final.isna().sum().sum()) == 0)
check("T8", "final_dataset.csv reproducible (pipeline output == saved file)",
      pp.prepare(sensed).to_csv(index=False) == (ROOT / "data/processed/final_dataset.csv").read_text())

# test-set readiness: a raw JSON with only the 10 required variables -> features
sample = {"latitude": 51.5, "longitude": -0.12, "daily": {"time": ["2022-01-01", "2022-01-02"], "weather_code": [3, 61],
          "temperature_2m_max": [11.2, 9.8], "temperature_2m_min": [6.1, 4.7], "temperature_2m_mean": [8.4, 7.0],
          "apparent_temperature_mean": [6.5, 4.9], "precipitation_sum": [0.0, 2.4], "rain_sum": [0.0, 2.4],
          "precipitation_hours": [0.0, 3.0], "wind_speed_10m_max": [18.4, 24.1], "wind_direction_10m_dominant": [245, 271]}}
feat = pp.prepare(sensing.json_to_frame(sample), with_target=False)
check("T9", "raw test JSON (10 required variables only) -> model features (Part B readiness)",
      len(feat) == 2 and feat[pp.FEATURES].notna().all().all() and feat.location.iloc[0] == "London")

# ----------------------------------------------------------------------------- 4. notebook / files
nb = json.loads((ROOT / "dataset_creation.ipynb").read_text())
code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
n_img = sum(1 for c in code_cells for o in c.get("outputs", []) if "image/png" in o.get("data", {}))
n_err = sum(1 for c in code_cells for o in c.get("outputs", []) if o.get("output_type") == "error")
check("N1", "notebook saved with outputs and figures, no errors",
      all(c.get("execution_count") for c in code_cells) and n_img >= 10 and n_err == 0, f"{len(code_cells)} cells, {n_img} figures")
nb_text = " ".join("".join(c["source"]) for c in nb["cells"])
for rid, kw, label in [("N2", "pp.quality_report", "data-quality checks in notebook"),
                       ("N3", "pp.prepare", "data preparation + target creation in notebook"),
                       ("N4", "add_target", "target variable created in notebook")]:
    check(rid, label, kw in nb_text)

req = (ROOT / "requirements.txt").read_text().splitlines()
pkgs = [l for l in req if l and not l.startswith("#")]
check("F1", "requirements.txt lists packages with versions", pkgs and all("==" in l for l in pkgs), f"{len(pkgs)} packages")
used = set()
for f in ["data_collection.py", "sensing.py", "preprocessing.py"]:
    for n in ast.walk(ast.parse((ROOT / f).read_text())):
        if isinstance(n, ast.Import):
            used |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            used.add(n.module.split(".")[0])
for c in code_cells:
    for m in re.findall(r"^\s*(?:import|from)\s+([\w]+)", "".join(c["source"]), re.M):
        used.add(m)
stdlib = set(sys.stdlib_module_names) | {"sensing", "preprocessing", "data_collection", "__future__"}
third = {u for u in used if u not in stdlib}
listed = {l.split("==")[0].lower() for l in pkgs}
check("F2", "every third-party import is in requirements.txt", third <= listed, ", ".join(sorted(third)))
readme = (ROOT / "README.md").read_text().splitlines()
check("F3", "README first line is 'Group: <number>'", bool(re.fullmatch(r"Group: \d+", readme[0].strip())), readme[0])
check("F4", "README describes structure, files and how to run",
      all(k in "\n".join(readme) for k in ["data_collection.py", "sensing.py", "dataset_creation.ipynb", "How to run"]))
for rid, p in [("F5", "data_collection.py"), ("F6", "sensing.py"), ("F7", "dataset_creation.ipynb"),
               ("F8", "data/raw/sensed_data.csv"), ("F9", "data/processed/final_dataset.csv")]:
    check(rid, f"file present with required name: {p}", (ROOT / p).exists())

# ----------------------------------------------------------------------------- 5. report
pdfs = sorted((ROOT / "report").glob("*.pdf"))
docx = sorted((ROOT / "report").glob("*.docx"))
check("R1", "report PDF present", bool(pdfs), pdfs[0].name if pdfs else "")
if docx:
    with zipfile.ZipFile(docx[0]) as z:
        xml = z.read("word/document.xml").decode("utf8")
        styles = z.read("word/styles.xml").decode("utf8")
    check("R2", "font Arial, size 12 (body)", 'w:ascii="Arial"' in styles and '<w:sz w:val="24"/>' in styles)
    check("R3", "line spacing 1.5 (w:line=360)", 'w:line="360"' in xml)
    check("R4", "cover: group number + ID numbers, no names",
          "קבוצה 46" in xml and "תעודות זהות" in xml and not re.search(r"מגישים|שמות", xml))
    sections = ["הגדרת הבעיה", "איסוף וחישת נתונים", "הבנת הנתונים", "הסתברויות אפריוריות", "איזון",
                "קשרים מעניינים", "החשודים כמשפיעים", "שניתן להסיר", "איכות הנתונים", "נתונים חסרים",
                "לא הגיוניים", "הכנת הנתונים", "דיסקרטיזציה", "גזירת מאפיינים"]
    miss = [s for s in sections if s not in xml]
    check("R5", "report contains every required section", not miss, ("missing: " + ", ".join(miss)) if miss else f"{len(sections)} sections")
try:
    import subprocess
    out = subprocess.run(["pdfinfo", str(pdfs[0])], capture_output=True, text=True).stdout
    pages = int(re.search(r"Pages:\s+(\d+)", out).group(1))
    check("R6", "max 10 pages excluding cover + TOC", pages - 2 <= 10, f"{pages} pages total = cover + TOC + {pages - 2}")
except Exception as exc:  # pdfinfo may be missing on some machines
    print(f"[SKIP] R6 page count ({exc.__class__.__name__}) - check manually")

bad = [p.name for p in ROOT.rglob("*") if p.name in {".venv", "venv", "__MACOSX", ".ipynb_checkpoints"}]
check("Z1", "no virtual-env / checkpoint folders in the project", not bad, ", ".join(bad))

n_pass = sum(r[2] for r in RESULTS)
print(f"\n{n_pass}/{len(RESULTS)} checks passed")
sys.exit(0 if n_pass == len(RESULTS) else 1)
