from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

try:
    from zoneinfo import ZoneInfo
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Python 3.9+ required for zoneinfo") from exc

NY = ZoneInfo("America/New_York")


def floor_to_hour(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


def to_utc_millis(dt: datetime) -> int:
    if dt.tzinfo is None:
        raise ValueError("Datetime must be timezone-aware")
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)


def hour_range_last_n_full_hours(now_utc: datetime, n: int) -> list[tuple[datetime, datetime]]:
    if now_utc.tzinfo is None:
        raise ValueError("now_utc must be timezone-aware")
    end_anchor = floor_to_hour(now_utc.astimezone(timezone.utc))
    start_anchor = end_anchor - timedelta(hours=n)
    out: list[tuple[datetime, datetime]] = []
    cur = start_anchor
    while cur < end_anchor:
        nxt = cur + timedelta(hours=1)
        out.append((cur, nxt))
        cur = nxt
    return out


def append_df_csv(path: Path, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    df.to_csv(path, mode="a", header=write_header, index=False)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def save_config(path: Path, obj: Any) -> None:
    payload = asdict(obj) if hasattr(obj, "__dataclass_fields__") else obj
    save_json(path, payload)


def sleep_s(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def clip_prob(p: float, eps: float) -> float:
    return float(min(max(p, eps), 1.0 - eps))


def brier(p: float, y: int) -> float:
    return float((p - y) ** 2)


def logloss(p: float, y: int, eps: float) -> float:
    pp = clip_prob(p, eps)
    return float(-(y * math.log(pp) + (1 - y) * math.log(1.0 - pp)))


def calibration_table(df: pd.DataFrame, p_col: str, y_col: str, n_bins: int = 10) -> pd.DataFrame:
    tmp = df[[p_col, y_col]].dropna().copy()
    if tmp.empty:
        return pd.DataFrame(columns=["bin_lo", "bin_hi", "n", "p_mean", "y_mean"])
    tmp["p"] = pd.to_numeric(tmp[p_col], errors="coerce").clip(0, 1)
    tmp = tmp.dropna(subset=["p", y_col])
    if tmp.empty:
        return pd.DataFrame(columns=["bin_lo", "bin_hi", "n", "p_mean", "y_mean"])
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    tmp["bin"] = pd.cut(tmp["p"], bins=bins, include_lowest=True, right=False)
    g = (
        tmp.groupby("bin", observed=False)
        .agg(n=("p", "size"), p_mean=("p", "mean"), y_mean=(y_col, "mean"))
        .reset_index()
    )
    g["bin_lo"] = g["bin"].apply(lambda x: float(x.left))
    g["bin_hi"] = g["bin"].apply(lambda x: float(x.right))
    g = g.drop(columns=["bin"]).sort_values(["bin_lo"]).reset_index(drop=True)
    return g[["bin_lo", "bin_hi", "n", "p_mean", "y_mean"]]


def brier_decomposition_from_bins(calib: pd.DataFrame, y_bar: float) -> dict[str, float]:
    if calib.empty or calib["n"].sum() <= 0:
        return {"REL": float("nan"), "RES": float("nan"), "UNC": float("nan")}
    n_total = float(calib["n"].sum())
    w = calib["n"].astype(float) / n_total
    rel = float(np.sum(w * (calib["p_mean"] - calib["y_mean"]) ** 2))
    res = float(np.sum(w * (calib["y_mean"] - y_bar) ** 2))
    unc = float(y_bar * (1.0 - y_bar))
    return {"REL": rel, "RES": res, "UNC": unc}


def ensure_datetime_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str) and not x.strip():
            return None
        return float(x)
    except Exception:
        return None


def first_non_null(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None
