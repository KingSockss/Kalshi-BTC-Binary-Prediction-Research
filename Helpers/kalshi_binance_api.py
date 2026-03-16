from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests

from Helpers.config import BINANCE_BASE_URL, BINANCE_KLINES_PATH, KALSHI_BASE_URL, KALSHI_SERIES_PREFIX, RepoConfig
from Helpers.utils import NY, safe_float, sleep_s


def build_kalshi_event_ticker(dt_et: datetime) -> str:
    yy = dt_et.strftime("%y")
    mon = dt_et.strftime("%b").upper()
    dd = dt_et.strftime("%d")
    hh = dt_et.strftime("%H")
    return f"{KALSHI_SERIES_PREFIX}-{yy}{mon}{dd}{hh}"


def create_session(cfg: RepoConfig) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "kalshi-eval-repo/1.0"})
    if cfg.kalshi_access_key:
        session.headers["KALSHI-ACCESS-KEY"] = cfg.kalshi_access_key
    if cfg.kalshi_access_signature:
        session.headers["KALSHI-ACCESS-SIGNATURE"] = cfg.kalshi_access_signature
    if cfg.kalshi_access_timestamp:
        session.headers["KALSHI-ACCESS-TIMESTAMP"] = cfg.kalshi_access_timestamp
    return session


def fetch_binance_klines_1m(
    session: requests.Session,
    cfg: RepoConfig,
    symbol: str,
    start_utc_ms: int,
    end_utc_ms: int,
    interval: str = "1m",
    limit: int = 1000,
) -> pd.DataFrame:
    rows: list[list[Any]] = []
    cur = start_utc_ms

    while cur < end_utc_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cur,
            "endTime": end_utc_ms,
            "limit": limit,
        }
        r = session.get(
            BINANCE_BASE_URL + BINANCE_KLINES_PATH,
            params=params,
            timeout=(cfg.timeout_connect_s, cfg.timeout_read_s),
        )
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        rows.extend(data)
        last_open_time = int(data[-1][0])
        next_cur = last_open_time + 60_000
        if next_cur <= cur:
            break
        cur = next_cur
        sleep_s(cfg.binance_sleep_s)
        if len(data) < limit:
            break

    if not rows:
        raise RuntimeError("No Binance klines returned for requested window")

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time_ms",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time_ms",
            "quote_asset_volume",
            "num_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    return df.sort_values("open_time").reset_index(drop=True)


def kalshi_request_json(
    session: requests.Session,
    cfg: RepoConfig,
    path: str,
    params: Optional[dict[str, Any]] = None,
    *,
    allow_404: bool = False,
) -> Optional[dict[str, Any]]:
    url = f"{KALSHI_BASE_URL}{path}"
    r = session.get(url, params=params, timeout=(cfg.timeout_connect_s, cfg.timeout_read_s))
    if allow_404 and r.status_code == 404:
        return None
    if r.status_code != 200:
        raise RuntimeError(f"GET {r.url} failed: {r.status_code} {r.text[:300]}")
    sleep_s(cfg.kalshi_sleep_s)
    return r.json()


def get_historical_cutoff(session: requests.Session, cfg: RepoConfig) -> Optional[pd.Timestamp]:
    data = kalshi_request_json(session, cfg, "/historical/cutoff")
    if not data:
        return None
    val = data.get("market_settled_ts")
    if not val:
        return None
    return pd.to_datetime(val, utc=True, errors="coerce")


def kalshi_get_event_markets(session: requests.Session, cfg: RepoConfig, event_ticker: str) -> list[dict[str, Any]]:
    data = kalshi_request_json(
        session,
        cfg,
        f"/events/{event_ticker}",
        params={"with_nested_markets": "true"},
    )
    if not data:
        return []
    return data.get("markets") or (data.get("event", {}) or {}).get("markets") or []


def kalshi_pick_ladder_around_spot(
    markets: list[dict[str, Any]],
    spot: float,
    offsets: tuple[int, ...],
) -> list[tuple[int, dict[str, Any]]]:
    ladder: list[tuple[float, dict[str, Any]]] = []
    for market in markets:
        strike = safe_float(market.get("floor_strike") or market.get("strike") or market.get("yes_sub_title"))
        if strike is not None and math.isfinite(strike):
            ladder.append((strike, market))
    if not ladder:
        raise RuntimeError("No usable strike-bearing markets in event")
    ladder.sort(key=lambda x: x[0])
    strikes = [x[0] for x in ladder]
    center = int(np.argmin([abs(k - float(spot)) for k in strikes]))
    out: list[tuple[int, dict[str, Any]]] = []
    seen: set[str] = set()
    for off in offsets:
        j = center + int(off)
        if 0 <= j < len(ladder):
            m = ladder[j][1]
            ticker = str(m.get("ticker") or "")
            if ticker and ticker not in seen:
                out.append((int(off), m))
                seen.add(ticker)
    return out


def _price01(value: Any) -> Optional[float]:
    x = safe_float(value)
    if x is None:
        return None
    # tolerate both old integer-cents and new fixed-point dollars
    if x > 1.0:
        x /= 100.0
    if 0.0 <= x <= 1.0:
        return float(x)
    return None


def _maybe_close01(candle: dict[str, Any], side_key: str) -> Optional[float]:
    block = candle.get(side_key)
    if isinstance(block, dict):
        return _price01(block.get("close") or block.get("close_dollars"))
    return None


def get_market_candlesticks(
    session: requests.Session,
    cfg: RepoConfig,
    market_ticker: str,
    start_ts: int,
    end_ts: int,
    period_interval: int = 1,
    settled_before_cutoff: bool = False,
) -> list[dict[str, Any]]:
    if settled_before_cutoff:
        path = f"/historical/markets/{market_ticker}/candlesticks"
    else:
        path = f"/series/{KALSHI_SERIES_PREFIX}/markets/{market_ticker}/candlesticks"
    params: dict[str, Any] = {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "period_interval": period_interval,
    }
    data = kalshi_request_json(session, cfg, path, params=params)
    return [] if not data else (data.get("candlesticks") or [])


def download_market_candle_series_for_hour(
    session: requests.Session,
    cfg: RepoConfig,
    market_ticker: str,
    hour_start_utc: datetime,
    hour_end_utc: datetime,
    settled_before_cutoff: bool,
) -> pd.DataFrame:
    start_ts = int(hour_start_utc.timestamp())
    end_ts = int(hour_end_utc.timestamp())
    candles = get_market_candlesticks(
        session=session,
        cfg=cfg,
        market_ticker=market_ticker,
        start_ts=start_ts,
        end_ts=end_ts,
        period_interval=1,
        settled_before_cutoff=settled_before_cutoff,
    )
    rows: list[dict[str, Any]] = []
    for c in candles:
        end_period_ts = c.get("end_period_ts")
        if not isinstance(end_period_ts, (int, float)):
            continue
        t_end = int(end_period_ts)
        if not (start_ts < t_end <= end_ts):
            continue
        t_start_utc = datetime.fromtimestamp(t_end - 60, tz=timezone.utc).replace(second=0, microsecond=0)
        rows.append(
            {
                "time_utc": t_start_utc,
                "time_ny": t_start_utc.astimezone(NY),
                "kalshi_yes_ask": _maybe_close01(c, "yes_ask"),
                "kalshi_yes_bid": _maybe_close01(c, "yes_bid"),
                "kalshi_no_ask": _maybe_close01(c, "no_ask"),
                "kalshi_no_bid": _maybe_close01(c, "no_bid"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "time_utc",
                "time_ny",
                "kalshi_yes_ask",
                "kalshi_yes_bid",
                "kalshi_no_ask",
                "kalshi_no_bid",
                "kalshi_yes_mid",
                "kalshi_no_mid",
            ]
        )
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df["kalshi_yes_mid"] = (df["kalshi_yes_ask"] + df["kalshi_yes_bid"]) / 2.0
    df["kalshi_no_mid"] = (df["kalshi_no_ask"] + df["kalshi_no_bid"]) / 2.0
    return df.drop_duplicates(subset=["time_utc"]).sort_values("time_utc").set_index("time_utc")


def get_market_metadata(session: requests.Session, cfg: RepoConfig, market_ticker: str, *, historical: bool) -> dict[str, Any]:
    path = f"/historical/markets/{market_ticker}" if historical else f"/markets/{market_ticker}"
    data = kalshi_request_json(session, cfg, path, allow_404=True)
    if not data:
        return {}
    return data.get("market") or data


def get_member_settlements(session: requests.Session, cfg: RepoConfig, market_ticker: str) -> list[dict[str, Any]]:
    # Optional supplement only. This endpoint is account-specific, so do not rely on it as the primary truth source.
    if not cfg.kalshi_access_key:
        return []
    data = kalshi_request_json(
        session,
        cfg,
        "/portfolio/settlements",
        params={"ticker": market_ticker},
        allow_404=True,
    )
    if not data:
        return []
    return data.get("settlements") or []


def parse_yes_outcome_from_market(market: dict[str, Any]) -> Optional[int]:
    candidates = [
        market.get("result"),
        market.get("outcome"),
        market.get("winning_outcome"),
        market.get("settlement_outcome"),
        market.get("status"),
    ]
    for item in candidates:
        if item is None:
            continue
        text = str(item).strip().lower()
        if text in {"yes", "true", "won_yes", "winner_yes", "settled_yes"}:
            return 1
        if text in {"no", "false", "won_no", "winner_no", "settled_no"}:
            return 0
    return None


def resolve_market_outcome(
    session: requests.Session,
    cfg: RepoConfig,
    market_ticker: str,
    strike: float,
    hour_end_utc: datetime,
    df_bin: pd.DataFrame,
    historical_cutoff: Optional[pd.Timestamp],
) -> dict[str, Any]:
    settled_before_cutoff = False
    if historical_cutoff is not None:
        settled_before_cutoff = pd.Timestamp(hour_end_utc) < historical_cutoff

    market = get_market_metadata(session, cfg, market_ticker, historical=settled_before_cutoff)
    y_market = parse_yes_outcome_from_market(market)
    if y_market is not None:
        return {
            "y": int(y_market),
            "settle_spot": np.nan,
            "settlement_source": "kalshi_market_metadata",
            "settled_before_cutoff": settled_before_cutoff,
        }

    personal_rows = get_member_settlements(session, cfg, market_ticker)
    for row in personal_rows:
        y_personal = parse_yes_outcome_from_market(row)
        if y_personal is not None:
            return {
                "y": int(y_personal),
                "settle_spot": np.nan,
                "settlement_source": "kalshi_portfolio_settlement",
                "settled_before_cutoff": settled_before_cutoff,
            }

    settle_idx = hour_end_utc - pd.Timedelta(minutes=1)
    if settle_idx in df_bin.index:
        settle_spot = float(df_bin.loc[settle_idx, "close"])
    else:
        prior = df_bin.loc[df_bin.index < hour_end_utc]
        if prior.empty:
            raise RuntimeError(f"No Binance fallback price available before {hour_end_utc.isoformat()}")
        settle_spot = float(prior.iloc[-1]["close"])

    return {
        "y": int(settle_spot > float(strike)),
        "settle_spot": float(settle_spot),
        "settlement_source": "binance_fallback",
        "settled_before_cutoff": settled_before_cutoff,
    }
