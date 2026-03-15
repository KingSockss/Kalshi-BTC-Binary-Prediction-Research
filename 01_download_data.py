from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import RepoConfig
from kalshi_binance_api import (
    build_kalshi_event_ticker,
    create_session,
    download_market_candle_series_for_hour,
    fetch_binance_klines_1m,
    get_historical_cutoff,
    kalshi_get_event_markets,
    kalshi_pick_ladder_around_spot,
    resolve_market_outcome,
)
from utils import NY, append_df_csv, hour_range_last_n_full_hours, save_config, save_json, to_utc_millis


def main() -> None:
    cfg = RepoConfig()
    run_id = datetime.now(tz=NY).strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.output_root / f"run_{run_id}"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    save_config(run_dir / "config.json", cfg)

    now_utc = datetime.now(timezone.utc)
    hours = hour_range_last_n_full_hours(now_utc, cfg.eval_hours)
    if not hours:
        raise RuntimeError("No hours produced")

    eval_start_utc = hours[0][0]
    eval_end_utc = hours[-1][1]
    buffer_start_utc = eval_start_utc - timedelta(minutes=cfg.garch_buffer_minutes)

    session = create_session(cfg)
    historical_cutoff = get_historical_cutoff(session, cfg)

    print(f"Downloading Binance from {buffer_start_utc.isoformat()} to {eval_end_utc.isoformat()}", flush=True)
    df_bin = fetch_binance_klines_1m(
        session=session,
        cfg=cfg,
        symbol=cfg.symbol,
        start_utc_ms=to_utc_millis(buffer_start_utc),
        end_utc_ms=to_utc_millis(eval_end_utc),
        interval=cfg.interval,
    )
    df_bin = df_bin.sort_values("open_time").reset_index(drop=True)
    df_bin.to_csv(raw_dir / "binance_1m.csv", index=False)

    df_bin_idx = df_bin.set_index("open_time").sort_index()
    df_bin_idx["log_price"] = np.log(df_bin_idx["close"])
    df_bin_idx["log_ret"] = df_bin_idx["log_price"].diff()

    events_csv = raw_dir / "events.csv"
    market_candles_csv = raw_dir / "kalshi_market_candles.csv"
    settlements_csv = raw_dir / "settlement_outcomes.csv"

    summary = {
        "eval_start_utc": eval_start_utc.isoformat(),
        "eval_end_utc": eval_end_utc.isoformat(),
        "historical_cutoff": None if historical_cutoff is None else str(historical_cutoff),
        "total_hours": len(hours),
    }
    save_json(raw_dir / "download_summary.json", summary)

    for hour_idx, (hour_start_utc, hour_end_utc) in enumerate(hours, start=1):
        hour_start_ny = hour_start_utc.astimezone(NY)
        hour_end_ny = hour_end_utc.astimezone(NY)
        event_ticker = build_kalshi_event_ticker(hour_end_ny)

        hour_slice = df_bin_idx.loc[(df_bin_idx.index >= hour_start_utc) & (df_bin_idx.index < hour_end_utc)].copy()
        if len(hour_slice) < 55:
            print(f"SKIP {hour_start_ny}: insufficient Binance bars ({len(hour_slice)})", flush=True)
            continue

        if hour_start_utc in df_bin_idx.index:
            spot_hour_start = float(df_bin_idx.loc[hour_start_utc, "close"])
        else:
            spot_hour_start = float(hour_slice.iloc[0]["close"])

        try:
            markets = kalshi_get_event_markets(session, cfg, event_ticker)
            if not markets:
                print(f"SKIP {hour_start_ny}: no Kalshi markets for {event_ticker}", flush=True)
                continue
            ladder = kalshi_pick_ladder_around_spot(markets, spot_hour_start, cfg.ladder_offsets)
            if not ladder:
                print(f"SKIP {hour_start_ny}: could not build ladder", flush=True)
                continue
        except Exception as exc:
            print(f"SKIP {hour_start_ny}: ladder fetch error: {exc}", flush=True)
            continue

        hour_events_rows: list[dict] = []
        hour_candle_rows: list[dict] = []
        hour_settlement_rows: list[dict] = []

        for off, market in ladder:
            mt = str(market.get("ticker") or "")
            strike = float(market.get("floor_strike") or market.get("strike"))
            if not mt:
                continue
            settlement = resolve_market_outcome(
                session=session,
                cfg=cfg,
                market_ticker=mt,
                strike=strike,
                hour_end_utc=hour_end_utc,
                df_bin=df_bin_idx,
                historical_cutoff=historical_cutoff,
            )
            try:
                kdf = download_market_candle_series_for_hour(
                    session=session,
                    cfg=cfg,
                    market_ticker=mt,
                    hour_start_utc=hour_start_utc,
                    hour_end_utc=hour_end_utc,
                    settled_before_cutoff=bool(settlement["settled_before_cutoff"]),
                )
            except Exception as exc:
                print(f"WARN {hour_start_ny}: candle download error for {mt}: {exc}", flush=True)
                kdf = pd.DataFrame()

            if not kdf.empty:
                tmp = kdf.reset_index()
                tmp["event_ticker"] = event_ticker
                tmp["market_ticker"] = mt
                tmp["hour_start_utc"] = hour_start_utc.isoformat()
                tmp["hour_end_utc"] = hour_end_utc.isoformat()
                tmp["strike_offset"] = int(off)
                tmp["strike"] = float(strike)
                tmp["spot_hour_start"] = float(spot_hour_start)
                hour_candle_rows.extend(tmp.to_dict("records"))

            event_id = f"{event_ticker}|{mt}"
            hour_events_rows.append(
                {
                    "event_id": event_id,
                    "event_ticker": event_ticker,
                    "market_ticker": mt,
                    "hour_start_utc": hour_start_utc.isoformat(),
                    "hour_end_utc": hour_end_utc.isoformat(),
                    "hour_start_ny": hour_start_ny.isoformat(),
                    "hour_end_ny": hour_end_ny.isoformat(),
                    "spot_hour_start": float(spot_hour_start),
                    "strike_offset": int(off),
                    "strike": float(strike),
                    "moneyness_hour_start": float(spot_hour_start - strike),
                    "settled_before_cutoff": bool(settlement["settled_before_cutoff"]),
                }
            )
            hour_settlement_rows.append(
                {
                    "event_id": event_id,
                    "event_ticker": event_ticker,
                    "market_ticker": mt,
                    "hour_start_utc": hour_start_utc.isoformat(),
                    "hour_end_utc": hour_end_utc.isoformat(),
                    "strike_offset": int(off),
                    "strike": float(strike),
                    "spot_hour_start": float(spot_hour_start),
                    "settle_spot": settlement["settle_spot"],
                    "y": int(settlement["y"]),
                    "settlement_source": settlement["settlement_source"],
                    "settled_before_cutoff": bool(settlement["settled_before_cutoff"]),
                }
            )

        append_df_csv(events_csv, pd.DataFrame(hour_events_rows))
        append_df_csv(market_candles_csv, pd.DataFrame(hour_candle_rows))
        append_df_csv(settlements_csv, pd.DataFrame(hour_settlement_rows))
        print(
            f"[{hour_idx}/{len(hours)}] {hour_start_ny.strftime('%Y-%m-%d %H:%M')} event={event_ticker} "
            f"markets={len(hour_events_rows)} candles={len(hour_candle_rows)} settlements={len(hour_settlement_rows)}",
            flush=True,
        )

    save_json(
        raw_dir / "manifest.json",
        {
            "run_dir": str(run_dir),
            "raw_dir": str(raw_dir),
            "files": [
                str(raw_dir / "binance_1m.csv"),
                str(events_csv),
                str(market_candles_csv),
                str(settlements_csv),
            ],
        },
    )
    print(f"Saved raw data to: {raw_dir}", flush=True)


if __name__ == "__main__":
    main()
