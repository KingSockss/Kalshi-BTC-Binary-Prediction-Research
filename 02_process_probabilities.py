from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from Helpers.config import RepoConfig
from Helpers.model_eval_utils import (
    add_realized_state_features,
    build_conditional_skill_matrix,
    bucket_summary,
    fit_with_fallbacks,
    simulate_terminal_prices_from_arch,
    simulate_terminal_prices_from_ewma,
)
from Helpers.plotting_utils import plot_calibration, plot_heatmap_skill, plot_one_line, plot_two_lines, save_plot
from Helpers.utils import NY, brier, brier_decomposition_from_bins, calibration_table, logloss, save_json


def _latest_run_dir(output_root: Path) -> Path:
    runs = sorted([p for p in output_root.glob("run_*") if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No run_* folders found under {output_root}")
    return runs[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default="", help="Optional explicit run dir. Defaults to latest run_*")
    args = parser.parse_args()

    cfg = RepoConfig()
    if cfg.seed is not None:
        np.random.seed(cfg.seed)

    run_dir = Path(args.run_dir) if args.run_dir else _latest_run_dir(cfg.output_root)
    raw_dir = run_dir / "raw"
    processed_dir = run_dir / "processed"
    plots_dir = run_dir / "plots"
    processed_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    binance_csv = raw_dir / "binance_1m.csv"
    candles_csv = raw_dir / "kalshi_market_candles.csv"
    settlements_csv = raw_dir / "settlement_outcomes.csv"
    if not binance_csv.exists() or not candles_csv.exists() or not settlements_csv.exists():
        raise FileNotFoundError("Missing one or more raw input files. Run 01_download_data.py first.")

    df_bin = pd.read_csv(binance_csv, low_memory=False)
    df_bin["open_time"] = pd.to_datetime(df_bin["open_time"], utc=True)
    df_bin = df_bin.set_index("open_time").sort_index()
    df_bin["log_price"] = np.log(df_bin["close"])
    df_bin["log_ret"] = df_bin["log_price"].diff()

    candles = pd.read_csv(candles_csv, low_memory=False)
    candles["time_utc"] = pd.to_datetime(candles["time_utc"], utc=True)
    settlements = pd.read_csv(settlements_csv, low_memory=False)

    key_cols = ["event_id", "event_ticker", "market_ticker", "hour_start_utc", "hour_end_utc", "strike_offset", "strike", "spot_hour_start", "settle_spot", "y", "settlement_source"]
    settlements = settlements[key_cols].copy()
    settlements["hour_start_utc"] = pd.to_datetime(settlements["hour_start_utc"], utc=True)
    settlements["hour_end_utc"] = pd.to_datetime(settlements["hour_end_utc"], utc=True)

    minute_rows: list[dict] = []
    fallback_counts_total: dict[str, int] = {"garch_t_const": 0, "garch_t_zero": 0, "garch_n_zero": 0, "ewma_t": 0}

    grouped = settlements.groupby(["hour_start_utc", "hour_end_utc"], sort=True)
    for (hour_start_utc, hour_end_utc), hour_settles in grouped:
        hour_start_ny = hour_start_utc.tz_convert(NY)
        hour_slice = df_bin.loc[(df_bin.index >= hour_start_utc) & (df_bin.index < hour_end_utc)].copy()
        if len(hour_slice) < 55:
            print(f"SKIP {hour_start_ny}: insufficient Binance bars ({len(hour_slice)})", flush=True)
            continue

        for ts in hour_slice.index[:60]:
            minutes_remaining = int(np.ceil((hour_end_utc - ts).total_seconds() / 60.0))
            if minutes_remaining <= 0:
                continue

            hist = df_bin.loc[df_bin.index <= ts]
            hist_rets = hist["log_ret"].dropna()
            if len(hist_rets) < 50:
                continue
            if len(hist_rets) > cfg.garch_fit_window_minutes:
                hist_rets = hist_rets.iloc[-cfg.garch_fit_window_minutes:]

            rets_raw = hist_rets.values.astype(float)
            rets_scaled = (rets_raw * cfg.returns_scale).astype(float)
            tag, res, meta = fit_with_fallbacks(rets_scaled, rets_raw, cfg)
            fallback_counts_total[tag] = fallback_counts_total.get(tag, 0) + 1

            spot_t = float(df_bin.loc[ts, "close"])
            if tag == "ewma_t":
                sigma2 = float(meta.get("ewma_sigma2", float("nan")))
                nu = float(meta.get("ewma_nu", cfg.ewma_nu))
                if not np.isfinite(sigma2) or sigma2 <= 0:
                    continue
                terminal = simulate_terminal_prices_from_ewma(s0=spot_t, sigma2=sigma2, nu=nu, horizon=minutes_remaining, n_paths=cfg.mc_paths)
            else:
                terminal = simulate_terminal_prices_from_arch(s0=spot_t, res=res, horizon=minutes_remaining, n_paths=cfg.mc_paths)

            minute_candles = candles.loc[
                (candles["hour_start_utc"] == hour_start_utc.isoformat())
                & (candles["time_utc"] == ts)
            ].copy()
            minute_candles = minute_candles.set_index("market_ticker", drop=False)
            minute_idx = int((ts - hour_start_utc).total_seconds() // 60)
            ts_ny = ts.tz_convert(NY)

            for _, settle_row in hour_settles.iterrows():
                mt = settle_row["market_ticker"]
                strike = float(settle_row["strike"])
                y = int(settle_row["y"])
                p_model = float(np.mean(terminal > strike))

                if mt in minute_candles.index:
                    rowk = minute_candles.loc[mt]
                    yes_ask = rowk.get("kalshi_yes_ask", np.nan)
                    yes_bid = rowk.get("kalshi_yes_bid", np.nan)
                    yes_mid = rowk.get("kalshi_yes_mid", np.nan)
                    no_ask = rowk.get("kalshi_no_ask", np.nan)
                    no_bid = rowk.get("kalshi_no_bid", np.nan)
                    no_mid = rowk.get("kalshi_no_mid", np.nan)
                else:
                    yes_ask = yes_bid = yes_mid = no_ask = no_bid = no_mid = np.nan

                if cfg.kalshi_score_quote == "mid":
                    p_kalshi = yes_mid if pd.notna(yes_mid) else yes_ask
                    quote_used = "mid" if pd.notna(yes_mid) else "ask_fallback"
                else:
                    p_kalshi = yes_ask
                    quote_used = "ask"

                minute_rows.append(
                    {
                        "event_id": settle_row["event_id"],
                        "event_ticker": settle_row["event_ticker"],
                        "market_ticker": mt,
                        "hour_start_utc": hour_start_utc.isoformat(),
                        "hour_end_utc": hour_end_utc.isoformat(),
                        "time_utc": ts.isoformat(),
                        "time_ny": ts_ny.isoformat(),
                        "minute_idx": minute_idx,
                        "minutes_to_expiry": minutes_remaining,
                        "spot_hour_start": float(settle_row["spot_hour_start"]),
                        "strike_offset": int(settle_row["strike_offset"]),
                        "strike": strike,
                        "spot_t": float(spot_t),
                        "settle_spot": settle_row["settle_spot"],
                        "y": y,
                        "settlement_source": settle_row["settlement_source"],
                        "moneyness": float(spot_t - strike),
                        "log_moneyness": float(np.log(spot_t / strike)) if spot_t > 0 and strike > 0 else np.nan,
                        "p_model": p_model,
                        "fit_tag": tag,
                        "kalshi_yes_ask": yes_ask,
                        "kalshi_yes_bid": yes_bid,
                        "kalshi_yes_mid": yes_mid,
                        "kalshi_no_ask": no_ask,
                        "kalshi_no_bid": no_bid,
                        "kalshi_no_mid": no_mid,
                        "p_kalshi_used": p_kalshi,
                        "kalshi_quote_used": quote_used,
                        "brier_model": brier(p_model, y),
                        "brier_kalshi": brier(float(p_kalshi), y) if pd.notna(p_kalshi) else np.nan,
                        "logloss_model": logloss(p_model, y, cfg.prob_eps),
                        "logloss_kalshi": logloss(float(p_kalshi), y, cfg.prob_eps) if pd.notna(p_kalshi) else np.nan,
                    }
                )

        print(f"Processed {hour_start_ny.strftime('%Y-%m-%d %H:%M')} ({len(hour_settles)} markets)", flush=True)

    minutes_df = pd.DataFrame(minute_rows)
    if minutes_df.empty:
        raise RuntimeError("No minute-level forecasts produced")

    minutes_csv = processed_dir / "minute_forecasts.csv"
    minutes_df.to_csv(minutes_csv, index=False)

    minutes_df = add_realized_state_features(minutes_df, df_bin=df_bin, cfg=cfg)
    minutes_states_csv = processed_dir / "minute_forecasts_with_states.csv"
    minutes_df.to_csv(minutes_states_csv, index=False)

    for col, name in [
        ("time_bucket", "bucket_summary_time.csv"),
        ("vol_regime", "bucket_summary_vol_regime.csv"),
        ("shock_state", "bucket_summary_shock.csv"),
        ("trend_state", "bucket_summary_trend.csv"),
        ("moneyness_bucket", "bucket_summary_moneyness.csv"),
    ]:
        bucket_summary(minutes_df, col).to_csv(processed_dir / name, index=False)

    cond = build_conditional_skill_matrix(minutes_df, cfg)
    cond.to_csv(processed_dir / "conditional_skill_matrix.csv", index=False)
    cond.sort_values(["passes_min_n", "brier_skill", "n"], ascending=[False, False, False]).to_csv(processed_dir / "conditional_skill_matrix_sorted.csv", index=False)

    def _mean_or_nan(s: pd.Series) -> float:
        s = pd.to_numeric(s, errors="coerce").dropna()
        return float(s.mean()) if not s.empty else float("nan")

    agg = (
        minutes_df.groupby("minute_idx", as_index=False)
        .agg(
            n=("p_model", "size"),
            mean_brier_model=("brier_model", "mean"),
            mean_brier_kalshi=("brier_kalshi", _mean_or_nan),
            mean_logloss_model=("logloss_model", "mean"),
            mean_logloss_kalshi=("logloss_kalshi", _mean_or_nan),
            mean_p_model=("p_model", "mean"),
            mean_p_kalshi=("p_kalshi_used", _mean_or_nan),
        )
        .sort_values("minute_idx")
        .reset_index(drop=True)
    )
    agg["minutes_to_expiry"] = 60 - agg["minute_idx"]
    agg["brier_skill"] = 1.0 - (agg["mean_brier_model"] / agg["mean_brier_kalshi"])
    agg["logloss_skill"] = 1.0 - (agg["mean_logloss_model"] / agg["mean_logloss_kalshi"])
    agg.to_csv(processed_dir / "minute_summary.csv", index=False)

    calib_model = calibration_table(minutes_df, "p_model", "y", n_bins=10)
    calib_kalshi = calibration_table(minutes_df, "p_kalshi_used", "y", n_bins=10)
    calib_model.to_csv(processed_dir / "calibration_overall_model.csv", index=False)
    calib_kalshi.to_csv(processed_dir / "calibration_overall_kalshi.csv", index=False)

    minutes_df["minute_bucket"] = minutes_df["minute_idx"].apply(lambda m: "early_0_9" if m <= 9 else ("mid_10_39" if m <= 39 else "late_40_59"))
    bucket_rows = []
    for bucket in ["early_0_9", "mid_10_39", "late_40_59"]:
        sub = minutes_df.loc[minutes_df["minute_bucket"] == bucket].copy()
        cm = calibration_table(sub, "p_model", "y", n_bins=10)
        ck = calibration_table(sub, "p_kalshi_used", "y", n_bins=10)
        cm["forecaster"] = "model"
        ck["forecaster"] = "kalshi"
        cm["bucket"] = bucket
        ck["bucket"] = bucket
        bucket_rows.extend([cm, ck])
        save_plot(plot_calibration(cm.drop(columns=["forecaster", "bucket"], errors="ignore"), ck.drop(columns=["forecaster", "bucket"], errors="ignore"), f"Calibration ({bucket})"), plots_dir / f"calibration_{bucket}.html")
    pd.concat(bucket_rows, ignore_index=True).to_csv(processed_dir / "calibration_by_bucket.csv", index=False)

    y_bar = float(minutes_df["y"].mean())
    dec_rows = [
        {"scope": "overall", "forecaster": "model", **brier_decomposition_from_bins(calib_model, y_bar=y_bar)},
        {"scope": "overall", "forecaster": "kalshi", **brier_decomposition_from_bins(calib_kalshi, y_bar=y_bar)},
    ]
    for bucket in ["early_0_9", "mid_10_39", "late_40_59"]:
        sub = minutes_df.loc[minutes_df["minute_bucket"] == bucket].copy()
        if sub.empty:
            continue
        yb = float(sub["y"].mean())
        dec_rows.append({"scope": bucket, "forecaster": "model", **brier_decomposition_from_bins(calibration_table(sub, "p_model", "y", 10), y_bar=yb)})
        dec_rows.append({"scope": bucket, "forecaster": "kalshi", **brier_decomposition_from_bins(calibration_table(sub, "p_kalshi_used", "y", 10), y_bar=yb)})
    pd.DataFrame(dec_rows).to_csv(processed_dir / "brier_decomposition.csv", index=False)

    save_plot(plot_two_lines(agg["minute_idx"].tolist(), agg["mean_brier_model"].tolist(), agg["mean_brier_kalshi"].tolist(), "Model mean Brier", f"Kalshi mean Brier ({cfg.kalshi_score_quote})", "Brier score by minute", "minute_idx", "Mean Brier"), plots_dir / "brier_by_minute.html")
    save_plot(plot_one_line(agg["minute_idx"].tolist(), agg["brier_skill"].tolist(), "Brier skill", "Brier skill vs Kalshi", "minute_idx", "Brier skill"), plots_dir / "brier_skill_by_minute.html")
    save_plot(plot_two_lines(agg["minute_idx"].tolist(), agg["mean_logloss_model"].tolist(), agg["mean_logloss_kalshi"].tolist(), "Model mean LogLoss", f"Kalshi mean LogLoss ({cfg.kalshi_score_quote})", "Log loss by minute", "minute_idx", "Mean LogLoss"), plots_dir / "logloss_by_minute.html")
    save_plot(plot_one_line(agg["minute_idx"].tolist(), agg["logloss_skill"].tolist(), "LogLoss skill", "LogLoss skill vs Kalshi", "minute_idx", "LogLoss skill"), plots_dir / "logloss_skill_by_minute.html")
    save_plot(plot_calibration(calib_model, calib_kalshi, "Calibration (overall)"), plots_dir / "calibration_overall.html")

    cond_ok = cond.loc[cond["passes_min_n"]].copy()
    if not cond_ok.empty:
        save_plot(plot_heatmap_skill(cond_ok, "time_bucket", "vol_regime", "brier_skill", "Conditional Brier skill: time × vol_regime", "time_bucket", "vol_regime"), plots_dir / "conditional_skill_heatmap_time_x_vol.html")
        save_plot(plot_heatmap_skill(cond_ok, "time_bucket", "shock_state", "brier_skill", "Conditional Brier skill: time × shock_state", "time_bucket", "shock_state"), plots_dir / "conditional_skill_heatmap_time_x_shock.html")
        save_plot(plot_heatmap_skill(cond_ok, "time_bucket", "moneyness_bucket", "brier_skill", "Conditional Brier skill: time × moneyness", "time_bucket", "moneyness_bucket"), plots_dir / "conditional_skill_heatmap_time_x_moneyness.html")
        save_plot(plot_heatmap_skill(cond_ok, "time_bucket", "trend_state", "brier_skill", "Conditional Brier skill: time × trend_state", "time_bucket", "trend_state"), plots_dir / "conditional_skill_heatmap_time_x_trend.html")

    save_json(
        processed_dir / "process_summary.json",
        {
            "run_dir": str(run_dir),
            "processed_at_ny": datetime.now(tz=NY).isoformat(),
            "rows": int(len(minutes_df)),
            "events": int(minutes_df["event_id"].nunique()),
            "overall_brier_model": float(minutes_df["brier_model"].mean()),
            "overall_brier_kalshi": float(minutes_df["brier_kalshi"].dropna().mean()) if minutes_df["brier_kalshi"].notna().any() else None,
            "overall_logloss_model": float(minutes_df["logloss_model"].mean()),
            "overall_logloss_kalshi": float(minutes_df["logloss_kalshi"].dropna().mean()) if minutes_df["logloss_kalshi"].notna().any() else None,
            "fallback_counts": fallback_counts_total,
        },
    )
    print(f"Saved processed outputs to: {processed_dir}", flush=True)


if __name__ == "__main__":
    main()
