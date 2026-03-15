from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from Helpers.config import RepoConfig
from Helpers.model_eval_utils import build_paper_trades, strategy_summary
from Helpers.plotting_utils import plot_one_line, save_plot
from Helpers.utils import NY, save_json


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
    run_dir = Path(args.run_dir) if args.run_dir else _latest_run_dir(cfg.output_root)
    processed_dir = run_dir / "processed"
    plots_dir = run_dir / "plots"
    processed_csv = processed_dir / "minute_forecasts_with_states.csv"
    if not processed_csv.exists():
        raise FileNotFoundError("Missing processed minute forecasts. Run 02_process_probabilities.py first.")

    minutes_df = pd.read_csv(processed_csv, low_memory=False)
    minutes_df["time_utc"] = pd.to_datetime(minutes_df["time_utc"], utc=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    paper_all = build_paper_trades(minutes_df, cfg)
    paper_b = build_paper_trades(minutes_df, cfg, minute_lo=cfg.paper_minute_lo_b, minute_hi=cfg.paper_minute_hi_b)
    paper_atm_lowvol = build_paper_trades(minutes_df, cfg, minute_lo=30, minute_hi=50, trade_filter_col="trade_allowed_atm_lowvol")

    if not paper_all.empty:
        paper_all.to_csv(processed_dir / "paper_trades_all.csv", index=False)
        save_plot(
            plot_one_line(paper_all["time_sort_dt"].tolist(), paper_all["equity"].tolist(), "Equity", f"Paper equity (all minutes) | C={cfg.paper_contracts_per_trade}", "time_utc", "equity"),
            plots_dir / "equity_all.html",
        )

    if not paper_b.empty:
        paper_b.to_csv(processed_dir / "paper_trades_30_55.csv", index=False)
        save_plot(
            plot_one_line(paper_b["time_sort_dt"].tolist(), paper_b["equity"].tolist(), "Equity", f"Paper equity ({cfg.paper_minute_lo_b}..{cfg.paper_minute_hi_b})", "time_utc", "equity"),
            plots_dir / "equity_30_55.html",
        )

    strat_summary = strategy_summary(minutes_df.loc[minutes_df["trade_allowed_atm_lowvol"].astype(bool)].copy(), label="ATM(z<=0.25) & LowVol(rolling30m tercile)")
    strat_summary.to_csv(processed_dir / "strategy_atm_lowvol_summary.csv", index=False)

    if not paper_atm_lowvol.empty:
        paper_atm_lowvol.to_csv(processed_dir / "paper_trades_atm_lowvol.csv", index=False)
        save_plot(
            plot_one_line(paper_atm_lowvol["time_sort_dt"].tolist(), paper_atm_lowvol["equity"].tolist(), "Equity", "Paper equity (ATM(z) & LowVol rolling30m)", "time_utc", "equity"),
            plots_dir / "equity_atm_lowvol.html",
        )

    if paper_all.empty:
        edge_by_minute = pd.DataFrame({
            "minute_idx": sorted(minutes_df["minute_idx"].dropna().astype(int).unique().tolist()),
            "n_trades": 0,
            "trade_rate": 0.0,
            "hit_rate": np.nan,
            "mean_pnl": np.nan,
            "median_pnl": np.nan,
        })
    else:
        denom = minutes_df.groupby("minute_idx").size().rename("n_rows").reset_index()
        tgrp = (
            paper_all.groupby("minute_idx", as_index=False)
            .agg(
                n_trades=("pnl_net", "size"),
                hit_rate=("pnl_net", lambda s: float((s > 0).mean())),
                mean_pnl=("pnl_net", "mean"),
                median_pnl=("pnl_net", "median"),
                std_pnl=("pnl_net", "std"),
                mean_edge=("edge_chosen_net", "mean"),
            )
            .sort_values("minute_idx")
        )
        edge_by_minute = denom.merge(tgrp, on="minute_idx", how="left")
        edge_by_minute["n_trades"] = edge_by_minute["n_trades"].fillna(0).astype(int)
        edge_by_minute["trade_rate"] = edge_by_minute["n_trades"] / edge_by_minute["n_rows"]
        edge_by_minute = edge_by_minute.drop(columns=["n_rows"])

    edge_by_minute.to_csv(processed_dir / "edge_by_minute.csv", index=False)
    save_plot(plot_one_line(edge_by_minute["minute_idx"].tolist(), edge_by_minute["trade_rate"].tolist(), "Trade rate", f"Edge trade rate by minute (threshold={cfg.edge_delta:.3f})", "minute_idx", "trade_rate"), plots_dir / "edge_trade_rate_by_minute.html")

    save_json(
        processed_dir / "backtest_summary.json",
        {
            "run_dir": str(run_dir),
            "backtested_at_ny": datetime.now(tz=NY).isoformat(),
            "paper_all_trades": int(len(paper_all)) if not paper_all.empty else 0,
            "paper_b_trades": int(len(paper_b)) if not paper_b.empty else 0,
            "paper_atm_lowvol_trades": int(len(paper_atm_lowvol)) if not paper_atm_lowvol.empty else 0,
            "paper_all_final_equity": float(paper_all["equity"].iloc[-1]) if not paper_all.empty else None,
            "paper_b_final_equity": float(paper_b["equity"].iloc[-1]) if not paper_b.empty else None,
            "paper_atm_lowvol_final_equity": float(paper_atm_lowvol["equity"].iloc[-1]) if not paper_atm_lowvol.empty else None,
        },
    )
    print(f"Saved backtest outputs to: {processed_dir}", flush=True)


if __name__ == "__main__":
    main()
