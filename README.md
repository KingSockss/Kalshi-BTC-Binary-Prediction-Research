# Kalshi BTC Binary Prediction Research

This repository evaluates whether a simple volatility-based probability model can outperform live Kalshi BTC hourly binary market prices.

The workflow:

1. Download 1-minute BTC spot data from Binance and 1-minute Kalshi candle data for a ladder of strikes around spot.
2. Fit a rolling volatility model each minute, simulate terminal BTC prices at hourly expiry, and compare model probabilities against Kalshi quotes.
3. Run a paper-trading backtest using simple edge thresholds and Kalshi-style fees.

## What The Repo Does

For each hourly Kalshi BTC event:

- Pulls 1-minute Binance `BTCUSDT` candles for a historical evaluation window.
- Builds a small strike ladder around the hour-start spot price.
- Downloads Kalshi minute candles for those markets.
- Resolves each market outcome from settlement data and/or spot at expiry.
- Recomputes a model probability every minute until expiry.
- Scores the model against Kalshi using Brier score, log loss, calibration, and conditional slices.
- Simulates paper trades when the model shows enough net edge over Kalshi ask prices.

The modeling stack is intentionally simple and robust:

- Primary fit: GARCH(1,1) with Student-t innovations and a constant mean.
- Fallbacks: zero-mean Student-t GARCH, zero-mean Gaussian GARCH, then EWMA variance with Student-t shocks.

## Repository Layout

```text
.
├── 01_download_data.py
├── 02_process_probabilities.py
├── 03_backtest_paper.py
├── README.md
└── Helpers/
    ├── config.py
    ├── kalshi_binance_api.py
    ├── model_eval_utils.py
    ├── plotting_utils.py
    ├── requirements.txt
    └── utils.py
```

## Requirements

- Python 3.9+ (`zoneinfo` is required)
- Internet access to Binance and Kalshi APIs
- Packages in [`Helpers/requirements.txt`](Helpers/requirements.txt)

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r Helpers/requirements.txt
```

## Configuration

All runtime settings live in [`Helpers/config.py`](Helpers/config.py) inside the `RepoConfig` dataclass.

Important defaults:

- `eval_hours = 24 * 90`: evaluate the last 90 full days of hourly markets
- `ladder_offsets = (-2, -1, 0, 1, 2)`: score 5 strikes around the nearest strike to spot
- `mc_paths = 10_000`: Monte Carlo paths per minute/market forecast
- `kalshi_score_quote = "ask"`: compare the model against Kalshi yes asks by default
- `edge_delta = 0.02` and `paper_extra_edge = 0.02`: minimum edge thresholds for paper trades

Optional Kalshi auth headers can also be set in `RepoConfig`, but most of the pipeline uses public market data.

One implementation detail to know up front: outputs default to `Helpers/outputs_eval/`, not the repo root, because `RepoConfig.output_root` is anchored from the `Helpers` directory.

## Quick Start

Run the scripts in order from the repository root.

### 1. Download raw market data

```bash
python 01_download_data.py
```

This creates a timestamped run directory like:

```text
Helpers/outputs_eval/run_YYYYMMDD_HHMMSS/
```

Raw outputs include:

- `raw/binance_1m.csv`
- `raw/events.csv`
- `raw/kalshi_market_candles.csv`
- `raw/settlement_outcomes.csv`
- `raw/download_summary.json`
- `raw/manifest.json`

### 2. Process probabilities and evaluate model skill

```bash
python 02_process_probabilities.py
```

By default this uses the latest `run_*` folder. To target a specific run:

```bash
python 02_process_probabilities.py --run-dir Helpers/outputs_eval/run_YYYYMMDD_HHMMSS
```

Main processed outputs:

- `processed/minute_forecasts.csv`
- `processed/minute_forecasts_with_states.csv`
- `processed/minute_summary.csv`
- `processed/conditional_skill_matrix.csv`
- `processed/conditional_skill_matrix_sorted.csv`
- `processed/calibration_overall_model.csv`
- `processed/calibration_overall_kalshi.csv`
- `processed/calibration_by_bucket.csv`
- `processed/brier_decomposition.csv`
- bucket summaries by time, volatility regime, shock state, trend state, and moneyness

Plots:

- `plots/brier_by_minute.html`
- `plots/brier_skill_by_minute.html`
- `plots/logloss_by_minute.html`
- `plots/logloss_skill_by_minute.html`
- `plots/calibration_overall.html`
- `plots/calibration_early_0_9.html`
- `plots/calibration_mid_10_39.html`
- `plots/calibration_late_40_59.html`

### 3. Run the paper-trading backtest

```bash
python 03_backtest_paper.py
```

Or point it at a specific run:

```bash
python 03_backtest_paper.py --run-dir Helpers/outputs_eval/run_YYYYMMDD_HHMMSS
```

Backtest outputs:

- `processed/paper_trades_all.csv`
- `processed/paper_trades_30_55.csv`
- `processed/paper_trades_atm_lowvol.csv`
- `processed/edge_by_minute.csv`
- `processed/strategy_atm_lowvol_summary.csv`
- `processed/backtest_summary.json`
- `plots/equity_all.html`
- `plots/equity_30_55.html`
- `plots/equity_atm_lowvol.html`
- `plots/edge_trade_rate_by_minute.html`

## How To Read The Results

- `minute_forecasts_with_states.csv` is the main analysis table. Each row is one market at one minute with the model probability, Kalshi quote snapshot, realized outcome, scoring metrics, and realized volatility/trend state labels.
- `minute_summary.csv` shows how model-vs-market skill changes as expiry approaches.
- `conditional_skill_matrix.csv` shows where the model helps or hurts across volatility, trend, shock, and moneyness regimes.
- `paper_trades_*.csv` shows the hypothetical trades taken under the current edge rules and fee assumptions.

## Notes And Caveats

- This is a research pipeline, not a production trading system.
- The scripts are designed around hourly Kalshi BTC markets whose tickers are built from the `KXBTCD` series prefix.
- If Kalshi candle data is missing for a minute, the model forecast is still produced but market-side metrics can be `NaN`.
- Quote comparison uses yes asks by default; processing also records bids and mids.
- Settlement logic may use different endpoints depending on whether a market is already in Kalshi historical data.
- The scripts use New York time when constructing hourly event tickers and reporting run timestamps.

## Suggested Workflow

If you are iterating on the research:

1. Tune parameters in [`Helpers/config.py`](Helpers/config.py).
2. Run `01_download_data.py`.
3. Run `02_process_probabilities.py`.
4. Inspect `minute_summary.csv`, `conditional_skill_matrix_sorted.csv`, and the HTML plots.
5. Run `03_backtest_paper.py`.
6. Compare `backtest_summary.json` and the `paper_trades_*.csv` outputs across runs.
