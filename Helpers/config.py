from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

# -----------------------------
# Global config
# -----------------------------

KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
BINANCE_BASE_URL = "https://api.binance.com"
BINANCE_KLINES_PATH = "/api/v3/klines"
KALSHI_SERIES_PREFIX = "KXBTCD"


@dataclass
class RepoConfig:
    # market/data scope
    symbol: str = "BTCUSDT"
    interval: str = "1m"
    eval_hours: int = 24 * 90
    ladder_offsets: tuple[int, ...] = (-2, -1, 0, 1, 2)

    # fitting / simulation
    mc_paths: int = 10_000
    garch_buffer_minutes: int = 2 * 24 * 60
    garch_fit_window_minutes: int = 1500
    returns_scale: float = 100.0
    seed: Optional[int] = 42

    # robustness
    mu_abs_max_scaled: float = 0.05
    ab_sum_max: float = 0.9995
    nu_min: float = 2.05
    nu_max: float = 200.0
    ewma_lambda: float = 0.94
    ewma_nu: float = 6.0

    # scoring / trading
    kalshi_score_quote: Literal["mid", "ask"] = "ask"
    prob_eps: float = 1e-4
    edge_delta: float = 0.02
    paper_contracts_per_trade: int = 1
    paper_extra_edge: float = 0.02
    paper_minute_lo_b: int = 30
    paper_minute_hi_b: int = 55

    # state features
    rv_win_5: int = 5
    rv_win_10: int = 10
    rv_win_15: int = 15
    rv_win_60: int = 60
    vol_shock_ratio: float = 2.0
    trend_z_threshold: float = 1.0
    moneyness_atm_pct: float = 0.001
    cond_min_n: int = 50
    vol_regime_window_minutes: int = 30
    vol_regime_min_periods: int = 20
    atm_z_thresh: float = 0.25

    # throttles
    kalshi_sleep_s: float = 0.10
    binance_sleep_s: float = 0.05
    timeout_connect_s: int = 8
    timeout_read_s: int = 30

    # local storage
    repo_root: Path = Path(__file__).resolve().parent
    output_root_dirname: str = "outputs_eval"

    # optional auth for personal settlement lookups
    kalshi_access_key: Optional[str] = None
    kalshi_access_signature: Optional[str] = None
    kalshi_access_timestamp: Optional[str] = None

    @property
    def output_root(self) -> Path:
        return self.repo_root / self.output_root_dirname
