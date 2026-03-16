from __future__ import annotations

import math
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from arch import arch_model

from Helpers.config import RepoConfig

FitTag = Literal["garch_t_const", "garch_t_zero", "garch_n_zero", "ewma_t"]


def _standardized_student_t(df: float, size: tuple[int, ...]) -> np.ndarray:
    if df <= 2.0:
        df = 2.01
    x = np.random.standard_t(df, size=size)
    return x / math.sqrt(df / (df - 2.0))


def _standardized_normal(size: tuple[int, ...]) -> np.ndarray:
    return np.random.standard_normal(size)


def _ewma_last_variance(returns: np.ndarray, lam: float) -> float:
    x = returns[np.isfinite(returns)]
    if x.size < 5:
        return float("nan")
    v = float(np.var(x[-min(50, x.size):], ddof=1))
    for r in x:
        v = lam * v + (1.0 - lam) * (float(r) ** 2)
    return max(v, 1e-18)


def _validate_arch_result(res: Any, cfg: RepoConfig, *, mean: str, dist: str) -> tuple[bool, str]:
    flag = getattr(res, "convergence_flag", 0)
    if flag != 0:
        return False, f"convergence_flag={flag}"

    p = res.params
    try:
        omega = float(p.get("omega", np.nan))
        alpha = float(p.get("alpha[1]", np.nan))
        beta = float(p.get("beta[1]", np.nan))
    except Exception:
        return False, "param_cast_error"

    if not np.isfinite([omega, alpha, beta]).all():
        return False, "nonfinite_params"
    if omega <= 0 or alpha < 0 or beta < 0:
        return False, "negativity_violation"
    if alpha + beta >= cfg.ab_sum_max:
        return False, f"alpha+beta too high ({alpha + beta:.6f})"

    if mean.lower() == "constant":
        mu = float(p.get("mu", p.get("Const", 0.0)))
        if not np.isfinite(mu):
            return False, "nonfinite_mu"
        if abs(mu) > cfg.mu_abs_max_scaled:
            return False, f"|mu| too large ({mu:.6f})"

    if dist.lower() in {"studentst", "students-t", "t"}:
        nu = float(p.get("nu", np.nan))
        if not np.isfinite(nu):
            return False, "nonfinite_nu"
        if not (cfg.nu_min <= nu <= cfg.nu_max):
            return False, f"nu out of bounds ({nu:.3f})"

    cv = np.asarray(res.conditional_volatility)
    if cv.size == 0 or (not np.isfinite(cv[-1])) or cv[-1] <= 0:
        return False, "bad_cond_vol"

    return True, "ok"


def _try_fit_arch(returns_scaled: np.ndarray, cfg: RepoConfig, *, mean: str, dist: str) -> tuple[Optional[Any], str]:
    try:
        am = arch_model(
            returns_scaled,
            mean=mean,
            vol="GARCH",
            p=1,
            q=1,
            dist=dist,
            rescale=False,
        )
        res = am.fit(disp="off")
        ok, reason = _validate_arch_result(res, cfg, mean=mean, dist=dist)
        if not ok:
            return None, reason
        return res, "ok"
    except Exception as exc:
        return None, f"exception: {type(exc).__name__}: {exc}"


def fit_with_fallbacks(returns_scaled: np.ndarray, returns_raw: np.ndarray, cfg: RepoConfig) -> tuple[FitTag, Optional[Any], dict[str, Any]]:
    res, reason = _try_fit_arch(returns_scaled, cfg, mean="Constant", dist="StudentsT")
    if res is not None:
        return "garch_t_const", res, {"reason": reason}

    res2, reason2 = _try_fit_arch(returns_scaled, cfg, mean="Zero", dist="StudentsT")
    if res2 is not None:
        return "garch_t_zero", res2, {"reason": reason2, "primary_failed": reason}

    res3, reason3 = _try_fit_arch(returns_scaled, cfg, mean="Zero", dist="Normal")
    if res3 is not None:
        return "garch_n_zero", res3, {"reason": reason3, "primary_failed": reason, "t_zero_failed": reason2}

    sigma2 = _ewma_last_variance(returns_raw, cfg.ewma_lambda)
    return "ewma_t", None, {
        "reason": "ewma_ok" if np.isfinite(sigma2) and sigma2 > 0 else "ewma_failed",
        "primary_failed": reason,
        "t_zero_failed": reason2,
        "n_zero_failed": reason3,
        "ewma_sigma2": float(sigma2),
        "ewma_nu": cfg.ewma_nu,
    }


def simulate_terminal_prices_from_arch(s0: float, res: Any, horizon: int, n_paths: int) -> np.ndarray:
    params = res.params
    mu = float(params.get("mu", params.get("Const", 0.0)))
    omega = float(params["omega"])
    alpha = float(params["alpha[1]"])
    beta = float(params["beta[1]"])
    nu = params.get("nu")

    cond_vol = np.asarray(res.conditional_volatility)
    sigma2 = np.full(n_paths, float(cond_vol[-1] ** 2), dtype=float)
    s = np.full(n_paths, s0, dtype=float)

    use_t = nu is not None
    nu_f = float(nu) if use_t else None

    for _ in range(horizon):
        z = _standardized_student_t(nu_f, size=(n_paths,)) if use_t else _standardized_normal((n_paths,))
        eps = np.sqrt(np.maximum(sigma2, 1e-18)) * z
        r_scaled = mu + eps
        sigma2 = omega + alpha * (eps ** 2) + beta * sigma2
        r = r_scaled / 100.0
        s *= np.exp(r)
    return s


def simulate_terminal_prices_from_ewma(s0: float, *, sigma2: float, nu: float, horizon: int, n_paths: int) -> np.ndarray:
    sigma = math.sqrt(max(float(sigma2), 1e-18))
    s = np.full(n_paths, s0, dtype=float)
    for _ in range(horizon):
        z = _standardized_student_t(float(nu), size=(n_paths,))
        r = sigma * z
        s *= np.exp(r)
    return s


def add_realized_state_features(minutes_df: pd.DataFrame, df_bin: pd.DataFrame, cfg: RepoConfig) -> pd.DataFrame:
    if minutes_df.empty:
        return minutes_df.copy()
    out = minutes_df.copy()
    out["time_utc_dt"] = pd.to_datetime(out["time_utc"], errors="coerce", utc=True)

    b = df_bin.copy().sort_index()
    eps = 1e-12
    b["rv_5"] = b["log_ret"].rolling(cfg.rv_win_5, min_periods=max(3, cfg.rv_win_5 // 2)).std()
    b["rv_10"] = b["log_ret"].rolling(cfg.rv_win_10, min_periods=max(5, cfg.rv_win_10 // 2)).std()
    b["rv_15"] = b["log_ret"].rolling(cfg.rv_win_15, min_periods=max(7, cfg.rv_win_15 // 2)).std()
    b["rv_60"] = b["log_ret"].rolling(cfg.rv_win_60, min_periods=max(20, cfg.rv_win_60 // 3)).std()
    b["ret_10"] = b["log_ret"].rolling(cfg.rv_win_10, min_periods=max(5, cfg.rv_win_10 // 2)).sum()
    b["z_trend"] = b["ret_10"].abs() / (b["rv_10"] * math.sqrt(float(cfg.rv_win_10)) + eps)
    b["vol_shock"] = b["rv_5"] / (b["rv_60"] + eps)

    w = int(cfg.vol_regime_window_minutes)
    mp = int(cfg.vol_regime_min_periods)
    b["rv15_q1_30m"] = b["rv_15"].rolling(w, min_periods=mp).quantile(1 / 3)
    b["rv15_q2_30m"] = b["rv_15"].rolling(w, min_periods=mp).quantile(2 / 3)

    b["vol_regime"] = "unknown"
    mask_valid = np.isfinite(b["rv_15"]) & np.isfinite(b["rv15_q1_30m"]) & np.isfinite(b["rv15_q2_30m"])
    b.loc[mask_valid & (b["rv_15"] <= b["rv15_q1_30m"]), "vol_regime"] = "low"
    b.loc[mask_valid & (b["rv_15"] > b["rv15_q1_30m"]) & (b["rv_15"] <= b["rv15_q2_30m"]), "vol_regime"] = "med"
    b.loc[mask_valid & (b["rv_15"] > b["rv15_q2_30m"]), "vol_regime"] = "high"

    join_cols = [
        "rv_5", "rv_10", "rv_15", "rv_60", "ret_10", "z_trend", "vol_shock",
        "rv15_q1_30m", "rv15_q2_30m", "vol_regime",
    ]
    b_join = b[join_cols].reset_index().rename(columns={"open_time": "time_utc_dt"})
    out = out.merge(b_join, on="time_utc_dt", how="left")

    def time_bucket(m: int) -> str:
        if pd.isna(m):
            return "unknown"
        m = int(m)
        if 0 <= m < 10:
            return "0_10"
        if 10 <= m < 20:
            return "10_20"
        if 20 <= m < 30:
            return "20_30"
        if 30 <= m < 40:
            return "30_40"
        if 40 <= m < 50:
            return "40_50"
        if 50 <= m < 60:
            return "50_60"
        return "out_of_range"

    out["time_bucket"] = out["minute_idx"].apply(time_bucket)
    out["shock_state"] = out["vol_shock"].apply(lambda x: "shock" if np.isfinite(x) and x >= cfg.vol_shock_ratio else "normal")
    out["trend_state"] = out["z_trend"].apply(lambda x: "trend" if np.isfinite(x) and x >= cfg.trend_z_threshold else "mean_revert")
    out["trend_dir"] = out["ret_10"].apply(lambda x: "up" if np.isfinite(x) and x > 0 else ("down" if np.isfinite(x) and x < 0 else "flat"))

    out["moneyness_pct"] = (pd.to_numeric(out["spot_t"], errors="coerce") - pd.to_numeric(out["strike"], errors="coerce")) / (pd.to_numeric(out["spot_t"], errors="coerce") + eps)
    out["moneyness_bucket"] = out["moneyness_pct"].apply(lambda x: "ATM" if np.isfinite(x) and abs(x) <= cfg.moneyness_atm_pct else ("deep_ITM" if np.isfinite(x) and x > 0 else ("deep_OTM" if np.isfinite(x) else "unknown")))

    logm = pd.to_numeric(out["log_moneyness"], errors="coerce")
    sigma = pd.to_numeric(out["rv_15"], errors="coerce")
    tau_m = pd.to_numeric(out["minutes_to_expiry"], errors="coerce")
    denom = sigma * np.sqrt(np.maximum(tau_m, 0.0)) + eps
    out["z_moneyness"] = logm / denom
    out["atm_z"] = np.isfinite(out["z_moneyness"]) & (out["z_moneyness"].abs() <= cfg.atm_z_thresh)
    out["is_low_vol_rolling30m"] = out["vol_regime"].astype(str).eq("low")
    out["trade_allowed_atm_lowvol"] = out["atm_z"].astype(bool) & out["is_low_vol_rolling30m"].astype(bool)
    return out


def build_conditional_skill_matrix(minutes_df: pd.DataFrame, cfg: RepoConfig) -> pd.DataFrame:
    if minutes_df.empty:
        return pd.DataFrame()
    df = minutes_df.copy()
    group_cols = ["time_bucket", "vol_regime", "shock_state", "trend_state", "trend_dir", "moneyness_bucket"]

    def _mean_or_nan(s: pd.Series) -> float:
        s = pd.to_numeric(s, errors="coerce").dropna()
        return float(s.mean()) if not s.empty else float("nan")

    g = (
        df.groupby(group_cols, as_index=False)
        .agg(
            n=("p_model", "size"),
            mean_brier_model=("brier_model", "mean"),
            mean_brier_kalshi=("brier_kalshi", _mean_or_nan),
            mean_logloss_model=("logloss_model", "mean"),
            mean_logloss_kalshi=("logloss_kalshi", _mean_or_nan),
            mean_p_model=("p_model", "mean"),
            mean_p_kalshi=("p_kalshi_used", _mean_or_nan),
        )
        .reset_index(drop=True)
    )
    g["brier_skill"] = 1.0 - (g["mean_brier_model"] / g["mean_brier_kalshi"])
    g["logloss_skill"] = 1.0 - (g["mean_logloss_model"] / g["mean_logloss_kalshi"])
    g.loc[~np.isfinite(g["brier_skill"]), "brier_skill"] = np.nan
    g.loc[~np.isfinite(g["logloss_skill"]), "logloss_skill"] = np.nan
    g["passes_min_n"] = g["n"] >= int(cfg.cond_min_n)
    return g


def bucket_summary(minutes_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if minutes_df.empty:
        return pd.DataFrame()

    def _mean_or_nan(s: pd.Series) -> float:
        s = pd.to_numeric(s, errors="coerce").dropna()
        return float(s.mean()) if not s.empty else float("nan")

    out = (
        minutes_df.groupby(group_col, as_index=False)
        .agg(
            n=("p_model", "size"),
            mean_brier_model=("brier_model", "mean"),
            mean_brier_kalshi=("brier_kalshi", _mean_or_nan),
            mean_logloss_model=("logloss_model", "mean"),
            mean_logloss_kalshi=("logloss_kalshi", _mean_or_nan),
        )
        .reset_index(drop=True)
    )
    out["brier_skill"] = 1.0 - (out["mean_brier_model"] / out["mean_brier_kalshi"])
    out["logloss_skill"] = 1.0 - (out["mean_logloss_model"] / out["mean_logloss_kalshi"])
    out.loc[~np.isfinite(out["brier_skill"]), "brier_skill"] = np.nan
    out.loc[~np.isfinite(out["logloss_skill"]), "logloss_skill"] = np.nan
    return out


def strategy_summary(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame([{"strategy": label, "n_rows": 0, "brier_model": np.nan, "brier_kalshi": np.nan, "brier_skill": np.nan, "logloss_model": np.nan, "logloss_kalshi": np.nan, "logloss_skill": np.nan}])
    b_model = float(pd.to_numeric(df["brier_model"], errors="coerce").mean())
    b_k = float(pd.to_numeric(df["brier_kalshi"], errors="coerce").dropna().mean()) if df["brier_kalshi"].notna().any() else float("nan")
    ll_model = float(pd.to_numeric(df["logloss_model"], errors="coerce").mean())
    ll_k = float(pd.to_numeric(df["logloss_kalshi"], errors="coerce").dropna().mean()) if df["logloss_kalshi"].notna().any() else float("nan")
    b_skill = (1.0 - (b_model / b_k)) if np.isfinite(b_model) and np.isfinite(b_k) and b_k != 0 else np.nan
    ll_skill = (1.0 - (ll_model / ll_k)) if np.isfinite(ll_model) and np.isfinite(ll_k) and ll_k != 0 else np.nan
    return pd.DataFrame([{"strategy": label, "n_rows": int(len(df)), "brier_model": b_model, "brier_kalshi": b_k, "brier_skill": b_skill, "logloss_model": ll_model, "logloss_kalshi": ll_k, "logloss_skill": ll_skill}])


def kalshi_fee_total_dollars(c: int, p: float) -> float:
    if c <= 0:
        return 0.0
    if p is None or not np.isfinite(p):
        return float("nan")
    p = min(max(float(p), 0.0), 1.0)
    raw = 0.07 * float(c) * p * (1.0 - p)
    return float(math.ceil(raw * 100.0 - 1e-12) / 100.0)


def build_paper_trades(
    minutes_df: pd.DataFrame,
    cfg: RepoConfig,
    *,
    minute_lo: Optional[int] = None,
    minute_hi: Optional[int] = None,
    trade_filter_col: Optional[str] = None,
) -> pd.DataFrame:
    if minutes_df.empty:
        return pd.DataFrame()
    df = minutes_df.copy()
    df["time_sort_dt"] = pd.to_datetime(df["time_utc"], errors="coerce", utc=True)
    if minute_lo is not None:
        df = df.loc[df["minute_idx"] >= int(minute_lo)].copy()
    if minute_hi is not None:
        df = df.loc[df["minute_idx"] <= int(minute_hi)].copy()
    if trade_filter_col is not None:
        if trade_filter_col not in df.columns:
            return pd.DataFrame()
        df = df.loc[df[trade_filter_col].astype(bool)].copy()
    for c in ["p_model", "y", "kalshi_yes_ask", "kalshi_no_ask"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    c = int(cfg.paper_contracts_per_trade)
    extra = float(cfg.paper_extra_edge)
    df["edge_yes"] = df["p_model"] - df["kalshi_yes_ask"]
    df["edge_no"] = (1.0 - df["p_model"]) - df["kalshi_no_ask"]
    df["fee_total_yes"] = df["kalshi_yes_ask"].apply(lambda p: kalshi_fee_total_dollars(c, p) if np.isfinite(p) else np.nan)
    df["fee_total_no"] = df["kalshi_no_ask"].apply(lambda p: kalshi_fee_total_dollars(c, p) if np.isfinite(p) else np.nan)
    df["fee_pc_yes"] = df["fee_total_yes"] / float(c)
    df["fee_pc_no"] = df["fee_total_no"] / float(c)
    df["net_edge_yes"] = df["edge_yes"] - df["fee_pc_yes"]
    df["net_edge_no"] = df["edge_no"] - df["fee_pc_no"]
    df["qual_yes"] = np.isfinite(df["net_edge_yes"]) & (df["net_edge_yes"] >= extra)
    df["qual_no"] = np.isfinite(df["net_edge_no"]) & (df["net_edge_no"] >= extra)

    rows = []
    for _, r in df.iterrows():
        qy = bool(r["qual_yes"]) if pd.notna(r["qual_yes"]) else False
        qn = bool(r["qual_no"]) if pd.notna(r["qual_no"]) else False
        if not (qy or qn):
            continue
        if qy and qn:
            side = "NO" if float(r["net_edge_no"]) > float(r["net_edge_yes"]) else "YES"
        else:
            side = "YES" if qy else "NO"
        entry = float(r["kalshi_yes_ask"] if side == "YES" else r["kalshi_no_ask"])
        fee_pc = float(r["fee_pc_yes"] if side == "YES" else r["fee_pc_no"])
        edge_gross = float(r["edge_yes"] if side == "YES" else r["edge_no"])
        edge_net = float(r["net_edge_yes"] if side == "YES" else r["net_edge_no"])
        y = int(r["y"])
        pnl_gross = (y - entry) if side == "YES" else ((1 - y) - entry)
        pnl_net = pnl_gross - fee_pc
        out = r.to_dict()
        out.update({
            "trade_side": side,
            "C": c,
            "entry_price": entry,
            "fee_per_contract": fee_pc,
            "edge_chosen_gross": edge_gross,
            "edge_chosen_net": edge_net,
            "pnl_gross": pnl_gross,
            "pnl_net": pnl_net,
        })
        rows.append(out)

    if not rows:
        return pd.DataFrame()
    trades = pd.DataFrame(rows).sort_values(["time_sort_dt", "event_id", "strike_offset", "trade_side"]).reset_index(drop=True)
    trades["trade_idx"] = np.arange(len(trades), dtype=int)
    trades["equity"] = trades["pnl_net"].cumsum()
    return trades
