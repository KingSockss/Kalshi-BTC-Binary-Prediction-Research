from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go


def save_plot(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")


def plot_one_line(x: list[Any], y: list[float], name: str, title: str, x_title: str, y_title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=name))
    fig.update_layout(title=title, xaxis=dict(title=x_title), yaxis=dict(title=y_title), hovermode="x unified")
    return fig


def plot_two_lines(x: list[Any], y1: list[float], y2: list[float], name1: str, name2: str, title: str, x_title: str, y_title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y1, mode="lines+markers", name=name1))
    fig.add_trace(go.Scatter(x=x, y=y2, mode="lines+markers", name=name2))
    fig.update_layout(title=title, xaxis=dict(title=x_title), yaxis=dict(title=y_title), hovermode="x unified")
    return fig


def plot_calibration(calib_model: pd.DataFrame, calib_mkt: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line=dict(dash="dash")))
    if not calib_model.empty:
        fig.add_trace(go.Scatter(x=calib_model["p_mean"], y=calib_model["y_mean"], mode="lines+markers", name="Model"))
    if not calib_mkt.empty:
        fig.add_trace(go.Scatter(x=calib_mkt["p_mean"], y=calib_mkt["y_mean"], mode="lines+markers", name="Kalshi"))
    fig.update_layout(title=title, xaxis=dict(title="Predicted probability", range=[0, 1]), yaxis=dict(title="Empirical frequency", range=[0, 1]), hovermode="closest")
    return fig


def plot_heatmap_skill(df: pd.DataFrame, x_col: str, y_col: str, value_col: str, title: str, x_title: str, y_title: str) -> go.Figure:
    pivot = df.pivot_table(index=y_col, columns=x_col, values=value_col, aggfunc="mean")
    fig = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns.astype(str).tolist(), y=pivot.index.astype(str).tolist(), colorbar=dict(title=value_col)))
    fig.update_layout(title=title, xaxis=dict(title=x_title), yaxis=dict(title=y_title))
    return fig
