"""Métricas de una curva contra el benchmark (buy & hold de SPY, que también paga su costo de entrada)."""
from __future__ import annotations

import numpy as np
import pandas as pd

TD = 252


def cagr(eq: pd.Series) -> float:
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return float((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1) if yrs > 0 else np.nan


def max_dd(eq: pd.Series) -> float:
    return float((eq / eq.cummax() - 1).min())


def summary(res, bench) -> dict:
    eq, b = res.equity, bench.equity.reindex(res.equity.index)
    r, rb = eq.pct_change().dropna(), b.pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    vol = float(r.std() * np.sqrt(TD))
    down = r[r < 0].std() * np.sqrt(TD)
    ex = (r - rb).dropna()
    te = ex.std() * np.sqrt(TD)
    beta = float(np.cov(r, rb)[0, 1] / rb.var()) if rb.var() > 0 else np.nan
    yr = eq.resample("YE").last().pct_change()
    yr.iloc[0] = eq.resample("YE").last().iloc[0] / eq.iloc[0] - 1
    ybr = b.resample("YE").last().pct_change()
    ybr.iloc[0] = b.resample("YE").last().iloc[0] / b.iloc[0] - 1
    traded = res.trades["weight_traded"].abs().sum() if len(res.trades) else 0.0
    out = {
        "estrategia": res.name,
        "desde": eq.index[0].date().isoformat(),
        "hasta": eq.index[-1].date().isoformat(),
        "cagr": cagr(eq),
        "cagr_bench": cagr(b),
        "exceso_cagr": cagr(eq) - cagr(b),
        "vol": vol,
        "sharpe": float(r.mean() / r.std() * np.sqrt(TD)) if r.std() > 0 else np.nan,
        "sortino": float(r.mean() * TD / down) if down > 0 else np.nan,
        "max_dd": max_dd(eq),
        "max_dd_bench": max_dd(b),
        "calmar": cagr(eq) / abs(max_dd(eq)) if max_dd(eq) < 0 else np.nan,
        "beta": beta,
        "info_ratio": float(ex.mean() * TD / te) if te > 0 else np.nan,
        "t_exceso": float(ex.mean() / ex.std() * np.sqrt(len(ex))) if ex.std() > 0 else np.nan,
        "exposicion_media": float(res.exposure.mean()),
        "rotacion_anual": float(traded / yrs) if yrs > 0 else np.nan,  # suma de |peso operado| por año
        "operaciones_anuales": float(len(res.trades) / yrs) if yrs > 0 else np.nan,
        "costo_anual": float(res.costs_paid / yrs / eq.mean()) if yrs > 0 else np.nan,  # aprox. % del patrimonio
        "anios_gana": int((yr > ybr).sum()),
        "anios": int(len(yr)),
    }
    return out


def yearly(res, bench) -> pd.DataFrame:
    def by_year(eq):
        y = eq.resample("YE").last()
        r = y.pct_change()
        r.iloc[0] = y.iloc[0] / eq.iloc[0] - 1
        return r
    df = pd.DataFrame({"estrategia": by_year(res.equity), "spy": by_year(bench.equity.reindex(res.equity.index))})
    df.index = df.index.year
    return df
