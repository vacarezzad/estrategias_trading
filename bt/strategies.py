"""Estrategias. Cada una es `f(px: Prices, **params) -> DataFrame de pesos objetivo` con la regla:
la fila t solo puede usar datos hasta el cierre de t (lo verifica tests/test_no_leakage.py).

Convenciones:
- Fila NaN = no tocar la cartera.
- Rebalanceos "mensuales" se deciden al cierre del primer día hábil del mes (no hace falta saber que
  mañana cambia el mes, así el test de truncado lo puede verificar) y se ejecutan a la apertura siguiente.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Prices

REGISTRY: dict[str, dict] = {}


def strategy(name: str, tickers, grid: dict | None = None, default: dict | None = None, desc: str = ""):
    def deco(fn):
        REGISTRY[name] = dict(fn=fn, tickers=list(tickers), grid=grid or {}, default=default or {}, desc=desc)
        return fn
    return deco


def first_day_of_month(idx: pd.DatetimeIndex) -> np.ndarray:
    m = idx.to_period("M")
    flag = np.ones(len(idx), dtype=bool)
    flag[1:] = m[1:] != m[:-1]
    return flag


def only_on_changes(w: pd.DataFrame) -> pd.DataFrame:
    """Deja filas solo cuando el objetivo cambia (evita rebalancear la deriva todos los días)."""
    changed = w.ne(w.shift()).any(axis=1)
    changed.iloc[0] = True
    return w.where(changed)


# ---------------------------------------------------------------- benchmark
@strategy("buy_hold_spy", ["SPY"], desc="Comprar SPY el primer día y no tocar nunca.")
def buy_hold_spy(px: Prices):
    w = pd.DataFrame(np.nan, px.index, ["SPY"])
    w.iloc[0] = 1.0
    return w


# ---------------------------------------------------------------- 1. filtro de tendencia
@strategy("sma_filter_spy", ["SPY"],
          grid={"n": [100, 150, 200, 250], "band": [0.0, 0.01, 0.02], "freq": ["D", "M"]},
          default={"n": 200, "band": 0.01, "freq": "M"},
          desc="SPY si el cierre está sobre su media de n días (con banda para evitar serruchos); si no, liquidez.")
def sma_filter_spy(px: Prices, n=200, band=0.01, freq="M"):
    c = px.close["SPY"]
    sma = c.rolling(n, min_periods=n).mean()
    state = pd.Series(np.nan, c.index)
    state[c > sma * (1 + band)] = 1.0
    state[c < sma * (1 - band)] = 0.0
    state = state.ffill()
    if freq == "M":
        state = state.where(first_day_of_month(c.index)).ffill()
    state[sma.isna()] = np.nan
    return only_on_changes(state.to_frame("SPY"))
