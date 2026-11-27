"""Afinado de las 4 estrategias que quedaron más cerca del B&H (ronda 1, 2026-11-27).

Tres palancas, sin tocar los parámetros "de manual" de cada regla:
1. activo: aplicar la regla a SPY o QQQ.
2. refugio: cuando la regla dice "afuera", ir a liquidez, GLD (oro) o IEF (bonos del Tesoro 7-10 años).
3. piso: en "afuera" mantener una parte (0% o 50%) del activo en vez de salir del todo.

Todas arrancan cuando hay datos de GLD e IEF (fines de 2004) para que las variantes sean comparables.
También hay una versión por acción sobre el universo de 30 CEDEARs de rotacion_cartera.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import strategies as S
from .data import Prices

BASES = {
    "ema2_cross": lambda px, t: S.ema2_cross(px, 50, 200, "W", t=t),
    "ema_cross_macd": lambda px, t: S.ema_cross_macd(px, 50, 200, "W", t=t),
    "sma_filter": lambda px, t: _sma(px, t),
    "trend_dip": lambda px, t: S.trend_dip(px, 200, "any", 0.02, t=t),
}
REFUGES = ["cash", "GLD", "IEF"]


def _sma(px, t, n=200, band=0.01):
    c = px.close[t]
    sma = c.rolling(n, min_periods=n).mean()
    st = S.state(c > sma * (1 + band), c < sma * (1 - band))
    return S.sample(st, t, "M", sma.notna())


def overlay(base: pd.DataFrame, px: Prices, t: str, refuge: str, floor: float) -> pd.DataFrame:
    st = base[t].ffill()  # 1 adentro, 0 afuera
    valid = st.notna() & px.close["GLD"].notna() & px.close["IEF"].notna()
    cols = [t] + ([refuge] if refuge != "cash" else [])
    w = pd.DataFrame(0.0, st.index, cols)
    w[t] = np.where(st == 1, 1.0, floor)
    if refuge != "cash":
        w[refuge] = np.where(st == 1, 0.0, 1.0 - floor)
    w[~valid] = np.nan
    return S.only_on_changes(w)


def _register(name):
    def fn(px, t="SPY", refuge="cash", floor=0.0):
        return overlay(BASES[name](px, t), px, t, refuge, floor)
    S.strategy(f"af_{name}", ["SPY", "QQQ", "GLD", "IEF"],
               grid={"t": ["SPY", "QQQ"], "refuge": REFUGES, "floor": [0.0, 0.5]},
               default={"t": "SPY", "refuge": "GLD", "floor": 0.0},
               desc=f"{name} con activo, refugio y piso de exposición configurables.")(fn)


for _n in BASES:
    _register(_n)


# ---------------------------------------------------------------- por acción, universo de 30 CEDEARs
UNIVERSE = ["NVDA", "GOOGL", "AMD", "QQQ", "MELI", "VIST", "META", "MSFT", "AMZN", "GLD", "AVGO", "MCD", "TSM",
            "ORCL", "TSLA", "MRNA", "GLOB", "AAPL", "KO", "BABA", "INTC", "PEP", "PBR", "V", "IBM", "WMT", "JPM",
            "ASML", "MRVL", "UNH"]


def _per_stock(px, rule, refuge):
    """1/30 por acción mientras su regla esté adentro; lo que sobra (acciones afuera o sin historia) va al refugio."""
    sts = {}
    for t in UNIVERSE:
        if rule == "always":
            sts[t] = px.close[t].notna().astype(float).where(px.close[t].notna())
        else:
            sts[t] = BASES[rule](px, t)[t].ffill()
    st = pd.DataFrame(sts).fillna(0.0)
    # revisión mensual para cartera de 30 nombres (limita operaciones)
    st.loc[~S.first_day_of_month(st.index)] = np.nan
    st = st.ffill()
    w = st / len(UNIVERSE)
    if refuge != "cash":
        w[refuge] = w.get(refuge, 0.0) + (1 - w.sum(axis=1))
    w = w.loc[:, ~w.columns.duplicated()]
    w[w.index < pd.Timestamp("2005-01-01")] = np.nan  # mismo arranque que el resto del afinado
    return S.only_on_changes(w)


@S.strategy("af_per_stock", UNIVERSE + ["SPY", "IEF"],
            grid={"rule": ["always", "ema2_cross", "sma_filter", "trend_dip"], "refuge": ["cash", "SPY"]},
            default={"rule": "ema2_cross", "refuge": "SPY"},
            desc="Regla de tendencia aplicada acción por acción sobre los 30 CEDEARs de rotación (1/30 cada una). "
                 "rule=always es el B&H equiponderado del universo, la referencia justa (sesgo de supervivencia).")
def af_per_stock(px, rule="ema2_cross", refuge="SPY"):
    return _per_stock(px, rule, refuge)
