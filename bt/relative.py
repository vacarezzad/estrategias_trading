"""Ronda 3: análisis técnico RELATIVO sobre el universo de 30 CEDEARs (solo precios).

En vez de "¿entro o salgo?", cada rebalanceo elige los k activos más fuertes y queda siempre invertido
(salvo el interruptor opcional de tendencia del SPY). Puntajes:
- mom12_1: retorno de 12 meses sin el último mes (momentum clásico de Jegadeesh-Titman).
- mom_vol: mom12_1 dividido por la volatilidad de 6 meses (fuerza relativa ajustada por riesgo).
- high52: cierre / máximo de 52 semanas (George-Hwang).
Buffer: lo que ya está en cartera se mantiene mientras siga dentro del top k+buffer (menos rotación).
Arranca en 2004 para que todas las variantes sean comparables.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import strategies as S
from .refine import UNIVERSE

START = pd.Timestamp("2004-01-01")


def scores(px, kind: str) -> pd.DataFrame:
    c = px.close[UNIVERSE]
    hist = c.notna().rolling(252).sum() >= 252  # al menos un año de historia
    if kind == "mom12_1":
        s = c.shift(21) / c.shift(252) - 1
    elif kind == "mom_vol":
        s = (c.shift(21) / c.shift(252) - 1) / (c.pct_change(fill_method=None).rolling(126).std() * np.sqrt(252))
    elif kind == "high52":
        s = c / c.rolling(252).max()
    else:
        raise ValueError(kind)
    return s.where(hist)


def rebalance_days(idx: pd.DatetimeIndex, freq: str) -> np.ndarray:
    first = S.first_day_of_month(idx)
    if freq == "2M":
        first &= (idx.month % 2 == 1)
    elif freq == "3M":
        first &= np.isin(idx.month, [2, 5, 8, 11])
    return first


def spy_regime(px, band=0.01) -> pd.Series:
    c = px.close["SPY"]
    sma = c.rolling(200).mean()
    return S.state(c > sma * (1 + band), c < sma * (1 - band))


@S.strategy("rel_strength", UNIVERSE + ["SPY"],
            grid={"score": ["mom12_1", "mom_vol", "high52"], "k": [5, 8], "freq": ["M", "2M"],
                  "buffer": [0, 2], "regime": ["none", "half"]},
            default={"score": "mom12_1", "k": 5, "freq": "M", "buffer": 2, "regime": "none"},
            desc="Top k de los 30 por fuerza relativa, pesos iguales, rebalanceo mensual/bimestral con buffer. "
                 "regime=half: si el SPY está bajo su SMA 200, invierte solo 50% (resto liquidez).")
def rel_strength(px, score="mom12_1", k=5, freq="M", buffer=2, regime="none"):
    sc = scores(px, score)
    idx = sc.index
    reb = rebalance_days(idx, freq) & (idx >= START)
    reg = spy_regime(px) if regime != "none" else pd.Series(1.0, idx)
    w = pd.DataFrame(np.nan, idx, UNIVERSE)
    held: list[str] = []
    for i in np.nonzero(reb)[0]:
        row = sc.iloc[i].dropna().sort_values(ascending=False)
        if len(row) < k:
            continue
        ranked = list(row.index)
        keep = [t for t in held if t in ranked[: k + buffer]]
        new = [t for t in ranked if t not in keep][: k - len(keep)]
        held = keep + new
        expo = 1.0 if reg.iloc[i] == 1 else 0.5
        w.iloc[i] = 0.0
        w.loc[idx[i], held] = expo / k
    return w


@S.strategy("ew_universe", UNIVERSE + ["SPY"],
            desc="Referencia: los 30 en partes iguales (los que tienen un año de historia), sin rebalancear salvo cuando entra un activo nuevo.")
def ew_universe(px):
    elig = scores(px, "high52").notna()
    idx = elig.index
    elig = elig.astype(float)
    elig.loc[~S.first_day_of_month(idx)] = np.nan
    elig = elig.ffill().fillna(0.0).astype(bool)
    w = elig.div(elig.sum(axis=1), axis=0)
    changed = elig.ne(elig.shift()).any(axis=1)
    changed.iloc[0] = True
    changed.iloc[idx.searchsorted(START)] = True
    w.loc[~(changed & (idx >= START)).values] = np.nan
    return w
