"""Carga de precios diarios ajustados (un CSV por ticker en data/prices, columnas Date,Open,High,Low,Close,Volume).

Todo queda alineado al calendario de SPY. No se rellenan aperturas faltantes: si un activo no cotiza
un día, ese día no se puede operar. Los cierres se rellenan hacia adelante solo para valuar posiciones.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRICES_DIR = os.path.join(ROOT, "data", "prices")
FIELDS = ("open", "high", "low", "close", "volume")


@dataclass
class Prices:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def tickers(self) -> list[str]:
        return list(self.close.columns)

    def slice(self, start=None, end=None) -> "Prices":
        return Prices(*(getattr(self, f).loc[start:end] for f in FIELDS))

    def upto(self, i: int) -> "Prices":
        """Datos hasta la fila i inclusive (para el test de leakage)."""
        return Prices(*(getattr(self, f).iloc[: i + 1] for f in FIELDS))


def _file(ticker: str) -> str:
    return os.path.join(PRICES_DIR, f"{ticker.replace('^', '_')}.csv")


def available() -> list[str]:
    return sorted(f[:-4].replace("_", "^", 1) if f.startswith("_") else f[:-4]
                  for f in os.listdir(PRICES_DIR) if f.endswith(".csv"))


def load(tickers, calendar: str = "SPY") -> Prices:
    tickers = list(dict.fromkeys([calendar, *tickers]))
    frames = {}
    for t in tickers:
        df = pd.read_csv(_file(t), parse_dates=["Date"], index_col="Date").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        frames[t] = df.rename(columns=str.lower)
    idx = frames[calendar].index
    out = {f: pd.DataFrame({t: frames[t][f].reindex(idx) for t in tickers}) for f in FIELDS}
    return Prices(**out)


def load_ccl() -> pd.Series:
    """CCL diario (ARS por USD) desde 2020, para expresar curvas en pesos."""
    s = pd.read_csv(os.path.join(ROOT, "data", "ccl_history.csv"), parse_dates=["fecha"], index_col="fecha")["ccl"]
    return s.sort_index()
