"""Motor de backtest de cartera por pesos objetivo.

Contrato con las estrategias
----------------------------
Una estrategia recibe `Prices` y devuelve un DataFrame de pesos objetivo (fechas x tickers) calculado
con información disponible **al cierre** de cada fecha. Una fila con todos NaN significa "no cambiar nada".
Los pesos suman <= 1; el resto queda en liquidez (USD sin rendimiento salvo que se indique `cash_ticker`).

El motor ejecuta la orden a la **apertura del día hábil siguiente** (nunca al mismo cierre que generó la
señal), cobra comisión + slippage por lado sobre el monto operado y valúa al cierre.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import Prices

IOL_COST_PER_SIDE = 0.0094  # comisión + derechos de mercado + IVA, aproximado, por lado


@dataclass
class Costs:
    per_side: float = IOL_COST_PER_SIDE
    slippage: float = 0.0  # prima/descuento del CEDEAR vs CCL y spread, por lado
    min_trade: float = 0.01  # no operar ajustes menores a este % del patrimonio

    @property
    def total(self) -> float:
        return self.per_side + self.slippage


@dataclass
class Result:
    name: str
    equity: pd.Series  # patrimonio al cierre, arranca en 1
    exposure: pd.Series  # % invertido al cierre
    weights: pd.DataFrame  # pesos al cierre
    trades: pd.DataFrame  # una fila por ejecución (fecha, ticker, monto, costo)
    costs_paid: float
    params: dict = field(default_factory=dict)


def run(name: str, target: pd.DataFrame, px: Prices, costs: Costs | None = None,
        start=None, end=None, cash_ticker: str | None = None, params: dict | None = None) -> Result:
    costs = costs or Costs()
    c = costs.total
    idx = px.index
    if start is not None:
        idx = idx[idx >= pd.Timestamp(start)]
    if end is not None:
        idx = idx[idx <= pd.Timestamp(end)]
    tickers = list(target.columns)
    O = px.open[tickers].reindex(idx)
    C = px.close[tickers].ffill().reindex(idx)
    # señal al cierre de t -> orden para la apertura de t+1
    shifted = target.reindex(px.index).shift(1)
    orders = shifted.reindex(idx)
    # si el backtest arranca a mitad de historia, la primera orden es el objetivo vigente a esa fecha
    if len(idx) and orders.iloc[0].isna().all():
        prev = shifted.loc[: idx[0]].dropna(how="all")
        if len(prev):
            orders.iloc[0] = prev.iloc[-1].values
    cash_ret = (px.close[cash_ticker].pct_change().reindex(idx).fillna(0.0).values
                if cash_ticker else np.zeros(len(idx)))

    n = len(tickers)
    sh = np.zeros(n)
    cash = 1.0
    eq_out, expo_out, w_out, trades = [], [], [], []
    paid = 0.0
    Ov, Cv, Ordv = O.values, C.values, orders.values

    for k, d in enumerate(idx):
        if k > 0 and cash > 0:
            cash *= 1 + cash_ret[k]
        row = Ordv[k]
        if not np.all(np.isnan(row)):
            op = Ov[k]
            tradable = np.isfinite(op)
            # valor actual a la apertura (activo sin apertura se valúa al último cierre y no se toca)
            last_c = Cv[k - 1] if k > 0 else Cv[k]
            mark = np.where(tradable, op, np.nan_to_num(last_c))
            cur = sh * np.nan_to_num(mark)
            eq = cash + cur.sum()
            w = np.nan_to_num(row)
            # dos pasadas: el objetivo se calcula sobre el patrimonio neto del costo de la propia orden
            eq_net = eq
            for _ in range(2):
                tgt = np.where(tradable, w * eq_net, cur)
                dv = tgt - cur
                dv[np.abs(dv) < costs.min_trade * eq] = 0.0
                cost = c * np.abs(dv).sum()
                eq_net = eq - cost
            if np.any(dv != 0):
                for j in np.nonzero(dv)[0]:
                    trades.append((d, tickers[j], float(dv[j] / eq), float(c * abs(dv[j]))))
                sh = sh + np.where(dv != 0, dv / np.where(tradable, op, 1.0), 0.0)
                cash -= dv.sum() + cost
                paid += cost
        val = sh * np.nan_to_num(Cv[k])
        eq_c = cash + val.sum()
        eq_out.append(eq_c)
        expo_out.append(val.sum() / eq_c if eq_c else 0.0)
        w_out.append(val / eq_c if eq_c else val * 0)

    tr = pd.DataFrame(trades, columns=["date", "ticker", "weight_traded", "cost"])
    return Result(
        name=name,
        equity=pd.Series(eq_out, idx, name=name),
        exposure=pd.Series(expo_out, idx, name=name),
        weights=pd.DataFrame(w_out, idx, tickers),
        trades=tr,
        costs_paid=paid,
        params=params or {},
    )
