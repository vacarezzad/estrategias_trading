"""Una posición a la vez sobre las 10 CEDEARs más líquidas: cuando una estrategia técnica da señal de compra,
se entra con el 100% en la MEJOR de las que dieron señal (mayor fuerza relativa) y se sale con la señal de venta
de esa misma estrategia. Mientras no hay posición: liquidez o SPY.

Modos de entrada:
- event: solo entra el día en que la señal se enciende (cruce, ruptura, etc.).
- state: si está afuera, entra en la mejor de las que están "adentro" según la regla (siempre en tendencia).
Señal al cierre de t, ejecución a la apertura de t+1 (lo hace el motor). Uso: python single_best.py
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from bt import data, engine, metrics, refine, relative
from bt.strategies import REGISTRY as R

T10 = relative.TOP10
BASES = ["sma_filter", "ema2_cross", "ema3_cross", "ema_ribbon", "macd", "bollinger_reversion", "bollinger_breakout",
         "stochastic", "atr_trailing", "ema_cross_macd", "ema2_atr_stop", "ribbon_atr", "trend_dip", "macd_stoch", "vote"]
GRID = {"slots": [1, 2], "rank": ["mom12_1", "mom3"], "park": ["cash", "SPY"], "mode": ["event", "state"]}
DEFAULT = {"slots": 1, "rank": "mom12_1", "park": "cash", "mode": "event"}
SPLIT, OOS = "2014-12-31", "2015-01-02"


def states(px, base) -> pd.DataFrame:
    out = {}
    for t in T10:
        if base == "sma_filter":
            w = refine._sma(px, t)
        else:
            w = R[base]["fn"](px, t=t, **R[base]["default"])
        out[t] = w[t].ffill()
    return pd.DataFrame(out)


def rank_scores(px, kind):
    c = px.close[T10]
    s = c.shift(21) / c.shift(252) - 1 if kind == "mom12_1" else c / c.shift(63) - 1
    return s


def single_best(px, st: pd.DataFrame, rank="mom12_1", park="cash", mode="event", slots=1) -> pd.DataFrame:
    sc = rank_scores(px, rank).values
    S = st.fillna(0.0).values
    on = (S >= 0.5)
    ev = on & ~np.vstack([np.zeros((1, on.shape[1]), bool), on[:-1]])
    idx = st.index
    cols = T10 + (["SPY"] if park == "SPY" else [])
    W = np.full((len(idx), len(cols)), np.nan)
    held: list[int] = []
    start = idx.searchsorted(relative.START)
    for i in range(start, len(idx)):
        changed = i == start
        keep = [h for h in held if on[i, h]]
        if len(keep) != len(held):
            held, changed = keep, True
        while len(held) < slots:
            cand = (ev[i] if mode == "event" else on[i]) & np.isfinite(sc[i])
            cand[held] = False
            if not cand.any():
                break
            held.append(int(np.nanargmax(np.where(cand, sc[i], -np.inf))))
            changed = True
        if changed:
            row = np.zeros(len(cols))
            row[held] = 1.0 / slots
            if park == "SPY":
                row[-1] = 1.0 - len(held) / slots
            W[i] = row
    return pd.DataFrame(W, idx, cols)


def combos():
    return [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]


def main():
    px = data.load(T10 + ["SPY"])
    costs = engine.Costs(slippage=0.002)
    bench = {s: engine.run("spy", R["buy_hold_spy"]["fn"](px.slice(s, None)), px, costs, start=s)
             for s in ["2004-01-02", OOS]}
    rows, grid_rows = [], []
    for base in BASES:
        st = states(px, base)
        best, best_sh = None, -np.inf
        for p in combos():
            w = single_best(px, st, **p)
            full = engine.run(base, w, px, costs, start="2004-01-02")
            is_ = engine.run(base, w, px, costs, start="2004-01-02", end=SPLIT)
            mf = metrics.summary(full, bench["2004-01-02"])
            mi = metrics.summary(is_, bench["2004-01-02"])
            grid_rows.append({"estrategia": base, "params": json.dumps(p), "cagr": mf["cagr"], "sharpe": mf["sharpe"],
                              "max_dd": mf["max_dd"], "ops": mf["operaciones_anuales"], "sharpe_is": mi["sharpe"]})
            if p == DEFAULT:
                rows.append(mf | {"variante": "default", "params": json.dumps(p)})
                oos = engine.run(base, w, px, costs, start=OOS)
                rows.append(metrics.summary(oos, bench[OOS]) | {"variante": "OOS default", "params": json.dumps(p)})
            if mi["sharpe"] > best_sh:
                best, best_sh = p, mi["sharpe"]
        oos = engine.run(base, single_best(px, st, **best), px, costs, start=OOS)
        rows.append(metrics.summary(oos, bench[OOS]) | {"variante": "OOS (elegida en IS)", "params": json.dumps(best)})
        print(base, flush=True)
    pd.DataFrame(rows).to_csv("results_top10/single_best_resumen.csv", index=False, float_format="%.4f")
    pd.DataFrame(grid_rows).to_csv("results_top10/single_best_grilla.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
