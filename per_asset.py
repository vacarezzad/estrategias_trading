"""Cada estrategia (parámetros default, sin elegir) sobre cada uno de los 30 activos del universo de rotación,
contra el buy & hold de ese mismo activo en el mismo período. Uso: python per_asset.py"""
import json
import numpy as np
import pandas as pd

from bt import data, engine, metrics, refine
from bt.strategies import REGISTRY as R

NAMES = ["sma_filter", "ema2_cross", "ema3_cross", "ema_ribbon", "macd", "bollinger_reversion", "bollinger_breakout",
         "stochastic", "atr_trailing", "ema_cross_macd", "ema2_atr_stop", "ribbon_atr", "trend_dip", "macd_stoch", "vote"]
costs = engine.Costs(slippage=0.002)
rows = []
for t in refine.UNIVERSE:
    px = data.load([t])
    for name in NAMES:
        fn = (lambda p, t=t: refine._sma(p, t)) if name == "sma_filter" else (lambda p, t=t, f=R[name]["fn"]: f(p, t=t, **{k: v for k, v in R[name]["default"].items()}))
        w = fn(px)
        valid = w.notna().any(axis=1)
        s = max(w.index[valid.values.argmax()], pd.Timestamp("2000-01-03"))
        res = engine.run(name, w, px, costs, start=s)
        bw = pd.DataFrame(np.nan, px.index, [t]); bw.loc[s] = 1.0
        bench = engine.run("bh", bw, px, costs, start=s)
        m = metrics.summary(res, bench)
        rows.append(dict(activo=t, estrategia=name, desde=m["desde"], cagr=m["cagr"], cagr_bh=m["cagr_bench"],
                         exceso=m["exceso_cagr"], sharpe=m["sharpe"], max_dd=m["max_dd"], max_dd_bh=m["max_dd_bench"],
                         ops=m["operaciones_anuales"], gana=m["cagr"] > m["cagr_bench"]))
    print(t, flush=True)
df = pd.DataFrame(rows)
df.to_csv("results/por_activo.csv", index=False, float_format="%.4f")
