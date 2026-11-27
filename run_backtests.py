"""Corre todas las estrategias registradas contra el buy & hold de SPY.

Para cada estrategia:
- default: los parámetros "de manual" fijados de antemano.
- grilla: TODAS las combinaciones (se reporta la distribución, no solo la mejor).
- IS/OOS: se elige la mejor combinación por Sharpe en el tramo de elección y se mide en el tramo de prueba,
  que nunca se usó para elegir. Es la cifra honesta para comparar con el B&H.

Uso: python run_backtests.py [--only nombre] [--slippage 0.002] [--split 2014-12-31]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd

from bt import data, engine, metrics
from bt import refine  # noqa: F401 (registra las estrategias afinadas)
from bt.strategies import REGISTRY

OUT = "results"  # se puede cambiar con --out


def combos(grid):
    keys = list(grid)
    return [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))] or [{}]


def first_signal(w: pd.DataFrame):
    valid = w.notna().any(axis=1)
    return w.index[valid.values.argmax()]


def run_one(name, spec, px, costs, start, end, params):
    w = spec["fn"](px, **params)
    s = max(pd.Timestamp(start), first_signal(w))
    res = engine.run(name, w, px, costs, start=s, end=end, params=params)
    bw = REGISTRY["buy_hold_spy"]["fn"](px.slice(s, end))
    bench = engine.run("buy_hold_spy", bw, px, costs, start=s, end=end)
    return res, bench


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="prefijo del nombre, ej. af_")
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--split", default="2014-12-31", help="fin del tramo de elección de parámetros")
    ap.add_argument("--cost", type=float, default=engine.IOL_COST_PER_SIDE)
    ap.add_argument("--slippage", type=float, default=0.002, help="prima CEDEAR/CCL + spread, por lado")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    global OUT
    OUT = a.out
    costs = engine.Costs(per_side=a.cost, slippage=a.slippage)
    os.makedirs(OUT, exist_ok=True)

    names = [n for n in REGISTRY if n != "buy_hold_spy" and (not a.only or n.startswith(a.only))]
    rows, grid_rows, curves, yearly = [], [], {}, {}
    for name in names:
        spec = REGISTRY[name]
        px = data.load(spec["tickers"])
        # 1) default, período completo
        res, bench = run_one(name, spec, px, costs, a.start, a.end, spec["default"])
        m = metrics.summary(res, bench) | {"variante": "default", "params": json.dumps(spec["default"])}
        rows.append(m)
        curves[name] = res.equity
        curves.setdefault("buy_hold_spy", bench.equity)
        yearly[name] = metrics.yearly(res, bench)
        # 2) grilla completa, período completo + IS/OOS
        best, best_sh = None, -np.inf
        for p in combos(spec["grid"]):
            r_full, b_full = run_one(name, spec, px, costs, a.start, a.end, p)
            r_is, b_is = run_one(name, spec, px, costs, a.start, a.split, p)
            mf, mi = metrics.summary(r_full, b_full), metrics.summary(r_is, b_is)
            grid_rows.append({"estrategia": name, "params": json.dumps(p), "cagr": mf["cagr"], "exceso_cagr": mf["exceso_cagr"],
                              "sharpe": mf["sharpe"], "max_dd": mf["max_dd"], "sharpe_is": mi["sharpe"]})
            if mi["sharpe"] > best_sh:
                best, best_sh = p, mi["sharpe"]
        if spec["grid"]:
            oos_start = (pd.Timestamp(a.split) + pd.Timedelta(days=1)).date().isoformat()
            r_oos, b_oos = run_one(name, spec, px, costs, oos_start, a.end, best)
            rows.append(metrics.summary(r_oos, b_oos) | {"variante": "OOS (elegida en IS)", "params": json.dumps(best)})
            r_def, b_def = run_one(name, spec, px, costs, oos_start, a.end, spec["default"])
            rows.append(metrics.summary(r_def, b_def) | {"variante": "OOS default", "params": json.dumps(spec["default"])})
        print(f"{name}: {len(combos(spec['grid']))} combinaciones")

    summ = pd.DataFrame(rows)
    summ.to_csv(f"{OUT}/resumen.csv", index=False, float_format="%.4f")
    pd.DataFrame(grid_rows).to_csv(f"{OUT}/grilla.csv", index=False, float_format="%.4f")
    pd.DataFrame(curves).to_csv(f"{OUT}/curvas.csv", float_format="%.5f")
    pd.concat(yearly, names=["estrategia", "anio"]).to_csv(f"{OUT}/anual.csv", float_format="%.4f")
    cols = ["estrategia", "variante", "desde", "cagr", "cagr_bench", "sharpe", "max_dd", "max_dd_bench",
            "exposicion_media", "operaciones_anuales", "costo_anual", "anios_gana", "anios", "t_exceso", "params"]
    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(summ[cols].to_string(index=False))
        g = pd.DataFrame(grid_rows)
        if len(g):
            print("\nGrilla (período completo): distribución del exceso de CAGR vs SPY")
            print(g.groupby("estrategia")[["exceso_cagr", "sharpe", "max_dd"]].describe(percentiles=[.25, .5, .75]).T.to_string())


if __name__ == "__main__":
    main()
