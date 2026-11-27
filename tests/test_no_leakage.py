"""Cada estrategia, calculada con datos truncados al día t, tiene que dar el mismo peso en t que calculada
con toda la historia. Si usa datos futuros (rolling sin shift, centrar ventanas, etc.), este test falla."""
import itertools

import numpy as np
import pandas as pd
import pytest

from bt import data, engine
from bt.strategies import REGISTRY


def _combos(grid):
    keys = list(grid)
    return [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))] or [{}]


@pytest.mark.parametrize("name", list(REGISTRY))
def test_truncation_invariance(name):
    spec = REGISTRY[name]
    px = data.load(spec["tickers"]).slice("2005-01-01", "2012-12-31")
    rng = np.random.default_rng(0)
    for params in _combos(spec["grid"])[:6]:
        full = spec["fn"](px, **params)
        for i in sorted(rng.choice(np.arange(300, len(px.index)), 12, replace=False)):
            part = spec["fn"](px.upto(i), **params).iloc[-1]
            a, b = full.iloc[i].values.astype(float), part.reindex(full.columns).values.astype(float)
            assert np.allclose(a, b, equal_nan=True), f"{name} {params} difiere en {px.index[i].date()}"


def test_execution_next_open_and_costs():
    px = data.load(["SPY"]).slice("2020-01-01", "2020-03-31")
    w = pd.DataFrame(np.nan, px.index, ["SPY"])
    w.iloc[0] = 1.0
    res = engine.run("t", w, px, engine.Costs(per_side=0.01))
    # día 0: señal, sin posición; día 1: compra a la apertura pagando 1%
    assert res.equity.iloc[0] == 1.0
    d1 = px.index[1]
    expected = 0.99 * px.close.at[d1, "SPY"] / px.open.at[d1, "SPY"]
    assert abs(res.equity.iloc[1] - expected) < 1e-3
    assert len(res.trades) == 1
