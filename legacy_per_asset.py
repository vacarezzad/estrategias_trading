"""Estrategias del notebook de Diego, una por una, sobre cada activo (30 del universo; las 10 líquidas marcadas),
contra el buy & hold del mismo activo y período, con costos IOL. Sin stop y con el stop de 2,5% del notebook."""
import numpy as np
import pandas as pd

from bt import data, engine, legacy, metrics, relative, strategies as S

costs = engine.Costs(slippage=0.002)
rows = []
for t in relative.UNIVERSE:
    px = data.load([t])
    c = px.close[t]
    for name in legacy.STRATS:
        for sl in (None, 0.025):
            st = legacy.state_for(px, t, name, sl)
            valid = c.notna() & (np.arange(len(c)) >= c.notna().values.argmax() + 252)  # 1 año de calentamiento
            w = S.sample(st, t, "D", pd.Series(valid, c.index))
            s = max(w.index[w.notna().any(axis=1).values.argmax()], pd.Timestamp("2000-01-03"))
            res = engine.run(name, w, px, costs, start=s)
            bw = pd.DataFrame(np.nan, px.index, [t]); bw.loc[s] = 1.0
            bench = engine.run("bh", bw, px, costs, start=s)
            m = metrics.summary(res, bench)
            rows.append(dict(activo=t, top10=t in relative.TOP10, estrategia=name, stop="2,5%" if sl else "no",
                             desde=m["desde"], cagr=m["cagr"], cagr_bh=m["cagr_bench"], exceso=m["exceso_cagr"],
                             sharpe=m["sharpe"], max_dd=m["max_dd"], max_dd_bh=m["max_dd_bench"],
                             ops=m["operaciones_anuales"], expo=m["exposicion_media"], gana=m["cagr"] > m["cagr_bench"]))
    print(t, flush=True)
pd.DataFrame(rows).to_csv("results/notebook_por_activo.csv", index=False, float_format="%.4f")
