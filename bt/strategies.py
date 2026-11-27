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


# ================================================================ indicadores técnicos (todos causales)
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def atr(px: Prices, t: str, n: int) -> pd.Series:
    h, l, c = px.high[t], px.low[t], px.close[t]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def first_day_of_week(idx: pd.DatetimeIndex) -> np.ndarray:
    w = idx.to_period("W")
    flag = np.ones(len(idx), dtype=bool)
    flag[1:] = w[1:] != w[:-1]
    return flag


def long_flat(t: str, entry: pd.Series, exit_: pd.Series, freq: str = "D", valid: pd.Series | None = None) -> pd.DataFrame:
    """Estado comprado/afuera con histéresis: entra cuando `entry`, sale cuando `exit_`, si no mantiene.
    freq "W" o "M": solo se revisa el primer día hábil de la semana/mes (menos operaciones)."""
    st = pd.Series(np.nan, entry.index)
    st[exit_.fillna(False).astype(bool)] = 0.0
    st[entry.fillna(False).astype(bool)] = 1.0
    st = st.ffill().fillna(0.0)
    if freq in ("W", "M"):
        flag = first_day_of_week(st.index) if freq == "W" else first_day_of_month(st.index)
        st = st.where(flag).ffill()
    if valid is not None:
        st[~valid.fillna(False).astype(bool)] = np.nan
        first = valid.fillna(False).values.argmax()
        st.iloc[first:] = st.iloc[first:].fillna(0.0)
    return only_on_changes(st.to_frame(t))


FREQS = ["D", "W"]


# ---------------------------------------------------------------- cruce de 2 EMAs
@strategy("ema2_cross", ["SPY"],
          grid={"fast": [10, 20, 50], "slow": [100, 150, 200], "freq": FREQS},
          default={"fast": 50, "slow": 200, "freq": "W"},
          desc="Comprado mientras la EMA rápida está sobre la lenta.")
def ema2_cross(px, fast=50, slow=200, freq="W", t="SPY"):
    c = px.close[t]
    f, s = ema(c, fast), ema(c, slow)
    return long_flat(t, f > s, f < s, freq, s.notna())


# ---------------------------------------------------------------- cruce de 3 EMAs
@strategy("ema3_cross", ["SPY"],
          grid={"emas": ["5-20-50", "10-30-100", "20-50-200", "9-21-55"], "freq": FREQS},
          default={"emas": "20-50-200", "freq": "W"},
          desc="Entra con las tres EMAs alineadas (rápida > media > lenta); sale cuando la rápida cruza bajo la media.")
def ema3_cross(px, emas="20-50-200", freq="W", t="SPY"):
    a, b, d = (int(x) for x in emas.split("-"))
    c = px.close[t]
    ea, eb, ed = ema(c, a), ema(c, b), ema(c, d)
    return long_flat(t, (ea > eb) & (eb > ed), ea < eb, freq, ed.notna())


# ---------------------------------------------------------------- EMA ribbon
@strategy("ema_ribbon", ["SPY"],
          grid={"base": [5, 10, 20], "k_in": [6, 8], "k_out": [3, 4], "freq": FREQS},
          default={"base": 10, "k_in": 8, "k_out": 4, "freq": "W"},
          desc="Abanico de 8 EMAs (base·1..8). Entra si el precio está sobre al menos k_in de ellas; sale si queda sobre k_out o menos.")
def ema_ribbon(px, base=10, k_in=8, k_out=4, freq="W", t="SPY"):
    c = px.close[t]
    rib = pd.concat([ema(c, base * i) for i in range(1, 9)], axis=1)
    above = rib.lt(c, axis=0).sum(axis=1)
    return long_flat(t, above >= k_in, above <= k_out, freq, rib.iloc[:, -1].notna())


# ---------------------------------------------------------------- MACD
@strategy("macd", ["SPY"],
          grid={"fast_slow_sig": ["12-26-9", "8-17-9", "19-39-9"], "zero_filter": [False, True], "freq": FREQS},
          default={"fast_slow_sig": "12-26-9", "zero_filter": False, "freq": "W"},
          desc="Comprado con MACD sobre su línea de señal (opcional: solo si además el MACD es positivo).")
def macd(px, fast_slow_sig="12-26-9", zero_filter=False, freq="W", t="SPY"):
    f, s, g = (int(x) for x in fast_slow_sig.split("-"))
    c = px.close[t]
    m = ema(c, f) - ema(c, s)
    sig = m.ewm(span=g, adjust=False, min_periods=g).mean()
    entry = (m > sig) & ((m > 0) if zero_filter else True)
    exit_ = (m < sig) | ((m < 0) if zero_filter else False)
    return long_flat(t, entry, exit_, freq, sig.notna())


# ---------------------------------------------------------------- Bollinger
@strategy("bollinger_reversion", ["SPY"],
          grid={"n": [10, 20], "k": [1.5, 2.0, 2.5], "trend_filter": [False, True]},
          default={"n": 20, "k": 2.0, "trend_filter": True},
          desc="Reversión: compra con cierre bajo la banda inferior, vende al volver a la media. Filtro opcional: solo si el precio está sobre la SMA 200.")
def bollinger_reversion(px, n=20, k=2.0, trend_filter=True, t="SPY"):
    c = px.close[t]
    mid, sd = c.rolling(n).mean(), c.rolling(n).std()
    entry = c < mid - k * sd
    if trend_filter:
        entry &= c > c.rolling(200).mean()
    return long_flat(t, entry, c > mid, "D", c.rolling(200 if trend_filter else n).mean().notna())


@strategy("bollinger_breakout", ["SPY"],
          grid={"n": [20, 50], "k": [1.0, 1.5, 2.0], "freq": FREQS},
          default={"n": 20, "k": 2.0, "freq": "W"},
          desc="Ruptura: compra con cierre sobre la banda superior, sale con cierre bajo la media.")
def bollinger_breakout(px, n=20, k=2.0, freq="W", t="SPY"):
    c = px.close[t]
    mid, sd = c.rolling(n).mean(), c.rolling(n).std()
    return long_flat(t, c > mid + k * sd, c < mid, freq, mid.notna())


# ---------------------------------------------------------------- Estocástico
@strategy("stochastic", ["SPY"],
          grid={"n": [5, 14], "low": [20, 30], "trend_filter": [False, True]},
          default={"n": 14, "low": 20, "trend_filter": True},
          desc="%K cruza hacia arriba a %D estando en sobreventa (< low): compra; %K > 80: vende. Filtro opcional SMA 200.")
def stochastic(px, n=14, low=20, trend_filter=True, t="SPY"):
    c, h, l = px.close[t], px.high[t], px.low[t]
    hh, ll = h.rolling(n).max(), l.rolling(n).min()
    k = 100 * (c - ll) / (hh - ll)
    d = k.rolling(3).mean()
    entry = (k > d) & (k.shift() <= d.shift()) & (k.shift() < low)
    if trend_filter:
        entry &= c > c.rolling(200).mean()
    return long_flat(t, entry, k > 80, "D", c.rolling(200 if trend_filter else n + 3).mean().notna())


# ---------------------------------------------------------------- ATR: tendencia con trailing stop
@strategy("atr_trailing", ["SPY"],
          grid={"entry_n": [50, 100, 200], "atr_n": [14, 22], "mult": [2.0, 3.0, 4.0]},
          default={"entry_n": 100, "atr_n": 22, "mult": 3.0},
          desc="Entra con cierre en máximo de entry_n días; sale si el cierre cae mult·ATR desde el máximo desde la entrada (chandelier).")
def atr_trailing(px, entry_n=100, atr_n=22, mult=3.0, t="SPY"):
    c = px.close[t]
    a = atr(px, t, atr_n)
    hi = c.rolling(entry_n).max()
    cv, av, hv = c.values, a.values, hi.values
    st = np.full(len(c), np.nan)
    inpos, peak = False, 0.0
    for i in range(len(c)):
        if not (np.isfinite(av[i]) and np.isfinite(hv[i])):
            continue
        if inpos:
            peak = max(peak, cv[i])
            if cv[i] < peak - mult * av[i]:
                inpos = False
        elif cv[i] >= hv[i]:
            inpos, peak = True, cv[i]
        st[i] = 1.0 if inpos else 0.0
    return only_on_changes(pd.Series(st, c.index).to_frame(t))


# ================================================================ combinaciones
def state(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    """Estado diario 0/1 con histéresis (sin muestreo semanal)."""
    st = pd.Series(np.nan, entry.index)
    st[exit_.fillna(False).astype(bool)] = 0.0
    st[entry.fillna(False).astype(bool)] = 1.0
    return st.ffill().fillna(0.0)


def chandelier(c: pd.Series, a: pd.Series, entry: pd.Series, mult: float, exit_extra: pd.Series | None = None) -> pd.Series:
    """Entra con `entry`; sale con trailing stop mult·ATR desde el máximo (o con `exit_extra`)."""
    cv, av, ev = c.values, a.values, entry.fillna(False).values
    xv = exit_extra.fillna(False).values if exit_extra is not None else np.zeros(len(c), bool)
    st = np.zeros(len(c))
    inpos, peak = False, 0.0
    for i in range(len(c)):
        if inpos:
            peak = max(peak, cv[i])
            if (np.isfinite(av[i]) and cv[i] < peak - mult * av[i]) or xv[i]:
                inpos = False
        elif ev[i]:
            inpos, peak = True, cv[i]
        st[i] = 1.0 if inpos else 0.0
    return pd.Series(st, c.index)


def sample(st: pd.Series, t: str, freq: str, valid: pd.Series) -> pd.DataFrame:
    if freq in ("W", "M"):
        flag = first_day_of_week(st.index) if freq == "W" else first_day_of_month(st.index)
        st = st.where(flag).ffill()
    st = st.copy()
    v = valid.fillna(False).astype(bool)
    st[~v] = np.nan
    first = v.values.argmax()
    st.iloc[first:] = st.iloc[first:].fillna(0.0)
    return only_on_changes(st.to_frame(t))


def _macd(c, f=12, s=26, g=9):
    m = ema(c, f) - ema(c, s)
    return m, m.ewm(span=g, adjust=False, min_periods=g).mean()


def _stoch(px, t, n=14):
    c, h, l = px.close[t], px.high[t], px.low[t]
    k = 100 * (c - l.rolling(n).min()) / (h.rolling(n).max() - l.rolling(n).min())
    return k, k.rolling(3).mean()


@strategy("ema_cross_macd", ["SPY"],
          grid={"fast": [20, 50], "slow": [100, 200], "freq": FREQS},
          default={"fast": 50, "slow": 200, "freq": "W"},
          desc="Régimen con cruce de EMAs; dentro del régimen alcista entra cuando el MACD cruza sobre su señal. Sale solo cuando se pierde el régimen.")
def ema_cross_macd(px, fast=50, slow=200, freq="W", t="SPY"):
    c = px.close[t]
    f, s = ema(c, fast), ema(c, slow)
    m, sig = _macd(c)
    return sample(state((f > s) & (m > sig), f < s), t, freq, s.notna())


@strategy("ema2_atr_stop", ["SPY"],
          grid={"fast": [20, 50], "slow": [100, 200], "mult": [3.0, 4.0, 5.0]},
          default={"fast": 50, "slow": 200, "mult": 4.0},
          desc="Entra con EMA rápida > lenta; sale con el cruce a la baja o con un trailing stop de mult·ATR(22), lo que ocurra primero.")
def ema2_atr_stop(px, fast=50, slow=200, mult=4.0, t="SPY"):
    c = px.close[t]
    f, s = ema(c, fast), ema(c, slow)
    up = f > s
    entry = up & ~up.shift(1, fill_value=False)  # solo en el cruce, no re-entra tras el stop hasta un nuevo cruce
    return sample(chandelier(c, atr(px, t, 22), entry, mult, f < s), t, "D", s.notna())


@strategy("ribbon_atr", ["SPY"],
          grid={"base": [5, 10, 20], "mult": [3.0, 4.0, 5.0]},
          default={"base": 10, "mult": 4.0},
          desc="Entra cuando el precio está sobre las 8 EMAs del ribbon; sale con trailing stop de mult·ATR(22).")
def ribbon_atr(px, base=10, mult=4.0, t="SPY"):
    c = px.close[t]
    rib = pd.concat([ema(c, base * i) for i in range(1, 9)], axis=1)
    entry = rib.lt(c, axis=0).sum(axis=1) == 8
    return sample(chandelier(c, atr(px, t, 22), entry, mult), t, "D", rib.iloc[:, -1].notna())


@strategy("trend_dip", ["SPY"],
          grid={"trend_n": [150, 200], "trigger": ["stoch", "bollinger", "any"], "exit_band": [0.0, 0.02]},
          default={"trend_n": 200, "trigger": "any", "exit_band": 0.02},
          desc="Compra caídas dentro de la tendencia: con precio sobre la SMA de trend_n, entra cuando el estocástico está en sobreventa o el precio toca la Bollinger inferior; mantiene hasta perder la SMA (con banda).")
def trend_dip(px, trend_n=200, trigger="any", exit_band=0.02, t="SPY"):
    c = px.close[t]
    sma = c.rolling(trend_n).mean()
    k, _ = _stoch(px, t)
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    trg = {"stoch": k < 20, "bollinger": c < mid - 2 * sd, "any": (k < 20) | (c < mid - 2 * sd)}[trigger]
    return sample(state((c > sma) & trg, c < sma * (1 - exit_band)), t, "D", sma.notna())


@strategy("macd_stoch", ["SPY"],
          grid={"low": [20, 30], "freq": FREQS},
          default={"low": 30, "freq": "W"},
          desc="Entra con MACD sobre su señal y estocástico que sale de sobreventa (< low); sale con MACD bajo su señal.")
def macd_stoch(px, low=30, freq="W", t="SPY"):
    c = px.close[t]
    m, sig = _macd(c)
    k, d = _stoch(px, t)
    stoch_ok = (k.rolling(10).min() < low) & (k > d)
    return sample(state((m > sig) & stoch_ok, m < sig), t, freq, sig.notna())


VOTERS = ["sma200", "ema50_200", "macd", "ribbon", "bb_mid", "stoch50"]


def _voters(px, t):
    c = px.close[t]
    m, sig = _macd(c)
    rib = pd.concat([ema(c, 10 * i) for i in range(1, 9)], axis=1)
    k, _ = _stoch(px, t)
    v = pd.DataFrame({
        "sma200": c > c.rolling(200).mean(),
        "ema50_200": ema(c, 50) > ema(c, 200),
        "macd": m > sig,
        "ribbon": rib.lt(c, axis=0).sum(axis=1) >= 6,
        "bb_mid": c > c.rolling(20).mean(),
        "stoch50": k > 50,
    }).astype(float)
    return v, ema(c, 200).notna() & c.rolling(200).mean().notna()


@strategy("vote", ["SPY"],
          grid={"mode": ["majority", "proportional"], "min_votes": [3, 4, 5], "freq": ["W", "M"]},
          default={"mode": "majority", "min_votes": 4, "freq": "W"},
          desc="Votación de 6 señales (SMA200, EMA 50/200, MACD, ribbon, Bollinger media, estocástico > 50). majority: 100% si votan al menos min_votes, si no afuera. proportional: % invertido = votos/6 (redondeado a tercios).")
def vote(px, mode="majority", min_votes=4, freq="W", t="SPY"):
    v, valid = _voters(px, t)
    n = v.sum(axis=1)
    if mode == "majority":
        st = (n >= min_votes).astype(float)
    else:
        st = (np.floor(n / 6 * 3) / 3).where(n >= min_votes - 2, 0.0)
    return sample(st, t, freq, valid)
