"""Estrategias del notebook de Diego (funciones_algotrading_lab.ipynb), reimplementadas sin pandas_ta/vectorbt.

Cada función devuelve (entradas, salidas) booleanas con datos hasta el cierre de t. La posición sigue la lógica de
vectorbt.from_signals: entra con una señal de entrada si está afuera, sale con una de salida si está adentro; si las dos
coinciden el mismo día, no hace nada.

Correcciones de leakage respecto del original (anotadas en cada caso):
- ichi_sma: el CHIKOU de finta es el cierre desplazado 26 días hacia ATRÁS en el gráfico, o sea close[t+26]: usa el
  futuro. Acá se compara el cierre de hoy contra el SENKOU de hace 26 días (misma relación, sin mirar adelante).
- atr_volatility: atr.quantile(0.2) se calculaba sobre TODA la serie (incluye el futuro). Acá es el cuantil expansivo.
- macd_threshold / macd_ema_threshold: el umbral del histograma estaba en dólares; acá es proporción del precio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Prices


# ---------------------------------------------------------------- indicadores
def sma(s, n): return s.rolling(n, min_periods=n).mean()
def ema(s, n): return s.ewm(span=n, adjust=False, min_periods=n).mean()


def wma(s, n):
    w = np.arange(1, n + 1)
    return s.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def rma(s, n): return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(c, n=14):
    d = c.diff()
    up, dn = rma(d.clip(lower=0), n), rma(-d.clip(upper=0), n)
    return 100 - 100 / (1 + up / dn)


def true_range(h, l, c):
    return pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)


def atr(h, l, c, n=14): return rma(true_range(h, l, c), n)


def macd(c, f=12, s=26, g=9):
    m = ema(c, f) - ema(c, s)
    sig = ema(m, g)
    return m, m - sig, sig


def bbands(c, n=20, k=2.0):
    mid, sd = sma(c, n), c.rolling(n).std(ddof=0)
    return mid - k * sd, mid, mid + k * sd


def stoch(h, l, c, k=14, d=3, smooth=3):
    raw = 100 * (c - l.rolling(k).min()) / (h.rolling(k).max() - l.rolling(k).min())
    K = sma(raw, smooth)
    return K, sma(K, d)


def adx(h, l, c, n=14):
    upm, dnm = h.diff(), -l.diff()
    pdm = pd.Series(np.where((upm > dnm) & (upm > 0), upm, 0.0), h.index)
    ndm = pd.Series(np.where((dnm > upm) & (dnm > 0), dnm, 0.0), h.index)
    a = atr(h, l, c, n)
    dip, din = 100 * rma(pdm, n) / a, 100 * rma(ndm, n) / a
    dx = 100 * (dip - din).abs() / (dip + din)
    return rma(dx, n), dip, din


def aroon(h, l, n=14):
    up = 100 * h.rolling(n + 1).apply(lambda x: x.argmax(), raw=True) / n
    dn = 100 * l.rolling(n + 1).apply(lambda x: x.argmin(), raw=True) / n
    return up, dn


def mfi(h, l, c, v, n=14):
    tp = (h + l + c) / 3
    mf = tp * v
    pos = mf.where(tp > tp.shift(), 0.0).rolling(n).sum()
    neg = mf.where(tp < tp.shift(), 0.0).rolling(n).sum()
    return 100 - 100 / (1 + pos / neg)


def obv(c, v): return (np.sign(c.diff()).fillna(0) * v).cumsum()


def cci(h, l, c, n=20):
    tp = (h + l + c) / 3
    m = sma(tp, n)
    md = tp.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - m) / (0.015 * md)


def kama(c, n=10, fast=2, slow=30):
    er = (c - c.shift(n)).abs() / c.diff().abs().rolling(n).sum()
    sc = (er * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2
    out = np.full(len(c), np.nan)
    cv, sv = c.values, sc.values
    for i in range(len(c)):
        if not np.isfinite(sv[i]):
            continue
        out[i] = cv[i] if not np.isfinite(out[i - 1]) else out[i - 1] + sv[i] * (cv[i] - out[i - 1])
    return pd.Series(out, c.index)


def psar(h, l, af0=0.02, step=0.02, mx=0.2):
    hv, lv = h.values, l.values
    n = len(hv)
    sar = np.full(n, np.nan)
    i0 = np.argmax(np.isfinite(hv))
    if n - i0 < 3:
        return pd.Series(sar, h.index)
    up, af, ep, s = True, af0, hv[i0], lv[i0]
    for i in range(i0 + 1, n):
        s = s + af * (ep - s)
        if up:
            s = min(s, lv[i - 1], lv[i - 2] if i - 2 >= i0 else lv[i - 1])
            if lv[i] < s:
                up, s, ep, af = False, ep, lv[i], af0
            elif hv[i] > ep:
                ep, af = hv[i], min(af + step, mx)
        else:
            s = max(s, hv[i - 1], hv[i - 2] if i - 2 >= i0 else hv[i - 1])
            if hv[i] > s:
                up, s, ep, af = True, ep, hv[i], af0
            elif lv[i] < ep:
                ep, af = lv[i], min(af + step, mx)
        sar[i] = s
    return pd.Series(sar, h.index)


def ichimoku(h, l, c):
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    senkou_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    return tenkan, kijun, senkou_b


# ---------------------------------------------------------------- estrategias (nombre del notebook)
def _x(s): return s.shift(1)


def adx_strategy(o, h, l, c, v):
    a, dp, dn = adx(h, l, c)
    return (a >= 25) & (dp > dn), (a >= 25) & (dp < dn)


def macd_strategy(o, h, l, c, v, f=12, s=26, g=9):
    m, hi, sg = macd(c, f, s, g)
    return (hi > 0) & (m > sg), (hi < 0) | (m < sg)


def macd_threshold_strategy(o, h, l, c, v, thr=0.001):
    m, hi, sg = macd(c)
    return (hi > thr * c) & (m > sg), (hi < -thr * c) | (m < sg)


def macd_ema_threshold_strategy(o, h, l, c, v, thr=0.001, n=200):
    m, hi, sg = macd(c)
    return (hi > thr * c) & (m > sg) & (c > ema(c, n)), (hi < -thr * c) | (m < sg)


def macd_divergence_strategy(o, h, l, c, v):
    m, hi, sg = macd(c)
    return (c < l.rolling(14).min()) & (m > sg), (c > h.rolling(14).max()) & (m < sg)


def stoch_strategy(o, h, l, c, v, low=20):
    K, D = stoch(h, l, c)
    return (K < low) & (D < low), (K > 100 - low) & (D > 100 - low)


def aroon_strategy(o, h, l, c, v, n=25, thr=70):
    up, dn = aroon(h, l, n)
    return (up > thr) & (_x(up) <= thr), (dn > thr) & (_x(dn) <= thr)


def double_ema_strategy(o, h, l, c, v, a=20, b=50):
    s, lg = ema(c, a), ema(c, b)
    return (s > lg) & (_x(s) <= _x(lg)), (s < lg) & (_x(s) >= _x(lg))


def double_sma_strategy(o, h, l, c, v, a=50, b=200):
    s, lg = sma(c, a), sma(c, b)
    return (s > lg) & (_x(s) <= _x(lg)), (s < lg) & (_x(s) >= _x(lg))


def double_sma_adjustable_thresholds(o, h, l, c, v, a=50, b=200, thr=0.01):
    s, lg = sma(c, a), sma(c, b)
    return (s > lg * (1 + thr)), (s <= lg) & (lg > s * (1 + thr))


def triple_sma_strategy(o, h, l, c, v, a=10, b=20, d=50):
    s, m, lg = sma(c, a), sma(c, b), sma(c, d)
    ent = (s > m) & (m > lg) & ((_x(s) <= _x(m)) | (_x(m) <= _x(lg)))
    ext = (s <= m) | ((m <= lg) & ((_x(s) > _x(m)) | (_x(m) > _x(lg))))  # precedencia como en el original
    return ent, ext


def triple_sma_adjustable_thresholds(o, h, l, c, v, a=10, b=20, d=50, thr=0.01):
    s, m, lg = sma(c, a), sma(c, b), sma(c, d)
    ent = (s > m) & (m > lg) & (s > m * (1 + thr)) & ((_x(s) <= _x(m)) | (_x(m) <= _x(lg)))
    ext = ((s <= m) | (m <= lg)) & (m > s * (1 + thr)) & ((_x(s) >= _x(m)) | (_x(m) >= _x(lg)))
    return ent, ext


def triple_ema_strategy(o, h, l, c, v, a=10, b=20, d=50):
    s, m, lg = ema(c, a), ema(c, b), ema(c, d)
    ent = (s > m) & (m > lg) & ((_x(s) <= _x(m)) | (_x(m) <= _x(lg)))
    ext = ((s <= m) | (m <= lg)) & ((_x(s) > _x(m)) | (_x(m) > _x(lg)))
    return ent, ext


def ema_rsi_strategy(o, h, l, c, v, a=20, b=50, low=30):
    s, lg, r = ema(c, a), ema(c, b), rsi(c)
    return (s > lg) & (_x(s) <= _x(lg)) & (r < low), (s < lg) & (_x(s) >= _x(lg)) & (r >= 100 - low)


def ema_mfi_strategy(o, h, l, c, v, a=20, b=50, low=30):
    s, lg, m = ema(c, a), ema(c, b), mfi(h, l, c, v)
    return (m < low) & (s > lg) & (_x(s) <= _x(lg)), (m > 100 - low) | ((s < lg) & (_x(s) >= _x(lg)))


def bollinger_rsi_strategy(o, h, l, c, v, n=20, k=2.0, low=30):
    lo, _, up = bbands(c, n, k)
    r = rsi(c)
    return (c <= lo) & (_x(c) > _x(lo)) & (r < low), (c >= up) & (_x(c) < _x(up)) & (r > 100 - low)


def bollinger_stoch_strategy(o, h, l, c, v, n=20, k=2.0, low=20):
    lo, _, up = bbands(c, n, k)
    K, _ = stoch(h, l, c)
    return (c <= lo) & (_x(c) > _x(lo)) & (K < low), (c >= up) & (_x(c) < _x(up)) & (K > 100 - low)


def ema_bollinger_strategy(o, h, l, c, v, n_ema=50, n=20, k=2.0):
    lo, _, up = bbands(c, n, k)
    e = ema(c, n_ema)
    return ((c > e) & (c < lo) & (_x(c) <= _x(e)) & (_x(c) >= _x(lo)),
            (c < e) & (c > up) & (_x(c) >= _x(e)) & (_x(c) <= _x(up)))


def bollinger_strategy(o, h, l, c, v, n=20, k=2.0):
    lo, _, up = bbands(c, n, k)
    return (c <= lo) & (_x(c) > _x(lo)), (c >= up) & (_x(c) < _x(up))


def bollinger_cci_strategy(o, h, l, c, v, n=20, k=2.0):
    lo, _, up = bbands(c, n, k)
    cc = cci(h, l, c)
    return (c <= lo) & (cc < -100) & (_x(c) > _x(lo)), ((c >= up) & (_x(c) < _x(up))) | (cc >= 100)


def bollinger_revesion_strategy(o, h, l, c, v, n=20, k=2.0):
    lo, _, up = bbands(c, n, k)
    return (_x(c) <= _x(lo)) & (c > lo), (_x(c) >= _x(up)) & (c < up)


def bollinger_w_pattern_strategy(o, h, l, c, v, n=20):
    lo, _, up = bbands(c, n, 2.0)
    mid = lo  # el original usa la banda inferior como "media" (iloc[:,0]); se respeta
    w = ((c < lo) & (c.shift(1) > lo.shift(1)) & (c.shift(2) > mid.shift(2)) & (c.shift(3) < mid.shift(3))
         & (c.shift(4) > mid.shift(4)) & (c.shift(5) < up.shift(5)))
    ext_m = (c >= up) | (c.shift(1) >= up.shift(1))
    return w & ~w.shift(1, fill_value=False), w.shift(6, fill_value=False) & ext_m.shift(6, fill_value=False)


def natr_strategy(o, h, l, c, v, thr=2.0):
    n = 100 * atr(h, l, c) / c
    return n < thr, n > thr


def natr_strategy_willr(o, h, l, c, v, thr=2.0):
    n = 100 * atr(h, l, c) / c
    wr = -100 * (h.rolling(14).max() - c) / (h.rolling(14).max() - l.rolling(14).min())
    return (n < thr) & (wr > -80), (n > thr) & (wr < -20)


def squeeze_stategy(o, h, l, c, v):
    lo, mid, up = bbands(c, 20, 2.0)
    kc_m, kc_r = ema(c, 20), 1.5 * atr(h, l, c, 20)
    on = (lo > kc_m - kc_r) & (up < kc_m + kc_r)
    off = (lo < kc_m - kc_r) & (up > kc_m + kc_r)
    return on, off


def eri_strategy(o, h, l, c, v):
    e = ema(c, 13)
    return (l - e) > 0, (h - e) < 0


def ao_strategy(o, h, l, c, v):
    mp = (h + l) / 2
    a = sma(mp, 5) - sma(mp, 34)
    return a > 0, a < 0


def kama_strategy(o, h, l, c, v):
    k = kama(c, 10, 10, 20)
    return (c > k) & (_x(c) <= _x(k)), (c < k) & (_x(c) >= _x(k))


def kama_aroon_downtrend_strategy(o, h, l, c, v):
    k = kama(c)
    _, dn = aroon(h, l, 14)
    return (c < k) & (dn > 0), c > k


def ichi_sma_strategy(o, h, l, c, v, n=50):
    tk, kj, sb = ichimoku(h, l, c)
    chik_minus_senk = c - sb.shift(26)  # sin leakage (ver docstring)
    s = sma(c, n)
    return ((tk - kj > 0.1) & (chik_minus_senk > 0) & (c > s), (tk - kj < -0.1) & (chik_minus_senk < 0) & (c < s))


def obv_mfi_strategy(o, h, l, c, v, low=30):
    m, ob = mfi(h, l, c, v), obv(c, v)
    om = ob.rolling(14).mean()
    return (m < low) & (ob > om) & (_x(ob) <= _x(om)), (m > 100 - low) | ((ob < om) & (_x(ob) >= _x(om)))


def mean_reversion_bearish_strategy(o, h, l, c, v, n=20):
    s = sma(c, n)
    return c < s - 2 * c.rolling(n).std(), c > s


def rsi_divergence_bearish_strategy(o, h, l, c, v):
    r = rsi(c)
    return (c < l.rolling(14).min()) & (r > _x(r)), r < 70


def wma_crossover_bearish_strategy(o, h, l, c, v, a=20, b=50):
    s, lg = wma(c, a), wma(c, b)
    return (s > lg) & (_x(s) <= _x(lg)), s < lg


def atr_volatility_bearish_strategy(o, h, l, c, v, n_atr=14, n=50):
    a = atr(h, l, c, n_atr)
    q = a.expanding(252).quantile(0.2)  # el original usaba el cuantil de toda la serie (leakage)
    s = sma(c, n)
    return (c > s) & (a < q), c < s


def sar_bearish_strategy(o, h, l, c, v):
    s = psar(h, l)
    return c > s, c < s


def pivot_points_bearish_strategy(o, h, l, c, v):
    pp = (_x(h) + _x(l) + _x(c)) / 3
    s1 = 2 * pp - _x(h)
    return c > s1, c < s1


def donchian_rsi_pullback_strategy(o, h, l, c, v):
    dcm = (h.rolling(20).max() + l.rolling(20).min()) / 2
    r = rsi(c)
    return (c <= dcm) & (r < 30), (c >= dcm) | (r > 70)


def ema_ribbon_strategy(o, h, l, c, v):
    es = [ema(c, n) for n in (20, 25, 30, 40, 45, 50, 55)]
    ent = np.logical_and.reduce([c > e for e in es]) & (c > ema(c, 10))
    ext = np.logical_or.reduce([c < e for e in es])
    return pd.Series(ent, c.index), pd.Series(ext, c.index)


STRATS = {f.__name__: f for f in [
    adx_strategy, macd_strategy, macd_threshold_strategy, macd_ema_threshold_strategy, macd_divergence_strategy,
    stoch_strategy, aroon_strategy, double_ema_strategy, double_sma_strategy, double_sma_adjustable_thresholds,
    triple_sma_strategy, triple_sma_adjustable_thresholds, triple_ema_strategy, ema_rsi_strategy, ema_mfi_strategy,
    bollinger_rsi_strategy, bollinger_stoch_strategy, ema_bollinger_strategy, bollinger_strategy,
    bollinger_cci_strategy, bollinger_revesion_strategy, bollinger_w_pattern_strategy, natr_strategy,
    natr_strategy_willr, squeeze_stategy, eri_strategy, ao_strategy, kama_strategy, kama_aroon_downtrend_strategy,
    ichi_sma_strategy, obv_mfi_strategy, mean_reversion_bearish_strategy, rsi_divergence_bearish_strategy,
    wma_crossover_bearish_strategy, atr_volatility_bearish_strategy, sar_bearish_strategy,
    pivot_points_bearish_strategy, donchian_rsi_pullback_strategy, ema_ribbon_strategy]}


def position(entries: pd.Series, exits: pd.Series, close: pd.Series, sl: float | None = None) -> pd.Series:
    """Estado 0/1 al cierre, como vectorbt.from_signals (conflicto = ignorar). sl: stop loss sobre el cierre de entrada."""
    e = entries.fillna(False).astype(bool).values
    x = exits.fillna(False).astype(bool).values
    cv = close.values
    st = np.zeros(len(e))
    inpos, px_in = False, 0.0
    for i in range(len(e)):
        if not np.isfinite(cv[i]):
            st[i] = 1.0 if inpos else 0.0
            continue
        if inpos:
            if (x[i] and not e[i]) or (sl is not None and cv[i] < px_in * (1 - sl)):
                inpos = False
        elif e[i] and not x[i]:
            inpos, px_in = True, cv[i]
        st[i] = 1.0 if inpos else 0.0
    return pd.Series(st, close.index)


def state_for(px: Prices, t: str, name: str, sl: float | None = None) -> pd.Series:
    o, h, l, c, v = (getattr(px, f)[t] for f in ("open", "high", "low", "close", "volume"))
    ent, ext = STRATS[name](o, h, l, c, v)
    st = position(ent, ext, c, sl)
    return st.where(c.notna())
