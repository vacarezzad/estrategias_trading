# estrategias_trading

Backtester de estrategias de trading contra el **buy & hold de SPY**, pensado para operar CEDEARs en IOL
(swing, sin apalancamiento). El análisis se hace sobre el subyacente en USD.

## Reglas del motor (para que los números sean honestos)

| Regla | Detalle |
|---|---|
| Sin mirar el futuro | La estrategia decide con datos hasta el **cierre** de t y el motor ejecuta a la **apertura de t+1**. `tests/test_no_leakage.py` recalcula cada estrategia con la historia truncada en fechas al azar y exige el mismo resultado: si alguien usa un rolling sin shift, el test falla. |
| Costos | 0,94% por lado (comisión + derechos + IVA de IOL) + 0,2% de slippage por lado (prima o descuento del CEDEAR contra el CCL y spread). El B&H también paga su entrada. |
| CCL | Medido en USD, la suba o baja del CCL afecta igual a la estrategia y al SPY; lo que importa es la prima del CEDEAR al entrar y salir, que va en el slippage (`--slippage`). |
| Sin cherry-picking | Se corre la grilla completa de parámetros y se reporta su distribución. La cifra para comparar es la **OOS**: mejores parámetros elegidos hasta 2014 y medidos 2015 en adelante. |
| Datos | Precios diarios ajustados por dividendos (yfinance desde GitHub Actions), un CSV por ticker en `data/prices`. |

## Uso

```
pip install -r requirements.txt
python -m pytest -q tests          # test de leakage y de ejecución
python run_backtests.py            # todas las estrategias -> results/
python run_backtests.py --only sma_filter_spy --slippage 0.005
```

Salidas en `results/`: `resumen.csv` (default, OOS elegida en IS y OOS default), `grilla.csv` (todas las combinaciones),
`curvas.csv` y `anual.csv`.

## Agregar una estrategia

En `bt/strategies.py`, una función que recibe `Prices` y devuelve pesos objetivo (fechas x tickers), decorada con
`@strategy(nombre, tickers, grid=..., default=...)`. Fila NaN = no tocar la cartera.

## Resultados hasta ahora (2026-11-27)

**Ronda 1, 16 estrategias técnicas sobre SPY** (`results/`, y sin costos en `results_sin_costos/`): MACD, Bollinger (reversión y ruptura),
cruce de 2 y 3 EMAs, ribbon, estocástico, ATR trailing, filtro SMA 200 y 5 combinaciones. Ninguna le gana al B&H del SPY, ni sin costos.
Las de muchas operaciones (MACD, estocástico, Bollinger, ribbon, votación: 6 a 17 por año) mueren con los costos de IOL.
Las mejores (cruce EMA 50/200, compra de caídas en tendencia) rinden ~5,5% contra 8,0% del SPY (2000–2026) con caídas de −21% a −36% contra −55%.

**Ronda 2, afinado** (`results_afinado/`, desde fines de 2004): activo (SPY/QQQ), refugio al salir (liquidez, GLD, IEF) y piso de exposición (0%/50%).

| Variante | 2004–2026 CAGR | Caída máx | 2015–2026 CAGR | Caída máx |
|---|---|---|---|---|
| SPY B&H | 10,4% | −55% | 12,8% | −34% |
| QQQ B&H | 14,3% | −53% | 17,3% | −35% |
| QQQ + cruce EMA 50/200, baja a 50% al cruzar a la baja | 12,7% | −41% | 15,8% | −29% |
| QQQ + cruce EMA 50/200, sale del todo | 11,0% | −34% | 14,2% | −29% |
| SPY + compra de caídas en tendencia, 50% en IEF al salir | 8,7% | −27% | 9,3% | −24% |

Le gana al SPY solo quien tiene un activo más agresivo (QQQ, o las acciones del universo de rotación). Las reglas de timing
siempre rinden menos que mantener ese mismo activo y a cambio recortan la caída. Elegir QQQ mirando hacia atrás tiene sesgo.

**Ronda 3, técnico relativo** (`bt/relative.py`, `results_relativo/`, `results_ew/`, desde 2004): top k de los 30 CEDEARs por fuerza relativa.

| Variante | 2004–2026 CAGR | Caída máx | Sharpe | 2015–2026 CAGR |
|---|---|---|---|---|
| SPY B&H | 10,4% | −55% | | 12,8% |
| Los 30 en partes iguales (B&H) | 22,4% | −52% | 1,02 | 27,4% |
| Top 5 por momentum 12-1, mensual, buffer top 7 | 25,3% | −44% | 0,90 | 33,4% |
| ídem bimestral, 50% si SPY < SMA 200 | 25,2% | −40% | 0,94 | 29,2% |
| Elegida por Sharpe hasta 2014 (mom/vol, top 8, bimestral, 50%) | 18,5% | −36% | 0,91 | 18,5% |

Le gana al SPY por mucho, pero casi todo viene del universo (las 30 de hoy son ganadoras conocidas). Contra el B&H
equiponderado de los mismos 30 agrega ~3 pp por año con t ≈ 1,4: no es significativo, y la variante elegida en IS pierde fuera de muestra.
El máximo de 52 semanas no funciona. El modelo ML de rotacion_cartera (48%, t ≈ 3,5 contra el equiponderado) sigue siendo claramente mejor.

**Solo las 10 más líquidas** (NVDA, GOOGL, AMD, QQQ, MELI, VIST, META, MSFT, AMZN, GLD; `results_top10/`, desde 2004):
- Timing por activo (15 estrategias): la mejor es el filtro SMA 200, con 16,3% de CAGR promedio contra 20,3% del B&H de cada activo. Le gana en 2 de 10.
- Top 3 por momentum 12-1, mensual: 28,3% contra 24,5% del equiponderado de las 10 y 10,4% del SPY. Caída de −61%, t ≈ 1,4 contra el equiponderado.
  Desde 2015: 45% contra 37,8% del equiponderado y 12,8% del SPY. Con 50% invertido cuando el SPY está bajo la SMA 200, la caída baja a −39% y empata con el equiponderado.
  Las 32 combinaciones de la grilla le ganan al SPY (17,8% a 28,3%), pero casi todas pierden contra el equiponderado.
- Sesgo: las 10 más líquidas de hoy son las ganadoras de la última década; el equiponderado de las 10 ya hace 24,5%.
