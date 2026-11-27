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

## Resultados hasta ahora

| Estrategia | Período | CAGR | SPY B&H | MaxDD | Operaciones/año |
|---|---|---|---|---|---|
| Filtro SMA 200 mensual sobre SPY | 2000–2026 | 5,3% | 8,0% | −35% (SPY −55%) | 1,4 |
| ídem, OOS 2015–2026 | 2015–2026 | 3,2% | 12,8% | −35% (SPY −34%) | 1,9 |

Con costos de IOL la versión diaria del filtro (≈7 operaciones por año) queda en −1,9% anual: las señales diarias no sirven.
