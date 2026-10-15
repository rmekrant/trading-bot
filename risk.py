"""Gestion du risque : ATR, stop-loss, taille de position (leçon 4)."""

import pandas as pd


def atr(df, period=14):
    """Average True Range : amplitude moyenne d'une bougie (mesure de volatilité)."""
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def position_size(capital, risk_pct, entry, stop, max_alloc=1.0):
    """Nombre d'unités à acheter pour ne perdre que `risk_pct` du capital si le stop est touché.

    Plafonné à `max_alloc` x capital (donc jamais de levier)."""
    distance = entry - stop
    if distance <= 0 or entry <= 0:
        return 0.0
    units = capital * risk_pct / distance
    return min(units, capital * max_alloc / entry)
