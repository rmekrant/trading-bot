"""Stratégies. Chaque stratégie renvoie une Series de positions :
1.0 = en position (long), 0.0 = hors marché. Spot uniquement, pas de short, pas de levier.

Le signal d'une bougie est calculé avec sa clôture : il est donc exécuté
à la bougie SUIVANTE (c'est géré dans backtest.py et bot.py, pas ici).
"""

import numpy as np
import pandas as pd


def _hold(entry, exit_):
    """Transforme des événements d'entrée/sortie en état 0/1 (la sortie l'emporte)."""
    state = pd.Series(np.nan, index=entry.index)
    state[entry] = 1.0
    state[exit_] = 0.0
    return state.ffill().fillna(0.0)


def rsi(close, period=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_dn = down.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return 100 - 100 / (1 + avg_up / avg_dn)


class Strategy:
    name = "base"
    family = ""

    def positions(self, df):
        raise NotImplementedError


class SmaCross(Strategy):
    """Tendance : en position quand la moyenne rapide est au-dessus de la lente."""

    family = "tendance"

    def __init__(self, fast=20, slow=50):
        self.fast, self.slow = fast, slow
        self.name = f"SMA {fast}/{slow}"

    def positions(self, df):
        fast = df["close"].rolling(self.fast).mean()
        slow = df["close"].rolling(self.slow).mean()
        return ((fast > slow) & slow.notna()).astype(float)


class Breakout(Strategy):
    """Cassure (Donchian) : achat au-dessus du plus haut des N dernières bougies,
    sortie sous le plus bas des M dernières."""

    family = "tendance"

    def __init__(self, entry_n=20, exit_n=10):
        self.entry_n, self.exit_n = entry_n, exit_n
        self.name = f"Breakout {entry_n}/{exit_n}"

    def positions(self, df):
        highest = df["high"].rolling(self.entry_n).max().shift(1)
        lowest = df["low"].rolling(self.exit_n).min().shift(1)
        return _hold(df["close"] > highest, df["close"] < lowest)


class RsiReversion(Strategy):
    """Retour à la moyenne : achat quand le RSI est survendu, sortie quand il remonte.
    trend_n : si renseigné, n'achète que si le prix est au-dessus de sa SMA(trend_n)."""

    family = "mean reversion"

    def __init__(self, period=14, buy_below=30, exit_above=55, trend_n=None):
        self.period, self.buy_below, self.exit_above, self.trend_n = (
            period, buy_below, exit_above, trend_n,
        )
        self.name = f"RSI {period} ({buy_below}/{exit_above})"

    def positions(self, df):
        r = rsi(df["close"], self.period)
        entry = r < self.buy_below
        if self.trend_n:
            entry &= df["close"] > df["close"].rolling(self.trend_n).mean()
        return _hold(entry, r > self.exit_above)


def default_strategies():
    """Stratégies retenues par le banc d'essai (strategy_lab.py) pour des bougies de 4 h :
    trois familles de tendance différentes, qui ont chacune résisté au contrôle sur 5 actifs."""
    from candidates import DonchianTrend, KeltnerBreakout, SupertrendStrat

    return [SupertrendStrat(21, 5), DonchianTrend(24, 12, 125), KeltnerBreakout(24, 1.5)]


def strategy_positions(strategies, df):
    """DataFrame : une colonne de positions (0/1) par stratégie."""
    return pd.concat({s.name: s.positions(df) for s in strategies}, axis=1)


def combine(pos_df, min_votes=2):
    """Position combinée : 1 si au moins `min_votes` stratégies sont en position."""
    return (pos_df.sum(axis=1) >= min_votes).astype(float)
