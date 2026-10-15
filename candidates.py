"""Stratégies candidates pour les bougies d'une heure (et grilles de paramètres à tester).

Chaque stratégie renvoie 1.0 (en position) ou 0.0 (hors marché), long uniquement, comme dans strategies.py.
Le but n'est PAS de croire en l'une d'elles, mais de les tester toutes avec la même méthode
(strategy_lab.py) et de ne garder que celles qui résistent.
"""

import itertools

import numpy as np
import pandas as pd

from risk import atr
from strategies import Strategy, _hold, rsi


def _trend_ok(df, n):
    """Filtre de tendance : le prix est au-dessus de sa moyenne sur n bougies (0 = pas de filtre)."""
    if not n:
        return pd.Series(True, index=df.index)
    return df["close"] > df["close"].rolling(n).mean()


class DonchianTrend(Strategy):
    """Cassure du plus haut des N dernières bougies, sortie sous le plus bas des M dernières."""

    family = "cassure"

    def __init__(self, entry_n, exit_n, trend_n):
        self.entry_n, self.exit_n, self.trend_n = entry_n, exit_n, trend_n
        self.name = f"Cassure {entry_n}/{exit_n} filtre {trend_n}"

    def positions(self, df):
        high = df["high"].rolling(self.entry_n).max().shift(1)
        low = df["low"].rolling(self.exit_n).min().shift(1)
        entry = (df["close"] > high) & _trend_ok(df, self.trend_n)
        return _hold(entry, df["close"] < low)


class EmaCross(Strategy):
    family = "croisement EMA"

    def __init__(self, fast, slow):
        self.fast, self.slow = fast, slow
        self.name = f"EMA {fast}/{slow}"

    def positions(self, df):
        fast = df["close"].ewm(span=self.fast, adjust=False).mean()
        slow = df["close"].ewm(span=self.slow, adjust=False).mean()
        warm = pd.Series(np.arange(len(df)) >= self.slow, index=df.index)
        return ((fast > slow) & warm).astype(float)


class RsiDip(Strategy):
    """Achat d'un repli (RSI bas) uniquement quand la tendance de fond est haussière."""

    family = "repli RSI"

    def __init__(self, period, buy_below, exit_above, trend_n):
        self.period, self.buy_below, self.exit_above, self.trend_n = period, buy_below, exit_above, trend_n
        self.name = f"RSI {period} ({buy_below}/{exit_above}) filtre {trend_n}"

    def positions(self, df):
        r = rsi(df["close"], self.period)
        entry = (r < self.buy_below) & _trend_ok(df, self.trend_n)
        return _hold(entry, r > self.exit_above)


class BollingerRevert(Strategy):
    """Achat sous la bande basse de Bollinger (en tendance haussière), sortie à la moyenne."""

    family = "Bollinger"

    def __init__(self, n, k, trend_n):
        self.n, self.k, self.trend_n = n, k, trend_n
        self.name = f"Bollinger {n} x{k} filtre {trend_n}"

    def positions(self, df):
        mid = df["close"].rolling(self.n).mean()
        sd = df["close"].rolling(self.n).std()
        entry = (df["close"] < mid - self.k * sd) & _trend_ok(df, self.trend_n)
        return _hold(entry, df["close"] > mid)


class Momentum(Strategy):
    """En position quand le prix a monté de plus de `thr` sur les L dernières bougies."""

    family = "momentum"

    def __init__(self, lookback, thr):
        self.lookback, self.thr = lookback, thr
        self.name = f"Momentum {lookback}h > {thr:.0%}"

    def positions(self, df):
        ret = df["close"] / df["close"].shift(self.lookback) - 1
        return (ret > self.thr).astype(float)


class SupertrendStrat(Strategy):
    family = "Supertrend"

    def __init__(self, period, mult):
        self.period, self.mult = period, mult
        self.name = f"Supertrend {period} x{mult}"

    def positions(self, df):
        a = atr(df, self.period).to_numpy()
        close = df["close"].to_numpy()
        hl2 = ((df["high"] + df["low"]) / 2).to_numpy()
        n = len(df)
        out = np.zeros(n)
        valid = ~np.isnan(a)
        if not valid.any():
            return pd.Series(out, index=df.index)
        first = int(np.argmax(valid))
        ub, lb = hl2 + self.mult * a, hl2 - self.mult * a
        fub, flb, direction = ub.copy(), lb.copy(), np.zeros(n)
        for i in range(first + 1, n):
            fub[i] = ub[i] if (ub[i] < fub[i - 1] or close[i - 1] > fub[i - 1]) else fub[i - 1]
            flb[i] = lb[i] if (lb[i] > flb[i - 1] or close[i - 1] < flb[i - 1]) else flb[i - 1]
            if direction[i - 1] <= 0 and close[i] > fub[i - 1]:
                direction[i] = 1
            elif direction[i - 1] >= 0 and close[i] < flb[i - 1]:
                direction[i] = -1
            else:
                direction[i] = direction[i - 1]
        return pd.Series((direction == 1).astype(float), index=df.index)


class MacdTrend(Strategy):
    family = "MACD"

    def __init__(self, fast, slow, signal, trend_n):
        self.fast, self.slow, self.signal, self.trend_n = fast, slow, signal, trend_n
        self.name = f"MACD {fast}/{slow}/{signal} filtre {trend_n}"

    def positions(self, df):
        macd = df["close"].ewm(span=self.fast, adjust=False).mean() - df["close"].ewm(span=self.slow, adjust=False).mean()
        sig = macd.ewm(span=self.signal, adjust=False).mean()
        warm = pd.Series(np.arange(len(df)) >= self.slow, index=df.index)
        return ((macd > sig) & _trend_ok(df, self.trend_n) & warm).astype(float)


class KeltnerBreakout(Strategy):
    """Cassure au-dessus de EMA(n) + mult x ATR(n), sortie sous l'EMA."""

    family = "Keltner"

    def __init__(self, n, mult):
        self.n, self.mult = n, mult
        self.name = f"Keltner {n} x{mult}"

    def positions(self, df):
        mid = df["close"].ewm(span=self.n, adjust=False).mean()
        band = atr(df, self.n)
        return _hold(df["close"] > mid + self.mult * band, df["close"] < mid)


def _grid(cls, **params):
    keys = list(params)
    return [cls(**dict(zip(keys, values))) for values in itertools.product(*params.values())]


def candidate_families(tf_hours=1.0):
    """Dictionnaire famille -> liste de stratégies (une par combinaison de paramètres).

    Les fenêtres d'observation sont pensées en HEURES puis converties en nombre de bougies :
    avec des bougies de 4 h, une fenêtre de 96 h devient 24 bougies. Ainsi on teste les mêmes
    horizons de temps quelle que soit la taille des bougies. Les noms des stratégies indiquent
    des nombres de BOUGIES."""

    def w(hours):
        return max(2, int(round(hours / tf_hours)))

    def unique(strategies):
        seen, out = set(), []
        for s in strategies:
            if s.name not in seen:
                seen.add(s.name)
                out.append(s)
        return out

    t_opts = (0, w(200), w(500))
    donchian = unique([DonchianTrend(w(e), w(x), t) for e in (24, 48, 96, 168) for x in (12, 24, 48) if x < e for t in t_opts])
    ema = unique([EmaCross(w(f), w(s)) for f in (8, 12, 24, 48) for s in (48, 96, 200, 400) if f < s])
    rsi_dip = _grid(RsiDip, period=(2, 5, 14), buy_below=(10, 20, 30), exit_above=(50, 60, 70), trend_n=(w(200), w(500)))
    boll = unique([BollingerRevert(w(n), k, t) for n in (20, 48, 96) for k in (1.5, 2.0, 2.5) for t in (w(200), w(500))])
    mom = unique([Momentum(w(lb), thr) for lb in (24, 72, 168, 336, 720) for thr in (0.0, 0.03)])
    st = _grid(SupertrendStrat, period=(10, 14, 21, 48), mult=(2, 3, 4, 5))
    macd = [MacdTrend(f, s, g, t) for (f, s, g) in ((12, 26, 9), (24, 52, 18), (48, 104, 36)) for t in t_opts]
    kelt = unique([KeltnerBreakout(w(n), m) for n in (24, 48, 96) for m in (1.5, 2.0, 3.0)])
    return {
        "Cassure (Donchian)": donchian,
        "Croisement EMA": ema,
        "Repli RSI en tendance": rsi_dip,
        "Bollinger en tendance": boll,
        "Momentum": mom,
        "Supertrend": st,
        "MACD en tendance": macd,
        "Keltner": kelt,
    }
