"""Récupération des données de marché (ccxt) + données simulées pour les tests."""

import os
import time

import numpy as np
import pandas as pd

EXCHANGES = ("binance", "kraken", "coinbase")
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
TF_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def _exchange(name):
    import ccxt

    return getattr(ccxt, name)({"enableRateLimit": True})


def _to_df(rows):
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    return df.astype(float)


def fetch_ohlcv(symbol, timeframe="1d", limit=500):
    """Dernières bougies (la dernière peut être encore en formation)."""
    last_error = None
    for name in EXCHANGES:
        try:
            rows = _exchange(name).fetch_ohlcv(symbol, timeframe, limit=limit)
            if rows:
                return _to_df(rows)
        except Exception as e:  # réseau, pays bloqué, symbole inconnu...
            last_error = e
    raise RuntimeError(f"Aucun exchange n'a répondu pour {symbol} : {last_error}")


def fetch_last_price(symbol):
    """Dernier prix du marché, ou None si indisponible."""
    for name in EXCHANGES:
        try:
            price = _exchange(name).fetch_ticker(symbol)["last"]
            if price:
                return float(price)
        except Exception:
            continue
    return None


def fetch_history(symbol, timeframe="1d", years=5, cache=True):
    """Historique long (pagination) avec cache CSV de 24 h dans ./data."""
    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", f"{symbol.replace('/', '_')}_{timeframe}_{years:g}y.csv")
    if cache and os.path.exists(path) and time.time() - os.path.getmtime(path) < 86_400:
        return pd.read_csv(path, index_col=0, parse_dates=True)

    step = TF_MS[timeframe]
    last_error = None
    for name in EXCHANGES:
        try:
            ex = _exchange(name)
            now = ex.milliseconds()
            since = now - int(years * 365 * 86_400_000)
            rows = []
            for _ in range(300):  # garde-fou anti boucle infinie
                batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
                if not batch:
                    break
                rows.extend(batch)
                new_since = batch[-1][0] + step
                if new_since <= since or new_since >= now:
                    break
                since = new_since
            if rows:
                df = _to_df(rows)
                print(f"Données : {name}, {len(df)} bougies")
                if cache:
                    df.to_csv(path)
                return df
        except Exception as e:
            last_error = e
            print(f"{name} indisponible : {e}")
    raise RuntimeError(f"Aucun exchange n'a répondu : {last_error}")


def drop_unclosed(df, timeframe):
    """Retire la dernière bougie si elle n'est pas encore terminée."""
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    close_time = df.index + pd.Timedelta(milliseconds=TF_MS[timeframe])
    return df[close_time <= now]


def synthetic_ohlcv(n=1500, seed=42, start="2021-01-01", freq="D"):
    """Marché fictif avec des phases haussières, baissières et latérales.
    Sert uniquement à tester le code hors ligne : ne PROUVE RIEN sur le vrai marché."""
    rng = np.random.default_rng(seed)
    mu = np.zeros(n)
    sigma = np.zeros(n)
    regimes = {"bull": (0.0015, 0.03), "bear": (-0.0015, 0.035), "range": (0.0, 0.02)}
    i = 0
    while i < n:
        length = int(rng.integers(60, 250))
        m, s = regimes[rng.choice(list(regimes), p=[0.4, 0.25, 0.35])]
        mu[i:i + length] = m
        sigma[i:i + length] = s
        i += length

    ret = rng.normal(mu, sigma)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[100.0], close[:-1]]) * (1 + rng.normal(0, 0.002, n))
    spread = np.abs(rng.normal(0, sigma))
    high = np.maximum(open_, close) * (1 + spread * 0.5)
    low = np.minimum(open_, close) * (1 - spread * 0.5)
    volume = rng.lognormal(10, 0.5, n)
    idx = pd.date_range(start, periods=n, freq=freq, name="timestamp")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
