"""Moteur de backtest réaliste (long uniquement, spot).

- Le signal calculé à la clôture de la bougie i-1 est exécuté à l'OUVERTURE de la bougie i
  (aucun look-ahead).
- Frais et slippage appliqués à chaque ordre.
- Stop-loss ATR vérifié dans la bougie (sur le plus bas) ; gap sous le stop = exécution à l'ouverture.
- Taille de position calculée pour risquer risk_pct du capital.
- Après un stop, on ne rachète pas tant que le signal n'est pas repassé à 0 (évite les allers-retours).
"""

import numpy as np
import pandas as pd

from risk import atr, position_size

PERIODS_PER_YEAR = {"1d": 365, "4h": 2190, "1h": 8760, "15m": 35040}
TRADE_COLS = ["entry_time", "exit_time", "entry_price", "exit_price", "units", "pnl", "ret_pct", "reason"]


def run_backtest(df, signal, cfg, atr_series=None):
    atr_s = atr_series if atr_series is not None else atr(df)
    idx = df.index
    o, l, c = df["open"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    sig = signal.reindex(idx).fillna(0.0).to_numpy()
    av = atr_s.reindex(idx).to_numpy()
    n = len(df)

    cash = cfg.capital
    units = stop = cost = entry_px = 0.0
    entry_i = 0
    blocked = False
    equity = np.full(n, cfg.capital, dtype=float)
    in_pos = np.zeros(n, dtype=bool)
    trades = []

    def exit_trade(i, raw_price, reason):
        nonlocal cash, units
        px = raw_price * (1 - cfg.slippage)
        proceeds = units * px * (1 - cfg.fee)
        pnl = proceeds - cost
        trades.append({
            "entry_time": idx[entry_i], "exit_time": idx[i],
            "entry_price": entry_px, "exit_price": px, "units": units,
            "pnl": pnl, "ret_pct": pnl / cost, "reason": reason,
        })
        cash += proceeds
        units = 0.0

    for i in range(1, n):
        want = sig[i - 1]
        if want == 0.0:
            blocked = False

        # 1) Actions à l'ouverture de la bougie i (signal de la bougie i-1)
        if units > 0 and want == 0.0:
            exit_trade(i, o[i], "signal")
        elif units == 0 and want == 1.0 and not blocked and not np.isnan(av[i - 1]):
            px = o[i] * (1 + cfg.slippage)
            stop_px = px - cfg.atr_mult * av[i - 1]
            q = position_size(cash, cfg.risk_pct, px, stop_px, cfg.max_alloc)
            q = min(q, cash / (px * (1 + cfg.fee)))
            if q > 0:
                cost = q * px * (1 + cfg.fee)
                cash -= cost
                units, stop, entry_px, entry_i = q, stop_px, px, i

        # 2) Stop-loss touché pendant la bougie i
        if units > 0 and l[i] <= stop:
            exit_trade(i, min(o[i], stop), "stop")
            blocked = True

        equity[i] = cash + units * c[i]
        in_pos[i] = units > 0

    return {
        "equity": pd.Series(equity, index=idx),
        "trades": pd.DataFrame(trades, columns=TRADE_COLS),
        "exposure": float(in_pos.mean()),
    }


def buy_and_hold(df, cfg):
    eq = cfg.capital * (1 - cfg.fee) * df["close"] / df["close"].iloc[0]
    return {"equity": eq, "trades": pd.DataFrame(columns=TRADE_COLS), "exposure": 1.0}


def compute_stats(result, timeframe="1d"):
    eq, tr = result["equity"], result["trades"]
    ppy = PERIODS_PER_YEAR.get(timeframe, 365)
    ret = eq.pct_change().dropna()
    years = max((eq.index[-1] - eq.index[0]).total_seconds() / (365.25 * 86400), 1e-9)
    growth = eq.iloc[-1] / eq.iloc[0]
    wins, losses = tr[tr["pnl"] > 0], tr[tr["pnl"] <= 0]
    gross_loss = abs(losses["pnl"].sum())
    return {
        "rendement": growth - 1,
        "annuel": growth ** (1 / years) - 1 if growth > 0 else -1.0,
        "drawdown": (eq / eq.cummax() - 1).min(),
        "sharpe": ret.mean() / ret.std() * np.sqrt(ppy) if ret.std() > 0 else np.nan,
        "exposition": result["exposure"],
        "trades": len(tr),
        "win%": len(wins) / len(tr) if len(tr) else np.nan,
        "profit factor": wins["pnl"].sum() / gross_loss if gross_loss > 0 else np.nan,
        "gain moyen/trade": tr["ret_pct"].mean() if len(tr) else np.nan,
    }
