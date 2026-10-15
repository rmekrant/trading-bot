"""Backtest de toutes les stratégies + de leur combinaison, avec séparation
in-sample / out-of-sample et comparaison au buy & hold.

Exemples :
    python run_backtest.py
    python run_backtest.py --symbol ETH/USDT --years 5
    python run_backtest.py --fee 0.003
    python run_backtest.py --synthetic        # données fictives, sans internet
"""

import argparse
import dataclasses

import matplotlib.pyplot as plt
import pandas as pd

from backtest import buy_and_hold, compute_stats, run_backtest
from config import Config
from data import fetch_history, synthetic_ohlcv
from risk import atr
from strategies import combine, default_strategies, strategy_positions

PCT_COLS = ["rendement", "annuel", "drawdown", "exposition", "win%", "gain moyen/trade"]


def format_table(rows):
    table = pd.DataFrame(rows).T
    for col in table.columns:
        if col in PCT_COLS:
            table[col] = table[col].map(lambda v: "-" if pd.isna(v) else f"{v:.1%}")
        elif col == "trades":
            table[col] = table[col].astype(int)
        else:
            table[col] = table[col].map(lambda v: "-" if pd.isna(v) else f"{v:.2f}")
    return table


def evaluate(df, curves, cfg, timeframe, a, b):
    """Évalue chaque courbe de position sur df.iloc[a:b]."""
    seg = df.iloc[a:b]
    atr_full = atr(df)
    rows = {}
    for name, pos in curves.items():
        res = run_backtest(seg, pos.iloc[a:b], cfg, atr_full.iloc[a:b])
        rows[name] = compute_stats(res, timeframe)
    rows["BUY & HOLD"] = compute_stats(buy_and_hold(seg, cfg), timeframe)
    return format_table(rows), seg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--timeframe", default="1d")
    p.add_argument("--years", type=float, default=5)
    p.add_argument("--split", type=float, default=0.7, help="part des données en in-sample")
    p.add_argument("--fee", type=float, default=0.001)
    p.add_argument("--slippage", type=float, default=0.0005)
    p.add_argument("--risk", type=float, default=0.01)
    p.add_argument("--atr-mult", type=float, default=2.0)
    p.add_argument("--min-votes", type=int, default=2)
    p.add_argument("--max-alloc", type=float, default=1.0)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--show", action="store_true", help="afficher le graphique")
    args = p.parse_args()

    cfg = dataclasses.replace(
        Config(), fee=args.fee, slippage=args.slippage, risk_pct=args.risk,
        atr_mult=args.atr_mult, min_votes=args.min_votes, max_alloc=args.max_alloc,
    )

    if args.synthetic:
        df = synthetic_ohlcv(1800)
        label = "DONNÉES FICTIVES"
    else:
        df = fetch_history(args.symbol, args.timeframe, args.years)
        label = args.symbol
    print(f"\n{label} | {df.index[0].date()} -> {df.index[-1].date()} | {len(df)} bougies")
    print(f"Frais {cfg.fee:.2%} + slippage {cfg.slippage:.2%} par ordre | risque {cfg.risk_pct:.1%}/trade "
          f"| stop {cfg.atr_mult} x ATR")

    strategies = default_strategies()
    pos_df = strategy_positions(strategies, df)
    curves = {col: pos_df[col] for col in pos_df.columns}
    curves[f"COMBO ({cfg.min_votes} votes sur {len(strategies)})"] = combine(pos_df, cfg.min_votes)

    cut = int(len(df) * args.split)
    pd.set_option("display.width", 200)
    for title, a, b in [
        ("PÉRIODE ENTIÈRE", 0, len(df)),
        (f"IN-SAMPLE (premiers {args.split:.0%}) : sert à choisir/ajuster", 0, cut),
        (f"OUT-OF-SAMPLE (derniers {1 - args.split:.0%}) : le test honnête", cut, len(df)),
    ]:
        table, _ = evaluate(df, curves, cfg, args.timeframe, a, b)
        print(f"\n=== {title} ===")
        print(table.to_string())

    # Graphique
    atr_full = atr(df)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, pos in curves.items():
        res = run_backtest(df, pos, cfg, atr_full)
        ax.plot(res["equity"], label=name, linewidth=2.2 if name.startswith("COMBO") else 1)
    ax.plot(buy_and_hold(df, cfg)["equity"], label="BUY & HOLD", color="black", linestyle="--")
    ax.axvline(df.index[cut], color="red", linestyle=":", label="début out-of-sample")
    ax.set_title(f"{label} : capital au cours du temps (départ = {cfg.capital:.0f})")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig("resultat.png", dpi=120)
    print("\nGraphique enregistré : resultat.png")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
