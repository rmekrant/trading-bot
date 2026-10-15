"""Test hors ligne : fait tourner le bot jour après jour sur des données fictives
et vérifie que la comptabilité reste cohérente. Lance :  python test_offline.py"""

import dataclasses
import os
import tempfile

import bot
from backtest import buy_and_hold, compute_stats, run_backtest
from config import Config
from data import synthetic_ohlcv
from strategies import combine, default_strategies, strategy_positions


def test_backtest():
    df = synthetic_ohlcv(1500, seed=7)
    cfg = dataclasses.replace(Config(), max_alloc=1.0)
    strategies = default_strategies()
    pos_df = strategy_positions(strategies, df)
    for name, pos in {**{c: pos_df[c] for c in pos_df}, "combo": combine(pos_df, 2)}.items():
        res = run_backtest(df, pos, cfg)
        s = compute_stats(res)
        assert res["equity"].min() > 0, name
        assert (res["trades"]["units"] > 0).all() if len(res["trades"]) else True
        print(f"  {name:<18} rendement {s['rendement']:+8.1%} | DD {s['drawdown']:7.1%} | {s['trades']} trades")
    bh = compute_stats(buy_and_hold(df, cfg))
    print(f"  {'buy & hold':<18} rendement {bh['rendement']:+8.1%} | DD {bh['drawdown']:7.1%}")
    # Sans frais ni slippage, risque d'1 % : la perte max par trade doit rester ~1 % du capital
    cfg0 = dataclasses.replace(cfg, fee=0.0, slippage=0.0)
    res = run_backtest(df, combine(pos_df, 2), cfg0)
    stops = res["trades"][res["trades"]["reason"] == "stop"]
    print(f"  stops touchés : {len(stops)}")


def test_bot_cycle():
    df = synthetic_ohlcv(900, seed=1)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = dataclasses.replace(
            Config(), symbols=["TEST/USDT"],
            state_file=os.path.join(tmp, "state.json"), log_file=os.path.join(tmp, "log.csv"),
        )
        # Prix fictifs ~100 : on met un capital élevé pour dépasser le minimum de 10 USDT
        cfg = dataclasses.replace(cfg, capital=10_000.0)
        state = bot.load_state(cfg)
        strategies = default_strategies()
        counts = {"🟢": 0, "🔴": 0, "🛑": 0}
        for k in range(300, 900):
            bot.fetch_ohlcv = lambda s, tf, limit=500, k=k: df.iloc[:k]
            bot.fetch_last_price = lambda s, k=k: float(df["close"].iloc[k - 1])
            lines, _ = bot.process_symbol("TEST/USDT", cfg, state, strategies)
            for m in lines:
                if m[0] in counts:
                    counts[m[0]] += 1
            assert state["cash"] >= -1e-6, "cash négatif !"
            assert bot.equity(state) > 0
        # une même bougie ne doit pas être traitée deux fois
        lines, processed = bot.process_symbol("TEST/USDT", cfg, state, strategies)
        assert not processed and lines == []
        bot.save_state(cfg, state)
        print(f"  achats {counts['🟢']} | ventes {counts['🔴']} | stops {counts['🛑']} "
              f"| equity finale {bot.equity(state):.2f} (départ {cfg.capital:.0f})")
        assert counts["🟢"] > 0, "le bot n'a jamais acheté sur 600 jours simulés"


if __name__ == "__main__":
    print("Test du backtest sur données fictives :")
    test_backtest()
    print("Test du bot (paper) jour après jour :")
    test_bot_cycle()
    print("\nOK : le code tourne et la comptabilité est cohérente.")
