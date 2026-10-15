"""Bot de SIGNAUX + PAPER TRADING (aucun argent réel, aucun ordre envoyé à un exchange).

Il analyse les bougies FERMÉES, combine les stratégies, calcule stop et taille de position,
t'envoie l'alerte (console + Telegram) et simule le trade dans un portefeuille fictif.

Usage :
    python bot.py                    # un cycle puis arrêt (à mettre dans un cron / planificateur)
    python bot.py --loop             # tourne en continu (vérifie toutes les 60 min)
    python bot.py --test-telegram    # vérifie que Telegram est bien configuré
    python bot.py --reset            # repart d'un portefeuille paper neuf
"""

import argparse
import csv
import dataclasses
import datetime as dt
import json
import os
import time

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from config import Config
from data import drop_unclosed, fetch_last_price, fetch_ohlcv
from notifier import Notifier
from risk import atr, position_size
from strategies import combine, default_strategies, strategy_positions

MIN_BARS = 200  # le filtre de tendance de la cassure a besoin de 125 bougies
MIN_NOTIONAL = 10.0  # on ignore les positions de moins de 10 USDT


# ----------------------------------------------------------------------------
# État du portefeuille paper
# ----------------------------------------------------------------------------
def load_state(cfg):
    if os.path.exists(cfg.state_file):
        with open(cfg.state_file, encoding="utf-8") as f:
            return json.load(f)
    return {"cash": cfg.capital, "positions": {}, "last_bar": {}, "blocked": {}, "last_price": {}}


def save_state(cfg, state):
    with open(cfg.state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def equity(state):
    total = state["cash"]
    for sym, pos in state["positions"].items():
        total += pos["units"] * state["last_price"].get(sym, pos["entry_price"])
    return total


def log_trade(cfg, row):
    new_file = not os.path.exists(cfg.log_file)
    with open(cfg.log_file, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["date", "symbole", "action", "prix", "unites", "stop", "raison", "pnl"])
        w.writerow(row)


def close_paper(cfg, state, symbol, raw_price, reason):
    pos = state["positions"].pop(symbol)
    px = raw_price * (1 - cfg.slippage)
    proceeds = pos["units"] * px * (1 - cfg.fee)
    pnl = proceeds - pos["cost"]
    state["cash"] += proceeds
    log_trade(cfg, [dt.datetime.now().isoformat(timespec="seconds"), symbol, "VENTE",
                    round(px, 6), pos["units"], "", reason, round(pnl, 2)])
    return px, pnl, pnl / pos["cost"]


# ----------------------------------------------------------------------------
# Logique d'un symbole
# ----------------------------------------------------------------------------
def process_symbol(symbol, cfg, state, strategies):
    """Retourne (lignes de message, bougie_traitée?)."""
    df = drop_unclosed(fetch_ohlcv(symbol, cfg.timeframe, limit=500), cfg.timeframe)
    if len(df) < MIN_BARS:
        return [f"{symbol}: pas assez de données ({len(df)} bougies)"], False

    bar = df.index[-1].isoformat()
    if state["last_bar"].get(symbol) == bar:
        return [], False  # cette bougie a déjà été analysée
    state["last_bar"][symbol] = bar

    pos_df = strategy_positions(strategies, df)
    signal = combine(pos_df, cfg.min_votes)
    sig_now = signal.iloc[-1]
    votes = int(pos_df.iloc[-1].sum())
    agree = [name for name, v in pos_df.iloc[-1].items() if v == 1.0]
    trailing = int(np.cumprod((signal.to_numpy()[::-1] == 1.0).astype(int)).sum())

    price = fetch_last_price(symbol) or float(df["close"].iloc[-1])
    state["last_price"][symbol] = price
    if sig_now == 0.0:
        state["blocked"][symbol] = False

    pos = state["positions"].get(symbol)
    lines = []

    if pos:
        after = df[df.index > pd.Timestamp(pos["entry_bar"])]
        if len(after) and after["low"].min() <= pos["stop"]:
            px, pnl, pct = close_paper(cfg, state, symbol, pos["stop"], "stop-loss")
            state["blocked"][symbol] = True
            lines.append(f"🛑 STOP-LOSS touché sur {symbol}\n"
                         f"   Sortie ~{px:.4f} | résultat {pnl:+.2f} USDT ({pct:+.1%})")
        elif sig_now == 0.0:
            px, pnl, pct = close_paper(cfg, state, symbol, price, "signal")
            lines.append(f"🔴 VENDRE {symbol} à ~{px:.4f}\n"
                         f"   Résultat {pnl:+.2f} USDT ({pct:+.1%}) | les stratégies ne sont plus d'accord")
        else:
            lines.append(f"⏸ {symbol}: position ouverte depuis {pos['entry_bar'][:10]} "
                         f"(entrée {pos['entry_price']:.4f}, stop {pos['stop']:.4f}, "
                         f"actuel {price:.4f}, {price / pos['entry_price'] - 1:+.1%})")
    else:
        fresh = trailing <= cfg.max_signal_age
        if sig_now == 1.0 and fresh and not state["blocked"].get(symbol):
            a = atr(df).iloc[-1]
            entry = price * (1 + cfg.slippage)
            stop = entry - cfg.atr_mult * a
            units = position_size(equity(state), cfg.risk_pct, entry, stop, cfg.max_alloc)
            units = min(units, state["cash"] / (entry * (1 + cfg.fee)))
            if np.isnan(a) or units * entry < MIN_NOTIONAL:
                lines.append(f"{symbol}: signal d'achat mais position trop petite ou ATR indisponible, ignoré")
            else:
                cost = units * entry * (1 + cfg.fee)
                state["cash"] -= cost
                state["positions"][symbol] = {
                    "units": units, "entry_price": entry, "stop": stop,
                    "entry_bar": bar, "cost": cost,
                }
                risk_amt = units * (entry - stop)
                log_trade(cfg, [dt.datetime.now().isoformat(timespec="seconds"), symbol, "ACHAT",
                                round(entry, 6), units, round(stop, 6), "signal", ""])
                lines.append(
                    f"🟢 ACHETER {symbol} à ~{entry:.4f}\n"
                    f"   Stop-loss : {stop:.4f} ({stop / entry - 1:+.1%})\n"
                    f"   Taille : {units:.6f} unités (~{units * entry:.2f} USDT)\n"
                    f"   Risque max : {risk_amt:.2f} USDT ({risk_amt / equity(state):.1%} du capital)\n"
                    f"   D'accord : {', '.join(agree)}"
                )
        else:
            note = " (signal trop ancien, on n'entre pas en cours de route)" if sig_now == 1.0 and not fresh else ""
            lines.append(f"{symbol}: aucun signal ({votes}/{len(strategies)} stratégies en position){note}")
    return lines, True


def run_cycle(cfg, state, strategies, notifier):
    messages, any_new = [], False
    for symbol in cfg.symbols:
        try:
            lines, processed = process_symbol(symbol, cfg, state, strategies)
        except Exception as e:
            print(f"[erreur] {symbol} : {e}")
            continue
        messages += lines
        any_new |= processed
    save_state(cfg, state)

    has_event = any(m[0] in "🟢🔴🛑" for m in messages)
    if any_new and (has_event or cfg.notify_status):
        eq = equity(state)
        header = f"📊 Bot signaux (PAPER) | {dt.date.today()}"
        footer = f"💼 Portefeuille fictif : {eq:.2f} USDT ({eq / cfg.capital - 1:+.1%})"
        notifier.send("\n".join([header, *messages, footer]))
    elif any_new:
        print(f"[{dt.datetime.now():%H:%M}] nouvelle bougie analysée, aucun signal : pas d'alerte envoyée")
    else:
        print(f"[{dt.datetime.now():%H:%M}] pas de nouvelle bougie fermée, rien à faire")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--loop", action="store_true")
    p.add_argument("--interval", type=int, default=60, help="minutes entre deux vérifications (--loop)")
    p.add_argument("--symbols", nargs="+")
    p.add_argument("--timeframe")
    p.add_argument("--capital", type=float)
    p.add_argument("--reset", action="store_true")
    p.add_argument("--test-telegram", action="store_true")
    args = p.parse_args()

    cfg = Config()
    overrides = {k: v for k, v in {"symbols": args.symbols, "timeframe": args.timeframe,
                                    "capital": args.capital}.items() if v}
    cfg = dataclasses.replace(cfg, **overrides)
    notifier = Notifier()

    if args.test_telegram:
        if not notifier.telegram_enabled:
            print("Telegram non configuré : renseigne TELEGRAM_TOKEN et TELEGRAM_CHAT_ID dans .env")
        else:
            notifier.send("✅ Test : ton bot peut t'envoyer des messages.")
        return

    if args.reset:
        for f in (cfg.state_file, cfg.log_file):
            if os.path.exists(f):
                os.remove(f)
        print("Portefeuille paper réinitialisé.")

    state = load_state(cfg)
    strategies = default_strategies()
    print(f"Bot démarré | {cfg.symbols} | {cfg.timeframe} | mode PAPER (aucun ordre réel)")
    while True:
        run_cycle(cfg, state, strategies, notifier)
        if not args.loop:
            break
        time.sleep(args.interval * 60)


if __name__ == "__main__":
    main()
