"""Banc d'essai des stratégies (bougies de 1 h, 4 h ou 1 jour).

Méthode (pour ne pas se raconter d'histoires) :
  1. Les données de chaque actif sont coupées en deux : 60 % ENTRAÎNEMENT, 40 % TEST (jamais utilisé pour choisir).
  2. Pour chaque famille de stratégies, on choisit les paramètres qui marchent le mieux à l'ENTRAÎNEMENT,
     en regardant la médiane sur TOUS les actifs (pas un seul).
  3. On applique ce choix, sans le modifier, sur la période TEST, actif par actif, avec les frais réels
     puis avec des frais DOUBLES.
  4. Verdict "PROMETTEUR" seulement si tous les tests ci-dessous sont réussis (dont : gagnante dans
     les DEUX moitiés de la période test, pour ne pas dépendre d'une seule phase de marché).

Même "PROMETTEUR" ne prouve rien : ça veut seulement que la stratégie a survécu à un contrôle sévère.
Plus on teste de stratégies, plus il y a de chances qu'une réussisse par hasard.

Exemples :
    python strategy_lab.py                          # 5 actifs, 5 ans de bougies de 4 h
    python strategy_lab.py --timeframe 1d           # bougies journalières
    python strategy_lab.py --timeframe 1h --years 3 # bougies d'une heure
    python strategy_lab.py --synthetic random       # test du banc d'essai sur du hasard pur (doit tout rejeter)
    python strategy_lab.py --synthetic trend        # test sur des données avec une vraie tendance (doit en trouver)
"""

import argparse
import dataclasses
import time

import numpy as np
import pandas as pd

from backtest import compute_stats, run_backtest
from candidates import candidate_families
from config import Config
from data import fetch_history
from risk import atr

DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
MIN_YEARS = 1.0  # moins d'un an de données : trop court pour conclure
TF_HOURS = {"1h": 1.0, "4h": 4.0, "1d": 24.0}

# Seuils du verdict
MIN_POSITIVE_SHARE = 0.6    # au moins 60 % des actifs gagnants sur la période TEST
MIN_TRADES_TEST = 60        # nombre total de trades sur le TEST, tous actifs confondus
MIN_PROFIT_FACTOR = 1.15    # gains cumulés / pertes cumulées (médiane des actifs)
MIN_GRID_POSITIVE = 0.4     # au moins 40 % des combinaisons de la famille gagnantes à l'entraînement


def synthetic_hourly(seed, n=26000, trend=False, tf_hours=1.0):
    """Données fictives. random = marché sans aucun avantage possible ; trend = tendances persistantes."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, 0.004 * np.sqrt(tf_hours), n)
    drift = np.zeros(n)
    if trend:
        for i in range(1, n):
            drift[i] = (1 - 0.002 * tf_hours) * drift[i - 1] + rng.normal(0, 0.00003 * tf_hours)
    ret = eps + drift
    close = 100 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[100.0], close[:-1]])
    wick = np.abs(rng.normal(0, 0.0015, (2, n)))
    high = np.maximum(open_, close) * (1 + wick[0])
    low = np.minimum(open_, close) * (1 - wick[1])
    idx = pd.date_range("2022-01-01", periods=n, freq=f"{int(tf_hours)}h", name="timestamp")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": rng.lognormal(10, 0.5, n)}, index=idx)


def load_data(args):
    data = {}
    if args.synthetic:
        for k, sym in enumerate(DEFAULT_SYMBOLS):
            n = int(args.years * 8760 / TF_HOURS[args.timeframe])
            data[sym] = synthetic_hourly(args.seed + k, n=n, trend=(args.synthetic == "trend"),
                                         tf_hours=TF_HOURS[args.timeframe])
        return data
    for sym in args.symbols:
        try:
            df = fetch_history(sym, args.timeframe, args.years)
        except Exception as e:
            print(f"  {sym} ignoré : {e}")
            continue
        min_bars = int(MIN_YEARS * 8760 / TF_HOURS[args.timeframe])
        if len(df) < min_bars:
            print(f"  {sym} ignoré : seulement {len(df)} bougies (il en faut au moins {min_bars})")
            continue
        data[sym] = df
    return data


def run(df, pos, cfg, atr_full, a, b, tf="4h"):
    res = run_backtest(df.iloc[a:b], pos.iloc[a:b], cfg, atr_full.iloc[a:b])
    return compute_stats(res, tf)


def fmt_pct(x):
    return "-" if pd.isna(x) else f"{x:+.1%}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--timeframe", choices=list(TF_HOURS), default="4h")
    p.add_argument("--years", type=float, default=5)
    p.add_argument("--fee", type=float, default=0.001)
    p.add_argument("--slippage", type=float, default=0.0005)
    p.add_argument("--split", type=float, default=0.6, help="part des données utilisée pour l'entraînement")
    p.add_argument("--synthetic", choices=["random", "trend"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--only", help="ne tester qu'une famille (partie du nom)")
    args = p.parse_args()

    cfg = dataclasses.replace(Config(), fee=args.fee, slippage=args.slippage, max_alloc=1.0)
    cfg2 = dataclasses.replace(cfg, fee=args.fee * 2, slippage=args.slippage * 2)

    print("Chargement des données (la première fois, ça peut prendre quelques minutes)...")
    data = load_data(args)
    if len(data) < 3:
        print("\nMoins de 3 actifs disponibles : impossible de conclure sérieusement.")
        print("Si Binance est bloqué chez toi, dis-le moi : on changera de source de données.")
        return
    symbols = list(data)
    n_min = min(len(d) for d in data.values())
    print(f"\n{len(symbols)} actifs : {', '.join(symbols)} | {n_min} bougies de {args.timeframe} chacun "
          f"(~{n_min * TF_HOURS[args.timeframe] / 8760:.1f} an) | frais {cfg.fee:.2%} + slippage {cfg.slippage:.2%} par ordre")

    cut = {s: int(len(d) * args.split) for s, d in data.items()}
    atrs = {s: atr(d) for s, d in data.items()}
    families = candidate_families(TF_HOURS[args.timeframe])
    if args.only:
        families = {k: v for k, v in families.items() if args.only.lower() in k.lower()}
    total_combos = sum(len(v) for v in families.values())
    print(f"{len(families)} familles, {total_combos} combinaisons de paramètres au total.")
    print(f"Les nombres dans les noms des stratégies sont des nombres de bougies de {args.timeframe}.\n")

    summary, details = [], []
    done, t0 = 0, time.time()
    for fam, strategies in families.items():
        # 1) Choix des paramètres sur l'ENTRAÎNEMENT uniquement
        scored = []
        for strat in strategies:
            per_sym, trades = [], 0
            for s in symbols:
                pos = strat.positions(data[s])
                st = run(data[s], pos, cfg, atrs[s], 0, cut[s], args.timeframe)
                per_sym.append(st["sharpe"] if not pd.isna(st["sharpe"]) else -9.0)
                trades += st["trades"]
            score = float(np.median(per_sym)) if trades >= 30 else -9.0
            scored.append((score, strat))
            done += 1
            if done % 20 == 0:
                print(f"  ... {done}/{total_combos} combinaisons testées ({time.time() - t0:.0f} s)")
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best = scored[0]
        grid_positive = float(np.mean([sc > 0 for sc, _ in scored]))

        # 2) Application sur le TEST, sans rien changer
        rets, pfs, rets2, bh, trades_test, dds = [], [], [], [], 0, []
        h1, h2 = [], []
        for s in symbols:
            pos = best.positions(data[s])
            end = len(data[s])
            mid = cut[s] + (end - cut[s]) // 2
            st = run(data[s], pos, cfg, atrs[s], cut[s], end, args.timeframe)
            st2 = run(data[s], pos, cfg2, atrs[s], cut[s], end, args.timeframe)
            h1.append(run(data[s], pos, cfg, atrs[s], cut[s], mid, args.timeframe)["rendement"])
            h2.append(run(data[s], pos, cfg, atrs[s], mid, end, args.timeframe)["rendement"])
            seg = data[s]["close"].iloc[cut[s]:]
            bh_ret = seg.iloc[-1] / seg.iloc[0] - 1
            pf = 0.0 if pd.isna(st["profit factor"]) else st["profit factor"]
            rets.append(st["rendement"]); rets2.append(st2["rendement"]); pfs.append(pf)
            bh.append(bh_ret); dds.append(st["drawdown"]); trades_test += st["trades"]
            details.append({"famille": fam, "strategie": best.name, "actif": s, "rendement_test": st["rendement"],
                            "rendement_frais_x2": st2["rendement"], "buy_hold": bh_ret, "drawdown": st["drawdown"],
                            "trades": st["trades"], "profit_factor": pf})

        pos_share = float(np.mean([r > 0 for r in rets]))
        med_pf = float(np.median(pfs))
        med_ret2 = float(np.median(rets2))
        checks = {
            f"gagnante sur >= {MIN_POSITIVE_SHARE:.0%} des actifs": pos_share >= MIN_POSITIVE_SHARE,
            f">= {MIN_TRADES_TEST} trades": trades_test >= MIN_TRADES_TEST,
            f"profit factor médian >= {MIN_PROFIT_FACTOR}": med_pf >= MIN_PROFIT_FACTOR,
            "encore gagnante avec des frais x2": med_ret2 > 0,
            f"pas un coup isolé (>= {MIN_GRID_POSITIVE:.0%} des paramètres gagnants)": grid_positive >= MIN_GRID_POSITIVE,
            "gagnante dans les 2 moitiés de la période test": float(np.median(h1)) > 0 and float(np.median(h2)) > 0,
        }
        verdict = "PROMETTEUR" if all(checks.values()) else "rejetée"
        failed = [k for k, v in checks.items() if not v]
        summary.append({
            "famille": fam, "paramètres retenus": best.name, "verdict": verdict,
            "rendement médian TEST": fmt_pct(float(np.median(rets))),
            "buy&hold médian": fmt_pct(float(np.median(bh))),
            "actifs gagnants": f"{int(round(pos_share * len(symbols)))}/{len(symbols)}",
            "trades TEST": trades_test, "profit factor": round(med_pf, 2),
            "rend. frais x2": fmt_pct(med_ret2),
            "moitié 1 / moitié 2": f"{fmt_pct(float(np.median(h1)))} / {fmt_pct(float(np.median(h2)))}", "drawdown médian": fmt_pct(float(np.median(dds))),
            "% params gagnants": f"{grid_positive:.0%}",
            "pourquoi rejetée": "; ".join(failed),
        })

    table = pd.DataFrame(summary)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 60)
    print("\n=== RÉSULTATS SUR LA PÉRIODE TEST (jamais utilisée pour choisir) ===\n")
    print(table.drop(columns=["pourquoi rejetée"]).to_string(index=False))
    print("\nPourquoi les stratégies rejetées l'ont été :")
    for row in summary:
        if row["verdict"] != "PROMETTEUR":
            print(f"  - {row['famille']} : {row['pourquoi rejetée']}")

    winners = [r for r in summary if r["verdict"] == "PROMETTEUR"]
    print()
    if winners:
        print(f"{len(winners)} famille(s) PROMETTEUSE(S) : {', '.join(r['paramètres retenus'] for r in winners)}")
        print(f"Attention : sur {len(families)} familles testées, une réussite par pur hasard reste possible.")
        print("La suite logique : les suivre en paper trading plusieurs semaines avant tout argent réel.")
    else:
        print("Aucune stratégie ne résiste au contrôle. C'est un résultat utile : ça t'évite de suivre")
        print("des signaux qui auraient probablement fait perdre de l'argent après frais.")

    table.to_csv("lab_resultats.csv", index=False)
    pd.DataFrame(details).to_csv("lab_details.csv", index=False)
    print("\nFichiers écrits : lab_resultats.csv (résumé) et lab_details.csv (détail par actif).")


if __name__ == "__main__":
    main()
