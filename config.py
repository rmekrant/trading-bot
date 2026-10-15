"""Configuration centrale du bot. Modifie les valeurs ici."""

from dataclasses import dataclass, field


@dataclass
class Config:
    # --- Marchés ---
    symbols: list = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"])
    timeframe: str = "4h"          # "4h" = une bougie toutes les 4 heures (validé par strategy_lab.py) ; "1h" et "1d" existent aussi

    # --- Capital et risque ---
    capital: float = 1000.0        # capital de départ (paper trading)
    risk_pct: float = 0.01         # 1 % du capital risqué par trade (leçon 4)
    atr_mult: float = 2.0          # stop-loss = prix d'entrée - 2 x ATR
    max_alloc: float = 0.5         # max 50 % du capital dans une seule position (pas de levier)

    # --- Coûts réalistes ---
    fee: float = 0.001             # 0,1 % de frais par ordre (achat ET vente)
    slippage: float = 0.0005       # 0,05 % de glissement par ordre

    # --- Stratégies ---
    min_votes: int = 2             # nb de stratégies qui doivent être d'accord pour acheter
    max_signal_age: int = 2        # on n'achète que si le signal a moins de N bougies

    # --- Bot ---
    notify_status: bool = False    # True = message récapitulatif à chaque bougie ; False = seulement quand il y a une alerte (achat, vente, stop)
    state_file: str = "paper_state.json"
    log_file: str = "trades_log.csv"
