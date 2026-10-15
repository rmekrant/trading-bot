# trading-bot : bot de signaux crypto + paper trading

Projet éducatif en Python. Le bot analyse le marché crypto (bougies 4 h, 5 actifs), calcule le stop-loss
et la taille de position, envoie une alerte Telegram quand il y a un signal, et simule les trades dans un
portefeuille fictif.

**Il n'envoie aucun ordre à un exchange et n'utilise aucune clé API d'exchange.**
Aucune rentabilité n'est revendiquée : la plupart des traders particuliers perdent de l'argent.
Ceci n'est pas un conseil financier.

## Ce que fait le projet

- Données de marché via `ccxt` (Binance, avec Kraken/Coinbase en secours) : BTC, ETH, SOL, BNB, XRP.
- 3 stratégies de suivi de tendance combinées par vote (2 sur 3) : Supertrend, cassure de Donchian
  avec filtre de tendance, cassure de Keltner.
- Stop-loss à 2 x ATR, taille de position pour risquer 1 % du capital par trade (max 50 % par position).
- Alertes Telegram uniquement sur un vrai événement (achat, vente, stop-loss).
- Paper trading avec frais (0,1 %) et glissement (0,05 %) par ordre.

## Méthode

- **Backtest** sans look-ahead : signal sur la bougie i-1, exécution à l'ouverture de la bougie i,
  stops simulés dans la bougie, frais et glissement inclus.
- **`strategy_lab.py`** : séparation entraînement/test 60/40, paramètres choisis sur l'entraînement,
  validation sur 5 actifs avec des critères stricts (profit factor, nombre de trades, résistance à des
  frais doublés, stabilité des paramètres).
- Constat : en bougies 1 h, aucune stratégie ne survit aux frais ; en 4 h, 5 familles passent les critères
  (mais elles sont corrélées : c'est une seule idée, le suivi de tendance).

## Installation

```bash
pip install -r requirements.txt
python test_offline.py       # vérifie le code sans internet
```

## Utilisation

```bash
python run_backtest.py --timeframe 4h --years 5     # backtest BTC/USDT
python strategy_lab.py --timeframe 4h --years 5     # compare toutes les familles de stratégies
python bot.py                                       # un cycle d'analyse (à planifier toutes les heures)
python bot.py --test-telegram                       # teste l'envoi Telegram
python bot.py --reset                               # repart d'un portefeuille fictif neuf
```

Telegram : copie `.env.example` en `.env` et remplis `TELEGRAM_TOKEN` et `TELEGRAM_CHAT_ID`.
Le fichier `.env` est ignoré par Git : ne le publie jamais.

## Fichiers

| Fichier | Rôle |
| --- | --- |
| `config.py` | Réglages (marchés, capital, risque, frais) |
| `data.py` | Données ccxt + données fictives pour les tests |
| `strategies.py` | Stratégies par défaut et vote |
| `candidates.py` | Stratégies candidates testées par le lab |
| `risk.py` | ATR, stop-loss, taille de position |
| `backtest.py` / `run_backtest.py` | Moteur et lancement du backtest |
| `strategy_lab.py` | Recherche et validation de stratégies |
| `bot.py` | Signaux, paper trading, alertes |
| `notifier.py` | Envoi Telegram |
| `test_offline.py` | Tests sans internet |

## Limites

- Pas de levier, pas de short, pas d'exécution automatique d'ordres.
- Pas de garantie d'avantage statistique : c'est ce que le paper trading doit vérifier.
- La période de test n'est pas vierge (le timeframe a été choisi après avoir vu l'échec du 1 h).
