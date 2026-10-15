"""Envoi des alertes : console toujours, Telegram si configuré (.env)."""

import os
import sys

import requests

# Sous Windows, la sortie redirigée vers un fichier (bot.log) n'est pas en UTF-8 par défaut
# et plante sur les emojis : on force l'UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class Notifier:
    def __init__(self):
        self.token = os.getenv("TELEGRAM_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")

    @property
    def telegram_enabled(self):
        return bool(self.token and self.chat_id)

    def send(self, text):
        print(text)
        if not self.telegram_enabled:
            return
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                data={"chat_id": self.chat_id, "text": text},
                timeout=15,
            )
            if not r.ok:
                print(f"[Telegram] erreur {r.status_code} : {r.text[:200]}")
        except requests.RequestException as e:
            print(f"[Telegram] envoi impossible : {e}")
