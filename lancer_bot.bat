@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
python bot.py >> bot.log 2>&1
