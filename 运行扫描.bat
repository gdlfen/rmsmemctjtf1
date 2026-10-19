@echo off
chcp 65001 >nul
cd /d "%~dp0"
python scripts\scan.py --n 12 --depth 420
pause
