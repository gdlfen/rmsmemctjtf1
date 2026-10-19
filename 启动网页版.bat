@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 正在启动 A 股双轨投机雷达...
python -m streamlit run app.py --server.headless false
pause
