@echo off
cd /d %~dp0
.venv\Scripts\python run_terminal.py --demo %*
pause
