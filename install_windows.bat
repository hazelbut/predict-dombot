@echo off
cd /d %~dp0
where py >nul 2>nul && (py -3 -m venv .venv) || (python -m venv .venv)
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
if not exist .env copy .env.example .env
echo.
echo Done. Fill in .env (notepad .env), then run start_windows.bat
echo Demo without keys: start_demo_windows.bat
pause
