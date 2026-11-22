@echo off
cd /d "%~dp0\.."
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" src\train.py --data team_data --epochs 10 %*
) else (
    python src\train.py --data team_data --epochs 10 %*
)
pause
