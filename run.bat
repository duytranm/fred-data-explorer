@echo off
setlocal
cd /d "%~dp0"
title FRED Data Explorer

if not exist "venv\Scripts\python.exe" (
    echo First time setup - this only happens once...
    python -m venv venv
    call venv\Scripts\python.exe -m pip install --quiet --upgrade pip
)

echo Checking dependencies...
call venv\Scripts\python.exe -m pip install --quiet -r requirements.txt

call venv\Scripts\python.exe setup_api_key.py --check >nul 2>&1
if errorlevel 1 (
    echo.
    echo ============================================================
    echo  One more step: you need a free FRED API key ^(takes ~1 min^).
    echo  Opening the signup page in your browser...
    echo ============================================================
    start "" "https://fred.stlouisfed.org/docs/api/api_key.html"
    call venv\Scripts\python.exe setup_api_key.py
)

echo.
echo Starting FRED Data Explorer - your browser will open automatically...
call venv\Scripts\python.exe -m streamlit run app.py
pause
