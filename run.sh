#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ ! -f "venv/bin/python" ]; then
    echo "First time setup - this only happens once..."
    python3 -m venv venv
    ./venv/bin/python -m pip install --quiet --upgrade pip
fi

echo "Checking dependencies..."
./venv/bin/python -m pip install --quiet -r requirements.txt

if ! ./venv/bin/python setup_api_key.py --check >/dev/null 2>&1; then
    echo ""
    echo "============================================================"
    echo " One more step: you need a free FRED API key (takes ~1 min)."
    echo " Opening the signup page in your browser..."
    echo "============================================================"
    if command -v open >/dev/null; then
        open "https://fred.stlouisfed.org/docs/api/api_key.html"
    elif command -v xdg-open >/dev/null; then
        xdg-open "https://fred.stlouisfed.org/docs/api/api_key.html"
    else
        echo "Visit: https://fred.stlouisfed.org/docs/api/api_key.html"
    fi
    ./venv/bin/python setup_api_key.py
fi

echo ""
echo "Starting FRED Data Explorer - your browser will open automatically..."
./venv/bin/python -m streamlit run app.py
