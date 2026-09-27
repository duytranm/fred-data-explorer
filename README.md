# FRED Data Explorer

An interactive Streamlit app for pulling data from [FRED](https://fred.stlouisfed.org/) (Federal Reserve Economic Data), plotting multiple series over time, and exploring relationships between them (e.g. the Beveridge curve, the Phillips curve, or your own ratios).

## Features

- Pull any FRED series by ID, view raw / YoY % change / period % change / normalized values
- Overlay multiple series, with a secondary y-axis for series on different scales
- Build derived series from two existing ones (ratios, sums, differences, % of)
- Plot one series against another as a time-ordered scatter (Beveridge curve, Phillips curve, or custom)
- Manually group scatter points by color/shape — select on the chart (box/lasso), by a date-range slider, or auto-grouped by year/decade
- Publication-style chart exports (matplotlib, with NBER recession shading) and interactive-style exports (PNG/JPEG/SVG/PDF), plus CSV downloads
- Light theme, quick date-range presets (1Y/5Y/10Y/20Y/YTD/Max)

## Two ways to run this

- **Google Colab portal** (see below) - nothing installed, just a browser and a free Google account. Best for trying it out or sharing with someone who has no Python set up.
- **Local one-click script** (see below) - needs Python 3.10+ on your machine. Best for regular use.

## Zero-install: Google Colab

Open `FRED_Streamlit_Colab.ipynb` in Google Colab (colab.research.google.com, then File - Upload notebook, or open it from Google Drive), then use the Runtime menu, Run all (or Ctrl+F9 / Cmd+F9).

That one action installs everything, asks for your FRED API key, starts the app, and prints a public link to open it at. Leave the last cell running while you use the app.

## Setup — one click

**Requirements:** Python 3.10+ installed, and that's it.

- **Windows:** double-click `run.bat`
- **macOS/Linux:** double-click `run.sh` (or run `chmod +x run.sh` once, then `./run.sh`)

That single script:
1. Creates a private virtual environment and installs all dependencies (first run only — instant after that)
2. Detects you don't have a FRED API key yet, opens the free signup page in your browser, and prompts you to paste the key in once it's stored securely in your OS's credential manager
3. Launches the app and opens it in your browser

Every run after the first just launches straight into the app — no setup steps repeat.

### Manual setup (if you'd rather not use the script)

```bash
pip install -r requirements.txt
python setup_api_key.py      # one time — paste your key from https://fred.stlouisfed.org/docs/api/api_key.html
streamlit run app.py
```

Other `setup_api_key.py` options:
- `--check` — confirms a key is stored
- `--remove` — deletes the stored key
- Alternatively, set the `FRED_API_KEY` environment variable instead of using `keyring` — useful on systems where `keyring` has no backend available (see Troubleshooting).

## Deploy on Streamlit Community Cloud

Share a live link with anyone — no install on their end.

1. Push this folder to a GitHub repo (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and click **New app**.
3. Pick the repo/branch and set the main file path to `app.py`.
4. (Optional) Before deploying, open **Advanced settings → Secrets** and paste:
   ```toml
   FRED_API_KEY = "your_fred_api_key_here"
   ```
   (get a free key at [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html) if you don't have one)
5. Click **Deploy**. The app reads the key from Streamlit's secrets automatically — `keyring`/Windows Credential Manager isn't available on Cloud, and the app already falls back past it.

Setting a secret is optional: the app also has a "Your own FRED API key" field at the top of its sidebar, so anyone using the deployed app can paste their own free key for their session instead of relying on yours. Configuring the secret just means the app works out of the box without visitors needing their own key.

To test the secrets path locally first: copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`, fill in your key (that file is gitignored, so it never gets pushed), then `streamlit run app.py`.

## Troubleshooting

- **First image export is slow**: the "publication-style" export uses `kaleido`, which downloads a small headless Chromium the first time you generate an image. This is one-time and automatic.
- **`keyring` errors on Linux** (e.g. "No recommended backend was available"): install a backend such as `keyrings.alt` (`pip install keyrings.alt`), or skip `keyring` entirely and set the `FRED_API_KEY` environment variable instead.
- **"No FRED API key found"**: run `python setup_api_key.py --check` to confirm a key is stored, or verify `FRED_API_KEY` is set in your current shell session.

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app |
| `fred_data.py` | FRED data-fetching helpers (`FredDatasets`, recession shading, API key lookup) |
| `setup_api_key.py` | One-time CLI to store your FRED API key |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | Light theme configuration |
| `run.bat` / `run.sh` | One-click local launcher: sets up everything and starts the app |
| `FRED_Streamlit_Colab.ipynb` | Zero-install portal: runs the same app inside Google Colab with a public link |
| `FRED_Data_Explorer.ipynb` | Lightweight standalone notebook version (no Streamlit) with example charts to copy/adapt |
