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
- New York Fed [Markets Data APIs](https://markets.newyorkfed.org/static/docs/markets-api.html) (no key needed):
  - Chart reference rates (SOFR, EFFR, OBFR, TGCR, BGCR, plus volumes and SOFR averages/index), SOMA holdings, Fed repo/reverse repo totals and any primary dealer statistic alongside FRED series. Pick them under **NY Fed Markets series** in the sidebar, or type IDs like `NYFED:SOFR` or `NYFED:PD:PDPOSMBS-TOT`.
  - The **NY Fed Markets** tab browses every area of the API (operations, SOMA, primary dealers, market share, guide sheets, or any custom endpoint path) as tables with CSV/JSON downloads and quick charts, with an optional switch to the API's beta endpoints.
- Bureau of Economic Analysis data via BEA's official [`beaapi`](https://github.com/us-bea/beaapi) library (free key: [apps.bea.gov/API/signup](https://apps.bea.gov/API/signup/)):
  - The **BEA** tab covers every BEA API dataset (NIPA, GDP by industry, Regional, ITA, IIP, Input-Output, MNE and more), with parameter choices loaded from BEA's metadata.
  - Chart any series from a fetched table, then add it to the Time Series and Relationship tabs to combine it with FRED and NY Fed data.
  - Provide the key as `BEA_API_KEY` in Streamlit secrets or an env var, or locally with `python setup_api_key.py --bea`.
- U.S. Census Bureau [Data API](https://www.census.gov/data/developers/data-sets.html) (free key: [api.census.gov/data/key_signup.html](https://api.census.gov/data/key_signup.html)):
  - The **Census** tab searches all ~850 aggregate and time-series datasets (ACS, decennial census, County Business Patterns, economic indicators, international trade, QWI and more; microdata optional). It picks tables and variables, geography levels and time ranges, pre-filled from Census's own example query for each dataset.
  - Time series can be added to the Time Series and Relationship tabs; cross-sections (e.g. one ACS year by state) are shown as a ranked bar chart.
  - Browsing works without a key; fetching data needs `CENSUS_API_KEY` in Streamlit secrets or an env var, or locally `python setup_api_key.py --census`.
- Bureau of Labor Statistics [Public Data API v2](https://www.bls.gov/developers/api_signature_v2.htm) (free key: [data.bls.gov/registrationEngine](https://data.bls.gov/registrationEngine/)):
  - The **BLS** tab browses each survey's most popular series or takes any series IDs, over a year-range slider, splitting long ranges into as many requests as BLS's limits need.
  - BLS series chart alongside everything else as `BLS:<series ID>` (e.g. `BLS:LNS14000000`), from the **BLS series** sidebar picker or typed anywhere a series ID is accepted.
  - Works without a key, but only 25 requests a day (shared by everyone using a deployed app), 10 years per request and no series titles. `BLS_API_KEY` in Streamlit secrets or an env var (or locally `python setup_api_key.py --bls`) raises that to 500 a day and 20 years, with titles.

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
4. Before deploying, open **Advanced settings → Secrets** and paste:
   ```toml
   FRED_API_KEY = "your_fred_api_key_here"
   BEA_API_KEY = "your_bea_api_key_here"
   CENSUS_API_KEY = "your_census_api_key_here"
   BLS_API_KEY = "your_bls_api_key_here"
   ```
   (free keys: [FRED](https://fred.stlouisfed.org/docs/api/api_key.html), [BEA](https://apps.bea.gov/API/signup/), [Census](https://api.census.gov/data/key_signup.html)). Without `FRED_API_KEY`, FRED series won't load; without `BEA_API_KEY`, the BEA tab is unavailable; without `CENSUS_API_KEY`, the Census tab can browse but not fetch data; without `BLS_API_KEY`, BLS data is limited to 25 requests a day. NY Fed data needs no key.
5. Click **Deploy**. The app reads the keys from Streamlit's secrets automatically — `keyring`/Windows Credential Manager isn't available on Cloud, and the app already falls back past it.

To test the secrets path locally first: copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`, fill in your keys (that file is gitignored, so it never gets pushed), then `streamlit run app.py`.

## Troubleshooting

- **First image export is slow**: the "publication-style" export uses `kaleido`, which downloads a small headless Chromium the first time you generate an image. This is one-time and automatic.
- **`keyring` errors on Linux** (e.g. "No recommended backend was available"): install a backend such as `keyrings.alt` (`pip install keyrings.alt`), or skip `keyring` entirely and set the `FRED_API_KEY` environment variable instead.
- **"No FRED API key found"**: run `python setup_api_key.py --check` to confirm a key is stored, or verify `FRED_API_KEY` is set in your current shell session.

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app |
| `fred_data.py` | FRED data-fetching helpers (`FredDatasets`, recession shading, API key lookup) |
| `nyfed_data.py` | New York Fed Markets Data API client (charted `NYFED:` series, response-to-table flattening) |
| `bea_data.py` | BEA API helpers on top of `beaapi` (key lookup, parameter metadata, period parsing, series extraction) |
| `census_data.py` | Census Data API client (dataset catalog, variables, geography, queries, time parsing) |
| `bls_data.py` | BLS Public Data API v2 client (surveys, popular series, chunked data requests, period parsing) |
| `setup_api_key.py` | One-time CLI to store your FRED API key (or your BEA / Census / BLS key, with `--bea` / `--census` / `--bls`) |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | Light theme configuration |
| `run.bat` / `run.sh` | One-click local launcher: sets up everything and starts the app |
| `FRED_Streamlit_Colab.ipynb` | Zero-install portal: runs the same app inside Google Colab with a public link |
| `FRED_Data_Explorer.ipynb` | Lightweight standalone notebook version (no Streamlit) with example charts to copy/adapt |
