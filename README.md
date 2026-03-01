# Bazaar Price Tracker

Bazaar Price Tracker powers a Minecraft Hypixel SkyBlock "Bazaar mod" by collecting bazaar history, training per-item ML models, and serving real-time entry recommendations via a Flask API.

---

## Quick Start

1. **Create & activate environment**:
   - `python -m venv .venv && source .venv/bin/activate`
2. **Install dependencies**:
   - `pip install -r requirements.txt`
3. **Train models** (per item, fetches data from the Coflnet API):
   - `python Backend/LGBMfulldata.py`
4. **Start the API server**:
   - `python Backend/flask_api.py`
5. Point your **Bazaar mod** or open `Frontend/website.html` to view predictions (default API: `http://localhost:5001`).


## Project Structure

```
Backend/
  flask_api.py              # Flask API server
  LGBMfulldata.py           # Model training, prediction, and analysis
  bazaar_full_items_ids.json # List of item IDs to train on
Frontend/
  website.html              # Web dashboard for viewing predictions
Utils/
  data_utils.py             # Coflnet API data fetching, caching, rate-limiting, proxy rotation
  mayor_utils.py            # SkyBlock mayor perk extraction (40 binary features)
  event_utils.py            # SkyBlock event timing features (festivals, dark auction, etc.)
  load_proxies.py           # Proxy pool loader for faster parallel data fetching
Testing Files/
  Find_Highest_Demand.py    # Ranks items by volume × median price to find high-value items
  refine.py                 # Filters items to those with ≥200k data entries
  find_label_clip_values.py # Computes label distribution percentiles per item
Model_Files/                # Trained model artifacts (model, scaler, features per item)
bazaar_data/                # Cached raw bazaar history (gzip-compressed pickle)
csv files/                  # Intermediate CSV files with computed features and targets
```


## Architecture

**Data pipeline** (`Utils/data_utils.py`)
- Fetches bazaar price history from the [Coflnet API](https://sky.coflnet.com/api) in chunked time intervals (~23 hours each).
- Supports both synchronous and async (`aiohttp`) fetching modes, with optional proxy rotation for parallelism.
- Implements rate limiting (30 requests per 10-second window) and binary search for the oldest available data.
- Caches fetched data locally as gzip-compressed pickle files in `bazaar_data/` and supports incremental updates.

**Feature engineering** (`Backend/LGBMfulldata.py`, `Utils/event_utils.py`, `Utils/mayor_utils.py`)
- Cyclical time encodings (hour-of-day, day-of-week as sin/cos).
- Lagged price returns, volume, rolling mean/std/momentum over multiple windows.
- Spread-based features: spread percentage, spread volatility, spread momentum.
- SkyBlock-specific features: proximity to Season of Jerry, Jerry Festival, Spooky Festival, Dark Auction, and Jacob's Contest.
- Mayor perk features: 40 binary flags representing the current mayor's active perks.

**Target construction** (`build_entry_targets`)
- For each data point, looks forward over a 3-hour horizon and computes: expected return (median), profit probability, max profit/loss, risk-reward ratio, win rates at 1%/2% thresholds, MAE/MFE, and time-to-first-up/down.
- Applies a 10-minute minimum delay to exclude near-term noise.
- The primary label (`entry_label`) is the median forward return after bazaar tax (1.25%).

**Model training** (`train_model_system`)
- Trains one LightGBM regression model per item using `StandardScaler`-normalized features.
- Hyperparameter tuning via Optuna (30 trials), optimizing a sign-penalty-weighted RMSE that penalizes predictions with the wrong direction more heavily.
- Saves three artifacts per item to `Model_Files/`: `_entry_model.pkl`, `_entry_scaler.pkl`, `_entry_features.pkl`.
- `test_train_model_system` runs the same pipeline with an 80/20 time-based split and logs RMSE, MAE, R², sign accuracy, and percent error stats.

**Prediction** (`predict_entries`, `analyze_entries`)
- Fetches the most recent 3 hours of live data from the Coflnet API for the item.
- Generates predictions at 5-minute intervals over the next 3 hours by reusing the latest feature row with updated timestamps.
- `analyze_entries` filters to positive-score future entries and sorts by time-to-entry, then by score.

**API server** (`Backend/flask_api.py`)
- Flask + CORS server on port 5001.
- On startup, loads all saved model artifacts and current mayor data.
- A background daemon thread re-runs predictions for all items every ~10 seconds and caches results to `predictions_cache.json`.
- Returns 503 for non-health endpoints until models are loaded.

**Frontend** (`Frontend/website.html`)
- Single-page web dashboard with a retro terminal aesthetic.
- Polls the `/predictions` endpoint every 30 seconds and displays a ranked table of entry opportunities.
- Supports filtering by minimum score, sorting by time or score, and pagination.


## API Endpoints

All endpoints return JSON. Served by `Backend/flask_api.py` on port 5001.

- `GET /` – Service metadata and available endpoint listing.
- `GET /health` – Liveness probe with `model_trained` flag and timestamp.
- `GET /items` – Full list of bazaar item IDs (fetched live from Coflnet).
- `GET /predictions?limit=100&min_score=0.0` – Cached predictions: best upcoming positive entry per item, ranked by time-to-entry then score.
- `GET /predict/<item_id>` – Fresh on-demand prediction for a specific item (not cached).
- `GET /investments?limit=10` – Aggregated investment ideas from cached predictions.


## In-Web Preview

![Website In Action](/Test_Image.png)

## Item Artifacts Link
- Use this link to access data scraped from the start of the Bazaar until about November 2025 (data to be updated later) -> https://huggingface.co/datasets/1amuel/Bazaar-ML-Model


## Development Notes

- Prefer running the API behind a reverse proxy if you expose it outside localhost.
- `Model_Files/`, `bazaar_data/`, and `csv files/` are gitignored — they contain large generated artifacts.
- When changing the feature set or model behavior in `LGBMfulldata.py`, regenerate all artifacts before restarting the API.
- To speed up bulk data fetching, configure a proxy pool via `Utils/load_proxies.py` with a `proxies.txt` file (supports `IP:PORT:USER:PASS` format).
- `Testing Files/` scripts are one-off utilities for data exploration — run them from the project root (e.g. `python -m Testing\ Files.Find_Highest_Demand`).

