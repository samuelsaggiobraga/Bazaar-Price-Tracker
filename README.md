# Bazaar Price Tracker

Bazaar Price Tracker powers a Minecraft Hypixel SkyBlock "Bazaar mod" by collecting bazaar history, training per-item ML models, and serving real-time entry recommendations via a Flask API.

---

## Quick Start

1. **Create & activate environment** (example):
   - `python -m venv .venv && source .venv/bin/activate`
2. **Install dependencies**:
   - `pip install -r requirements.txt`
3. **Start the API server**:
   - `python flask_api.py`
4. Point your **Bazaar mod** to the Flask API base URL (default: `http://localhost:5001`).


## Architecture

**High-level flow**
- `data_utils.py`: Fetches and aggregates bazaar history from the Coflnet API, with rate-limiting, optional proxy rotation, and async chunked downloads.
- `LGBMfulldata.py`: Trains LightGBM-based models per item and exposes `predict_entries` / `analyze_entries` to rank profitable buy/sell entries.
- `mayor_utils.py`: Pulls current mayor perks and feeds them into the feature pipeline.
- `flask_api.py`: Flask + CORS API that:
  - Loads models & feature metadata from `Model_Files`.
  - Runs a background loop that refreshes predictions and caches them to disk.
  - Exposes JSON endpoints for health checks, listing items, fetching predictions, and investment ideas.
- `Find_Highest_Demand.py`, `refine.py`: Utility scripts for deeper analytics, experimentation, and data preparation.

**Runtime components**
- **Model artifacts** live in `Model_Files` (one model/scaler/feature file per item).
- **Background worker** periodically recomputes predictions and stores them in `predictions_cache.json`.
- **Mod client** hits the Flask API to render recommended bazaar entries in-game.


## API Overview

Key endpoints exposed by `flask_api.py` (exact paths may evolve):
- `GET /` – Basic service metadata and readiness flag.
- `GET /health` – Liveness/readiness probe.
- `GET /items` – Items with trained entry models.
- `GET /predictions` – Latest cached predictions for all items.
- `GET /predict/<item_id>` – Fresh prediction and ranked entries for a specific item.
- `GET /investments` – Aggregated investment ideas based on current predictions.


## In-Game Preview

![Bazaar Mod in action](Screenshot%202026-01-27%20at%205.40.01%E2%80%AFPM.png)

## Item Artifacts Link
- Use this link to access data scraped from the start of the Bazaar until about November 2025 (data to be updated later) -> https://huggingface.co/datasets/1amuel/Bazaar-ML-Model


## Development Notes

- Prefer running the API behind a reverse proxy if you expose it outside localhost.
- Keep `Model_Files` out of version control unless you specifically want to ship trained artifacts.
- When changing the feature set or model behavior in `LGBMfulldata.py`, regenerate artifacts before restarting the API.

