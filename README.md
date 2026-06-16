# Probability Cup Bot

An explainable football forecasting bot for the SportsPredict Probability Cup.

## Day 1 Scope

Day 1 builds the foundation:

- Load settings from environment variables.
- Keep API keys out of source code.
- Fetch Probability Cup events, lobbies, matches, and markets.
- Store raw API responses for debugging.
- Store structured records in SQLite for later modeling.

No model predictions or submissions happen on Day 1.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then edit `.env` and set `SPORTSPREDICT_API_KEY`.

## Run Day 1 Fetch

```bash
python scripts/01_fetch_sportspredict.py
```

The script fetches data from the SportsPredict API, saves raw JSON snapshots in `data/raw/`, and writes structured rows into SQLite at `data/probability_cup.sqlite`.

## Why The Project Is Structured This Way

`src/` contains reusable code that the rest of the project imports.

`scripts/` contains terminal entry points. Scripts should stay small and call reusable code from `src/`.

`data/raw/` keeps exact API responses so we can later explain what information the model saw.

`data/processed/` will hold cleaned football data from Day 2 onward.
