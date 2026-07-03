# Probability Cup Bot

An explainable football forecasting bot for the SportsPredict Probability Cup.

## Current Goal

We are now working on a two-day deployment path:

- Day 1: build the football model and produce a dry-run prediction report.
- Day 2: submit predictions through the SportsPredict API and monitor results.

See `docs/two_day_deploy_plan.md` for the current execution plan.

## Day 1 Scope

Day 1 builds the foundation:

- Load settings from environment variables.
- Keep API keys out of source code.
- Fetch Probability Cup events, lobbies, matches, and markets.
- Store raw API responses for debugging.
- Store structured records in SQLite for later modeling.
- Prepare recent international football results.
- Build time-weighted Elo ratings and team features.

No live prediction submissions happen on Day 1.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then edit `.env` and set `SPORTSPREDICT_API_KEY`. If you want automatic bookmaker odds,
also set `ODDS_API_KEY` from The Odds API.

## Run Day 1 Fetch

```bash
python scripts/01_fetch_sportspredict.py
```

The script fetches data from the SportsPredict API, saves raw JSON snapshots in `data/raw/`, and writes structured rows into SQLite at `data/probability_cup.sqlite`.

## Prepare Football Data

```bash
python scripts/02_prepare_football_data.py
```

This downloads the current international results CSV, keeps completed matches from 2020 onward, excludes future `NA` score rows, and builds time-weighted Elo ratings.

The model starts teams from the official FIFA/Coca-Cola men's ranking published on `2026-06-11`, then updates ratings with recent match results.

The model weights older football lightly:

- 2020: `0.25x`
- 2021: `0.35x`
- 2022: `0.50x`
- 2023: `0.70x`
- 2024: `0.85x`
- 2025-2026: `1.00x`

Competition weights are also different:

- 2022 FIFA World Cup results get the most weight.
- Major continental tournaments such as Euros, Copa America, AFCON, Asian Cup, Gold Cup, and OFC Nations Cup are second priority.
- World Cup qualifiers, other qualifiers, and friendlies are lower priority.
- Lower-priority matches are adjusted by confederation strength: UEFA/CONMEBOL, then CAF, AFC, CONCACAF, and OFC.

Outputs are written to:

- `data/processed/recent_international_matches.csv`
- `data/processed/team_elo_ratings.csv`
- `data/processed/team_features.csv`
- `reports/elo_summary.md`

## Analyze SportsPredict Markets

```bash
python scripts/03_analyze_markets.py
```

This reads the stored SportsPredict markets from SQLite, parses every market question, and prints:

- parser coverage
- market type counts
- metric counts
- real examples for each market type

The parser lives in `src/question_parser.py`.

## Generate Dry-Run Predictions

First train the logistic regression model:

```bash
python scripts/04_train_ml_models.py
```

This trains logistic regression models for market types where historical match results provide real labels:

- home win
- away win
- over 2.5 goals
- both teams score
- home scores at least 1
- away scores at least 1

The trained bundle is written to `models/goal_market_logistic.joblib`.

Then generate predictions:

```bash
python scripts/04_predict_markets.py
```

This combines parsed markets, team features, Elo ratings, logistic regression, Poisson goal probabilities, and market-specific priors to produce predictions for every current market.

Outputs are written to:

- `reports/dry_run_predictions.csv`
- `reports/dry_run_predictions.md`

Winner and compatible goal markets use logistic regression when the trained model is available. Other goal markets use Poisson. Cards, corners, fouls, offsides, penalties, and player markets use conservative baseline priors with small team-strength adjustments.

## Add Odds And Player Form

For automatic match-winner and total-goals odds, use The Odds API:

```bash
python scripts/09_fetch_odds_api_baselines.py --list-sports
python scripts/09_fetch_odds_api_baselines.py --sport-key soccer_fifa_world_cup --markets h2h,totals
```

This writes de-vigged bookmaker baselines to `data/processed/odds_market_baselines.csv`.

For special markets such as corners, shots on target, cards, fouls, offsides, and player props,
fill bookmaker over/under odds manually in `data/external/bookmaker_prop_odds.csv`:

```bash
python scripts/08_prepare_bookmaker_prop_baselines.py --template
python scripts/08_prepare_bookmaker_prop_baselines.py
```

For player markets, generate the player form sheet and fill:

- `club_goals_2025_26`: goals for the player's club in the 2025/26 season
- `country_starts_last_10`: starts in the player's last 10 international matches

```bash
python scripts/07_prepare_player_form_template.py
```

## Why The Project Is Structured This Way

`src/` contains reusable code that the rest of the project imports.

`scripts/` contains terminal entry points. Scripts should stay small and call reusable code from `src/`.

`data/raw/` keeps exact API responses so we can later explain what information the model saw.

`data/processed/` will hold cleaned football data from Day 2 onward.

## Contributors

Sohan Kadam  
Aditya Kukreti  
Gopal Moranya



