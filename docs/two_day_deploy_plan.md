# Two-Day Deploy Plan

The new goal is to ship a working SportsPredict bot in two days, submit predictions, and use leaderboard/ranking feedback to decide whether to improve this model or switch strategies.

## Definition Of Done

The bot is deployable when it can:

- Refresh SportsPredict markets.
- Build team ratings and football features.
- Parse most market question types.
- Produce probabilities from 1 to 99.
- Write a dry-run report for review.
- Submit or update predictions in batches.
- Save every generated prediction locally.
- Re-run safely without duplicating work.

## Day 1: Model And Dry Run

Primary outcome: a dry-run prediction report for all open markets.

Steps:

1. Install new dependencies.

   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Add football data loader.

   Script/module targets:

   - `src/football_data.py`
   - `scripts/02_prepare_football_data.py`

   Minimum data needed:

   - Match date
   - Home team
   - Away team
   - Home goals
   - Away goals
   - Tournament or competition
   - Neutral flag if available

   Data window:

   - Use completed international matches from 2020 onward.
   - Exclude rows with `NA` scores.
   - Exclude future fixtures from training.
   - Weight old matches lightly and recent matches strongly.

3. Build custom Elo ratings.

   Script/module targets:

   - `src/elo.py`
   - `scripts/02_prepare_football_data.py`
   - `tests/test_elo_features.py`

   Elo implementation:

   - Start ranked teams from the current FIFA/Coca-Cola men's ranking points.
   - Use 1350 as the fallback prior for teams missing from FIFA rankings.
   - Process matches chronologically.
   - Calculate expected score.
   - Update after win/draw/loss.
   - Apply a small goal-difference multiplier.
   - Apply time decay:
     - 2020: 0.25x
     - 2021: 0.35x
     - 2022: 0.50x
     - 2023: 0.70x
     - 2024: 0.85x
     - 2025-2026: 1.00x
   - Apply competition priority:
     - 2022 FIFA World Cup: highest weight
     - Euros, Copa America, AFCON, Asian Cup, Gold Cup, OFC Nations Cup: second tier
     - World Cup qualifiers, other qualifiers, and friendlies: lower weight
   - Apply confederation hierarchy to lower-priority match weights:
     - UEFA and CONMEBOL
     - CAF
     - AFC
     - CONCACAF
     - OFC

4. Build form and goal features.

   Script/module targets:

   - `src/features.py`
   - `scripts/02_prepare_football_data.py`

   Version 1 features:

   - Elo difference
   - FIFA rank
   - FIFA ranking points
   - Confederation
   - Weighted points since 2020
   - Weighted goal difference since 2020
   - Last 5 points
   - Last 10 points
   - Last 10 win rate
   - Last 5 goals for
   - Last 5 goals against
   - Matches played since 2020
   - Recent match count since 2024
   - Neutral match flag
   - Tournament weight
   - Simple expected goals estimate

5. Build market parser.

   Script/module targets:

   - `src/question_parser.py`
   - `tests/test_question_parser.py`

   Supported first:

   - Team win
   - Team score
   - Total goals over/under
   - Both teams score plus total goals
   - Halftime tied/winning

   Conservative fallback:

   - Shots on target
   - Cards
   - Corners
   - Fouls
   - Offsides
   - Player shots/scorer markets

6. Generate dry-run predictions.

   Script/module targets:

   - `src/predictor.py`
   - `scripts/05_predict_markets.py`

   Output:

   - `reports/dry_run_predictions.csv`
   - `reports/dry_run_predictions.md`

## Day 2: Submission And Monitoring

Primary outcome: predictions submitted to SportsPredict and a repeatable deployment command.

Steps:

1. Review dry-run report.

   We check:

   - Probabilities are never outside 1 to 99.
   - Known strong teams have reasonable probabilities.
   - Fallback predictions are conservative.
   - No obvious team-name parsing failures.

2. Add prediction storage.

   Module target:

   - extend `src/storage.py`

   New tables:

   - `generated_predictions`
   - `submissions`

3. Add submission engine.

   Script/module targets:

   - `src/submitter.py`
   - `scripts/06_submit_predictions.py`

   Safety behavior:

   - Default mode is dry-run.
   - Real submission requires `--confirm`.
   - Batch size is max 50 predictions.
   - Existing predictions should be updated instead of duplicated where possible.

4. Submit predictions.

   Command:

   ```bash
   python scripts/06_submit_predictions.py --confirm
   ```

5. Add result/ranking checks.

   Script/module targets:

   - `src/evaluator.py`
   - `scripts/07_check_results.py`

   Metrics:

   - Brier score
   - Prediction count
   - Settled prediction count
   - Average confidence
   - Calibration buckets

6. Deploy/run loop.

   Commands:

   ```bash
   python scripts/01_fetch_sportspredict.py
   python scripts/05_predict_markets.py
   python scripts/06_submit_predictions.py --confirm
   python scripts/07_check_results.py
   ```

## Accuracy Upgrade Path After Ranking Check

If ranking is weak, improve in this order:

1. Blend in bookmaker implied probability.
2. Add real odds-margin removal.
3. Add Dixon-Coles or full Poisson model.
4. Add player/team stat feeds for shots/cards/corners.
5. Add calibration from our settled SportsPredict results.
6. Consider ML ensemble only after the classical baseline is measured.

## Deployment Meaning

For this competition, deployment means the bot submits predictions through the SportsPredict API. We do not need a web app first. A web dashboard can come later if we want nicer monitoring.
