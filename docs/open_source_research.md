# Open-Source Football Model Research

This note captures practical feature ideas from open-source football prediction and betting projects. The goal is not to copy a project, but to choose features and model types that are realistic to build quickly.

## Sources Checked

- Octopy: https://github.com/octosport/octopy
- Football League Monte Carlo Simulation: https://github.com/vickyfriss/football-league-predictions
- Sports Prediction Bot: https://github.com/louisoddie999/sports-prediction-bot
- Penaltyblog: https://github.com/martineastwood/penaltyblog
- Statistical football prediction background: https://en.wikipedia.org/wiki/Statistical_association_football_predictions
- Elo + Poisson World Cup model paper: https://arxiv.org/abs/1806.01930

## Repeated Patterns

The useful repeated patterns are:

- Team strength ratings: Elo, Pi ratings, Massey ratings, Colley ratings.
- Goal models: Poisson, Dixon-Coles, bivariate Poisson.
- Recent form: last N matches, weighted toward recent games.
- Attack and defense strength: goals for and goals against, ideally adjusted by opponent strength.
- Betting odds: convert odds into implied probabilities and remove bookmaker margin.
- Calibration: Platt scaling or another probability calibration step.
- Simulation: Monte Carlo is useful for tournament/league outcomes, but not required for initial SportsPredict market submission.

## What We Will Use First

For a two-day build, we will use:

- Custom Elo rating as the main team-strength feature.
- Recent form from last 5 and last 10 matches.
- Goals for and goals against averages.
- Poisson expected goals for goal-related markets.
- Logistic regression for direct team-win probabilities if enough training data is available.
- Simple calibration and probability clipping to avoid reckless overconfidence.

## What We Will Defer

These are useful but too expensive for the first deployable version:

- Full Dixon-Coles model fitting.
- Player-level xG or event data.
- Real-time lineups and injury feeds.
- Monte Carlo tournament simulation.
- Deep learning.

## Feature Set For Version 1

Team-level features:

- `team_elo`
- `opponent_elo`
- `elo_diff`
- `team_recent_points_last_5`
- `opponent_recent_points_last_5`
- `team_goal_for_avg_last_5`
- `team_goal_against_avg_last_5`
- `opponent_goal_for_avg_last_5`
- `opponent_goal_against_avg_last_5`
- `team_win_rate_last_10`
- `opponent_win_rate_last_10`
- `neutral_match`

Market-level features and heuristics:

- Team win: Elo/logistic probability.
- Team score at least 1: Poisson from expected goals.
- Total goals thresholds: Poisson total-goals distribution.
- Both teams score: Poisson scoring probability for both teams.
- Shots, cards, corners, offsides, fouls: conservative baselines adjusted slightly by team strength when no reliable data exists.
- Player shot/scorer markets: conservative fallback unless player data is added.

