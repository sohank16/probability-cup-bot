from __future__ import annotations

from dataclasses import dataclass

from src.fifa_rankings import FifaRanking, lookup_ranking
from src.football_data import FootballMatch, combined_match_weight, tournament_weight
from src.elo import BASE_RATING


@dataclass(frozen=True)
class TeamFeatures:
    team: str
    team_elo: float
    fifa_rank: int | None
    fifa_points: float | None
    confederation: str
    weighted_points_since_2020: float
    weighted_goal_difference_since_2020: float
    last_5_points: int
    last_10_points: int
    last_5_goals_for: float
    last_5_goals_against: float
    last_10_win_rate: float
    matches_played_since_2020: int
    recent_match_count_since_2024: int


@dataclass(frozen=True)
class MatchupFeatures:
    team: str
    opponent: str
    team_elo: float
    opponent_elo: float
    elo_diff: float
    weighted_points_since_2020: float
    weighted_goal_difference_since_2020: float
    last_5_points: int
    last_10_points: int
    last_5_goals_for: float
    last_5_goals_against: float
    last_10_win_rate: float
    matches_played_since_2020: int
    recent_match_count_since_2024: int
    neutral_match: bool
    tournament_weight: float


@dataclass(frozen=True)
class TeamMatchResult:
    match: FootballMatch
    team: str
    opponent: str
    goals_for: int
    goals_against: int
    points: int
    won: bool
    weight: float


def points_for(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def expand_team_results(matches: list[FootballMatch]) -> dict[str, list[TeamMatchResult]]:
    return expand_team_results_with_weights(matches)


def expand_team_results_with_weights(
    matches: list[FootballMatch],
    team_confederations: dict[str, str] | None = None,
) -> dict[str, list[TeamMatchResult]]:
    results: dict[str, list[TeamMatchResult]] = {}

    for match in sorted(matches, key=lambda item: item.match_date):
        home_points = points_for(match.home_score, match.away_score)
        away_points = points_for(match.away_score, match.home_score)
        weight = combined_match_weight(match, team_confederations)

        team_rows = [
            TeamMatchResult(
                match=match,
                team=match.home_team,
                opponent=match.away_team,
                goals_for=match.home_score,
                goals_against=match.away_score,
                points=home_points,
                won=home_points == 3,
                weight=weight,
            ),
            TeamMatchResult(
                match=match,
                team=match.away_team,
                opponent=match.home_team,
                goals_for=match.away_score,
                goals_against=match.home_score,
                points=away_points,
                won=away_points == 3,
                weight=weight,
            ),
        ]

        for row in team_rows:
            results.setdefault(row.team, []).append(row)

    return results


def average(values: list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def build_team_features(
    matches: list[FootballMatch],
    ratings: dict[str, float],
    rankings_by_team: dict[str, FifaRanking] | None = None,
    team_confederations: dict[str, str] | None = None,
) -> dict[str, TeamFeatures]:
    results_by_team = expand_team_results_with_weights(matches, team_confederations)
    features: dict[str, TeamFeatures] = {}

    for team, results in results_by_team.items():
        ranking = lookup_ranking(rankings_by_team or {}, team)
        last_5 = results[-5:]
        last_10 = results[-10:]
        weighted_points = sum(result.points * result.weight for result in results)
        weighted_goal_difference = sum(
            (result.goals_for - result.goals_against) * result.weight
            for result in results
        )
        recent_count = sum(1 for result in results if result.match.match_date.year >= 2024)

        features[team] = TeamFeatures(
            team=team,
            team_elo=ratings.get(team, BASE_RATING),
            fifa_rank=ranking.rank if ranking else None,
            fifa_points=ranking.points if ranking else None,
            confederation=ranking.confederation if ranking else "",
            weighted_points_since_2020=weighted_points,
            weighted_goal_difference_since_2020=weighted_goal_difference,
            last_5_points=sum(result.points for result in last_5),
            last_10_points=sum(result.points for result in last_10),
            last_5_goals_for=average([result.goals_for for result in last_5]),
            last_5_goals_against=average([result.goals_against for result in last_5]),
            last_10_win_rate=(
                sum(1 for result in last_10 if result.won) / len(last_10)
                if last_10
                else 0.0
            ),
            matches_played_since_2020=len(results),
            recent_match_count_since_2024=recent_count,
        )

    return features


def build_matchup_features(
    team: str,
    opponent: str,
    team_features: dict[str, TeamFeatures],
    neutral_match: bool,
    tournament: str,
) -> MatchupFeatures:
    team_row = team_features.get(team) or TeamFeatures(
        team=team,
        team_elo=BASE_RATING,
        fifa_rank=None,
        fifa_points=None,
        confederation="",
        weighted_points_since_2020=0.0,
        weighted_goal_difference_since_2020=0.0,
        last_5_points=0,
        last_10_points=0,
        last_5_goals_for=0.0,
        last_5_goals_against=0.0,
        last_10_win_rate=0.0,
        matches_played_since_2020=0,
        recent_match_count_since_2024=0,
    )
    opponent_row = team_features.get(opponent) or TeamFeatures(
        team=opponent,
        team_elo=BASE_RATING,
        fifa_rank=None,
        fifa_points=None,
        confederation="",
        weighted_points_since_2020=0.0,
        weighted_goal_difference_since_2020=0.0,
        last_5_points=0,
        last_10_points=0,
        last_5_goals_for=0.0,
        last_5_goals_against=0.0,
        last_10_win_rate=0.0,
        matches_played_since_2020=0,
        recent_match_count_since_2024=0,
    )

    return MatchupFeatures(
        team=team,
        opponent=opponent,
        team_elo=team_row.team_elo,
        opponent_elo=opponent_row.team_elo,
        elo_diff=team_row.team_elo - opponent_row.team_elo,
        weighted_points_since_2020=team_row.weighted_points_since_2020,
        weighted_goal_difference_since_2020=team_row.weighted_goal_difference_since_2020,
        last_5_points=team_row.last_5_points,
        last_10_points=team_row.last_10_points,
        last_5_goals_for=team_row.last_5_goals_for,
        last_5_goals_against=team_row.last_5_goals_against,
        last_10_win_rate=team_row.last_10_win_rate,
        matches_played_since_2020=team_row.matches_played_since_2020,
        recent_match_count_since_2024=team_row.recent_match_count_since_2024,
        neutral_match=neutral_match,
        tournament_weight=tournament_weight(tournament),
    )
