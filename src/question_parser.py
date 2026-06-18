from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ParsedMarket:
    market_type: str
    question: str
    team: str | None = None
    opponent: str | None = None
    player: str | None = None
    metric: str | None = None
    threshold: int | None = None
    period: str = "match"
    comparison: str | None = None
    extra: dict[str, str | int | bool] = field(default_factory=dict)

    @property
    def is_supported(self) -> bool:
        return self.market_type != "unknown"


def normalize_question(question: str) -> str:
    return " ".join(question.strip().rstrip("?").split())


def clean_entity(value: str) -> str:
    return value.strip(" ,.")


def period_from_text(text: str) -> str:
    lower = text.lower()
    if "at halftime" in lower or "at half-time" in lower:
        return "halftime"
    if "first half" in lower:
        return "first_half"
    if "second half" in lower:
        return "second_half"
    return "match"


def metric_from_text(text: str) -> str:
    lower = text.lower()
    if "shot on target" in lower or "shots on target" in lower:
        return "shots_on_target"
    if "corner kick" in lower or "corner kicks" in lower:
        return "corners"
    if "card" in lower or "cards" in lower:
        return "cards"
    if "foul" in lower or "fouls" in lower:
        return "fouls"
    if "offside" in lower:
        return "offsides"
    if "goal" in lower or "goals" in lower:
        return "goals"
    if "penalty" in lower:
        return "penalty"
    return "unknown_metric"


def parse_market_question(question: str) -> ParsedMarket:
    text = normalize_question(question)
    period = period_from_text(text)

    parser_rules = [
        parse_penalty_or_red_card,
        parse_both_teams_score_and_total_goals,
        parse_halftime_state,
        parse_match_winner,
        parse_player_goal_or_assist,
        parse_player_goal,
        parse_both_teams_metric_at_least,
        parse_player_shot_on_target,
        parse_team_first_goal_combo,
        parse_team_first_goal_period,
        parse_team_score_at_least,
        parse_team_score_period,
        parse_team_total_goals,
        parse_total_goals,
        parse_total_metric_threshold,
        parse_team_metric_threshold,
        parse_team_metric_comparison,
    ]

    for parser in parser_rules:
        parsed = parser(text, period)
        if parsed is not None:
            return parsed

    return ParsedMarket(market_type="unknown", question=text, period=period)


def parse_penalty_or_red_card(text: str, period: str) -> ParsedMarket | None:
    if re.fullmatch(r"Will a penalty kick be awarded OR a red card be shown(?: in the match)?", text):
        return ParsedMarket(
            market_type="penalty_or_red_card",
            question=text,
            metric="penalty_or_red_card",
            period=period,
        )
    if re.fullmatch(r"Will a penalty kick be awarded(?: in the match)?", text):
        return ParsedMarket(
            market_type="penalty_awarded",
            question=text,
            metric="penalty",
            period=period,
        )
    return None


def parse_both_teams_score_and_total_goals(text: str, period: str) -> ParsedMarket | None:
    match = re.fullmatch(
        r"Will both teams score AND the match have (?P<threshold>\d+) or more total goals",
        text,
    )
    if not match:
        return None
    return ParsedMarket(
        market_type="both_teams_score_and_total_goals",
        question=text,
        metric="goals",
        threshold=int(match.group("threshold")),
        period=period,
        comparison="over_or_equal",
        extra={"both_teams_score": True},
    )


def parse_halftime_state(text: str, period: str) -> ParsedMarket | None:
    tied = re.fullmatch(r"At halftime, will the match be tied", text)
    if tied:
        return ParsedMarket(
            market_type="halftime_tied",
            question=text,
            metric="goals",
            period="halftime",
        )

    winning = re.fullmatch(r"At halftime, will (?P<team>.+?) be winning", text)
    if winning:
        return ParsedMarket(
            market_type="halftime_team_winning",
            question=text,
            team=clean_entity(winning.group("team")),
            metric="goals",
            period="halftime",
        )
    return None


def parse_match_winner(text: str, period: str) -> ParsedMarket | None:
    match = re.fullmatch(r"Will (?P<team>.+?) win the match", text)
    if not match:
        return None
    return ParsedMarket(
        market_type="match_winner",
        question=text,
        team=clean_entity(match.group("team")),
        metric="goals",
        period=period,
    )


def parse_player_goal_or_assist(text: str, period: str) -> ParsedMarket | None:
    match = re.fullmatch(
        r"Will (?P<player>.+?) score or assist a goal \(excluding own goals\)",
        text,
    )
    if not match:
        return None
    return ParsedMarket(
        market_type="player_goal_or_assist",
        question=text,
        player=clean_entity(match.group("player")),
        metric="goal_or_assist",
        period=period,
    )


def parse_player_goal(text: str, period: str) -> ParsedMarket | None:
    match = re.fullmatch(
        r"Will (?P<player>.+?) score a goal \(excluding own goals\)",
        text,
    )
    if not match:
        return None
    return ParsedMarket(
        market_type="player_goal",
        question=text,
        player=clean_entity(match.group("player")),
        metric="goals",
        threshold=1,
        period=period,
        comparison="over_or_equal",
    )


def parse_player_shot_on_target(text: str, period: str) -> ParsedMarket | None:
    match = re.fullmatch(
        r"Will (?P<player>.+?) have at least (?P<threshold>\d+) shot(?:s)? on target(?: in the second half)?",
        text,
    )
    if not match:
        return None
    return ParsedMarket(
        market_type="player_shot_on_target",
        question=text,
        player=clean_entity(match.group("player")),
        metric="shots_on_target",
        threshold=int(match.group("threshold")),
        period=period,
        comparison="over_or_equal",
    )


def parse_team_first_goal_combo(text: str, period: str) -> ParsedMarket | None:
    match = re.fullmatch(
        r"Will (?P<team>.+?) score the first goal of the game and (?P<opponent>.+?) score in the second half",
        text,
    )
    if not match:
        return None
    return ParsedMarket(
        market_type="team_first_goal_and_opponent_second_half_score",
        question=text,
        team=clean_entity(match.group("team")),
        opponent=clean_entity(match.group("opponent")),
        metric="goals",
        period="match_and_second_half",
    )


def parse_team_first_goal_period(text: str, period: str) -> ParsedMarket | None:
    match = re.fullmatch(r"Will (?P<team>.+?) score the first goal of the second half", text)
    if not match:
        return None
    return ParsedMarket(
        market_type="team_first_goal_of_second_half",
        question=text,
        team=clean_entity(match.group("team")),
        metric="goals",
        period="second_half",
    )


def parse_team_score_at_least(text: str, period: str) -> ParsedMarket | None:
    match = re.fullmatch(r"Will (?P<team>.+?) score at least (?P<threshold>\d+) goal(?:s)?", text)
    if not match:
        return None
    return ParsedMarket(
        market_type="team_score_at_least",
        question=text,
        team=clean_entity(match.group("team")),
        metric="goals",
        threshold=int(match.group("threshold")),
        period=period,
        comparison="over_or_equal",
    )


def parse_team_score_period(text: str, period: str) -> ParsedMarket | None:
    match = re.fullmatch(r"Will (?P<team>.+?) score in the (?P<period_text>first|second) half", text)
    if not match:
        return None
    return ParsedMarket(
        market_type="team_score_in_period",
        question=text,
        team=clean_entity(match.group("team")),
        metric="goals",
        threshold=1,
        period=f"{match.group('period_text')}_half",
        comparison="over_or_equal",
    )


def parse_team_total_goals(text: str, period: str) -> ParsedMarket | None:
    match = re.fullmatch(r"Will (?P<team>.+?) score (?P<threshold>\d+) or more total goals", text)
    if not match:
        return None
    return ParsedMarket(
        market_type="team_total_goals_over",
        question=text,
        team=clean_entity(match.group("team")),
        metric="goals",
        threshold=int(match.group("threshold")),
        period=period,
        comparison="over_or_equal",
    )


def parse_total_goals(text: str, period: str) -> ParsedMarket | None:
    over_match = re.fullmatch(
        r"Will (?:the match|the second half) have (?P<threshold>\d+) or more (?:total )?goals",
        text,
    )
    if over_match:
        return ParsedMarket(
            market_type="total_goals_over",
            question=text,
            metric="goals",
            threshold=int(over_match.group("threshold")),
            period=period,
            comparison="over_or_equal",
        )

    under_match = re.fullmatch(r"Will the match have (?P<threshold>\d+) or fewer total goals", text)
    if under_match:
        return ParsedMarket(
            market_type="total_goals_under",
            question=text,
            metric="goals",
            threshold=int(under_match.group("threshold")),
            period=period,
            comparison="under_or_equal",
        )

    half_more = re.fullmatch(r"Will the second half have more (?:total )?goals than the first half", text)
    if half_more:
        return ParsedMarket(
            market_type="second_half_more_goals_than_first",
            question=text,
            metric="goals",
            period="second_half_vs_first_half",
            comparison="greater_than",
        )
    return None


def parse_both_teams_metric_at_least(text: str, period: str) -> ParsedMarket | None:
    match = re.fullmatch(
        r"(?:Will|At halftime, will) both teams have at least (?P<threshold>\d+) (?P<metric_text>shot(?:s)? on target)(?: in the second half)?",
        text,
    )
    if not match:
        return None
    return ParsedMarket(
        market_type="both_teams_metric_at_least",
        question=text,
        metric=metric_from_text(match.group("metric_text")),
        threshold=int(match.group("threshold")),
        period=period,
        comparison="over_or_equal",
    )


def parse_total_metric_threshold(text: str, period: str) -> ParsedMarket | None:
    match = re.fullmatch(
        r"Will there be (?P<threshold>\d+) or more total (?P<metric_text>cards shown|shots on target|corner kicks)(?: in the match| in the second half)?",
        text,
    )
    if not match:
        return None
    return ParsedMarket(
        market_type="total_metric_over",
        question=text,
        metric=metric_from_text(match.group("metric_text")),
        threshold=int(match.group("threshold")),
        period=period,
        comparison="over_or_equal",
    )


def parse_team_metric_threshold(text: str, period: str) -> ParsedMarket | None:
    match = re.fullmatch(
        r"Will (?P<team>.+?) (?P<verb>have|receive|be caught offside) (?:(?:at least )?(?P<threshold_a>\d+)|(?P<threshold_b>\d+) or more) (?P<metric_text>shots on target|shot on target|corner kicks|corner kick|cards?|card|times)(?: in the (?:first|second) half)?",
        text,
    )
    if not match:
        return None

    metric_text = match.group("metric_text")
    verb = match.group("verb")
    if verb == "be caught offside":
        metric = "offsides"
    else:
        metric = metric_from_text(metric_text)

    return ParsedMarket(
        market_type="team_metric_over",
        question=text,
        team=clean_entity(match.group("team")),
        metric=metric,
        threshold=int(match.group("threshold_a") or match.group("threshold_b")),
        period=period,
        comparison="over_or_equal",
    )


def parse_team_metric_comparison(text: str, period: str) -> ParsedMarket | None:
    patterns = [
        r"(?:Will|In the second half, will|At halftime, will) (?P<team>.+?) have more (?P<metric_text>shots on target|corner kicks) than (?P<opponent>.+?)(?: in the second half)?",
        r"Will (?P<team>.+?) commit more (?P<metric_text>fouls) than (?P<opponent>.+?)",
        r"Will (?P<team>.+?) receive more (?P<metric_text>cards) than (?P<opponent>.+?)",
        r"Will (?P<team>.+?) finish with more (?P<metric_text>corner kicks) than (?P<opponent>.+?)",
        r"Will (?P<team>.+?) score more (?P<metric_text>goals) than (?P<opponent>.+?) in the second half",
    ]
    for pattern in patterns:
        match = re.fullmatch(pattern, text)
        if match:
            return ParsedMarket(
                market_type="team_metric_more_than_opponent",
                question=text,
                team=clean_entity(match.group("team")),
                opponent=clean_entity(match.group("opponent")),
                metric=metric_from_text(match.group("metric_text")),
                period=period,
                comparison="greater_than",
            )
    return None
