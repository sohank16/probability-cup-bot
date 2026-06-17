from __future__ import annotations

from src.fifa_rankings import normalize_team_name


TEAM_ALIASES = {
    "ALG": "Algeria",
    "ARG": "Argentina",
    "AUS": "Australia",
    "AUT": "Austria",
    "BEL": "Belgium",
    "BIH": "Bosnia and Herzegovina",
    "BRA": "Brazil",
    "CAN": "Canada",
    "CIV": "Ivory Coast",
    "COD": "DR Congo",
    "COL": "Colombia",
    "CPV": "Cape Verde",
    "CRO": "Croatia",
    "CZE": "Czech Republic",
    "ECU": "Ecuador",
    "EGY": "Egypt",
    "ENG": "England",
    "ESP": "Spain",
    "FRA": "France",
    "GER": "Germany",
    "GHA": "Ghana",
    "IRN": "Iran",
    "IRQ": "Iraq",
    "JOR": "Jordan",
    "JPN": "Japan",
    "KOR": "South Korea",
    "KSA": "Saudi Arabia",
    "MAR": "Morocco",
    "MEX": "Mexico",
    "NED": "Netherlands",
    "NOR": "Norway",
    "PAN": "Panama",
    "PAR": "Paraguay",
    "POR": "Portugal",
    "QAT": "Qatar",
    "RSA": "South Africa",
    "SCO": "Scotland",
    "SEN": "Senegal",
    "SUI": "Switzerland",
    "SWE": "Sweden",
    "TUN": "Tunisia",
    "TUR": "Turkey",
    "URU": "Uruguay",
    "USA": "United States",
    "UZB": "Uzbekistan",
}

NAME_ALIASES = {
    "curacao": "Curaçao",
    "czech republic": "Czech Republic",
    "czechia": "Czech Republic",
    "dr congo": "DR Congo",
    "haiti": "Haiti",
    "new zealand": "New Zealand",
    "turkiye": "Turkey",
    "united states": "United States",
}


def canonical_team_name(value: str) -> str:
    cleaned = value.strip()
    if cleaned in TEAM_ALIASES:
        return TEAM_ALIASES[cleaned]
    return NAME_ALIASES.get(normalize_team_name(cleaned), cleaned)


def split_match_name(match_name: str) -> tuple[str, str]:
    if " vs " not in match_name:
        raise ValueError(f"Could not split match name: {match_name}")
    left, right = match_name.split(" vs ", maxsplit=1)
    return canonical_team_name(left), canonical_team_name(right)


def opponent_for(team: str, home_team: str, away_team: str) -> str | None:
    normalized_team = normalize_team_name(canonical_team_name(team))
    if normalized_team == normalize_team_name(home_team):
        return away_team
    if normalized_team == normalize_team_name(away_team):
        return home_team
    return None
