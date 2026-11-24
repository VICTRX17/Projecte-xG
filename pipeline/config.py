"""Project-wide configuration: paths, data source URLs, selected competitions and pitch constants.

All other modules import their constants from here so that changing a path or
a competition only requires editing this file.
"""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

STATSBOMB_RAW_BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
REQUEST_TIMEOUT_SECONDS = 30

# StatsBomb pitch coordinates: 120 x 80, always attacking left to right.
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
GOAL_X = 120.0
GOAL_CENTER_Y = 40.0
GOAL_POST_LOW_Y = 36.0
GOAL_POST_HIGH_Y = 44.0


@dataclass(frozen=True)
class CompetitionSeason:
    """A (competition, season) pair from the StatsBomb open-data catalogue.

    Attributes:
        competition_id: StatsBomb competition identifier.
        season_id: StatsBomb season identifier within the competition.
        label: Human-readable name used in reports.
    """

    competition_id: int
    season_id: int
    label: str


# IDs are resolved against competitions.json at runtime; the pipeline fails
# explicitly if a pair is not present in the catalogue.
SELECTED_COMPETITIONS = (
    CompetitionSeason(competition_id=11, season_id=90, label="La Liga 2020/21"),
    CompetitionSeason(competition_id=43, season_id=3, label="FIFA World Cup 2018"),
    CompetitionSeason(competition_id=43, season_id=106, label="FIFA World Cup 2022"),
)
