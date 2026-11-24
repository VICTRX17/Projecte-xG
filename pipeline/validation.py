"""Validation of raw and processed data.

Every check raises ``ValidationError`` with a precise message on failure, so
problems surface immediately instead of producing a silently wrong dataset.
"""

from typing import Any, Dict, List

import pandas as pd

from xg_pipeline.config import GOAL_X, PITCH_LENGTH, PITCH_WIDTH
from xg_pipeline.features import FEATURE_COLUMNS, NON_FEATURE_COLUMNS, TARGET_COLUMN

REQUIRED_COLUMNS = (
    "shot_id",
    "match_id",
    "competition",
    "season",
    "x",
    "y",
    "outcome",
    "body_part",
    "shot_type",
    *FEATURE_COLUMNS,
    TARGET_COLUMN,
)


class ValidationError(ValueError):
    """Raised when a dataset violates a required rule."""


def validate_raw_events(events: Any, source: str) -> None:
    """Check that a raw event file has the expected top-level structure.

    Args:
        events: Parsed JSON content.
        source: Description of the source file, used in error messages.

    Raises:
        ValidationError: If the content is not a list of event dictionaries.
    """
    if not isinstance(events, list):
        raise ValidationError(f"{source}: expected a list of events, got {type(events).__name__}")
    if len(events) == 0:
        raise ValidationError(f"{source}: event list is empty")
    for index, event in enumerate(events):
        if not isinstance(event, dict) or "type" not in event:
            raise ValidationError(f"{source}: event at index {index} has no 'type' field")


def validate_processed(df: pd.DataFrame) -> Dict[str, Any]:
    """Run integrity checks on the final shot dataset.

    Args:
        df: Processed shot DataFrame.

    Returns:
        A dictionary of summary statistics about the dataset.

    Raises:
        ValidationError: If a required column is missing, shot IDs are duplicated,
            the target is not binary, or geometry values are outside their domain.
    """
    if df.empty:
        raise ValidationError("Processed dataset is empty")

    missing_columns: List[str] = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValidationError(f"Missing required columns: {missing_columns}")

    if df["shot_id"].duplicated().any():
        raise ValidationError("Duplicate shot_id values found")

    target_values = set(df[TARGET_COLUMN].unique())
    if not target_values.issubset({0, 1}):
        raise ValidationError(f"Target must be binary, found values {sorted(target_values)}")

    if (df["distance"] < 0).any():
        raise ValidationError("Negative distance values found")

    if ((df["angle_rad"] < 0) | (df["angle_rad"] > 3.1416)).any():
        raise ValidationError("Angle outside [0, pi] found")

    if ((df["x"] < 0) | (df["x"] > PITCH_LENGTH)).any():
        raise ValidationError("Shot x coordinate outside the pitch")

    if ((df["y"] < 0) | (df["y"] > PITCH_WIDTH)).any():
        raise ValidationError("Shot y coordinate outside the pitch")

    leaked = [c for c in FEATURE_COLUMNS if c in NON_FEATURE_COLUMNS]
    if leaked:
        raise ValidationError(f"Leakage guard: columns marked as both feature and non-feature: {leaked}")

    return summarise(df)


def summarise(df: pd.DataFrame) -> Dict[str, Any]:
    """Return headline statistics used in the run report.

    Args:
        df: Processed shot DataFrame.

    Returns:
        A dictionary with shot counts, goal rate, and per-competition counts.
    """
    per_competition = {}
    for name, group in df.groupby("competition"):
        per_competition[str(name)] = int(len(group))

    return {
        "n_shots": int(len(df)),
        "n_matches": int(df["match_id"].nunique()),
        "goal_rate": round(float(df[TARGET_COLUMN].mean()), 4),
        "shots_per_competition": per_competition,
        "penalties": int(df["is_penalty"].sum()),
        "max_distance_to_goal": round(float(df["distance"].max()), 2),
        "goal_x_reference": GOAL_X,
    }
