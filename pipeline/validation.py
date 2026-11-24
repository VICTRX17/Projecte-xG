"""Validation of processed shot data."""

from typing import Any, Dict, List

import pandas as pd

from pipeline.config import GOAL_X, PITCH_LENGTH, PITCH_WIDTH
from pipeline.features import FEATURE_COLUMNS, NON_FEATURE_COLUMNS, TARGET_COLUMN

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


def validate_processed(df: pd.DataFrame) -> Dict[str, Any]:
    """Run integrity checks on the final shot dataset and return its summary."""
    if df.empty:
        raise ValidationError("Processed dataset is empty")

    missing_columns: List[str] = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValidationError(f"Missing required columns: {missing_columns}")

    if df["shot_id"].isna().any():
        raise ValidationError("Missing shot_id values found")
    if df["shot_id"].duplicated().any():
        raise ValidationError("Duplicate shot_id values found")

    if df[TARGET_COLUMN].isna().any():
        raise ValidationError("Missing target values found")
    target_values = set(df[TARGET_COLUMN].unique())
    if not target_values.issubset({0, 1}):
        raise ValidationError(f"Target must be binary, found values {sorted(target_values)}")

    if df[["x", "y", "distance", "angle_rad"]].isna().any().any():
        raise ValidationError("Missing coordinate or geometry values found")
    if (df["distance"] < 0).any():
        raise ValidationError("Negative distance values found")
    if ((df["angle_rad"] < 0) | (df["angle_rad"] > 3.1416)).any():
        raise ValidationError("Angle outside [0, pi] found")
    if ((df["x"] < 0) | (df["x"] > PITCH_LENGTH)).any():
        raise ValidationError("Shot x coordinate outside the pitch")
    if ((df["y"] < 0) | (df["y"] > PITCH_WIDTH)).any():
        raise ValidationError("Shot y coordinate outside the pitch")

    leaked = [column for column in FEATURE_COLUMNS if column in NON_FEATURE_COLUMNS]
    if leaked:
        raise ValidationError(
            f"Leakage guard: columns marked as both feature and non-feature: {leaked}"
        )

    return summarise(df)


def summarise(df: pd.DataFrame) -> Dict[str, Any]:
    """Return headline statistics used in the run report."""
    per_competition = {
        str(name): int(len(group))
        for name, group in df.groupby("competition")
    }

    return {
        "n_shots": int(len(df)),
        "n_matches": int(df["match_id"].nunique()),
        "goal_rate": round(float(df[TARGET_COLUMN].mean()), 4),
        "shots_per_competition": per_competition,
        "penalties": int(df["is_penalty"].sum()),
        "max_distance_to_goal": round(float(df["distance"].max()), 2),
        "goal_x_reference": GOAL_X,
    }
