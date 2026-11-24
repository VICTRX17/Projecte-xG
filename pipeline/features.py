"""Feature engineering and cleaning for the shot dataset.

Responsibilities:
    * compute geometry (distance and angle to the goal mouth) from coordinates;
    * build the binary target ``goal`` (1 = goal, 0 = any other outcome);
    * derive play-type flags such as penalty and open play;
    * handle missing values explicitly and count what was dropped.

The module deliberately does not train or score anything.
"""

import math
from typing import Dict, Tuple

import pandas as pd

from pipeline.config import (
    GOAL_CENTER_Y,
    GOAL_POST_HIGH_Y,
    GOAL_POST_LOW_Y,
    GOAL_X,
    PITCH_LENGTH,
    PITCH_WIDTH,
)

TARGET_COLUMN = "goal"
GOAL_OUTCOME = "Goal"

# Columns that must never be used as model inputs in future work.
# - outcome: the target is derived from it, so it leaks the answer.
# - statsbomb_xg: StatsBomb's own prediction; reserved as a benchmark.
# - end_x / end_y: the ball's destination is only known after the shot.
# - identifiers and raw context columns are kept for traceability.
NON_FEATURE_COLUMNS = (
    "shot_id",
    "match_id",
    "competition",
    "season",
    "period",
    "minute",
    "second",
    "team",
    "player",
    "outcome",
    "statsbomb_xg",
    "end_x",
    "end_y",
    TARGET_COLUMN,
)

# Columns available at the moment of the shot and suitable as model inputs.
FEATURE_COLUMNS = (
    "x",
    "y",
    "distance",
    "angle_rad",
    "body_part",
    "shot_type",
    "technique",
    "first_time",
    "one_on_one",
    "under_pressure",
    "play_pattern",
    "is_penalty",
    "is_open_play",
)


def compute_distance(x: float, y: float) -> float:
    """Return the straight-line distance from a shot to the centre of the goal.

    Args:
        x: Shot x coordinate on the 120 x 80 pitch.
        y: Shot y coordinate on the 120 x 80 pitch.

    Returns:
        Distance in pitch units (same units as the coordinates).
    """
    return math.hypot(GOAL_X - x, GOAL_CENTER_Y - y)


def compute_angle(x: float, y: float) -> float:
    """Return the angle in radians subtended by the goal mouth from a shot location.

    The angle is formed by the two goal posts as seen from the shot. A wider
    angle means a more central, closer shot, and it is 0 for shots from the
    byline and approaches 0 as the shot moves far out wide.

    Args:
        x: Shot x coordinate.
        y: Shot y coordinate.

    Returns:
        Angle in radians, in the range [0, pi].
    """
    dx = GOAL_X - x
    ax, ay = dx, GOAL_POST_LOW_Y - y
    bx, by = dx, GOAL_POST_HIGH_Y - y
    dot = ax * bx + ay * by
    cross = ax * by - ay * bx
    return abs(math.atan2(cross, dot))


def validate_coordinates(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Drop shots with missing or off-pitch coordinates and count them.

    Args:
        df: Raw shot DataFrame with ``x`` and ``y`` columns.

    Returns:
        A tuple of the cleaned DataFrame and a dictionary of drop counts.
    """
    report = {"missing_location": 0, "off_pitch": 0}

    missing_mask = df["x"].isna() | df["y"].isna()
    report["missing_location"] = int(missing_mask.sum())
    df = df.loc[~missing_mask].copy()

    off_pitch_mask = (
        (df["x"] < 0) | (df["x"] > PITCH_LENGTH) | (df["y"] < 0) | (df["y"] > PITCH_WIDTH)
    )
    report["off_pitch"] = int(off_pitch_mask.sum())
    df = df.loc[~off_pitch_mask].copy()

    return df, report


def add_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``distance`` and ``angle_rad`` columns computed from coordinates.

    Args:
        df: Shot DataFrame with valid ``x`` and ``y`` columns.

    Returns:
        A new DataFrame with the two geometry columns added.
    """
    df = df.copy()
    distances = []
    angles = []
    for x, y in zip(df["x"], df["y"]):
        distances.append(compute_distance(x, y))
        angles.append(compute_angle(x, y))
    df["distance"] = distances
    df["angle_rad"] = angles
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add the binary target column ``goal`` derived from the shot outcome.

    Args:
        df: Shot DataFrame with an ``outcome`` column.

    Returns:
        A new DataFrame with ``goal`` set to 1 for goals and 0 otherwise.
    """
    df = df.copy()
    goals = []
    for outcome in df["outcome"]:
        goals.append(1 if outcome == GOAL_OUTCOME else 0)
    df[TARGET_COLUMN] = goals
    return df


def add_play_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean flags for penalties and open-play shots.

    Args:
        df: Shot DataFrame with a ``shot_type`` column.

    Returns:
        A new DataFrame with ``is_penalty`` and ``is_open_play`` columns.
    """
    df = df.copy()
    df["is_penalty"] = (df["shot_type"] == "Penalty").astype(int)
    df["is_open_play"] = (df["shot_type"] == "Open Play").astype(int)
    return df


def fill_categorical_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Replace missing categorical values with an explicit ``"Unknown"`` label.

    Missing values are kept visible rather than silently imputed, so that
    later analysis can measure whether they carry information.

    Args:
        df: Shot DataFrame.

    Returns:
        A new DataFrame with missing categorical values filled.
    """
    df = df.copy()
    for column in ("body_part", "shot_type", "technique", "play_pattern"):
        df[column] = df[column].fillna("Unknown")
    # StatsBomb omits boolean keys when they are false, so a missing flag means False.
    for column in ("first_time", "one_on_one", "under_pressure"):
        df[column] = df[column].fillna(False).astype(bool)
    return df


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Run the full cleaning and feature-engineering sequence.

    Args:
        df: Flattened shot DataFrame as produced by ``extraction``.

    Returns:
        The processed DataFrame and a dictionary reporting how many rows were
        removed at each step.
    """
    df, coordinate_report = validate_coordinates(df)
    df = add_geometry(df)
    df = add_target(df)
    df = add_play_flags(df)
    df = fill_categorical_missing(df)
    return df, coordinate_report
