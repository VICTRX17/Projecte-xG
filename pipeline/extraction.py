"""Extract and normalize shot rows from StatsBombPy event DataFrames."""

from typing import Any, Dict, List

import pandas as pd


def _clean_value(value: Any) -> Any:
    """Convert pandas missing values to None without treating lists as scalars."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        return None if pd.isna(value) else value
    except (TypeError, ValueError):
        return value


def _coordinate(location: Any, index: int) -> Any:
    """Safely return one coordinate from a StatsBomb location."""
    location = _clean_value(location)
    if not isinstance(location, (list, tuple)) or len(location) <= index:
        return None
    return _clean_value(location[index])


def _end_coordinate(end_location: Any, index: int) -> Any:
    """Safely return one coordinate from a shot end location."""
    return _coordinate(end_location, index)


def flatten_shot(event: Dict[str, Any], match_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Map one flattened StatsBombPy shot row to the project's column names."""
    location = event.get("location")
    end_location = event.get("shot_end_location")

    return {
        "shot_id": _clean_value(event.get("id")),
        "match_id": match_meta["match_id"],
        "competition": match_meta["competition"],
        "season": match_meta["season"],
        "period": _clean_value(event.get("period")),
        "minute": _clean_value(event.get("minute")),
        "second": _clean_value(event.get("second")),
        "team": _clean_value(event.get("team")),
        "player": _clean_value(event.get("player")),
        "x": _coordinate(location, 0),
        "y": _coordinate(location, 1),
        "shot_type": _clean_value(event.get("shot_type")),
        "body_part": _clean_value(event.get("shot_body_part")),
        "technique": _clean_value(event.get("shot_technique")),
        "first_time": _clean_value(event.get("shot_first_time")),
        "one_on_one": _clean_value(event.get("shot_one_on_one")),
        "under_pressure": _clean_value(event.get("under_pressure")),
        "play_pattern": _clean_value(event.get("play_pattern")),
        "outcome": _clean_value(event.get("shot_outcome")),
        # Benchmark and post-shot metadata; never use these as model features.
        "statsbomb_xg": _clean_value(event.get("shot_statsbomb_xg")),
        "end_x": _end_coordinate(end_location, 0),
        "end_y": _end_coordinate(end_location, 1),
    }


def extract_shot_events(events: pd.DataFrame) -> pd.DataFrame:
    """Return shot events from a StatsBombPy event DataFrame."""
    if "type" not in events.columns:
        raise ValueError("StatsBombPy events DataFrame has no 'type' column")
    return events.loc[events["type"] == "Shot"].copy()


def extract_match_shots(
    events: pd.DataFrame, match_meta: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Flatten all supplied shot events into the project's row schema.

    The downloader normally supplies only regular-period shots through
    get_shots(); this function also accepts a full event DataFrame.
    """
    shots = extract_shot_events(events)
    return [
        flatten_shot(event, match_meta)
        for event in shots.to_dict(orient="records")
    ]
