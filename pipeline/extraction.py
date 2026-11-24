"""Extract shot events from raw StatsBomb event data.

Each StatsBomb shot event is nested JSON. This module flattens the fields we
need into a plain dictionary, one per shot, and attaches match-level metadata.
Only information known at the moment of the shot is read here; the shot's end
location and StatsBomb's own xG are read solely as metadata/benchmark columns.
"""

from typing import Any, Dict, List, Optional

SHOT_EVENT_TYPE = "Shot"


def _get_nested(record: Dict[str, Any], *keys: str) -> Any:
    """Safely walk nested dictionaries, returning None if any key is missing.

    Args:
        record: The dictionary to traverse.
        *keys: Sequence of keys to follow.

    Returns:
        The value at the end of the path, or None.
    """
    current: Any = record
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def extract_shot_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return only the shot events from a list of raw match events.

    Args:
        events: Raw event dictionaries for one match.

    Returns:
        The subset whose ``type.name`` is ``"Shot"``, in original order.
    """
    shots = []
    for event in events:
        if _get_nested(event, "type", "name") == SHOT_EVENT_TYPE:
            shots.append(event)
    return shots


def flatten_shot(event: Dict[str, Any], match_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one raw shot event into a row-shaped dictionary.

    Args:
        event: A single raw shot event.
        match_meta: Metadata for the match containing the shot. Must include
            ``match_id``, ``competition`` and ``season``.

    Returns:
        A dictionary with one key per output column. Missing source fields
        are represented as None so that downstream code can handle them.
    """
    shot = event.get("shot") or {}
    location = event.get("location") or [None, None]

    return {
        "shot_id": event.get("id"),
        "match_id": match_meta["match_id"],
        "competition": match_meta["competition"],
        "season": match_meta["season"],
        "period": event.get("period"),
        "minute": event.get("minute"),
        "second": event.get("second"),
        "team": _get_nested(event, "team", "name"),
        "player": _get_nested(event, "player", "name"),
        "x": location[0] if len(location) > 0 else None,
        "y": location[1] if len(location) > 1 else None,
        "shot_type": _get_nested(shot, "type", "name"),
        "body_part": _get_nested(shot, "body_part", "name"),
        "technique": _get_nested(shot, "technique", "name"),
        "first_time": shot.get("first_time"),
        "one_on_one": shot.get("one_on_one"),
        "under_pressure": event.get("under_pressure"),
        "play_pattern": _get_nested(event, "play_pattern", "name"),
        "outcome": _get_nested(shot, "outcome", "name"),
        # Benchmark and metadata only: NOT features (see features.NON_FEATURE_COLUMNS).
        "statsbomb_xg": shot.get("statsbomb_xg"),
        "end_x": _end_coordinate(shot, 0),
        "end_y": _end_coordinate(shot, 1),
    }


def _end_coordinate(shot: Dict[str, Any], index: int) -> Optional[float]:
    """Return one coordinate of a shot's end location, or None if absent."""
    end_location = shot.get("end_location")
    if not isinstance(end_location, list) or len(end_location) <= index:
        return None
    return end_location[index]


def extract_match_shots(
    events: List[Dict[str, Any]], match_meta: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Extract and flatten every shot from one match.

    Args:
        events: Raw event dictionaries for the match.
        match_meta: Metadata used to label each shot.

    Returns:
        One flattened dictionary per shot event.
    """
    rows = []
    for event in extract_shot_events(events):
        rows.append(flatten_shot(event, match_meta))
    return rows
