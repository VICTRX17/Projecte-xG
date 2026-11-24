"""StatsBomb Open Data access through the StatsBombPy library."""

from typing import Any

import pandas as pd
from statsbombpy import sb


class DownloadError(RuntimeError):
    """Raised when StatsBombPy cannot retrieve the requested data."""


class StatsBombDownloader:
    """Small interface around StatsBombPy used by the project pipeline."""

    def get_competitions(self) -> pd.DataFrame:
        """Return the available competition and season combinations."""
        try:
            return sb.competitions()
        except Exception as exc:
            raise DownloadError(f"Could not retrieve StatsBomb competitions: {exc}") from exc

    def get_matches(self, competition_id: int, season_id: int) -> pd.DataFrame:
        """Return matches for one competition and season."""
        try:
            return sb.matches(
                competition_id=competition_id,
                season_id=season_id,
            )
        except Exception as exc:
            raise DownloadError(
                f"Could not retrieve matches for competition {competition_id}, "
                f"season {season_id}: {exc}"
            ) from exc

    def get_events(self, match_id: int) -> pd.DataFrame:
        """Return the flattened event DataFrame for one match."""
        try:
            return sb.events(match_id=match_id)
        except Exception as exc:
            raise DownloadError(
                f"Could not retrieve events for match {match_id}: {exc}"
            ) from exc

    def get_shots(self, match_id: int) -> pd.DataFrame:
        """Return shots from regular match periods, excluding shootouts."""
        events = self.get_events(match_id)

        required_columns = {"type", "period"}
        missing = required_columns.difference(events.columns)
        if missing:
            raise DownloadError(
                f"StatsBombPy events for match {match_id} are missing columns: "
                f"{sorted(missing)}"
            )

        return events.loc[
            (events["type"] == "Shot") & (events["period"] < 5)
        ].copy()

    def validate_selection(self, competitions: Any) -> None:
        """Check selected competition-season pairs against StatsBomb's catalogue."""
        catalogue = self.get_competitions()
        known = set(
            zip(catalogue["competition_id"], catalogue["season_id"])
        )

        for competition in competitions:
            key = (competition.competition_id, competition.season_id)
            if key not in known:
                raise DownloadError(
                    f"Competition '{competition.label}' {key} "
                    "not found in StatsBomb catalogue"
                )
