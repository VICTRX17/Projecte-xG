"""Download and cache StatsBomb Open Data JSON files.

Files are stored under ``data/raw`` mirroring the upstream repository layout,
so a file is only fetched once unless ``force`` is requested.
"""

import json
from pathlib import Path
from typing import Any, Iterable, List

import requests

from xg_pipeline.config import (
    RAW_DIR,
    REQUEST_TIMEOUT_SECONDS,
    STATSBOMB_RAW_BASE_URL,
    CompetitionSeason,
)


class DownloadError(RuntimeError):
    """Raised when a remote file cannot be fetched or parsed."""


class StatsBombDownloader:
    """Fetches StatsBomb open-data files with a local on-disk cache.

    Attributes:
        raw_dir: Local directory where raw JSON files are cached.
        base_url: Root URL of the StatsBomb ``data`` folder.
        timeout: HTTP timeout in seconds.
        force: If True, ignore the cache and re-download every file.
    """

    def __init__(
        self,
        raw_dir: Path = RAW_DIR,
        base_url: str = STATSBOMB_RAW_BASE_URL,
        timeout: int = REQUEST_TIMEOUT_SECONDS,
        force: bool = False,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.force = force

    def get_json(self, relative_path: str) -> Any:
        """Return parsed JSON for a path relative to the StatsBomb ``data`` folder.

        Args:
            relative_path: Path such as ``"competitions.json"`` or ``"events/123.json"``.

        Returns:
            The parsed JSON object (list or dict).

        Raises:
            DownloadError: If the file cannot be fetched, or it is not valid JSON.
        """
        local_path = self.raw_dir / relative_path
        if local_path.exists() and not self.force:
            return self._read_local(local_path)

        url = f"{self.base_url}/{relative_path}"
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise DownloadError(f"Could not fetch {url}: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise DownloadError(f"Response from {url} is not valid JSON") from exc

        local_path.parent.mkdir(parents=True, exist_ok=True)
        with local_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle)
        return data

    def get_competitions(self) -> List[dict]:
        """Return the StatsBomb competition catalogue (``competitions.json``)."""
        return self.get_json("competitions.json")

    def get_matches(self, competition: CompetitionSeason) -> List[dict]:
        """Return the match list for one competition season.

        Args:
            competition: The competition and season to load.

        Returns:
            List of match dictionaries as provided by StatsBomb.
        """
        path = f"matches/{competition.competition_id}/{competition.season_id}.json"
        return self.get_json(path)

    def get_events(self, match_id: int) -> List[dict]:
        """Return all events (passes, shots, etc.) for one match.

        Args:
            match_id: StatsBomb match identifier.

        Returns:
            List of event dictionaries for the match.
        """
        return self.get_json(f"events/{match_id}.json")

    def validate_selection(self, competitions: Iterable[CompetitionSeason]) -> None:
        """Check that every selected competition season exists in the catalogue.

        Args:
            competitions: The competition seasons requested by the user.

        Raises:
            DownloadError: If a competition or season ID is not in ``competitions.json``.
        """
        catalogue = self.get_competitions()
        known = {(row["competition_id"], row["season_id"]) for row in catalogue}
        for competition in competitions:
            key = (competition.competition_id, competition.season_id)
            if key not in known:
                raise DownloadError(
                    f"Competition '{competition.label}' {key} not found in StatsBomb catalogue"
                )

    @staticmethod
    def _read_local(path: Path) -> Any:
        """Read a cached JSON file, raising a clear error if it is corrupt."""
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise DownloadError(f"Cached file {path} is unreadable: {exc}") from exc
