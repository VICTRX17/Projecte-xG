"""End-to-end orchestration of the data pipeline.

Steps:
    1. Verify the selected competitions exist in the StatsBomb catalogue.
    2. For each competition season, read its matches.
    3. For each match, download (or load from cache) its events and extract shots.
    4. Build features and the binary target.
    5. Validate and write ``data/processed/shots.csv`` and ``run_report.json``.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from pipeline.config import PROCESSED_DIR, SELECTED_COMPETITIONS, CompetitionSeason
from pipeline.download import DownloadError, StatsBombDownloader
from pipeline.extraction import extract_match_shots
from pipeline.features import build_features
from pipeline.validation import ValidationError, validate_processed, validate_raw_events

PROCESSED_FILENAME = "shots.csv"
REPORT_FILENAME = "run_report.json"


class ShotPipeline:
    """Builds the shot-level dataset from StatsBomb open data.

    Attributes:
        competitions: Competition seasons to include.
        downloader: Object responsible for fetching raw JSON files.
        output_dir: Directory where processed outputs are written.
        max_matches: Optional cap on matches per run, for quick test runs.
    """

    def __init__(
        self,
        competitions: Sequence[CompetitionSeason] = SELECTED_COMPETITIONS,
        downloader: Optional[StatsBombDownloader] = None,
        output_dir: Path = PROCESSED_DIR,
        max_matches: Optional[int] = None,
    ) -> None:
        self.competitions = list(competitions)
        self.downloader = downloader or StatsBombDownloader()
        self.output_dir = Path(output_dir)
        self.max_matches = max_matches

    def run(self) -> pd.DataFrame:
        """Execute the pipeline and write outputs.

        Returns:
            The processed shot DataFrame (one row per shot).

        Raises:
            DownloadError: If a remote file cannot be fetched.
            ValidationError: If raw or processed data violate a rule.
        """
        self.downloader.validate_selection(self.competitions)

        rows, skipped_matches = self._collect_rows()
        if not rows:
            raise ValidationError("No shots were extracted from the selected competitions")

        raw_df = pd.DataFrame(rows)
        clean_df, drop_report = build_features(raw_df)
        summary = validate_processed(clean_df)

        self._write_outputs(clean_df, summary, drop_report, skipped_matches, len(raw_df))
        return clean_df

    def _collect_rows(self) -> tuple:
        """Gather flattened shot rows from every selected match.

        Returns:
            A tuple of (list of shot dictionaries, list of skipped match descriptions).
        """
        rows: List[Dict[str, Any]] = []
        skipped: List[str] = []
        processed_matches = 0

        for competition in self.competitions:
            matches = self.downloader.get_matches(competition)
            for match in matches:
                if self.max_matches is not None and processed_matches >= self.max_matches:
                    return rows, skipped
                match_id = match["match_id"]
                match_meta = {
                    "match_id": match_id,
                    "competition": competition.label,
                    "season": competition.season_id,
                }
                try:
                    events = self.downloader.get_events(match_id)
                    validate_raw_events(events, source=f"events/{match_id}.json")
                except (DownloadError, ValidationError) as exc:
                    # A single broken match should not stop the whole run; it is reported instead.
                    skipped.append(f"match {match_id}: {exc}")
                    continue

                rows.extend(extract_match_shots(events, match_meta))
                processed_matches += 1

        return rows, skipped

    def _write_outputs(
        self,
        df: pd.DataFrame,
        summary: Dict[str, Any],
        drop_report: Dict[str, int],
        skipped_matches: List[str],
        raw_shot_count: int,
    ) -> None:
        """Write the processed CSV and a JSON run report.

        Args:
            df: Final processed DataFrame.
            summary: Statistics from ``validate_processed``.
            drop_report: Rows removed by each cleaning step.
            skipped_matches: Descriptions of matches that could not be processed.
            raw_shot_count: Number of shots before cleaning.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_dir / PROCESSED_FILENAME, index=False)

        report = {
            "raw_shots_extracted": raw_shot_count,
            "rows_dropped": drop_report,
            "skipped_matches": skipped_matches,
            "summary": summary,
        }
        with (self.output_dir / REPORT_FILENAME).open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
