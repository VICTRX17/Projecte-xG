"""End-to-end orchestration of the StatsBombPy-to-shot-dataset pipeline."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from pipeline.config import PROCESSED_DIR, SELECTED_COMPETITIONS, CompetitionSeason
from pipeline.download import DownloadError, StatsBombDownloader
from pipeline.extraction import extract_match_shots
from pipeline.features import build_features
from pipeline.validation import ValidationError, validate_processed

PROCESSED_FILENAME = "shots.csv"
REPORT_FILENAME = "run_report.json"


class ShotPipeline:
    """Build a shot-level dataset from selected StatsBomb competitions."""

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
        """Download events, extract shots, engineer features and write outputs."""
        self.downloader.validate_selection(self.competitions)

        rows, skipped_matches = self._collect_rows()
        if not rows:
            raise ValidationError("No shots were extracted from the selected competitions")

        raw_df = pd.DataFrame(rows)
        clean_df, drop_report = build_features(raw_df)
        summary = validate_processed(clean_df)

        self._write_outputs(clean_df, summary, drop_report, skipped_matches, len(raw_df))
        return clean_df

    def _collect_rows(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Collect shots from every selected match, recording failed matches."""
        rows: List[Dict[str, Any]] = []
        skipped: List[str] = []
        processed_matches = 0

        for competition in self.competitions:
            matches = self.downloader.get_matches(
                competition.competition_id,
                competition.season_id,
            )

            for _, match in matches.iterrows():
                if self.max_matches is not None and processed_matches >= self.max_matches:
                    return rows, skipped

                match_id = int(match["match_id"])
                match_meta = {
                    "match_id": match_id,
                    "competition": competition.label,
                    "season": competition.season_id,
                }

                try:
                    shots = self.downloader.get_shots(match_id)
                    rows.extend(extract_match_shots(shots, match_meta))
                except (DownloadError, ValueError, KeyError) as exc:
                    skipped.append(f"match {match_id}: {exc}")
                    continue

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
        """Write the processed CSV and a JSON report for this pipeline run."""
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
