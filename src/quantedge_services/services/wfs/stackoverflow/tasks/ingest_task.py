"""StackOverflowIngestionTask — filter and normalise the raw StackOverflow parquet."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from quantedge_services.api.schemas.foundations.domain_adapt_schemas import (
    StackOverflowIngestionRequest,
    StackOverflowIngestionResult,
)


class StackOverflowIngestionTask:
    """Reads raw parquet, filters rows whose ``tags`` column contains any of the
    requested tag strings, extracts ``title + body`` as ``text``, and writes a
    filtered parquet to ``output_dir``."""

    def execute(self, request: StackOverflowIngestionRequest) -> StackOverflowIngestionResult:
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_parquet(request.input_path)
        filtered = self._filter_by_tags(df, request.tags_filter)
        filtered = filtered.copy()
        filtered["text"] = filtered["title"].fillna("") + " " + filtered["body"].fillna("")

        output_path = output_dir / "stackoverflow_filtered.parquet"
        filtered.to_parquet(output_path, index=False)

        return StackOverflowIngestionResult(
            output_path=str(output_path),
            filtered_count=len(filtered),
            status="success",
        )

    @staticmethod
    def _filter_by_tags(df: pd.DataFrame, tags_filter: list[str]) -> pd.DataFrame:
        if not tags_filter or "tags" not in df.columns:
            return df
        mask = df["tags"].astype(str).apply(
            lambda t: any(tag.lower() in t.lower() for tag in tags_filter)
        )
        return df[mask]
