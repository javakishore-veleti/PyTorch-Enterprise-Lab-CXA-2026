"""StackOverflowDownloadTask — simulate download of StackOverflow Posts dataset."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from quantedge_services.api.schemas.foundations.domain_adapt_schemas import (
    StackOverflowDownloadRequest,
    StackOverflowDownloadResult,
)

_SYNTHETIC_RECORD_COUNT = 20


class StackOverflowDownloadTask:
    """Simulates download of the StackOverflow Posts dataset (~10 GB XML on Kaggle).

    In test/CI mode always produces a small synthetic parquet so no network or
    Kaggle credentials are required.
    """

    def execute(self, request: StackOverflowDownloadRequest) -> StackOverflowDownloadResult:
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "stackoverflow_raw.parquet"
        df = self._build_synthetic_dataframe()
        df.to_parquet(output_path, index=False)
        return StackOverflowDownloadResult(
            output_path=str(output_path),
            record_count=len(df),
            status="success",
        )

    @staticmethod
    def _build_synthetic_dataframe() -> pd.DataFrame:
        rows = []
        tags_pool = ["java", "python", "spring-boot", "elasticsearch", "javascript"]
        for i in range(_SYNTHETIC_RECORD_COUNT):
            tag = tags_pool[i % len(tags_pool)]
            rows.append(
                {
                    "id": i + 1,
                    "post_type": "question" if i % 2 == 0 else "answer",
                    "tags": f"<{tag}>",
                    "title": f"How to use {tag} feature {i}?",
                    "body": f"<p>Body text for post {i} about {tag}.</p>",
                    "score": i * 2,
                    "answer_count": i % 5,
                }
            )
        return pd.DataFrame(rows)
