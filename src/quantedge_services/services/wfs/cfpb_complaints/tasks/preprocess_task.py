"""CFPBPreprocessTask — text cleaning, label encoding, DistilBERT tokenisation."""
from __future__ import annotations
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any
import pandas as pd
from quantedge_services.api.schemas.foundations.cfpb_schemas import CFPBPreprocessRequest, CFPBPreprocessResponse
from quantedge_services.core.logging import StructuredLogger

NARRATIVE_COL = "Consumer complaint narrative"
PRODUCT_COL   = "Product"


class CFPBPreprocessTask:
    """Reads cleaned parquet from CFPBIngestionTask, tokenises narratives,
    encodes Product labels as integers. Saves tokenised parquet + label_map JSON."""

    def __init__(self) -> None:
        self._logger = StructuredLogger(name=__name__)

    def execute(
        self,
        request: CFPBPreprocessRequest,
        parquet_path: str,
    ) -> tuple[CFPBPreprocessResponse, pd.DataFrame, dict[str, int]]:
        try:
            from transformers import AutoTokenizer
            from datasets import Dataset

            df = pd.read_parquet(parquet_path)
            before = len(df)

            df[NARRATIVE_COL] = (
                df[NARRATIVE_COL]
                .str.lower()
                .str.replace(r"<[^>]+>", " ", regex=True)
                .str.replace(r"[^a-z0-9\s.,!?'-]", " ", regex=True)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )
            df = df[df[NARRATIVE_COL].str.len() >= 20].reset_index(drop=True)
            after = len(df)

            unique_products = sorted(df[PRODUCT_COL].unique())
            label_map: dict[str, int] = {p: i for i, p in enumerate(unique_products)}
            df["label"] = df[PRODUCT_COL].map(label_map)

            tokenizer = AutoTokenizer.from_pretrained(request.model_name)
            hf_ds = Dataset.from_pandas(df[[NARRATIVE_COL, "label"]])

            def _tokenize(batch: dict[str, Any]) -> dict[str, Any]:
                return tokenizer(
                    batch[NARRATIVE_COL],
                    truncation=True,
                    padding="max_length",
                    max_length=request.max_length,
                )

            hf_ds = hf_ds.map(_tokenize, batched=True, batch_size=request.batch_size)
            hf_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

            out_dir = Path("data/cfpb/tokenised")
            out_dir.mkdir(parents=True, exist_ok=True)
            tokenised_path = out_dir / f"cfpb_tokenised_{request.execution_id[:8]}.parquet"
            hf_ds.to_parquet(str(tokenised_path))

            label_map_path = out_dir / f"label_map_{request.execution_id[:8]}.json"
            label_map_path.write_text(json.dumps(label_map, indent=2))

            counts: Counter[str] = Counter(df[PRODUCT_COL])
            total = sum(counts.values())
            class_weights = {k: round(total / v, 4) for k, v in counts.items()}

            self._logger.info(
                "cfpb_preprocessed",
                execution_id=request.execution_id,
                rows_before=before, rows_after=after,
                n_classes=len(label_map),
            )
            return CFPBPreprocessResponse(
                execution_id=request.execution_id,
                rows_after_filter=after,
                n_classes=len(label_map),
                class_weights=class_weights,
                status="success",
            ), df, label_map

        except Exception as exc:
            self._logger.error("cfpb_preprocess_failed", error=str(exc))
            return CFPBPreprocessResponse(
                execution_id=request.execution_id,
                rows_after_filter=0, n_classes=0,
                class_weights={}, status="failed", error=str(exc),
            ), pd.DataFrame(), {}
