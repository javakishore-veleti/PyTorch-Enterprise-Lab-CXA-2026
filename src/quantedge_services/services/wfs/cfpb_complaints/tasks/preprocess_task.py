"""CFPBPreprocessTask — tokenize narratives, compute class weights."""

from __future__ import annotations

from collections import Counter
from typing import Any

from datasets import Dataset
from transformers import AutoTokenizer

from quantedge_services.api.schemas.foundations.cfpb_schemas import (
    CFPBPreprocessRequest,
    CFPBPreprocessResponse,
)
from quantedge_services.core.logging import StructuredLogger

NARRATIVE_COL = "Consumer complaint narrative"
PRODUCT_COL = "Product"


class CFPBPreprocessTask:
    def __init__(self) -> None:
        self._logger = StructuredLogger(name=__name__)

    def execute(
        self,
        request: CFPBPreprocessRequest,
        dataset: Dataset,
    ) -> tuple[CFPBPreprocessResponse, Dataset]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(request.model_name)
            before = len(dataset)
            dataset = dataset.filter(lambda x: x[NARRATIVE_COL] is not None)
            after = len(dataset)
            self._logger.info("narrative_filter", before=before, after=after)

            def _tokenize(batch: dict[str, Any]) -> dict[str, Any]:
                return tokenizer(
                    batch[NARRATIVE_COL],
                    truncation=True, padding="max_length",
                    max_length=request.max_length,
                )

            dataset = dataset.map(_tokenize, batched=True, batch_size=request.batch_size)
            counts: Counter[str] = Counter(dataset[PRODUCT_COL])
            total = sum(counts.values())
            class_weights = {label: round(total / count, 4) for label, count in counts.items()}

            return CFPBPreprocessResponse(
                execution_id=request.execution_id,
                rows_after_filter=after,
                n_classes=len(class_weights),
                class_weights=class_weights,
                status="success",
            ), dataset

        except Exception as exc:
            self._logger.error("cfpb_preprocess_failed", error=str(exc))
            return CFPBPreprocessResponse(
                execution_id=request.execution_id, rows_after_filter=0,
                n_classes=0, class_weights={}, status="failed", error=str(exc),
            ), Dataset.from_dict({})
