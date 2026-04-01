from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from quantedge_services.api.schemas.foundations.attention_schemas import (
    CMAPSSIngestionRequest, CMAPSSIngestionResponse,
)
from quantedge_services.core.logging import StructuredLogger

# 14 sensor feature indices (0-based within sensor columns 1-21)
# Sensors: s1,s2,s3,s4,s7,s8,s9,s11,s12,s13,s14,s15,s17,s20
# Indices:  0, 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 19
_SENSOR_INDICES = [0, 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 19]
_N_FEATURES = len(_SENSOR_INDICES)


class CMAPSSIngestionTask:
    """Reads CMAPSS train text file, builds sliding-window sequences, saves parquet."""

    def __init__(self) -> None:
        self._logger = StructuredLogger(name=__name__)

    def execute(self, request: CMAPSSIngestionRequest) -> CMAPSSIngestionResponse:
        raw_dir = Path(request.raw_dir)
        parquet_dir = Path(request.parquet_dir)
        subset = request.subset
        seq_len = request.seq_len
        train_file = raw_dir / f"train_{subset}.txt"

        if not train_file.exists():
            self._logger.info("cmapss_ingest_skipped", file=str(train_file))
            return CMAPSSIngestionResponse(
                execution_id=request.execution_id,
                sequences_created=0,
                parquet_path="",
                num_sensors=_N_FEATURES,
                max_rul=0,
                status="skipped",
                error=f"File not found: {train_file}",
            )

        try:
            # CMAPSS: space-separated, no header, 26 columns
            col_names = (
                ["engine_id", "cycle", "op1", "op2", "op3"]
                + [f"s{i}" for i in range(1, 22)]
            )
            df = pd.read_csv(train_file, sep=r"\s+", header=None, names=col_names)

            # Compute RUL per row
            max_cycle = df.groupby("engine_id")["cycle"].max().rename("max_cycle")
            df = df.join(max_cycle, on="engine_id")
            df["rul"] = df["max_cycle"] - df["cycle"]

            sensor_cols = [f"s{i}" for i in range(1, 22)]
            selected_sensors = [sensor_cols[i] for i in _SENSOR_INDICES]

            rows: list[dict] = []
            feature_cols = [f"f{i}" for i in range(seq_len * _N_FEATURES)]

            for eid, group in df.groupby("engine_id"):
                group = group.sort_values("cycle").reset_index(drop=True)
                data = group[selected_sensors].values.astype(np.float32)
                rul_vals = group["rul"].values.astype(np.float32)

                if len(data) < seq_len:
                    continue

                for start in range(len(data) - seq_len + 1):
                    X_seq = data[start: start + seq_len]  # [seq_len, n_features]
                    y_val = float(rul_vals[start + seq_len - 1])
                    row = dict(zip(feature_cols, X_seq.flatten().tolist()))
                    row["rul"] = y_val
                    row["engine_id"] = int(eid)
                    rows.append(row)

            if not rows:
                return CMAPSSIngestionResponse(
                    execution_id=request.execution_id,
                    sequences_created=0,
                    parquet_path="",
                    num_sensors=_N_FEATURES,
                    max_rul=0,
                    status="skipped",
                    error="No sequences created",
                )

            out_df = pd.DataFrame(rows)
            max_rul = int(out_df["rul"].max())
            parquet_dir.mkdir(parents=True, exist_ok=True)
            parquet_path = parquet_dir / f"{subset}_seq{seq_len}.parquet"
            out_df.to_parquet(parquet_path, index=False)

            return CMAPSSIngestionResponse(
                execution_id=request.execution_id,
                sequences_created=len(rows),
                parquet_path=str(parquet_path),
                num_sensors=_N_FEATURES,
                max_rul=max_rul,
                status="success",
                error=None,
            )
        except Exception as exc:
            self._logger.error("cmapss_ingest_failed", error=str(exc))
            return CMAPSSIngestionResponse(
                execution_id=request.execution_id,
                sequences_created=0,
                parquet_path="",
                num_sensors=_N_FEATURES,
                max_rul=0,
                status="failed",
                error=str(exc),
            )
