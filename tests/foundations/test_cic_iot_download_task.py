from __future__ import annotations
import pytest
from pathlib import Path


class TestCICIoTDownloadTask:
    def test_download_skips_if_already_present(self, tmp_path):
        from quantedge_services.services.wfs.cic_iot.tasks.download_task import CICIoTDownloadTask
        from quantedge_services.api.schemas.foundations.profiler_schemas import CICIoTDownloadRequest
        # Create a dummy file
        dest = tmp_path / "cic_raw"
        dest.mkdir()
        (dest / "dummy.csv").write_text("a,b,c\n1,2,3\n")
        task = CICIoTDownloadTask()
        req = CICIoTDownloadRequest(
            execution_id="dl-skip-001",
            kaggle_dataset="dhoogla/ciciot2023",
            dest_dir=str(dest),
            force_redownload=False,
        )
        resp = task.execute(req)
        assert resp.status in ("success", "download_skipped")

    def test_download_skips_gracefully_without_kaggle(self, tmp_path):
        from quantedge_services.services.wfs.cic_iot.tasks.download_task import CICIoTDownloadTask
        from quantedge_services.api.schemas.foundations.profiler_schemas import CICIoTDownloadRequest
        dest = tmp_path / "cic_empty"
        task = CICIoTDownloadTask()
        req = CICIoTDownloadRequest(
            execution_id="dl-skip-002",
            kaggle_dataset="dhoogla/ciciot2023",
            dest_dir=str(dest),
            force_redownload=False,
        )
        resp = task.execute(req)
        assert resp.status in ("success", "download_skipped")
