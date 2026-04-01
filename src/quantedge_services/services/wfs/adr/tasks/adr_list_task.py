"""ADRListTask — scans a directory for ADR markdown files and parses metadata."""
from __future__ import annotations
import re
from pathlib import Path
from quantedge_services.api.schemas.foundations.monitoring_schemas import (
    ADRListRequest, ADRListResult,
)


class ADRListTask:
    """Scans docs/adr (or configured directory) and returns parsed ADR metadata."""

    def execute(self, request: ADRListRequest) -> ADRListResult:
        adr_dir = Path(request.adr_dir)
        adrs: list[dict] = []

        if adr_dir.exists():
            for md_file in sorted(adr_dir.glob("*.md")):
                parsed = self._parse_adr(md_file)
                if parsed:
                    adrs.append(parsed)

        return ADRListResult(
            adrs=adrs,
            total_count=len(adrs),
            status="completed",
        )

    def _parse_adr(self, path: Path) -> dict | None:
        try:
            lines = path.read_text().splitlines()
        except OSError:
            return None

        adr_id = ""
        title = ""
        status = "Unknown"

        if lines:
            # Parse "# ADR-NNN: Title"
            first = lines[0].strip()
            m = re.match(r"^#\s+(ADR-\d+):\s+(.+)$", first)
            if m:
                adr_id = m.group(1)
                title = m.group(2)

        for line in lines:
            m = re.match(r"\*\*Status:\*\*\s*(.+)", line.strip())
            if m:
                status = m.group(1).strip()
                break

        if not adr_id:
            return None

        return {"id": adr_id, "title": title, "status": status, "path": str(path)}
