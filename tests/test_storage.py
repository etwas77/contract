from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from candidate_reader.models import CandidateCV, CandidateCVData, SourceMetadata
from candidate_reader.storage import atomic_write_model, load_cached


def _candidate_cv(checksum: str) -> CandidateCV:
    return CandidateCV(
        document_id=f"candidate_cv-{checksum[:12]}",
        source=SourceMetadata(
            filename="cv.txt",
            path="cv.txt",
            sha256=checksum,
            size_bytes=1,
            extracted_at=datetime.now(UTC),
            language="English",
        ),
        data=CandidateCVData(language="English"),
    )


def test_atomic_write_and_cache_reuse(tmp_path: Path) -> None:
    checksum = "a" * 64
    candidate = _candidate_cv(checksum)
    target = tmp_path / "candidate_cv.json"

    atomic_write_model(target, candidate)

    assert load_cached(target, CandidateCV, checksum) == candidate
    assert load_cached(target, CandidateCV, "b" * 64) is None


def test_invalid_cache_is_not_reused(tmp_path: Path) -> None:
    target = tmp_path / "candidate_cv.json"
    target.write_text("{broken", encoding="utf-8")

    assert load_cached(target, CandidateCV, "a" * 64) is None
