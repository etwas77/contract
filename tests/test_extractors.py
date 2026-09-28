from __future__ import annotations

from pathlib import Path

import pytest

from candidate_reader.errors import ExtractionError, SourceReferenceError
from candidate_reader.extractors import extract_document, validate_source_reference
from candidate_reader.models import DocumentType, SourceLocation


def test_extract_txt_preserves_line_references(tmp_path: Path) -> None:
    source = tmp_path / "constraints.txt"
    source.write_text("First rule\n  nested rule\n", encoding="utf-8")

    document = extract_document(source, DocumentType.CONSTRAINTS)

    assert document.line_count == 2
    assert "2:   nested rule" in document.prompt_text()
    validate_source_reference(
        SourceLocation(line_start=2, excerpt="nested rule"),
        document,
    )


def test_invalid_reference_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "constraints.txt"
    source.write_text("Only one line", encoding="utf-8")
    document = extract_document(source, DocumentType.CONSTRAINTS)

    with pytest.raises(SourceReferenceError):
        validate_source_reference(
            SourceLocation(line_start=2, excerpt="missing"),
            document,
        )


def test_unsupported_extension_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "candidate.docx"
    source.write_text("content", encoding="utf-8")

    with pytest.raises(ExtractionError, match="Unsupported"):
        extract_document(source, DocumentType.CV)
