from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from candidate_reader.errors import ExtractionError, SourceReferenceError
from candidate_reader.models import DocumentType, ExtractionWarning, SourceLocation


@dataclass(frozen=True)
class ExtractedSegment:
    text: str
    page: int | None = None
    line_start: int | None = None
    line_end: int | None = None


@dataclass(frozen=True)
class ExtractedDocument:
    source_path: Path
    document_type: DocumentType
    sha256: str
    size_bytes: int
    segments: tuple[ExtractedSegment, ...]
    warnings: tuple[ExtractionWarning, ...] = ()

    @property
    def page_count(self) -> int | None:
        pages = [segment.page for segment in self.segments if segment.page is not None]
        return max(pages) if pages else None

    @property
    def line_count(self) -> int | None:
        lines = [segment.line_end for segment in self.segments if segment.line_end is not None]
        return max(lines) if lines else None

    def prompt_text(self) -> str:
        parts: list[str] = []
        for segment in self.segments:
            if segment.page is not None:
                marker = f"[PAGE {segment.page}]"
            else:
                marker = f"[LINES {segment.line_start}-{segment.line_end}]"
            parts.append(f"{marker}\n{segment.text}")
        return "\n\n".join(parts)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_pdf_text(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _extract_pdf(path: Path, document_type: DocumentType) -> ExtractedDocument:
    try:
        reader = PdfReader(path)
    except Exception as exc:
        raise ExtractionError(f"Cannot open PDF '{path.name}': {exc}") from exc

    segments: list[ExtractedSegment] = []
    warnings: list[ExtractionWarning] = []
    for page_number, page in enumerate(reader.pages, 1):
        try:
            text = _normalize_pdf_text(page.extract_text() or "")
        except Exception as exc:
            raise ExtractionError(
                f"Cannot extract page {page_number} from '{path.name}': {exc}"
            ) from exc
        if not text:
            warnings.append(
                ExtractionWarning(
                    code="empty_pdf_page",
                    message=f"Page {page_number} has no extractable text",
                    source=None,
                )
            )
        segments.append(ExtractedSegment(text=text, page=page_number))

    if not segments or not any(segment.text for segment in segments):
        raise ExtractionError(f"PDF '{path.name}' contains no extractable text")

    return ExtractedDocument(
        source_path=path,
        document_type=document_type,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        segments=tuple(segments),
        warnings=tuple(warnings),
    )


def _extract_txt(path: Path, document_type: DocumentType) -> ExtractedDocument:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ExtractionError(f"TXT file '{path.name}' is not valid UTF-8") from exc
    except OSError as exc:
        raise ExtractionError(f"Cannot read TXT file '{path.name}': {exc}") from exc

    lines = text.splitlines()
    if not any(line.strip() for line in lines):
        raise ExtractionError(f"TXT file '{path.name}' is empty")
    return ExtractedDocument(
        source_path=path,
        document_type=document_type,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        segments=tuple(
            ExtractedSegment(
                text=f"{number}: {line}",
                line_start=number,
                line_end=number,
            )
            for number, line in enumerate(lines, 1)
        ),
    )


def extract_document(path: Path, document_type: DocumentType) -> ExtractedDocument:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ExtractionError(f"File does not exist: {resolved}")
    suffix = resolved.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(resolved, document_type)
    if suffix == ".txt":
        return _extract_txt(resolved, document_type)
    raise ExtractionError(f"Unsupported file type '{suffix}'. Use PDF or TXT.")


def _searchable(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def validate_source_reference(
    reference: SourceLocation,
    document: ExtractedDocument,
) -> None:
    if reference.page is not None:
        matches = [segment for segment in document.segments if segment.page == reference.page]
        if not matches:
            raise SourceReferenceError(f"Referenced page {reference.page} does not exist")
    else:
        assert reference.line_start is not None
        line_end = reference.line_end or reference.line_start
        if document.line_count is None or line_end > document.line_count:
            raise SourceReferenceError(
                f"Referenced lines {reference.line_start}-{line_end} do not exist"
            )
        matches = [
            segment
            for segment in document.segments
            if segment.line_start is not None
            and segment.line_end is not None
            and segment.line_start <= reference.line_start
            and segment.line_end >= line_end
        ]
    haystack = _searchable("\n".join(segment.text for segment in matches))
    needle = _searchable(reference.excerpt)
    if needle not in haystack:
        raise SourceReferenceError(
            f"Source excerpt was not found at {reference.page or reference.line_start}"
        )
