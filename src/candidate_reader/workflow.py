from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Protocol

from candidate_reader.config import Settings
from candidate_reader.errors import ConfigurationError, ExtractionError
from candidate_reader.extractors import ExtractedDocument, extract_document
from candidate_reader.models import (
    CandidateConstraints,
    CandidateCV,
    CandidateDocument,
    DocumentType,
)
from candidate_reader.storage import atomic_write_model, load_cached, output_path
from candidate_reader.structurer import CandidateStructurer

AuthorizeCallback = Callable[[tuple[str, ...]], bool]


class StructurerBudget(Protocol):
    reserved_usd: Decimal


class Structurer(Protocol):
    @property
    def budget(self) -> StructurerBudget: ...

    def structure_cv(self, document: ExtractedDocument) -> CandidateCV: ...

    def structure_constraints(
        self,
        document: ExtractedDocument,
    ) -> CandidateConstraints: ...


@dataclass(frozen=True)
class DocumentResult:
    document: CandidateDocument
    output_path: Path
    reused: bool


@dataclass(frozen=True)
class ReadResult:
    cv: DocumentResult
    constraints: DocumentResult
    reserved_cost_usd: str


class CandidateReadWorkflow:
    def __init__(
        self,
        settings: Settings,
        structurer_factory: Callable[[Settings], Structurer] = CandidateStructurer,
    ) -> None:
        self._settings = settings
        self._structurer_factory = structurer_factory

    def run(
        self,
        cv_path: Path,
        constraints_path: Path,
        *,
        force: bool = False,
        authorize: AuthorizeCallback,
    ) -> ReadResult:
        if cv_path.expanduser().resolve() == constraints_path.expanduser().resolve():
            raise ExtractionError("The same file cannot be used for both candidate roles")

        cv_source = extract_document(cv_path, DocumentType.CV)
        constraints_source = extract_document(constraints_path, DocumentType.CONSTRAINTS)
        cv_target = output_path(self._settings.structured_data_dir, DocumentType.CV)
        constraints_target = output_path(
            self._settings.structured_data_dir,
            DocumentType.CONSTRAINTS,
        )

        cached_cv = None if force else load_cached(cv_target, CandidateCV, cv_source.sha256)
        cached_constraints = (
            None
            if force
            else load_cached(
                constraints_target,
                CandidateConstraints,
                constraints_source.sha256,
            )
        )

        pending: list[str] = []
        if cached_cv is None:
            pending.append(cv_source.source_path.name)
        if cached_constraints is None:
            pending.append(constraints_source.source_path.name)

        structurer: Structurer | None = None
        if pending:
            if not authorize(tuple(pending)):
                raise ConfigurationError("Candidate data transmission was not authorized")
            structurer = self._structurer_factory(self._settings)

        cv_result = self._process_cv(cv_source, cv_target, cached_cv, structurer)
        constraints_result = self._process_constraints(
            constraints_source,
            constraints_target,
            cached_constraints,
            structurer,
        )
        reserved = "0.0000" if structurer is None else f"{structurer.budget.reserved_usd:.4f}"
        return ReadResult(
            cv=cv_result,
            constraints=constraints_result,
            reserved_cost_usd=reserved,
        )

    @staticmethod
    def _process_cv(
        source: ExtractedDocument,
        target: Path,
        cached: CandidateCV | None,
        structurer: Structurer | None,
    ) -> DocumentResult:
        if cached is not None:
            return DocumentResult(cached, target, reused=True)
        if structurer is None:
            raise ConfigurationError("CV generation requires a configured structurer")
        generated = structurer.structure_cv(source)
        atomic_write_model(target, generated)
        return DocumentResult(generated, target, reused=False)

    @staticmethod
    def _process_constraints(
        source: ExtractedDocument,
        target: Path,
        cached: CandidateConstraints | None,
        structurer: Structurer | None,
    ) -> DocumentResult:
        if cached is not None:
            return DocumentResult(cached, target, reused=True)
        if structurer is None:
            raise ConfigurationError("Constraint generation requires a configured structurer")
        generated = structurer.structure_constraints(source)
        atomic_write_model(target, generated)
        return DocumentResult(generated, target, reused=False)
