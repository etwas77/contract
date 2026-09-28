from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Protocol

from candidate_reader.config import Settings
from candidate_reader.errors import ConfigurationError, StructuredOutputError
from candidate_reader.models import CandidateConstraint, CandidateConstraints, CandidateCV
from company_reader.models import CompanyOffer
from comparison.loader import load_inputs
from comparison.models import (
    COMPARISON_VERSION,
    CandidateRoleAssessment,
    ComparisonMethod,
    ComparisonReportData,
    ComparisonStatus,
    ComparisonSummary,
    ConstraintResult,
    EvidenceReference,
    OverallOutcome,
    SemanticBatch,
    SemanticQuestion,
    VerificationBatch,
    VerificationStatus,
)
from comparison.report import render_report
from comparison.rules import compare_rules
from comparison.semantic import (
    SEMANTIC_PROMPT_VERSION,
    VERIFIER_PROMPT_VERSION,
    ComparisonAgent,
    build_semantic_questions,
)
from comparison.storage import load_cached_result, write_outputs, write_report

AuthorizeCallback = Callable[[tuple[str, ...], Decimal], bool]


class AnalyzerBudget(Protocol):
    reserved_usd: Decimal


class Analyzer(Protocol):
    @property
    def budget(self) -> AnalyzerBudget: ...

    def prepare(
        self,
        questions: list[SemanticQuestion],
        cv: CandidateCV,
        offer: CompanyOffer,
    ) -> Decimal: ...

    def run(
        self,
        questions: list[SemanticQuestion],
    ) -> tuple[SemanticBatch, VerificationBatch]: ...


@dataclass(frozen=True)
class ComparisonRunResult:
    data: ComparisonReportData
    data_path: Path
    report_path: Path
    reused: bool


class ComparisonWorkflow:
    def __init__(
        self,
        settings: Settings,
        analyzer_factory: Callable[[Settings], Analyzer] | None = None,
    ) -> None:
        self._settings = settings
        self._analyzer_factory = analyzer_factory or ComparisonAgent

    def run(
        self,
        cv_path: Path,
        constraints_path: Path,
        offer_path: Path,
        *,
        report_path: Path = Path("report.md"),
        force: bool = False,
        authorize: AuthorizeCallback,
    ) -> ComparisonRunResult:
        cv, constraints, offer = load_inputs(cv_path, constraints_path, offer_path)
        paths = {
            "candidate_cv": cv_path,
            "candidate_constraints": constraints_path,
            "company_offer": offer_path,
        }
        input_hashes = {name: _file_hash(path) for name, path in paths.items()}
        identity = _cache_identity(input_hashes, self._settings.openai_model)
        data_path = self._settings.structured_data_dir / "comparison_result.json"

        cached = None if force else load_cached_result(data_path, identity)
        if cached is not None:
            if not report_path.is_file():
                write_report(report_path, render_report(cached))
            return ComparisonRunResult(cached, data_path, report_path, reused=True)

        deterministic, unresolved = compare_rules(constraints, offer)
        questions = build_semantic_questions(unresolved, offer)
        semantic_results: list[ConstraintResult] = []
        assessment: CandidateRoleAssessment | None = None
        reserved = Decimal("0")
        warnings: list[str] = []

        if questions:
            analyzer = self._analyzer_factory(self._settings)
            reserved = analyzer.prepare(questions, cv, offer)
            if not authorize(tuple(path.name for path in paths.values()), reserved):
                raise ConfigurationError("Structured comparison transmission was not authorized")
            semantic, verification = analyzer.run(questions)
            semantic_results, warnings = _merge_semantic(
                unresolved,
                questions,
                semantic,
                verification,
                constraints,
            )
            assessment = semantic.candidate_role_assessment

        results = _ordered_results(
            constraints,
            [*deterministic, *semantic_results],
        )
        summary = _summary(results)
        data = ComparisonReportData(
            generated_at=datetime.now(UTC),
            cache_identity=identity,
            input_hashes=input_hashes,
            model=self._settings.openai_model if questions else None,
            reserved_cost_usd=f"{reserved:.4f}",
            results=results,
            candidate_role_assessment=assessment,
            summary=summary,
            warnings=warnings,
        )
        write_outputs(data_path, report_path, data, render_report(data))
        return ComparisonRunResult(data, data_path, report_path, reused=False)


def _merge_semantic(
    constraints: list[CandidateConstraint],
    questions: list[SemanticQuestion],
    semantic: SemanticBatch,
    verification: VerificationBatch,
    all_constraints: CandidateConstraints,
) -> tuple[list[ConstraintResult], list[str]]:
    constraint_by_id = {item.constraint_id: item for item in constraints}
    question_by_id = {item.question_id: item for item in questions}
    expected_pairs = {
        (question.question_id, constraint_id)
        for question in questions
        for constraint_id in question.constraint_ids
    }
    answer_by_pair = {
        (answer.question_id, answer.constraint_id): answer for answer in semantic.answers
    }
    decision_by_pair = {
        (decision.question_id, decision.constraint_id): decision
        for decision in verification.decisions
    }
    if set(answer_by_pair) != expected_pairs:
        raise StructuredOutputError(
            "Semantic agent did not answer every unresolved constraint once"
        )
    if set(decision_by_pair) != expected_pairs:
        raise StructuredOutputError("Verifier did not decide every semantic answer once")

    results: list[ConstraintResult] = []
    warnings: list[str] = []
    document_id = all_constraints.document_id
    for pair in sorted(expected_pairs):
        answer = answer_by_pair[pair]
        decision = decision_by_pair[pair]
        question = question_by_id[answer.question_id]
        constraint = constraint_by_id[answer.constraint_id]
        evidence_by_id = {item.evidence_id: item for item in question.evidence}
        invalid_ids = set(answer.evidence_ids) - set(evidence_by_id)
        accepted = decision.accepted and not invalid_ids
        defects = list(decision.defects)
        if invalid_ids:
            defects.append(f"Unknown evidence IDs: {', '.join(sorted(invalid_ids))}")
        if not accepted:
            warnings.append(
                f"{constraint.constraint_id}: semantic conclusion rejected"
                + (f" ({'; '.join(defects)})" if defects else "")
            )
        results.append(
            ConstraintResult(
                constraint_id=constraint.constraint_id,
                category=constraint.category,
                priority=constraint.priority,
                requirement=constraint.requirement,
                status=answer.status if accepted else ComparisonStatus.UNKNOWN,
                method=ComparisonMethod.SEMANTIC,
                expected=_expected(constraint),
                actual=answer.actual if accepted else None,
                rationale=(
                    answer.rationale
                    if accepted
                    else (
                        "Verifier rejected the semantic conclusion; manual review is required. "
                        f"Original analysis: {answer.rationale}"
                    )
                ),
                candidate_evidence=EvidenceReference(
                    evidence_id=f"constraint:{constraint.constraint_id}",
                    document_id=document_id,
                    document_type="candidate_constraints",
                    field=f"constraints.{constraint.constraint_id}",
                    source=constraint.source,
                ),
                company_evidence=[
                    evidence_by_id[evidence_id]
                    for evidence_id in answer.evidence_ids
                    if evidence_id in evidence_by_id
                ],
                verification=(
                    VerificationStatus.VERIFIED if accepted else VerificationStatus.REJECTED
                ),
                warnings=defects,
            )
        )
    return results, warnings


def _ordered_results(
    constraints: CandidateConstraints,
    results: list[ConstraintResult],
) -> list[ConstraintResult]:
    by_id = {result.constraint_id: result for result in results}
    expected = {constraint.constraint_id for constraint in constraints.data.constraints}
    if set(by_id) != expected:
        missing = sorted(expected - set(by_id))
        extra = sorted(set(by_id) - expected)
        raise StructuredOutputError(
            f"Comparison result coverage mismatch; missing={missing}, extra={extra}"
        )
    return [by_id[item.constraint_id] for item in constraints.data.constraints]


def _summary(results: list[ConstraintResult]) -> ComparisonSummary:
    def count(priority: str, status: ComparisonStatus) -> int:
        return sum(
            result.priority.value == priority and result.status == status for result in results
        )

    mandatory_violated = count("mandatory", ComparisonStatus.VIOLATED)
    mandatory_unknown = count("mandatory", ComparisonStatus.UNKNOWN)
    if mandatory_violated:
        outcome = OverallOutcome.REQUIREMENTS_NOT_MET
    elif mandatory_unknown:
        outcome = OverallOutcome.MANUAL_REVIEW_REQUIRED
    else:
        outcome = OverallOutcome.REQUIREMENTS_MET
    return ComparisonSummary(
        mandatory_satisfied=count("mandatory", ComparisonStatus.SATISFIED),
        mandatory_violated=mandatory_violated,
        mandatory_unknown=mandatory_unknown,
        mandatory_not_applicable=count("mandatory", ComparisonStatus.NOT_APPLICABLE),
        preferred_satisfied=count("preferred", ComparisonStatus.SATISFIED),
        preferred_violated=count("preferred", ComparisonStatus.VIOLATED),
        preferred_unknown=count("preferred", ComparisonStatus.UNKNOWN),
        preferred_not_applicable=count("preferred", ComparisonStatus.NOT_APPLICABLE),
        overall_outcome=outcome,
    )


def _file_hash(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ConfigurationError(f"Cannot hash structured input '{path}': {exc}") from exc


def _cache_identity(input_hashes: dict[str, str], model: str) -> str:
    parts = [
        COMPARISON_VERSION,
        SEMANTIC_PROMPT_VERSION,
        VERIFIER_PROMPT_VERSION,
        model,
        *(f"{key}:{value}" for key, value in sorted(input_hashes.items())),
    ]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _expected(constraint: CandidateConstraint) -> str | None:
    if constraint.expected_value is None:
        return None
    return " ".join(part for part in (constraint.expected_value, constraint.unit) if part)
