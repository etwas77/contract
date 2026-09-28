from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from candidate_reader.models import (
    ConstraintCategory,
    ConstraintPriority,
    SourceLocation,
    StrictModel,
)

COMPARISON_VERSION: Literal["1.0"] = "1.0"


class ComparisonStatus(StrEnum):
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class ComparisonMethod(StrEnum):
    DETERMINISTIC = "deterministic"
    SEMANTIC = "semantic"


class VerificationStatus(StrEnum):
    NOT_REQUIRED = "not_required"
    VERIFIED = "verified"
    REJECTED = "rejected"


class OverallOutcome(StrEnum):
    REQUIREMENTS_MET = "requirements_met"
    REQUIREMENTS_NOT_MET = "requirements_not_met"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"


class EvidenceReference(StrictModel):
    evidence_id: str
    document_id: str
    document_type: str
    field: str
    source: SourceLocation


class ConstraintResult(StrictModel):
    constraint_id: str
    category: ConstraintCategory
    priority: ConstraintPriority
    requirement: str
    status: ComparisonStatus
    method: ComparisonMethod
    expected: str | None = None
    actual: str | None = None
    rationale: str
    candidate_evidence: EvidenceReference
    company_evidence: list[EvidenceReference] = Field(default_factory=list)
    verification: VerificationStatus = VerificationStatus.NOT_REQUIRED
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_verification(self) -> ConstraintResult:
        if (
            self.method == ComparisonMethod.DETERMINISTIC
            and self.verification != VerificationStatus.NOT_REQUIRED
        ):
            raise ValueError("Deterministic results do not require LLM verification")
        if (
            self.method == ComparisonMethod.SEMANTIC
            and self.verification == VerificationStatus.NOT_REQUIRED
        ):
            raise ValueError("Semantic results require verification")
        return self


class SemanticQuestion(StrictModel):
    question_id: str
    constraint_ids: list[str] = Field(min_length=1)
    requirement: str
    expected: str | None = None
    condition: str | None = None
    offer_facts: dict[str, str | bool | None]
    evidence: list[EvidenceReference]


class SemanticAnswer(StrictModel):
    question_id: str
    constraint_id: str
    status: ComparisonStatus
    actual: str | None = None
    rationale: str
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: int = Field(ge=0, le=100)

    @model_validator(mode="after")
    def require_evidence_for_conclusion(self) -> SemanticAnswer:
        if (
            self.status in {ComparisonStatus.SATISFIED, ComparisonStatus.VIOLATED}
            and not self.evidence_ids
        ):
            raise ValueError("Satisfied or violated semantic answers require evidence")
        return self


class CandidateRoleAssessment(StrictModel):
    summary: str
    matched_skills: list[str] = Field(default_factory=list)
    possible_gaps: list[str] = Field(default_factory=list)
    cv_evidence: list[SourceLocation] = Field(default_factory=list)
    offer_evidence: list[SourceLocation] = Field(default_factory=list)
    confidence: int = Field(ge=0, le=100)


class SemanticBatch(StrictModel):
    answers: list[SemanticAnswer] = Field(default_factory=list)
    candidate_role_assessment: CandidateRoleAssessment


class VerificationDecision(StrictModel):
    question_id: str
    constraint_id: str
    accepted: bool
    defects: list[str] = Field(default_factory=list)


class VerificationBatch(StrictModel):
    decisions: list[VerificationDecision] = Field(default_factory=list)


class ComparisonSummary(StrictModel):
    mandatory_satisfied: int = Field(ge=0)
    mandatory_violated: int = Field(ge=0)
    mandatory_unknown: int = Field(ge=0)
    mandatory_not_applicable: int = Field(ge=0)
    preferred_satisfied: int = Field(ge=0)
    preferred_violated: int = Field(ge=0)
    preferred_unknown: int = Field(ge=0)
    preferred_not_applicable: int = Field(ge=0)
    overall_outcome: OverallOutcome


class ComparisonReportData(StrictModel):
    comparison_version: Literal["1.0"] = COMPARISON_VERSION
    generated_at: datetime
    cache_identity: str
    input_hashes: dict[str, str]
    model: str | None = None
    reserved_cost_usd: str = "0.0000"
    results: list[ConstraintResult]
    candidate_role_assessment: CandidateRoleAssessment | None = None
    summary: ComparisonSummary
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_results(self) -> ComparisonReportData:
        ids = [result.constraint_id for result in self.results]
        if len(ids) != len(set(ids)):
            raise ValueError("Each constraint must appear exactly once")
        return self
