from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

SCHEMA_VERSION: Literal["1.0"] = "1.0"
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DocumentType(StrEnum):
    CV = "candidate_cv"
    CONSTRAINTS = "candidate_constraints"
    COMPANY_OFFER = "company_offer"


class SourceLocation(StrictModel):
    page: int | None = Field(default=None, ge=1)
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    section: str | None = None
    excerpt: NonEmptyStr

    @model_validator(mode="after")
    def validate_coordinates(self) -> SourceLocation:
        has_page = self.page is not None
        has_lines = self.line_start is not None
        if has_page == has_lines:
            raise ValueError("Provide either a page or a line range")
        if self.line_end is not None and self.line_start is None:
            raise ValueError("line_end requires line_start")
        if (
            self.line_start is not None
            and self.line_end is not None
            and self.line_end < self.line_start
        ):
            raise ValueError("line_end cannot precede line_start")
        return self


class ExtractionWarning(StrictModel):
    code: NonEmptyStr
    message: NonEmptyStr
    source: SourceLocation | None = None


class SourceMetadata(StrictModel):
    filename: NonEmptyStr
    path: NonEmptyStr
    sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    size_bytes: int = Field(ge=0)
    extracted_at: datetime
    language: NonEmptyStr
    page_count: int | None = Field(default=None, ge=1)
    warnings: list[ExtractionWarning] = Field(default_factory=list)


class SkillGroup(StrictModel):
    category: NonEmptyStr
    skills: list[NonEmptyStr] = Field(min_length=1)
    source: SourceLocation


class WorkExperience(StrictModel):
    start_date: str | None = None
    end_date: str | None = None
    employer: NonEmptyStr
    title: NonEmptyStr
    responsibilities: list[NonEmptyStr] = Field(default_factory=list)
    source: SourceLocation


class Education(StrictModel):
    start_date: str | None = None
    end_date: str | None = None
    institution: NonEmptyStr
    qualification: NonEmptyStr
    subject: str | None = None
    source: SourceLocation


class Project(StrictModel):
    name: NonEmptyStr
    description: NonEmptyStr
    technologies: list[NonEmptyStr] = Field(default_factory=list)
    link: str | None = None
    source: SourceLocation


class LanguageSkill(StrictModel):
    language: NonEmptyStr
    proficiency: str | None = None
    source: SourceLocation


class Certification(StrictModel):
    name: NonEmptyStr
    source: SourceLocation


class SensitiveContact(StrictModel):
    address: str | None = None
    phone: str | None = None
    email: str | None = None


class CandidateCVData(StrictModel):
    language: NonEmptyStr
    professional_title: str | None = None
    professional_summary: str | None = None
    skills: list[SkillGroup] = Field(default_factory=list)
    work_experience: list[WorkExperience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    languages: list[LanguageSkill] = Field(default_factory=list)
    certifications: list[Certification] = Field(default_factory=list)
    sensitive_contact: SensitiveContact | None = None
    warnings: list[ExtractionWarning] = Field(default_factory=list)


class ConstraintPriority(StrEnum):
    MANDATORY = "mandatory"
    PREFERRED = "preferred"
    INFORMATIONAL = "informational"


class ConstraintCategory(StrEnum):
    EMPLOYMENT_DURATION = "employment_duration"
    COMPENSATION = "compensation"
    PROBATION = "probation"
    VARIABLE_COMPENSATION = "variable_compensation"
    TRAVEL = "travel"
    WORKING_TIME = "working_time"
    OVERTIME = "overtime"
    HOME_OFFICE = "home_office"
    VACATION = "vacation"
    OTHER = "other"


class ComparisonOperator(StrEnum):
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    EQUALS = "equals"
    REQUIRED = "required"
    PROHIBITED = "prohibited"
    CONDITIONAL = "conditional"
    CUSTOM = "custom"


class CandidateConstraint(StrictModel):
    constraint_id: Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]*$")]
    category: ConstraintCategory
    requirement: NonEmptyStr
    priority: ConstraintPriority
    operator: ComparisonOperator
    expected_value: str | None = None
    unit: str | None = None
    condition: str | None = None
    parent_constraint_id: str | None = None
    source: SourceLocation


class CandidateConstraintsData(StrictModel):
    language: NonEmptyStr
    constraints: list[CandidateConstraint] = Field(min_length=1)
    warnings: list[ExtractionWarning] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_constraint_links(self) -> CandidateConstraintsData:
        ids = [constraint.constraint_id for constraint in self.constraints]
        if len(ids) != len(set(ids)):
            raise ValueError("constraint_id values must be unique")
        known_ids = set(ids)
        for constraint in self.constraints:
            parent_id = constraint.parent_constraint_id
            if parent_id is not None and parent_id not in known_ids:
                raise ValueError(f"Unknown parent_constraint_id: {parent_id}")
            if parent_id == constraint.constraint_id:
                raise ValueError("A constraint cannot be its own parent")
        return self


class CandidateCV(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    document_id: NonEmptyStr
    document_type: Literal[DocumentType.CV] = DocumentType.CV
    source: SourceMetadata
    data: CandidateCVData


class CandidateConstraints(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    document_id: NonEmptyStr
    document_type: Literal[DocumentType.CONSTRAINTS] = DocumentType.CONSTRAINTS
    source: SourceMetadata
    data: CandidateConstraintsData


CandidateDocument = CandidateCV | CandidateConstraints
