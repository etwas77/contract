from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from candidate_reader.models import (
    SCHEMA_VERSION,
    ExtractionWarning,
    SourceLocation,
    SourceMetadata,
    StrictModel,
)

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class EmploymentDurationType(StrEnum):
    PERMANENT = "permanent"
    FIXED_TERM = "fixed_term"
    UNCLEAR = "unclear"


class MoneyAmount(StrictModel):
    amount: Decimal = Field(ge=0)
    currency: NonEmptyStr
    period: str | None = None


class EmploymentDuration(StrictModel):
    duration_type: EmploymentDurationType
    start_date: str | None = None
    end_date: str | None = None
    duration_months: Decimal | None = Field(default=None, ge=0)
    source: SourceLocation


class CompensationTerms(StrictModel):
    fixed_annual: MoneyAmount | None = None
    variable_annual: MoneyAmount | None = None
    target_annual: MoneyAmount | None = None
    discretionary_bonus: bool | None = None
    target_setting: str | None = None
    payment_timing: str | None = None
    source: SourceLocation


class WorkingTimeTerms(StrictModel):
    regular_weekly_hours: Decimal | None = Field(default=None, ge=0)
    included_overtime_weekly_hours: Decimal | None = Field(default=None, ge=0)
    overtime_treatment: str | None = None
    schedule_terms: str | None = None
    source: SourceLocation


class TravelTerms(StrictModel):
    required: bool | None = None
    geographic_scope: str | None = None
    frequency_limit: str | None = None
    transport_reimbursement: str | None = None
    accommodation_reimbursement: str | None = None
    per_diem: str | None = None
    source: SourceLocation


class RemoteWorkTerms(StrictModel):
    available: bool | None = None
    entitlement: bool | None = None
    arrangement: str | None = None
    source: SourceLocation


class ContractClause(StrictModel):
    section_number: str | None = None
    title: NonEmptyStr
    summary: NonEmptyStr
    employee_obligations: list[NonEmptyStr] = Field(default_factory=list)
    employer_obligations: list[NonEmptyStr] = Field(default_factory=list)
    monetary_values: list[MoneyAmount] = Field(default_factory=list)
    durations: list[NonEmptyStr] = Field(default_factory=list)
    source: SourceLocation


class OfferFactType(StrEnum):
    EMPLOYER = "employer"
    EMPLOYEE = "employee"
    POSITION = "position"
    WORKPLACE = "workplace"
    DUTY = "duty"
    DURATION_TYPE = "duration_type"
    START_DATE = "start_date"
    END_DATE = "end_date"
    DURATION_MONTHS = "duration_months"
    PROBATION_MONTHS = "probation_months"
    FIXED_ANNUAL = "fixed_annual"
    VARIABLE_ANNUAL = "variable_annual"
    TARGET_ANNUAL = "target_annual"
    CURRENCY = "currency"
    DISCRETIONARY_BONUS = "discretionary_bonus"
    TARGET_SETTING = "target_setting"
    PAYMENT_TIMING = "payment_timing"
    WEEKLY_HOURS = "weekly_hours"
    INCLUDED_OVERTIME_HOURS = "included_overtime_hours"
    OVERTIME_TREATMENT = "overtime_treatment"
    SCHEDULE_TERMS = "schedule_terms"
    VACATION_DAYS = "vacation_days"
    TRAVEL_REQUIRED = "travel_required"
    TRAVEL_SCOPE = "travel_scope"
    TRAVEL_FREQUENCY_LIMIT = "travel_frequency_limit"
    TRANSPORT_REIMBURSEMENT = "transport_reimbursement"
    ACCOMMODATION_REIMBURSEMENT = "accommodation_reimbursement"
    PER_DIEM = "per_diem"
    REMOTE_AVAILABLE = "remote_available"
    REMOTE_ENTITLEMENT = "remote_entitlement"
    REMOTE_ARRANGEMENT = "remote_arrangement"
    TERMINATION = "termination"
    CONFIDENTIALITY = "confidentiality"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    NON_COMPETE = "non_compete"
    SIDE_ACTIVITY = "side_activity"


class OfferFact(StrictModel):
    fact_type: OfferFactType
    value: NonEmptyStr
    unit: str | None = None
    source: SourceLocation


class CompanyOfferPageData(StrictModel):
    language: NonEmptyStr
    facts: list[OfferFact] = Field(default_factory=list)
    clauses: list[ContractClause] = Field(default_factory=list)
    warnings: list[ExtractionWarning] = Field(default_factory=list)


class CompanyOfferData(StrictModel):
    language: NonEmptyStr
    employer: str | None = None
    employee: str | None = None
    position: str | None = None
    workplace: str | None = None
    duties: list[NonEmptyStr] = Field(default_factory=list)
    employment_duration: EmploymentDuration | None = None
    probation_months: Decimal | None = Field(default=None, ge=0)
    probation_source: SourceLocation | None = None
    compensation: CompensationTerms | None = None
    working_time: WorkingTimeTerms | None = None
    vacation_days_per_year: Decimal | None = Field(default=None, ge=0)
    vacation_source: SourceLocation | None = None
    travel: TravelTerms | None = None
    remote_work: RemoteWorkTerms | None = None
    termination_terms: str | None = None
    termination_source: SourceLocation | None = None
    confidentiality_terms: str | None = None
    confidentiality_source: SourceLocation | None = None
    intellectual_property_terms: str | None = None
    intellectual_property_source: SourceLocation | None = None
    non_compete_terms: str | None = None
    non_compete_source: SourceLocation | None = None
    side_activity_terms: str | None = None
    side_activity_source: SourceLocation | None = None
    clauses: list[ContractClause] = Field(default_factory=list)
    warnings: list[ExtractionWarning] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_sources_for_normalized_terms(self) -> CompanyOfferData:
        paired_values = (
            ("probation_months", self.probation_months, self.probation_source),
            ("vacation_days_per_year", self.vacation_days_per_year, self.vacation_source),
            ("termination_terms", self.termination_terms, self.termination_source),
            (
                "confidentiality_terms",
                self.confidentiality_terms,
                self.confidentiality_source,
            ),
            (
                "intellectual_property_terms",
                self.intellectual_property_terms,
                self.intellectual_property_source,
            ),
            ("non_compete_terms", self.non_compete_terms, self.non_compete_source),
            ("side_activity_terms", self.side_activity_terms, self.side_activity_source),
        )
        for name, value, source in paired_values:
            if value is not None and source is None:
                raise ValueError(f"{name} requires its source reference")
        return self


class CompanyOffer(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    document_id: NonEmptyStr
    document_type: Literal["company_offer"] = "company_offer"
    source: SourceMetadata
    data: CompanyOfferData

    @model_validator(mode="after")
    def require_contract_clauses(self) -> CompanyOffer:
        if not self.data.clauses:
            raise ValueError("A company offer must contain at least one contract clause")
        return self
