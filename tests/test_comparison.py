from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from candidate_reader.config import Settings
from candidate_reader.models import (
    CandidateConstraint,
    CandidateConstraints,
    CandidateConstraintsData,
    CandidateCV,
    CandidateCVData,
    ComparisonOperator,
    ConstraintCategory,
    ConstraintPriority,
    SourceLocation,
    SourceMetadata,
)
from company_reader.models import (
    CompanyOffer,
    CompanyOfferData,
    CompensationTerms,
    ContractClause,
    EmploymentDuration,
    EmploymentDurationType,
    MoneyAmount,
    RemoteWorkTerms,
    TravelTerms,
    WorkingTimeTerms,
)
from comparison.models import (
    CandidateRoleAssessment,
    ComparisonStatus,
    SemanticAnswer,
    SemanticBatch,
    SemanticQuestion,
    VerificationBatch,
    VerificationDecision,
)
from comparison.rules import compare_rules
from comparison.workflow import ComparisonWorkflow


def _source(page: int = 1, excerpt: str = "source") -> SourceLocation:
    return SourceLocation(page=page, excerpt=excerpt)


def _metadata(filename: str) -> SourceMetadata:
    return SourceMetadata(
        filename=filename,
        path=filename,
        sha256="a" * 64,
        size_bytes=1,
        extracted_at=datetime.now(UTC),
        language="English",
        page_count=1,
    )


def _constraint(
    constraint_id: str,
    category: ConstraintCategory,
    operator: ComparisonOperator,
    expected: str | None = None,
    unit: str | None = None,
) -> CandidateConstraint:
    return CandidateConstraint(
        constraint_id=constraint_id,
        category=category,
        requirement=constraint_id.replace("_", " "),
        priority=ConstraintPriority.MANDATORY,
        operator=operator,
        expected_value=expected,
        unit=unit,
        source=SourceLocation(line_start=1, excerpt=constraint_id),
    )


def _constraints(*items: CandidateConstraint) -> CandidateConstraints:
    return CandidateConstraints(
        document_id="candidate-constraints",
        source=_metadata("constraints.json"),
        data=CandidateConstraintsData(language="English", constraints=list(items)),
    )


def _offer() -> CompanyOffer:
    source = _source()
    return CompanyOffer(
        document_id="company-offer",
        source=_metadata("offer.json"),
        data=CompanyOfferData(
            language="German",
            employment_duration=EmploymentDuration(
                duration_type=EmploymentDurationType.FIXED_TERM,
                duration_months=Decimal("24"),
                source=source,
            ),
            probation_months=Decimal("6"),
            probation_source=source,
            compensation=CompensationTerms(
                fixed_annual=MoneyAmount(amount=Decimal("55000"), currency="EUR", period="year"),
                variable_annual=MoneyAmount(amount=Decimal("15000"), currency="EUR", period="year"),
                target_annual=MoneyAmount(amount=Decimal("70000"), currency="EUR", period="year"),
                target_setting="Quarterly targets",
                source=source,
            ),
            working_time=WorkingTimeTerms(
                regular_weekly_hours=Decimal("40"),
                included_overtime_weekly_hours=Decimal("4"),
                overtime_treatment="Overtime is included in compensation",
                source=source,
            ),
            vacation_days_per_year=Decimal("30"),
            vacation_source=source,
            travel=TravelTerms(required=True, source=source),
            remote_work=RemoteWorkTerms(
                available=True,
                entitlement=False,
                source=source,
            ),
            clauses=[
                ContractClause(
                    section_number="1",
                    title="Offer",
                    summary="Synthetic offer",
                    source=source,
                )
            ],
        ),
    )


def _cv() -> CandidateCV:
    return CandidateCV(
        document_id="candidate-cv",
        source=_metadata("cv.json"),
        data=CandidateCVData(language="English", professional_title="Developer"),
    )


def test_rules_compare_typed_values_without_an_agent() -> None:
    constraints = _constraints(
        _constraint(
            "contract_permanent",
            ConstraintCategory.EMPLOYMENT_DURATION,
            ComparisonOperator.REQUIRED,
            "permanent",
        ),
        _constraint(
            "salary_min_70000_euro",
            ConstraintCategory.COMPENSATION,
            ComparisonOperator.MINIMUM,
            "70000",
            "euro",
        ),
        _constraint(
            "probation_max_6_months",
            ConstraintCategory.PROBATION,
            ComparisonOperator.MAXIMUM,
            "6",
            "months",
        ),
        _constraint(
            "variable_part_max_5000_euro_during_probation",
            ConstraintCategory.VARIABLE_COMPENSATION,
            ComparisonOperator.MAXIMUM,
            "5000",
            "euro",
        ),
        _constraint(
            "working_time_max_40_hours_week",
            ConstraintCategory.WORKING_TIME,
            ComparisonOperator.MAXIMUM,
            "40",
            "hours per week",
        ),
        _constraint(
            "homeoffice_desired",
            ConstraintCategory.HOME_OFFICE,
            ComparisonOperator.CUSTOM,
            "desired",
        ),
        _constraint(
            "vacation_min_30_days_per_year",
            ConstraintCategory.VACATION,
            ComparisonOperator.MINIMUM,
            "30",
            "days",
        ),
    )

    results, unresolved = compare_rules(constraints, _offer())
    statuses = {result.constraint_id: result.status for result in results}

    assert unresolved == []
    assert statuses["contract_permanent"] == ComparisonStatus.VIOLATED
    assert statuses["salary_min_70000_euro"] == ComparisonStatus.SATISFIED
    assert statuses["probation_max_6_months"] == ComparisonStatus.SATISFIED
    assert statuses["variable_part_max_5000_euro_during_probation"] == ComparisonStatus.VIOLATED
    assert statuses["working_time_max_40_hours_week"] == ComparisonStatus.SATISFIED
    assert statuses["homeoffice_desired"] == ComparisonStatus.SATISFIED
    assert statuses["vacation_min_30_days_per_year"] == ComparisonStatus.SATISFIED


def test_paid_overtime_satisfies_any_of_group() -> None:
    offer = _offer()
    assert offer.data.working_time is not None
    offer.data.working_time.overtime_treatment = "Overtime is paid separately"
    constraints = _constraints(
        _constraint(
            "overtime_paid",
            ConstraintCategory.OVERTIME,
            ComparisonOperator.REQUIRED,
            "paid",
        ),
        _constraint(
            "overtime_time_account",
            ConstraintCategory.OVERTIME,
            ComparisonOperator.REQUIRED,
            "special time account (gleitzeitkonto)",
        ),
    )

    results, _ = compare_rules(constraints, offer)

    assert results[0].status == ComparisonStatus.SATISFIED
    assert results[1].status == ComparisonStatus.NOT_APPLICABLE


class FakeBudget:
    reserved_usd = Decimal("0.1000")


class FakeAnalyzer:
    calls = 0

    def __init__(self, _settings: Settings) -> None:
        self.budget = FakeBudget()

    def prepare(
        self,
        questions: list[SemanticQuestion],
        _cv: CandidateCV,
        _offer: CompanyOffer,
    ) -> Decimal:
        assert questions
        return self.budget.reserved_usd

    def run(
        self,
        questions: list[SemanticQuestion],
    ) -> tuple[SemanticBatch, VerificationBatch]:
        type(self).calls += 1
        question = questions[0]
        answer = SemanticAnswer(
            question_id=question.question_id,
            constraint_id=question.constraint_ids[0],
            status=ComparisonStatus.UNKNOWN,
            rationale="The offer does not define this limitation.",
            confidence=100,
        )
        return (
            SemanticBatch(
                answers=[answer],
                candidate_role_assessment=CandidateRoleAssessment(
                    summary="The offer has insufficient technical role detail.",
                    confidence=100,
                ),
            ),
            VerificationBatch(
                decisions=[
                    VerificationDecision(
                        question_id=answer.question_id,
                        constraint_id=answer.constraint_id,
                        accepted=True,
                    )
                ]
            ),
        )


def test_workflow_writes_and_reuses_validated_outputs(tmp_path: Path) -> None:
    FakeAnalyzer.calls = 0
    cv_path = tmp_path / "candidate_cv.json"
    constraints_path = tmp_path / "candidate_constraints.json"
    offer_path = tmp_path / "company_offer.json"
    cv_path.write_text(_cv().model_dump_json(), encoding="utf-8")
    constraints_path.write_text(
        _constraints(
            _constraint(
                "contract_permanent",
                ConstraintCategory.EMPLOYMENT_DURATION,
                ComparisonOperator.REQUIRED,
                "permanent",
            ),
            _constraint(
                "variable_duration_only_in_probation",
                ConstraintCategory.VARIABLE_COMPENSATION,
                ComparisonOperator.REQUIRED,
                "probation only",
            ),
        ).model_dump_json(),
        encoding="utf-8",
    )
    offer_path.write_text(_offer().model_dump_json(), encoding="utf-8")
    settings = Settings(
        structured_data_dir=tmp_path / "output",
        openai_api_key="not-used",
    )
    workflow = ComparisonWorkflow(settings, analyzer_factory=FakeAnalyzer)
    report_path = tmp_path / "report.md"

    generated = workflow.run(
        cv_path,
        constraints_path,
        offer_path,
        report_path=report_path,
        authorize=lambda _files, _cost: True,
    )
    reused = workflow.run(
        cv_path,
        constraints_path,
        offer_path,
        report_path=report_path,
        authorize=lambda _files, _cost: False,
    )

    assert generated.reused is False
    assert reused.reused is True
    assert FakeAnalyzer.calls == 1
    assert generated.data.summary.mandatory_violated == 1
    assert generated.data.summary.mandatory_unknown == 1
    semantic_result = generated.data.results[1]
    assert semantic_result.rationale == "The offer does not define this limitation."
    assert "Candidate and company offer comparison" in report_path.read_text(encoding="utf-8")
    assert (tmp_path / "output" / "comparison_result.json").is_file()
