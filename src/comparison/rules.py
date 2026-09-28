from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal

from candidate_reader.models import CandidateConstraint, CandidateConstraints, SourceLocation
from company_reader.models import CompanyOffer
from comparison.models import (
    ComparisonMethod,
    ComparisonStatus,
    ConstraintResult,
    EvidenceReference,
    VerificationStatus,
)

RuleHandler = Callable[[CandidateConstraint, CompanyOffer], ConstraintResult]


def compare_rules(
    constraints: CandidateConstraints,
    offer: CompanyOffer,
) -> tuple[list[ConstraintResult], list[CandidateConstraint]]:
    results: list[ConstraintResult] = []
    unresolved: list[CandidateConstraint] = []
    statuses: dict[str, ComparisonStatus] = {}
    for constraint in constraints.data.constraints:
        parent_id = constraint.parent_constraint_id
        if parent_id is not None and statuses.get(parent_id) == ComparisonStatus.NOT_APPLICABLE:
            result = _result(
                constraint,
                offer,
                ComparisonStatus.NOT_APPLICABLE,
                f"Parent condition '{parent_id}' does not apply.",
            )
            results.append(result)
            statuses[constraint.constraint_id] = result.status
            continue
        handler = RULES.get(constraint.constraint_id)
        if handler is None:
            unresolved.append(constraint)
            continue
        result = handler(constraint, offer)
        if result.method == ComparisonMethod.SEMANTIC:
            unresolved.append(constraint)
        else:
            results.append(result)
            statuses[constraint.constraint_id] = result.status
    return results, unresolved


def _candidate_evidence(
    constraint: CandidateConstraint,
    document_id: str = "candidate_constraints",
) -> EvidenceReference:
    return EvidenceReference(
        evidence_id=f"constraint:{constraint.constraint_id}",
        document_id=document_id,
        document_type="candidate_constraints",
        field=f"constraints.{constraint.constraint_id}",
        source=constraint.source,
    )


def _company_evidence(
    offer: CompanyOffer,
    field: str,
    source: SourceLocation,
) -> EvidenceReference:
    return EvidenceReference(
        evidence_id=f"offer:{field}",
        document_id=offer.document_id,
        document_type="company_offer",
        field=field,
        source=source,
    )


def _result(
    constraint: CandidateConstraint,
    offer: CompanyOffer,
    status: ComparisonStatus,
    rationale: str,
    *,
    actual: str | None = None,
    field: str | None = None,
    source: SourceLocation | None = None,
) -> ConstraintResult:
    company_evidence = (
        [_company_evidence(offer, field, source)]
        if field is not None and source is not None
        else []
    )
    return ConstraintResult(
        constraint_id=constraint.constraint_id,
        category=constraint.category,
        priority=constraint.priority,
        requirement=constraint.requirement,
        status=status,
        method=ComparisonMethod.DETERMINISTIC,
        expected=_expected(constraint),
        actual=actual,
        rationale=rationale,
        candidate_evidence=_candidate_evidence(constraint),
        company_evidence=company_evidence,
    )


def _expected(constraint: CandidateConstraint) -> str | None:
    if constraint.expected_value is None:
        return None
    return " ".join(part for part in (constraint.expected_value, constraint.unit) if part)


def _unknown(
    constraint: CandidateConstraint, offer: CompanyOffer, rationale: str
) -> ConstraintResult:
    return _result(constraint, offer, ComparisonStatus.UNKNOWN, rationale)


def _semantic(
    constraint: CandidateConstraint,
    offer: CompanyOffer,
) -> ConstraintResult:
    del offer
    return ConstraintResult(
        constraint_id=constraint.constraint_id,
        category=constraint.category,
        priority=constraint.priority,
        requirement=constraint.requirement,
        status=ComparisonStatus.UNKNOWN,
        method=ComparisonMethod.SEMANTIC,
        expected=_expected(constraint),
        rationale="Requires semantic comparison of contract wording.",
        candidate_evidence=_candidate_evidence(constraint),
        verification=VerificationStatus.REJECTED,
    )


def _permanent(constraint: CandidateConstraint, offer: CompanyOffer) -> ConstraintResult:
    duration = offer.data.employment_duration
    if duration is None:
        return _unknown(constraint, offer, "The offer does not state an employment duration.")
    status = (
        ComparisonStatus.SATISFIED
        if duration.duration_type.value == "permanent"
        else ComparisonStatus.VIOLATED
    )
    return _result(
        constraint,
        offer,
        status,
        f"The offer describes the employment as {duration.duration_type.value}.",
        actual=duration.duration_type.value,
        field="employment_duration",
        source=duration.source,
    )


def _salary_minimum(constraint: CandidateConstraint, offer: CompanyOffer) -> ConstraintResult:
    compensation = offer.data.compensation
    money = compensation.target_annual if compensation else None
    if compensation is None or money is None:
        return _unknown(constraint, offer, "The annual target compensation is missing.")
    if money.currency.casefold() != "eur" or money.period != "year":
        return _unknown(constraint, offer, "The compensation currency or period is not comparable.")
    expected = Decimal(constraint.expected_value or "0")
    status = ComparisonStatus.SATISFIED if money.amount >= expected else ComparisonStatus.VIOLATED
    return _result(
        constraint,
        offer,
        status,
        f"Annual target compensation is EUR {money.amount}.",
        actual=f"{money.amount} EUR per year",
        field="compensation.target_annual",
        source=compensation.source,
    )


def _probation(constraint: CandidateConstraint, offer: CompanyOffer) -> ConstraintResult:
    value = offer.data.probation_months
    source = offer.data.probation_source
    if value is None or source is None:
        return _unknown(constraint, offer, "The probation duration is missing.")
    expected = Decimal(constraint.expected_value or "0")
    status = ComparisonStatus.SATISFIED if value <= expected else ComparisonStatus.VIOLATED
    return _result(
        constraint,
        offer,
        status,
        f"Probation is {value} months.",
        actual=f"{value} months",
        field="probation_months",
        source=source,
    )


def _variable_condition(constraint: CandidateConstraint, offer: CompanyOffer) -> ConstraintResult:
    compensation = offer.data.compensation
    variable = compensation.variable_annual if compensation else None
    if compensation is None or variable is None:
        return _result(
            constraint,
            offer,
            ComparisonStatus.NOT_APPLICABLE,
            "The offer has no stated variable compensation.",
        )
    active = variable.amount > 0
    return _result(
        constraint,
        offer,
        ComparisonStatus.SATISFIED if active else ComparisonStatus.NOT_APPLICABLE,
        "The variable-compensation condition applies." if active else "No variable amount applies.",
        actual=f"{variable.amount} {variable.currency} per year",
        field="compensation.variable_annual",
        source=compensation.source,
    )


def _variable_probation_amount(
    constraint: CandidateConstraint,
    offer: CompanyOffer,
) -> ConstraintResult:
    compensation = offer.data.compensation
    variable = compensation.variable_annual if compensation else None
    months = offer.data.probation_months
    if compensation is None or variable is None or months is None:
        return _unknown(
            constraint,
            offer,
            "Variable annual compensation or probation duration is missing.",
        )
    if variable.currency.casefold() != "eur" or variable.period != "year":
        return _unknown(constraint, offer, "Variable compensation cannot be converted safely.")
    probation_amount = variable.amount * months / Decimal(12)
    expected = Decimal(constraint.expected_value or "0")
    status = (
        ComparisonStatus.SATISFIED if probation_amount <= expected else ComparisonStatus.VIOLATED
    )
    return _result(
        constraint,
        offer,
        status,
        f"Prorated variable compensation over {months} months is EUR {probation_amount}.",
        actual=f"{probation_amount} EUR during probation",
        field="compensation.variable_annual",
        source=compensation.source,
    )


def _travel_condition(constraint: CandidateConstraint, offer: CompanyOffer) -> ConstraintResult:
    travel = offer.data.travel
    if travel is None or travel.required is None:
        return _unknown(
            constraint, offer, "The offer does not establish whether travel is required."
        )
    return _result(
        constraint,
        offer,
        ComparisonStatus.SATISFIED if travel.required else ComparisonStatus.NOT_APPLICABLE,
        "Business travel is required." if travel.required else "Business travel is not required.",
        actual=str(travel.required).lower(),
        field="travel.required",
        source=travel.source,
    )


def _travel_field(
    constraint: CandidateConstraint,
    offer: CompanyOffer,
    attribute: str,
    label: str,
) -> ConstraintResult:
    travel = offer.data.travel
    value = getattr(travel, attribute) if travel else None
    if travel is None or value is None:
        return _result(
            constraint,
            offer,
            ComparisonStatus.UNKNOWN,
            f"The offer does not specify {label}.",
            field="travel",
            source=travel.source if travel else None,
        )
    return _semantic(constraint, offer)


def _working_hours(constraint: CandidateConstraint, offer: CompanyOffer) -> ConstraintResult:
    working = offer.data.working_time
    value = working.regular_weekly_hours if working else None
    if working is None or value is None:
        return _unknown(constraint, offer, "Regular weekly working time is missing.")
    expected = Decimal(constraint.expected_value or "0")
    status = ComparisonStatus.SATISFIED if value <= expected else ComparisonStatus.VIOLATED
    return _result(
        constraint,
        offer,
        status,
        f"Regular working time is {value} hours per week.",
        actual=f"{value} hours per week",
        field="working_time.regular_weekly_hours",
        source=working.source,
    )


def _overtime_condition(constraint: CandidateConstraint, offer: CompanyOffer) -> ConstraintResult:
    working = offer.data.working_time
    overtime = working.included_overtime_weekly_hours if working else None
    if working is None or overtime is None:
        return _unknown(constraint, offer, "The offer does not quantify required overtime.")
    return _result(
        constraint,
        offer,
        ComparisonStatus.SATISFIED if overtime > 0 else ComparisonStatus.NOT_APPLICABLE,
        f"The offer includes {overtime} overtime hours per week.",
        actual=f"{overtime} hours per week",
        field="working_time.included_overtime_weekly_hours",
        source=working.source,
    )


def _overtime_alternative(constraint: CandidateConstraint, offer: CompanyOffer) -> ConstraintResult:
    working = offer.data.working_time
    if working is None or not working.overtime_treatment:
        return _unknown(constraint, offer, "The offer does not explain overtime treatment.")
    treatment = working.overtime_treatment.casefold()
    expected = (constraint.expected_value or "").casefold()
    paid = "paid" in treatment and "included" not in treatment
    account = "time account" in treatment or "gleitzeit" in treatment
    requested_paid = expected == "paid"
    satisfied = paid if requested_paid else account
    alternative_satisfied = account if requested_paid else paid
    status = (
        ComparisonStatus.SATISFIED
        if satisfied
        else (
            ComparisonStatus.NOT_APPLICABLE if alternative_satisfied else ComparisonStatus.VIOLATED
        )
    )
    return _result(
        constraint,
        offer,
        status,
        f"The offer states: {working.overtime_treatment}.",
        actual=working.overtime_treatment,
        field="working_time.overtime_treatment",
        source=working.source,
    )


def _home_office(constraint: CandidateConstraint, offer: CompanyOffer) -> ConstraintResult:
    remote = offer.data.remote_work
    if remote is None or remote.available is None:
        return _unknown(
            constraint, offer, "The offer does not state whether home office is available."
        )
    return _result(
        constraint,
        offer,
        ComparisonStatus.SATISFIED if remote.available else ComparisonStatus.VIOLATED,
        (
            "Home office is available, though it is not an entitlement."
            if remote.available and remote.entitlement is False
            else f"Home-office availability is {remote.available}."
        ),
        actual=remote.arrangement or str(remote.available),
        field="remote_work",
        source=remote.source,
    )


def _vacation(constraint: CandidateConstraint, offer: CompanyOffer) -> ConstraintResult:
    value = offer.data.vacation_days_per_year
    source = offer.data.vacation_source
    if value is None or source is None:
        return _unknown(constraint, offer, "Annual vacation is missing.")
    expected = Decimal(constraint.expected_value or "0")
    status = ComparisonStatus.SATISFIED if value >= expected else ComparisonStatus.VIOLATED
    return _result(
        constraint,
        offer,
        status,
        f"The offer provides {value} vacation days per year.",
        actual=f"{value} days per year",
        field="vacation_days_per_year",
        source=source,
    )


def _target_condition(constraint: CandidateConstraint, offer: CompanyOffer) -> ConstraintResult:
    compensation = offer.data.compensation
    setting = compensation.target_setting if compensation else None
    if compensation is None or not setting:
        return _result(
            constraint,
            offer,
            ComparisonStatus.NOT_APPLICABLE,
            "The offer does not make variable pay dependent on defined targets.",
        )
    return _result(
        constraint,
        offer,
        ComparisonStatus.SATISFIED,
        "The offer makes variable pay dependent on target agreements.",
        actual=setting,
        field="compensation.target_setting",
        source=compensation.source,
    )


RULES: dict[str, RuleHandler] = {
    "contract_permanent": _permanent,
    "salary_min_70000_euro": _salary_minimum,
    "probation_max_6_months": _probation,
    "salary_split_fixed_variable_condition": _variable_condition,
    "variable_duration_only_in_probation": _semantic,
    "variable_part_max_5000_euro_during_probation": _variable_probation_amount,
    "variable_depends_on_achievings_condition": _target_condition,
    "achievings_defined_on_paper_prior_start": _semantic,
    "achieved_targets_fixed_on_paper_end": _semantic,
    "variable_split_by_target": _semantic,
    "all_achieved_targets_paid": _semantic,
    "business_travel_condition": _travel_condition,
    "business_travel_max_2_weeks_3_months": lambda c, o: _travel_field(
        c, o, "frequency_limit", "travel frequency"
    ),
    "accommodation_paid_by_company": lambda c, o: _travel_field(
        c, o, "accommodation_reimbursement", "accommodation reimbursement"
    ),
    "travel_days_money_paid": lambda c, o: _travel_field(c, o, "per_diem", "travel per diem"),
    "working_time_max_40_hours_week": _working_hours,
    "overtime_condition": _overtime_condition,
    "overtime_paid": _overtime_alternative,
    "overtime_time_account": _overtime_alternative,
    "homeoffice_desired": _home_office,
    "vacation_min_30_days_per_year": _vacation,
}
