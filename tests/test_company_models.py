from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from candidate_reader.models import SourceLocation
from company_reader.models import (
    CompanyOfferData,
    CompanyOfferPageData,
    ContractClause,
    OfferFact,
    OfferFactType,
)
from company_reader.structurer import CompanyOfferStructurer


def test_normalized_company_term_requires_source() -> None:
    clause = ContractClause(
        section_number="1",
        title="Start",
        summary="Employment starts later",
        source=SourceLocation(line_start=1, excerpt="Start"),
    )

    with pytest.raises(ValidationError, match="probation_months requires"):
        CompanyOfferData(
            language="English",
            probation_months=Decimal("6"),
            clauses=[clause],
        )


def test_page_chunks_merge_normalized_terms_and_clauses() -> None:
    first = CompanyOfferPageData(
        language="English",
        facts=[
            OfferFact(
                fact_type=OfferFactType.EMPLOYER,
                value="Example Company",
                source=SourceLocation(line_start=1, excerpt="Example Company"),
            )
        ],
        clauses=[],
    )
    second = CompanyOfferPageData(
        language="English",
        facts=[
            OfferFact(
                fact_type=OfferFactType.POSITION,
                value="Software Developer",
                source=SourceLocation(line_start=2, excerpt="Software Developer"),
            ),
            OfferFact(
                fact_type=OfferFactType.VACATION_DAYS,
                value="30",
                unit="working days",
                source=SourceLocation(line_start=2, excerpt="30 working days"),
            ),
        ],
        clauses=[
            ContractClause(
                section_number="5",
                title="Vacation",
                summary="Annual vacation is 30 working days",
                source=SourceLocation(line_start=1, excerpt="5. Vacation"),
            )
        ],
    )

    merged = CompanyOfferStructurer._merge_pages([first, second])

    assert merged.employer == "Example Company"
    assert merged.position == "Software Developer"
    assert merged.vacation_days_per_year == Decimal("30")
    assert len(merged.clauses) == 1
