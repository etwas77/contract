from __future__ import annotations

import pytest
from pydantic import ValidationError

from candidate_reader.models import (
    CandidateConstraint,
    CandidateConstraintsData,
    ComparisonOperator,
    ConstraintCategory,
    ConstraintPriority,
    SourceLocation,
)


def test_source_location_requires_exactly_one_coordinate_type() -> None:
    with pytest.raises(ValidationError):
        SourceLocation(page=1, line_start=1, excerpt="text")

    with pytest.raises(ValidationError):
        SourceLocation(excerpt="text")


def test_constraint_parent_must_exist() -> None:
    constraint = CandidateConstraint(
        constraint_id="child",
        category=ConstraintCategory.COMPENSATION,
        requirement="Salary target",
        priority=ConstraintPriority.MANDATORY,
        operator=ComparisonOperator.MINIMUM,
        expected_value="70000",
        unit="EUR/year",
        parent_constraint_id="missing",
        source=SourceLocation(line_start=2, excerpt="salary"),
    )
    with pytest.raises(ValidationError, match="Unknown parent"):
        CandidateConstraintsData(language="English", constraints=[constraint])
