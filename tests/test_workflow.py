from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from candidate_reader.config import Settings
from candidate_reader.extractors import ExtractedDocument
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
    WorkExperience,
)
from candidate_reader.workflow import CandidateReadWorkflow


class FakeBudget:
    reserved_usd = Decimal("0.1234")


class FakeStructurer:
    calls = 0

    def __init__(self, settings: Settings) -> None:
        del settings
        self._budget = FakeBudget()

    @property
    def budget(self) -> FakeBudget:
        return self._budget

    @staticmethod
    def _metadata(document: ExtractedDocument, language: str) -> SourceMetadata:
        return SourceMetadata(
            filename=document.source_path.name,
            path=str(document.source_path),
            sha256=document.sha256,
            size_bytes=document.size_bytes,
            extracted_at=datetime.now(UTC),
            language=language,
            page_count=document.page_count,
        )

    def structure_cv(self, document: ExtractedDocument) -> CandidateCV:
        type(self).calls += 1
        return CandidateCV(
            document_id=f"candidate_cv-{document.sha256[:12]}",
            source=self._metadata(document, "English"),
            data=CandidateCVData(
                language="English",
                professional_title="Senior Software Developer",
                work_experience=[
                    WorkExperience(
                        start_date="2018",
                        end_date="present",
                        employer="Example Systems",
                        title="Senior Developer",
                        source=SourceLocation(
                            line_start=2,
                            excerpt="Example Systems, Senior Developer",
                        ),
                    )
                ],
            ),
        )

    def structure_constraints(self, document: ExtractedDocument) -> CandidateConstraints:
        type(self).calls += 1
        return CandidateConstraints(
            document_id=f"candidate_constraints-{document.sha256[:12]}",
            source=self._metadata(document, "English"),
            data=CandidateConstraintsData(
                language="English",
                constraints=[
                    CandidateConstraint(
                        constraint_id="permanent-contract",
                        category=ConstraintCategory.EMPLOYMENT_DURATION,
                        requirement="Employment contract must be permanent",
                        priority=ConstraintPriority.MANDATORY,
                        operator=ComparisonOperator.REQUIRED,
                        expected_value="permanent",
                        source=SourceLocation(
                            line_start=1,
                            excerpt="contract must be permanent",
                        ),
                    )
                ],
            ),
        )


def test_workflow_generates_then_reuses_without_agent_calls(tmp_path: Path) -> None:
    cv = tmp_path / "cv.txt"
    constraints = tmp_path / "constraints.txt"
    cv.write_text(
        "Senior Software Developer\n2018-present Example Systems, Senior Developer\n",
        encoding="utf-8",
    )
    constraints.write_text(
        "The employment contract must be permanent.\n",
        encoding="utf-8",
    )
    settings = Settings(
        openai_api_key="test",
        structured_data_dir=tmp_path / "structured_data",
    )
    FakeStructurer.calls = 0
    workflow = CandidateReadWorkflow(settings, structurer_factory=FakeStructurer)

    first = workflow.run(cv, constraints, authorize=lambda _: True)
    second = workflow.run(cv, constraints, authorize=lambda _: False)

    assert not first.cv.reused
    assert not first.constraints.reused
    assert first.reserved_cost_usd == "0.1234"
    assert second.cv.reused
    assert second.constraints.reused
    assert second.reserved_cost_usd == "0.0000"
    assert FakeStructurer.calls == 2


def test_changed_source_regenerates_only_that_document(tmp_path: Path) -> None:
    cv = tmp_path / "cv.txt"
    constraints = tmp_path / "constraints.txt"
    cv.write_text(
        "Senior Software Developer\n2018-present Example Systems, Senior Developer\n",
        encoding="utf-8",
    )
    constraints.write_text(
        "The employment contract must be permanent.\n",
        encoding="utf-8",
    )
    settings = Settings(
        openai_api_key="test",
        structured_data_dir=tmp_path / "structured_data",
    )
    FakeStructurer.calls = 0
    workflow = CandidateReadWorkflow(settings, structurer_factory=FakeStructurer)
    workflow.run(cv, constraints, authorize=lambda _: True)
    cv.write_text(
        "Senior Software Developer\n"
        "2018-present Example Systems, Senior Developer\n"
        "Additional detail\n",
        encoding="utf-8",
    )

    result = workflow.run(cv, constraints, authorize=lambda _: True)

    assert not result.cv.reused
    assert result.constraints.reused
    assert FakeStructurer.calls == 3
