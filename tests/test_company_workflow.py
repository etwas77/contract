from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from candidate_reader.config import Settings
from candidate_reader.extractors import ExtractedDocument
from candidate_reader.models import SourceLocation, SourceMetadata
from company_reader.models import (
    CompanyOffer,
    CompanyOfferData,
    CompensationTerms,
    ContractClause,
    EmploymentDuration,
    EmploymentDurationType,
    MoneyAmount,
)
from company_reader.workflow import CompanyReadWorkflow


class FakeBudget:
    reserved_usd = Decimal("0.0750")


class FakeOfferStructurer:
    calls = 0
    requires_external_transmission = False

    def __init__(self, settings: Settings) -> None:
        del settings
        self._budget = FakeBudget()

    @property
    def budget(self) -> FakeBudget:
        return self._budget

    def structure(self, document: ExtractedDocument) -> CompanyOffer:
        type(self).calls += 1
        return CompanyOffer(
            document_id=f"company_offer-{document.sha256[:12]}",
            source=SourceMetadata(
                filename=document.source_path.name,
                path=str(document.source_path),
                sha256=document.sha256,
                size_bytes=document.size_bytes,
                extracted_at=datetime.now(UTC),
                language="English",
                page_count=document.page_count,
            ),
            data=CompanyOfferData(
                language="English",
                employer="Example Company",
                position="Software Developer",
                employment_duration=EmploymentDuration(
                    duration_type=EmploymentDurationType.PERMANENT,
                    start_date="2030-01-01",
                    source=SourceLocation(
                        line_start=2,
                        excerpt="Employment starts on 1 January 2030 and is permanent",
                    ),
                ),
                compensation=CompensationTerms(
                    fixed_annual=MoneyAmount(
                        amount=Decimal("72000"),
                        currency="EUR",
                        period="year",
                    ),
                    source=SourceLocation(
                        line_start=6,
                        excerpt="fixed annual gross salary is EUR 72000",
                    ),
                ),
                clauses=[
                    ContractClause(
                        section_number="1",
                        title="Start",
                        summary="Permanent employment starts on 1 January 2030",
                        source=SourceLocation(line_start=1, excerpt="1. Start"),
                    ),
                    ContractClause(
                        section_number="3",
                        title="Compensation",
                        summary="Fixed annual gross salary is EUR 72000",
                        source=SourceLocation(line_start=5, excerpt="3. Compensation"),
                    ),
                ],
            ),
        )


def test_company_workflow_generates_then_reuses(tmp_path: Path) -> None:
    offer = tmp_path / "offer.txt"
    offer.write_text(
        "1. Start\n"
        "Employment starts on 1 January 2030 and is permanent.\n"
        "2. Role\n"
        "The employee works as a Software Developer.\n"
        "3. Compensation\n"
        "The fixed annual gross salary is EUR 72000.\n",
        encoding="utf-8",
    )
    settings = Settings(
        openai_api_key="test",
        structured_data_dir=tmp_path / "structured_data",
    )
    FakeOfferStructurer.calls = 0
    workflow = CompanyReadWorkflow(settings, structurer_factory=FakeOfferStructurer)

    first = workflow.run(offer, authorize=lambda _: True)
    second = workflow.run(offer, authorize=lambda _: False)

    assert not first.reused
    assert first.reserved_cost_usd == "0.0750"
    assert second.reused
    assert second.reserved_cost_usd == "0.0000"
    assert FakeOfferStructurer.calls == 1


def test_force_regenerates_company_offer(tmp_path: Path) -> None:
    offer = tmp_path / "offer.txt"
    offer.write_text(
        "1. Start\n"
        "Employment starts on 1 January 2030 and is permanent.\n"
        "2. Role\n"
        "The employee works as a Software Developer.\n"
        "3. Compensation\n"
        "The fixed annual gross salary is EUR 72000.\n",
        encoding="utf-8",
    )
    settings = Settings(
        openai_api_key="test",
        structured_data_dir=tmp_path / "structured_data",
    )
    FakeOfferStructurer.calls = 0
    workflow = CompanyReadWorkflow(settings, structurer_factory=FakeOfferStructurer)
    workflow.run(offer, authorize=lambda _: True)

    result = workflow.run(offer, force=True, authorize=lambda _: True)

    assert not result.reused
    assert FakeOfferStructurer.calls == 2
