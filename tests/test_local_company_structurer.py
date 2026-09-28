from __future__ import annotations

from pathlib import Path

from candidate_reader.extractors import ExtractedDocument, ExtractedSegment
from candidate_reader.models import DocumentType
from company_reader.local_structurer import LocalCompanyOfferStructurer
from company_reader.models import EmploymentDurationType


def test_local_structurer_parses_numbered_offer() -> None:
    document = ExtractedDocument(
        source_path=Path("offer.pdf"),
        document_type=DocumentType.COMPANY_OFFER,
        sha256="a" * 64,
        size_bytes=100,
        segments=(
            ExtractedSegment(
                page=1,
                text="Cover page",
            ),
            ExtractedSegment(
                page=2,
                text=(
                    "1. Beginn der Tätigkeit\n"
                    "Der Mitarbeiter wird ab dem 01.01.2030 beschäftigt.\n"
                    "2. Art der Tätigkeit\n"
                    "Er ist als Mitarbeiter im Bereich Software entwicklung "
                    "in der Geschäftsstelle Waldenbuch beschäftigt.\n"
                    "3. Einkommen\n"
                    "Das Jahres Brutto Zielgehalt beträgt EUR 70.000.\n"
                    "Das Brutto-Jahresgehalte Fixum beträgt EUR 55.000.\n"
                    "Das Brutto-Jahresgehalt variabel beträgt EUR 15.000.\n"
                    "6. Arbeitszeit\n"
                    "Die regelmäßige wöchentliche Arbeitszeit beträgt 40 Stunden.\n"
                    "7. Urlaub\n"
                    "Der jährliche Urlaub beträgt grundsätzlich 30 Arbeitstage.\n"
                    "Vertrag: Mitarbeiterverhältnis\n"
                    "© Architur Technology Entwicklungsgesellschaft mbH & CO.KG Seite 2\n"
                    "Version 2.25\n"
                    "Stand 25.09.2026\n"
                ),
            ),
            ExtractedSegment(
                page=3,
                text=(
                    "16. Beendigung des Arbeitsverhältnisses\n"
                    "Es wird eine Probezeit von 6 Monaten vereinbart. "
                    "Der Vertrag hat eine Dauer von insgesamt 24 Monaten.\n"
                ),
            ),
        ),
    )

    offer = LocalCompanyOfferStructurer(object()).structure(document)

    assert len(offer.data.clauses) == 6
    assert offer.data.position == "Mitarbeiter im Bereich Software entwicklung"
    assert offer.data.employment_duration is not None
    assert offer.data.employment_duration.duration_type == EmploymentDurationType.FIXED_TERM
    assert offer.data.employment_duration.duration_months == 24
    assert offer.data.probation_months == 6
    assert offer.data.compensation is not None
    assert offer.data.compensation.target_annual is not None
    assert offer.data.compensation.target_annual.amount == 70000
    assert offer.data.working_time is not None
    assert offer.data.working_time.regular_weekly_hours == 40
    assert offer.data.vacation_days_per_year == 30
    assert "Version 2.25" not in offer.data.clauses[4].summary
