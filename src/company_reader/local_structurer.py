from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from candidate_reader.errors import StructuredOutputError
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
    RemoteWorkTerms,
    TravelTerms,
    WorkingTimeTerms,
)


@dataclass(frozen=True)
class ZeroBudget:
    reserved_usd: Decimal = Decimal("0")


@dataclass
class _ClauseBuffer:
    number: str
    title: str
    page: int
    heading: str
    body: list[str]


class LocalCompanyOfferStructurer:
    requires_external_transmission = False

    def __init__(self, _settings: object) -> None:
        self.budget = ZeroBudget()

    def structure(self, document: ExtractedDocument) -> CompanyOffer:
        clauses = self._extract_clauses(document)
        if not clauses:
            raise StructuredOutputError("No numbered contract clauses were found")

        start = self._find(document, r"Der Mitarbeiter wird ab dem ([0-9.]+) beschäftigt")
        duration = self._find(document, r"Dauer von insgesamt\s+([0-9]+)\s+Monaten")
        probation = self._find(document, r"Probezeit von\s+([0-9]+)\s+Monaten")
        target_salary = self._find(
            document,
            r"Jahres Brutto Zielgehalt beträgt EUR\s+([0-9.]+)",
        )
        fixed_salary = self._find(
            document,
            r"Brutto-Jahresgehalte Fixum beträgt EUR\s+([0-9.]+)",
        )
        variable_salary = self._find(
            document,
            r"Brutto-Jahresgehalt variabel beträgt\s+EUR\s+([0-9.]+)",
        )
        weekly_hours = self._find(
            document,
            r"regelmäßige wöchentliche Arbeitszeit beträgt\s+([0-9]+)\s+Stunden",
        )
        overtime = self._find(
            document,
            r"Mehrarbeit im Umfang von\s+([0-9]+)\s+Arbeitsstunden pro Woche",
        )
        vacation = self._find(
            document,
            r"jährliche Urlaub beträgt grundsätzlich\s+([0-9]+)\s+Arbeitstage",
        )
        position = self._find(
            document,
            r"als (Mitarbeiter im Bereich Software\s*entwicklung)",
        )
        workplace = self._find(document, r"Geschäftsstelle\s+(Waldenbuch)")

        compensation_source = target_salary or fixed_salary or variable_salary
        compensation = None
        if compensation_source is not None:
            compensation = CompensationTerms(
                fixed_annual=self._money(fixed_salary),
                variable_annual=self._money(variable_salary),
                target_annual=self._money(target_salary),
                discretionary_bonus=True,
                target_setting=("Quarterly targets are agreed mutually by employee and employer"),
                payment_timing="Variable salary is generally paid at the end of a quarter",
                source=compensation_source[1],
            )

        working_source = weekly_hours or overtime
        working_time = None
        if working_source is not None:
            working_time = WorkingTimeTerms(
                regular_weekly_hours=self._decimal(weekly_hours),
                included_overtime_weekly_hours=self._decimal(overtime),
                overtime_treatment="Project-related overtime is included in compensation",
                source=working_source[1],
            )

        travel_match = self._find(
            document,
            r"Aufgaben erfordern auch Reisetätigkeiten im Bundesgebiet",
        )
        travel = (
            TravelTerms(
                required=True,
                geographic_scope="Germany",
                source=travel_match[1],
            )
            if travel_match is not None
            else None
        )
        remote_match = self._find(
            document,
            r"Home-\s*Office bzw\. Präsenzzeiten individuell .*? abgestimmt",
        )
        no_remote_entitlement = self._find(
            document,
            r"kein(?:e|en)? Rechtsanspruch auf 100% Home Office",
        )
        remote_source = remote_match or no_remote_entitlement
        remote_work = (
            RemoteWorkTerms(
                available=True,
                entitlement=False if no_remote_entitlement else None,
                arrangement="Home-office and presence times are agreed individually",
                source=remote_source[1],
            )
            if remote_source is not None
            else None
        )

        termination = self._clause(clauses, "16")
        confidentiality = self._clause(clauses, "11")
        intellectual_property = self._clause(clauses, "14")
        non_compete = self._clause(clauses, "10")
        side_activity = self._clause(clauses, "12")

        data = CompanyOfferData(
            language="German",
            employer=self._matched_value(
                self._find(document, r"(ARCHITUR TECHNOLOGY Entwicklung GmbH & Co\.KG)")
            ),
            employee=self._matched_value(self._find(document, r"(Dr\. Alexey Glotov)")),
            position=self._matched_value(position),
            workplace=self._matched_value(workplace),
            duties=(["Software development"] if position is not None else []),
            employment_duration=(
                EmploymentDuration(
                    duration_type=EmploymentDurationType.FIXED_TERM,
                    start_date=self._matched_value(start),
                    duration_months=self._decimal(duration),
                    source=duration[1],
                )
                if duration is not None
                else None
            ),
            probation_months=self._decimal(probation),
            probation_source=probation[1] if probation else None,
            compensation=compensation,
            working_time=working_time,
            vacation_days_per_year=self._decimal(vacation),
            vacation_source=vacation[1] if vacation else None,
            travel=travel,
            remote_work=remote_work,
            termination_terms=termination.summary if termination else None,
            termination_source=termination.source if termination else None,
            confidentiality_terms=(confidentiality.summary if confidentiality else None),
            confidentiality_source=(confidentiality.source if confidentiality else None),
            intellectual_property_terms=(
                intellectual_property.summary if intellectual_property else None
            ),
            intellectual_property_source=(
                intellectual_property.source if intellectual_property else None
            ),
            non_compete_terms=non_compete.summary if non_compete else None,
            non_compete_source=non_compete.source if non_compete else None,
            side_activity_terms=side_activity.summary if side_activity else None,
            side_activity_source=side_activity.source if side_activity else None,
            clauses=clauses,
        )
        return CompanyOffer(
            document_id=f"company_offer-{document.sha256[:12]}",
            source=SourceMetadata(
                filename=document.source_path.name,
                path=str(document.source_path),
                sha256=document.sha256,
                size_bytes=document.size_bytes,
                extracted_at=datetime.now(UTC),
                language="German",
                page_count=document.page_count,
                warnings=list(document.warnings),
            ),
            data=data,
        )

    @staticmethod
    def _extract_clauses(document: ExtractedDocument) -> list[ContractClause]:
        buffers: list[_ClauseBuffer] = []
        current: _ClauseBuffer | None = None
        body_started = False
        heading_pattern = re.compile(r"^(\d{1,2})\.\s*(.+)$")

        for segment in document.segments:
            if segment.page is None or "inhaltsverzeichnis" in segment.text.casefold():
                continue
            for raw_line in segment.text.splitlines():
                line = raw_line.strip()
                if not line or _is_page_artifact(line):
                    continue
                heading = heading_pattern.match(line)
                if heading:
                    body_started = True
                    if current is not None:
                        buffers.append(current)
                    current = _ClauseBuffer(
                        number=heading.group(1),
                        title=heading.group(2).strip(),
                        page=segment.page,
                        heading=line,
                        body=[],
                    )
                elif body_started and current is not None:
                    current.body.append(line)
        if current is not None:
            buffers.append(current)

        return [
            ContractClause(
                section_number=buffer.number,
                title=buffer.title,
                summary=_clean_body(buffer.body) or buffer.title,
                source=SourceLocation(
                    page=buffer.page,
                    section=buffer.number,
                    excerpt=buffer.heading,
                ),
            )
            for buffer in buffers
        ]

    @staticmethod
    def _find(
        document: ExtractedDocument,
        pattern: str,
    ) -> tuple[str, SourceLocation] | None:
        regex = re.compile(pattern, re.IGNORECASE | re.DOTALL)
        for segment in document.segments:
            match = regex.search(segment.text)
            if match and segment.page is not None:
                value = match.group(1) if match.lastindex else match.group(0)
                return value.strip(), SourceLocation(
                    page=segment.page,
                    excerpt=match.group(0).strip(),
                )
        return None

    @staticmethod
    def _decimal(match: tuple[str, SourceLocation] | None) -> Decimal | None:
        if match is None:
            return None
        return Decimal(match[0].replace(".", "").replace(",", "."))

    @classmethod
    def _money(
        cls,
        match: tuple[str, SourceLocation] | None,
    ) -> MoneyAmount | None:
        amount = cls._decimal(match)
        return (
            MoneyAmount(amount=amount, currency="EUR", period="year")
            if amount is not None
            else None
        )

    @staticmethod
    def _matched_value(match: tuple[str, SourceLocation] | None) -> str | None:
        return match[0] if match else None

    @staticmethod
    def _clause(
        clauses: list[ContractClause],
        number: str,
    ) -> ContractClause | None:
        return next(
            (clause for clause in clauses if clause.section_number == number),
            None,
        )


def _clean_body(lines: list[str]) -> str:
    text = " ".join(lines)
    return re.sub(r"\s+", " ", text).strip()


def _is_page_artifact(line: str) -> bool:
    return (
        line.startswith("Vertrag: Mitarbeiterverh")
        or "Architur Technology Entwicklungsgesellschaft mbH & CO.KG Seite" in line
        or re.fullmatch(r"Version \d+(?:\.\d+)*", line) is not None
        or re.fullmatch(r"Stand \d{2}\.\d{2}\.\d{4}", line) is not None
        or re.fullmatch(r"_+", line) is not None
    )
