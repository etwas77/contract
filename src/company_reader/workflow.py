from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Protocol

from candidate_reader.config import Settings
from candidate_reader.errors import ConfigurationError
from candidate_reader.extractors import ExtractedDocument, extract_document
from candidate_reader.models import DocumentType
from company_reader.local_structurer import LocalCompanyOfferStructurer
from company_reader.models import CompanyOffer
from company_reader.storage import company_offer_path, load_cached_offer, write_company_offer


class StructurerBudget(Protocol):
    reserved_usd: Decimal


class OfferStructurer(Protocol):
    requires_external_transmission: bool

    @property
    def budget(self) -> StructurerBudget: ...

    def structure(self, document: ExtractedDocument) -> CompanyOffer: ...


@dataclass(frozen=True)
class CompanyReadResult:
    offer: CompanyOffer
    output_path: Path
    reused: bool
    reserved_cost_usd: str


class CompanyReadWorkflow:
    def __init__(
        self,
        settings: Settings,
        structurer_factory: Callable[[Settings], OfferStructurer] | None = None,
    ) -> None:
        self._settings = settings
        self._structurer_factory = structurer_factory or LocalCompanyOfferStructurer

    def run(
        self,
        offer_path: Path,
        *,
        force: bool = False,
        authorize: Callable[[str], bool],
    ) -> CompanyReadResult:
        source = extract_document(offer_path, DocumentType.COMPANY_OFFER)
        target = company_offer_path(self._settings.structured_data_dir)
        cached = None if force else load_cached_offer(target, source.sha256)
        if cached is not None:
            return CompanyReadResult(cached, target, reused=True, reserved_cost_usd="0.0000")

        structurer = self._structurer_factory(self._settings)
        if structurer.requires_external_transmission and not authorize(source.source_path.name):
            raise ConfigurationError("Company offer transmission was not authorized")
        generated = structurer.structure(source)
        write_company_offer(target, generated)
        reserved = structurer.budget.reserved_usd
        return CompanyReadResult(
            generated,
            target,
            reused=False,
            reserved_cost_usd=f"{reserved:.4f}",
        )
