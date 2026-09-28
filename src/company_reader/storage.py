from pathlib import Path

from candidate_reader.storage import atomic_write_model, load_cached
from company_reader.models import CompanyOffer


def company_offer_path(directory: Path) -> Path:
    return directory / "company_offer.json"


def load_cached_offer(path: Path, expected_sha256: str) -> CompanyOffer | None:
    return load_cached(path, CompanyOffer, expected_sha256)


def write_company_offer(path: Path, offer: CompanyOffer) -> None:
    atomic_write_model(path, offer)
