from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from candidate_reader.errors import StorageError
from candidate_reader.models import CandidateConstraints, CandidateCV
from company_reader.models import CompanyOffer

ModelT = TypeVar("ModelT", bound=BaseModel)


def load_model(path: Path, model_type: type[ModelT]) -> ModelT:
    try:
        return model_type.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StorageError(f"Structured input does not exist: {path}") from exc
    except (OSError, UnicodeDecodeError, ValidationError) as exc:
        raise StorageError(f"Cannot load valid structured input '{path}': {exc}") from exc


def load_inputs(
    cv_path: Path,
    constraints_path: Path,
    offer_path: Path,
) -> tuple[CandidateCV, CandidateConstraints, CompanyOffer]:
    return (
        load_model(cv_path, CandidateCV),
        load_model(constraints_path, CandidateConstraints),
        load_model(offer_path, CompanyOffer),
    )
