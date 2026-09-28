from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from candidate_reader.errors import StorageError
from candidate_reader.models import SCHEMA_VERSION, DocumentType

ModelT = TypeVar("ModelT", bound=BaseModel)


def output_path(directory: Path, document_type: DocumentType) -> Path:
    filename = {
        DocumentType.CV: "candidate_cv.json",
        DocumentType.CONSTRAINTS: "candidate_constraints.json",
    }[document_type]
    return directory / filename


def load_cached(
    path: Path,
    model_type: type[ModelT],
    expected_sha256: str,
) -> ModelT | None:
    if not path.is_file():
        return None
    try:
        model = model_type.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError):
        return None
    source = getattr(model, "source", None)
    schema_version = getattr(model, "schema_version", None)
    if source is None or source.sha256 != expected_sha256 or schema_version != SCHEMA_VERSION:
        return None
    return model


def atomic_write_model(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(model.model_dump_json(indent=2))
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(path)
    except OSError as exc:
        raise StorageError(f"Cannot write structured data '{path}': {exc}") from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink(missing_ok=True)

    try:
        type(model).model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError) as exc:
        raise StorageError(f"Written structured data failed validation: {path}") from exc
