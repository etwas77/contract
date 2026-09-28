from __future__ import annotations

import os
import tempfile
from pathlib import Path

from pydantic import ValidationError

from candidate_reader.errors import StorageError
from comparison.models import ComparisonReportData


def load_cached_result(
    path: Path,
    cache_identity: str,
) -> ComparisonReportData | None:
    if not path.is_file():
        return None
    try:
        result = ComparisonReportData.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError):
        return None
    return result if result.cache_identity == cache_identity else None


def write_outputs(
    data_path: Path,
    report_path: Path,
    data: ComparisonReportData,
    report: str,
) -> None:
    data_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    data_temp = _write_temp(data_path, data.model_dump_json(indent=2) + "\n")
    report_temp = _write_temp(report_path, report)
    try:
        data_temp.replace(data_path)
        report_temp.replace(report_path)
    except OSError as exc:
        raise StorageError(f"Cannot replace comparison outputs: {exc}") from exc
    finally:
        data_temp.unlink(missing_ok=True)
        report_temp.unlink(missing_ok=True)
    try:
        ComparisonReportData.model_validate_json(data_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError) as exc:
        raise StorageError("Written comparison data failed validation") from exc


def write_report(path: Path, report: str) -> None:
    temporary = _write_temp(path, report)
    try:
        temporary.replace(path)
    except OSError as exc:
        raise StorageError(f"Cannot write report '{path}': {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _write_temp(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
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
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            return Path(temporary.name)
    except OSError as exc:
        raise StorageError(f"Cannot prepare output '{path}': {exc}") from exc
