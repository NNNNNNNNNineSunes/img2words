"""Small, durable storage for recognized OCR text."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_HISTORY_PATH = Path(__file__).resolve().parent.parent / "data" / "history.json"
_MISSING = object()


class HistoryError(RuntimeError):
    """Raised when OCR history cannot be read or safely written."""


@dataclass(frozen=True)
class HistoryEntry:
    """A stored OCR result with stable identity and optional capture time."""

    id: str
    text: str
    created_at: str | None


class HistoryStore:
    """Persist bounded OCR history in oldest-to-newest order."""

    def __init__(self, path: Path | str | None = None, max_entries: int = 100):
        if type(max_entries) is not int or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        self.path = Path(path) if path is not None else DEFAULT_HISTORY_PATH
        self.max_entries = max_entries
        self._lock = threading.RLock()

    def load_records(self) -> list[HistoryEntry]:
        with self._lock:
            value = self._read_value()
            if value is _MISSING:
                return []
            if isinstance(value, list):
                if any(not isinstance(text, str) for text in value):
                    raise HistoryError("Legacy OCR history must be a JSON list of strings")
                return [
                    HistoryEntry(
                        id=uuid.uuid5(uuid.NAMESPACE_URL, f"screenocr-legacy:{index}:{text}").hex,
                        text=text,
                        created_at=None,
                    )
                    for index, text in enumerate(value)
                ]
            if (
                not isinstance(value, dict)
                or set(value) != {"version", "records"}
                or type(value["version"]) is not int
                or value["version"] != 2
                or not isinstance(value["records"], list)
            ):
                raise HistoryError("OCR history must use the supported version 2 schema")

            records: list[HistoryEntry] = []
            seen_ids: set[str] = set()
            for item in value["records"]:
                if not isinstance(item, dict) or set(item) != {"id", "text", "created_at"}:
                    raise HistoryError("Each OCR history record must contain id, text, and created_at")
                record_id, text, created_at = item["id"], item["text"], item["created_at"]
                if not isinstance(record_id, str) or not record_id:
                    raise HistoryError("OCR history record id must be a non-empty string")
                if record_id in seen_ids:
                    raise HistoryError("OCR history record ids must be unique")
                if not isinstance(text, str):
                    raise HistoryError("OCR history record text must be a string")
                if created_at is not None and not self._valid_timestamp(created_at):
                    raise HistoryError("OCR history created_at must be a timezone-aware ISO timestamp or null")
                seen_ids.add(record_id)
                records.append(HistoryEntry(record_id, text, created_at))
            return records

    def load(self) -> list[str]:
        return [record.text for record in self.load_records()]

    def append(self, text: str) -> bool:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text.strip():
            return False
        with self._lock:
            records = self.load_records()
            created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            records.append(HistoryEntry(uuid.uuid4().hex, text, created_at))
            self._write(records[-self.max_entries :])
        return True

    def delete(self, record_id: str) -> bool:
        if not isinstance(record_id, str):
            raise TypeError("record_id must be a string")
        if not record_id:
            raise ValueError("record_id must not be empty")
        with self._lock:
            records = self.load_records()
            remaining = [record for record in records if record.id != record_id]
            if len(remaining) == len(records):
                return False
            self._write(remaining)
            return True

    def clear(self) -> None:
        with self._lock:
            self._write([])

    @staticmethod
    def _valid_timestamp(value: object) -> bool:
        if not isinstance(value, str):
            return False
        try:
            timestamp = datetime.fromisoformat(value)
        except ValueError:
            return False
        return timestamp.tzinfo is not None and timestamp.utcoffset() is not None

    def _read_value(self):
        try:
            with self.path.open("r", encoding="utf-8") as stream:
                return json.load(stream)
        except FileNotFoundError:
            return _MISSING
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise HistoryError(f"Could not read OCR history: {exc}") from exc

    def _write(self, records: list[HistoryEntry]) -> None:
        value = {
            "version": 2,
            "records": [
                {"id": record.id, "text": record.text, "created_at": record.created_at}
                for record in records
            ],
        }
        temporary_path: str | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_path = tempfile.mkstemp(
                prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
            )
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
                json.dump(value, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, self.path)
            temporary_path = None
        except (OSError, UnicodeError) as exc:
            raise HistoryError(f"Could not write OCR history: {exc}") from exc
        finally:
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass
