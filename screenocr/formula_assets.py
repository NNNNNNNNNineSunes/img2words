"""Stage-only read-only verification of pinned formula OCR assets (V31-M2A).

Import is pure: no filesystem access, no model probing, no heavy imports.
``verify_assets`` reads only a caller-chosen trusted local absolute directory and never
derives a path from manifest content.  Success means the four constant asset files are
regular local files with the pinned byte length and streamed SHA-256; it does NOT
install, download, or enable formula OCR, nor assert recognition accuracy.

Repair1 order: lexical drive-qualified absolute spelling checked before any FS call;
directory walked root-to-leaf so a reparse ancestor is refused before any descendant;
manifest and each asset leaf lstat-checked regular/non-link before open; reads capped at
pinned size + 1; cancel/deadline stop-checks around every phase.  A mapped remote drive
stays the caller's responsibility; syntax cannot prove the drive is not mapped.
"""
from __future__ import annotations

import hashlib, json, math, ntpath, os, stat, time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

__all__ = ["MANIFEST_NAME", "MANIFEST_MAX_BYTES", "AssetsUnavailable", "AssetsCancelled",
           "AssetsTimedOut", "AssetPaths", "manifest_bytes", "verify_assets"]

MANIFEST_NAME, MANIFEST_MAX_BYTES, MANIFEST_VERSION = "formula-assets.json", 65536, 1
ENGINE, ENGINE_VERSION, ASSET_RELEASE = "rapid-latex-ocr", "0.0.9", "v0.0.0"
MANIFEST_KEYS = frozenset({"v", "engine", "engine_version", "asset_release", "complete", "files"})
RECORD_KEYS = frozenset({"bytes", "sha256"})
CHUNK = 1 << 20
_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_HEX = frozenset("0123456789ABCDEF")
_DEVICES = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
    | {f"{base}{sup}" for base in ("COM", "LPT") for sup in ("\u00b9", "\u00b2", "\u00b3")})
_BAD_SPELLING = "model_dir must be an absolute local path with a drive letter"

# Pinned inventory (design-formula-setup.md 11-14): local SHA-256 of the four known G0
# files, not a publisher signature/digest, and no license or redistribution inference.
_SPECS: Mapping[str, Mapping[str, Any]] = {
    "image_resizer.onnx": {"bytes": 38967751, "sha256": "E0B075C39700F64D50400F39C8FC186BBB3B5D84D31864008313F376603ACA9D"},
    "encoder.onnx": {"bytes": 89008136, "sha256": "01BF5DC25539CA0CD5B1BD29296EA495977A6BA5F629DC4178277809D26E5E7D"},
    "decoder.onnx": {"bytes": 50952726, "sha256": "BD695497BF1B22279B7626F5916C79226E1E244C84355F8DA7EDFD2D921D0072"},
    "tokenizer.json": {"bytes": 24174, "sha256": "1DC27B18D6A518D0D5FF3F4BB7BD98521FE80AD39E5B2A246D4109F1BB9D5019"},
}
_HINT = (f"formula assets are not ready and formula stays disabled; run the manual "
         f"setup-formula.ps1 to verify the four pinned assets and write {MANIFEST_NAME}")


class AssetsUnavailable(RuntimeError):
    """Assets are missing, unreadable, incomplete, or do not match the pins."""


class AssetsCancelled(AssetsUnavailable):
    """Verification was cancelled before completion; no result was published."""


class AssetsTimedOut(AssetsUnavailable):
    """The absolute deadline passed; no result was published."""


@dataclass(frozen=True)
class AssetPaths:
    """Absolute local paths of the four verified pinned assets."""

    resizer: str
    encoder: str
    decoder: str
    tokenizer: str


def manifest_bytes() -> bytes:
    """Return the deterministic completion manifest for the pinned inventory."""
    document = {
        "v": MANIFEST_VERSION, "engine": ENGINE, "engine_version": ENGINE_VERSION,
        "asset_release": ASSET_RELEASE, "complete": True,
        "files": {name: {"bytes": int(spec["bytes"]), "sha256": str(spec["sha256"])}
                  for name, spec in _SPECS.items()},
    }
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _unavailable(detail: str) -> AssetsUnavailable:
    return AssetsUnavailable(f"{detail}; {_HINT}")


def _pairs(pairs: list) -> dict:
    """object_pairs_hook: reject duplicate keys instead of silently overwriting."""
    out: dict = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key {key!r}")
        out[key] = value
    return out


def _reject_constant(token: str) -> Any:
    """parse_constant hook: NaN/Infinity/-Infinity are never valid manifest numbers."""
    raise ValueError(f"non-finite number {token!r}")


def _is_link(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & _REPARSE)


def _stop(cancel_event: Any, deadline: Optional[float]) -> None:
    """Cooperative checkpoint: never interrupted mid-read, only between phases."""
    if cancel_event is not None and cancel_event.is_set():
        raise AssetsCancelled("asset verification was cancelled")
    if deadline is not None and time.monotonic() >= deadline:
        raise AssetsTimedOut("asset verification deadline expired")


def _lexical_dir(model_dir: Any) -> str:
    """Reject any spelling that is not a drive-qualified absolute local path (no FS call)."""
    if not isinstance(model_dir, str) or not model_dir:
        raise _unavailable("model_dir must be a non-empty absolute local path with a drive letter")
    if "\x00" in model_dir:
        raise _unavailable("model_dir must not contain a NUL character")
    drive = ntpath.splitdrive(model_dir)[0]
    if model_dir.startswith(("\\\\", "//")) or drive.startswith(("\\\\", "//")):
        raise _unavailable("UNC/network model directories are not accepted")
    if (len(drive) != 2 or not drive[0].isascii() or not drive[0].isalpha()
            or drive[1] != ":" or not ntpath.isabs(model_dir)):
        raise _unavailable(_BAD_SPELLING)
    for part in model_dir.replace("/", "\\").split("\\"):
        # Screen the RAW slash-normalized components, before ntpath.normpath can
        # collapse ``CONIN$\..`` away, and trim the stem's trailing spaces/dots so
        # ``CON ``/``CON.`` aliases cannot slip through as ordinary names.
        if part.split(".")[0].rstrip(" .").upper() in _DEVICES:
            raise _unavailable("model_dir must not name a reserved device")
    return ntpath.abspath(model_dir)


def _walk_dir(directory: str, cancel_event: Any, deadline: Optional[float]) -> None:
    """Stop-check every component from the drive root to the leaf; refuse links/non-dirs."""
    drive, rest = ntpath.splitdrive(directory)
    parts = [part for part in rest.replace("/", "\\").split("\\") if part]
    root = drive + "\\"
    chain = [root] + [ntpath.join(root, *parts[:index + 1]) for index in range(len(parts))]
    for index, component in enumerate(chain):
        leaf = index == len(chain) - 1
        _stop(cancel_event, deadline)
        try:
            info = os.lstat(component)
        except OSError as exc:
            what = "model directory is" if leaf else "model directory path is"
            raise _unavailable(f"{what} not readable ({exc.strerror or exc})") from None
        _stop(cancel_event, deadline)
        if _is_link(info):
            raise _unavailable("model directory is a reparse point or link; refusing to follow it"
                               if leaf else
                               "model directory path contains a reparse point or link")
        if not stat.S_ISDIR(info.st_mode):
            raise _unavailable("model_dir is not a directory" if leaf else
                               "model directory path component is not a directory")


def _read_manifest(path: str, cancel_event: Any, deadline: Optional[float]) -> Any:
    """Stat the constant manifest leaf, then bounded-read it and strictly parse it."""
    _stop(cancel_event, deadline)
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise _unavailable(f"{MANIFEST_NAME} is missing or unreadable "
                           f"({exc.strerror or exc})") from None
    _stop(cancel_event, deadline)
    if _is_link(info):
        raise _unavailable(f"{MANIFEST_NAME} is a reparse point or link; refusing to follow it")
    if not stat.S_ISREG(info.st_mode):
        raise _unavailable(f"{MANIFEST_NAME} is not a regular file")
    try:
        with open(path, "rb") as handle:
            _stop(cancel_event, deadline)
            blob = handle.read(MANIFEST_MAX_BYTES + 1)
            _stop(cancel_event, deadline)
    except AssetsUnavailable:
        raise
    except OSError as exc:
        raise _unavailable(f"{MANIFEST_NAME} is unreadable ({exc.strerror or exc})") from None
    if len(blob) > MANIFEST_MAX_BYTES:
        raise _unavailable(f"{MANIFEST_NAME} exceeds {MANIFEST_MAX_BYTES} bytes")
    try:
        return json.loads(blob.decode("utf-8"), object_pairs_hook=_pairs,
                          parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        # A deeply nested document inside the byte cap raises RecursionError, which is
        # not a ValueError; it must still surface as the typed error, not escape.
        raise _unavailable(f"{MANIFEST_NAME} is malformed ({exc})") from None


def _check_schema(document: Any) -> Mapping[str, Mapping[str, Any]]:
    """Enforce the exact v1 record: no unknown, duplicate, or mistyped fields."""
    def whole_int(value: Any, field: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise _unavailable(f"{MANIFEST_NAME} field {field} must be an integer")
        return value

    def digest(value: Any, field: str) -> str:
        if not isinstance(value, str) or len(value) != 64 or not set(value) <= _HEX:
            raise _unavailable(f"{MANIFEST_NAME} field {field} must be 64 uppercase hex chars")
        return value

    if not isinstance(document, dict):
        raise _unavailable(f"{MANIFEST_NAME} must contain a JSON object")
    if set(document) != MANIFEST_KEYS:
        raise _unavailable(f"{MANIFEST_NAME} keys must be exactly {sorted(MANIFEST_KEYS)}")
    if whole_int(document["v"], "v") != MANIFEST_VERSION:
        raise _unavailable(f"{MANIFEST_NAME} v must be {MANIFEST_VERSION}")
    if document["engine"] != ENGINE or document["engine_version"] != ENGINE_VERSION:
        raise _unavailable(f"{MANIFEST_NAME} engine must be {ENGINE} {ENGINE_VERSION}")
    if document["asset_release"] != ASSET_RELEASE:
        raise _unavailable(f"{MANIFEST_NAME} asset_release must be {ASSET_RELEASE}")
    if document["complete"] is not True:
        raise _unavailable(f"{MANIFEST_NAME} must declare complete=true")
    files = document["files"]
    if not isinstance(files, dict) or set(files) != set(_SPECS):
        raise _unavailable(f"{MANIFEST_NAME} files must be exactly {sorted(_SPECS)}")
    records: dict = {}
    for name, record in files.items():
        if not isinstance(record, dict) or set(record) != RECORD_KEYS:
            raise _unavailable(f"record {name} keys must be {sorted(RECORD_KEYS)}")
        records[name] = {"bytes": whole_int(record["bytes"], f"files.{name}.bytes"),
                         "sha256": digest(record["sha256"], f"files.{name}.sha256")}
    return records


def _stream_digest(name: str, path: str, expected_bytes: int, cancel_event: Any,
                   deadline: Optional[float]) -> tuple:
    """Stat the pinned size, then read at most ``expected_bytes + 1`` bytes and hash."""
    _stop(cancel_event, deadline)
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise _unavailable(f"missing asset ({exc.strerror or exc})") from None
    _stop(cancel_event, deadline)
    if _is_link(info):
        raise _unavailable("asset is a reparse point or link; refusing to follow it")
    if not stat.S_ISREG(info.st_mode):
        raise _unavailable("asset is not a regular file")
    if info.st_size != expected_bytes:
        raise _unavailable(f"{name} size {info.st_size} does not match pinned {expected_bytes}")
    hasher, count, limit = hashlib.sha256(), 0, expected_bytes + 1
    try:
        with open(path, "rb") as handle:
            _stop(cancel_event, deadline)
            while count < limit:
                _stop(cancel_event, deadline)
                chunk = handle.read(min(CHUNK, limit - count))
                if not chunk:
                    break
                count += len(chunk)
                hasher.update(chunk)
                _stop(cancel_event, deadline)
    except AssetsUnavailable:
        raise
    except OSError as exc:
        raise _unavailable(f"asset is unreadable ({exc.strerror or exc})") from None
    if count != expected_bytes:
        raise _unavailable(f"{name} size {count} does not match pinned {expected_bytes}")
    return count, hasher.hexdigest().upper()


def verify_assets(model_dir: str, *, require_manifest: bool = True, cancel_event: Any = None,
                  absolute_deadline: Optional[float] = None) -> AssetPaths:
    """Verify the four pinned assets inside a trusted local absolute directory.

    ``require_manifest=True`` is the runtime default and demands a complete manifest;
    ``require_manifest=False`` exists only for the explicit setup command verifying
    local source bytes before it writes a manifest.  A passed deadline is never
    extended, and nothing is written, downloaded, installed, or enabled here.
    """
    if not isinstance(require_manifest, bool):
        raise _unavailable("require_manifest must be a bool")
    if absolute_deadline is not None:
        if (isinstance(absolute_deadline, bool)
                or not isinstance(absolute_deadline, (int, float))):
            raise _unavailable("absolute_deadline must be a finite monotonic timestamp or None")
        try:
            absolute_deadline = float(absolute_deadline)
        except OverflowError:
            # An int beyond float range (e.g. 10**5000) is not a usable deadline.
            # Never interpolate it: Python refuses to render ints past 4300 digits.
            raise _unavailable("absolute_deadline must be a finite monotonic timestamp "
                               "within float range or None") from None
        if not math.isfinite(absolute_deadline):
            raise _unavailable("absolute_deadline must be a finite monotonic timestamp or None")
    _stop(cancel_event, absolute_deadline)
    directory = _lexical_dir(model_dir)
    _walk_dir(directory, cancel_event, absolute_deadline)
    paths = {name: ntpath.join(directory, name) for name in _SPECS}
    _stop(cancel_event, absolute_deadline)
    records = None
    if require_manifest:
        records = _check_schema(_read_manifest(ntpath.join(directory, MANIFEST_NAME),
                                               cancel_event, absolute_deadline))
        _stop(cancel_event, absolute_deadline)
    for name, path in paths.items():
        _stop(cancel_event, absolute_deadline)
        pinned_bytes, pinned_digest = int(_SPECS[name]["bytes"]), str(_SPECS[name]["sha256"])
        count, digest = _stream_digest(name, path, pinned_bytes, cancel_event, absolute_deadline)
        if digest != pinned_digest:
            raise _unavailable(f"{name} sha256 does not match the pinned local digest")
        if records is not None:
            if records[name]["bytes"] != count:
                raise _unavailable(f"{name} does not match the manifest byte count")
            if records[name]["sha256"] != digest:
                raise _unavailable(f"{name} does not match the manifest sha256")
        _stop(cancel_event, absolute_deadline)
    _stop(cancel_event, absolute_deadline)
    return AssetPaths(resizer=paths["image_resizer.onnx"], encoder=paths["encoder.onnx"],
                      decoder=paths["decoder.onnx"], tokenizer=paths["tokenizer.json"])
