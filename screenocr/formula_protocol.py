"""Strict, engine-neutral framed JSON protocol for the M1a formula worker.

Transport only.  A frame is exactly one 4-byte BIG-ENDIAN unsigned payload
length followed by a strict UTF-8 JSON object.  This module never imports an
optional ML dependency, Tk, or anything that touches the network/GPU, and it
performs no I/O at import time.

Nothing here rewrites the payload it carries: a returned ``raw_body`` is the
exact character sequence the worker produced (leading/trailing spaces, quotes,
backslashes and math spacing included).  Preserving the bytes is not a claim
that the characters are mathematically correct.
"""
from __future__ import annotations

import base64
import binascii
import json
import os
import re
import struct
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

__all__ = [
    "HEADER_SIZE",
    "MAX_REQUEST_ID_BYTES",
    "MAX_PNG_BYTES",
    "MAX_REQUEST_PAYLOAD_BYTES",
    "MAX_RAW_BODY_BYTES",
    "MAX_REPLY_PAYLOAD_BYTES",
    "MAX_ERROR_MESSAGE_BYTES",
    "ERROR_CODES",
    "SYS_PATH_PROFILES",
    "ProtocolError",
    "MetaExpectation",
    "Request",
    "Reply",
    "canonical_path",
    "validate_request_id",
    "validate_png_bytes",
    "validate_png_b64",
    "utf8_size",
    "encode_frame",
    "decode_frame",
    "build_request",
    "parse_request",
    "dumps_payload",
    "build_success",
    "build_error",
    "parse_reply",
]

HEADER_SIZE = 4
MAX_REQUEST_ID_BYTES = 64
_REQUEST_ID_RE = re.compile(r"\A[A-Za-z0-9_-]{1,64}\Z")

MAX_PNG_BYTES = 8 * 1024 * 1024
MAX_REQUEST_PAYLOAD_BYTES = 12 * 1024 * 1024
MAX_RAW_BODY_BYTES = 48 * 1024
MAX_REPLY_PAYLOAD_BYTES = 64 * 1024
MAX_ERROR_MESSAGE_BYTES = 256

STATUS_OK = "ok"
STATUS_ERROR = "error"
ERROR_CODES = (
    "unavailable",
    "invalid_input",
    "load_error",
    "inference_error",
    "worker_error",
)
SYS_PATH_PROFILES = ("stdlib_only", "formula_site_only")

REQUEST_KEYS = frozenset(("v", "request_id", "png_b64"))
SUCCESS_KEYS = frozenset(("v", "request_id", "status", "raw_body", "meta"))
ERROR_KEYS = frozenset(("v", "request_id", "status", "code", "message"))
META_KEYS = frozenset(
    (
        "python_executable",
        "isolated",
        "no_site",
        "no_bytecode",
        "utf8",
        "sys_path_profile",
        "integrity_rid",
    )
)


class ProtocolError(ValueError):
    """A frame, envelope or field violated the wire contract.

    The message is always bounded; a protocol failure is never turned into a
    copyable body.
    """

    def __init__(self, reason: str) -> None:
        text = reason if isinstance(reason, str) else repr(reason)
        text = text.replace("\x00", "\\0")
        encoded = text.encode("utf-8", "replace")
        if len(encoded) > MAX_ERROR_MESSAGE_BYTES:
            text = encoded[:MAX_ERROR_MESSAGE_BYTES].decode("utf-8", "ignore")
        super().__init__(text)


def _pairs_hook(pairs: Any) -> Dict[str, Any]:
    obj: Dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise ProtocolError(f"duplicate key {key!r}")
        obj[key] = value
    return obj


def _reject_constant(name: str) -> Any:
    raise ProtocolError(f"non-finite JSON constant {name!r} is not allowed")


def utf8_size(text: str) -> int:
    """Return the UTF-8 byte length, rejecting lone surrogates."""
    try:
        return len(text.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ProtocolError(f"string is not valid UTF-8 text: {exc}") from None


def canonical_path(path: object) -> str:
    """Return a case-folded absolute key for an interpreter/launch path.

    Comparison only: this never opens, executes or resolves content.
    """
    if not isinstance(path, str) or not path:
        raise ProtocolError("path must be a non-empty string")
    if "\x00" in path:
        raise ProtocolError("path must not contain NUL")
    # Lexical normalization only: *actual* can be untrusted child metadata, so
    # never resolve it through abspath/realpath/stat/open or any filesystem API.
    if not os.path.isabs(path):
        raise ProtocolError("path must be absolute")
    return os.path.normcase(os.path.normpath(path))


def _paths_match(actual: str, expected: str) -> bool:
    return canonical_path(actual) == canonical_path(expected)


def validate_request_id(request_id: object) -> str:
    if not isinstance(request_id, str):
        raise ProtocolError("request_id must be a string")
    if not _REQUEST_ID_RE.match(request_id):
        raise ProtocolError(
            "request_id must match [A-Za-z0-9_-]{1,64} and be ASCII"
        )
    return request_id


def validate_png_bytes(png_bytes: object) -> bytes:
    if isinstance(png_bytes, (bytes, bytearray, memoryview)):
        data = bytes(png_bytes)
    else:
        raise ProtocolError("png_bytes must be bytes")
    if not data:
        raise ProtocolError("png_bytes must not be empty")
    if len(data) > MAX_PNG_BYTES:
        raise ProtocolError(
            f"png_bytes exceeds {MAX_PNG_BYTES} bytes (got {len(data)})"
        )
    return data


def validate_png_b64(png_b64: object) -> bytes:
    """Decode canonical RFC4648 standard Base64 with re-encode equality."""
    if not isinstance(png_b64, str):
        raise ProtocolError("png_b64 must be a string")
    if not png_b64:
        raise ProtocolError("png_b64 must not be empty")
    try:
        raw = png_b64.encode("ascii")
    except UnicodeEncodeError:
        raise ProtocolError("png_b64 must be ASCII") from None
    if len(raw) % 4:
        raise ProtocolError("png_b64 length must be a multiple of 4")
    try:
        decoded = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ProtocolError(f"png_b64 is not valid standard Base64: {exc}") from None
    if base64.b64encode(decoded) != raw:
        raise ProtocolError("png_b64 is not canonical Base64 (re-encode mismatch)")
    return validate_png_bytes(decoded)


def encode_frame(payload: bytes, max_payload: int = MAX_REQUEST_PAYLOAD_BYTES) -> bytes:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ProtocolError("frame payload must be bytes")
    data = bytes(payload)
    if not data:
        raise ProtocolError("frame payload must not be zero-length")
    if len(data) > max_payload:
        raise ProtocolError(
            f"frame payload exceeds {max_payload} bytes (got {len(data)})"
        )
    return struct.pack(">I", len(data)) + data


def decode_frame(data: bytes, max_payload: int = MAX_REPLY_PAYLOAD_BYTES) -> bytes:
    """Return the single payload carried by *data*, rejecting anything else."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ProtocolError("frame must be bytes")
    buffer = bytes(data)
    if len(buffer) < HEADER_SIZE:
        raise ProtocolError("frame is truncated: fewer than 4 header bytes")
    (length,) = struct.unpack(">I", buffer[:HEADER_SIZE])
    if length == 0:
        raise ProtocolError("frame payload length must not be zero")
    if length > max_payload:
        raise ProtocolError(f"frame payload exceeds {max_payload} bytes (got {length})")
    body = buffer[HEADER_SIZE:]
    if len(body) < length:
        raise ProtocolError(
            f"frame is truncated: declared {length} bytes, received {len(body)}"
        )
    if len(body) > length:
        raise ProtocolError("frame has trailing bytes after the declared payload")
    return body


def _load_object(payload: bytes) -> Dict[str, Any]:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ProtocolError("payload must be bytes")
    buffer = bytes(payload)
    if not buffer:
        raise ProtocolError("payload must not be empty")
    if b"\x00" in buffer:
        raise ProtocolError("payload must not contain a raw NUL byte")
    try:
        text = buffer.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolError(f"payload is not valid UTF-8: {exc}") from None
    try:
        value = json.loads(
            text, object_pairs_hook=_pairs_hook, parse_constant=_reject_constant
        )
    except ProtocolError:
        raise
    except Exception as exc:
        raise ProtocolError(f"payload is not strict JSON: {exc}") from None
    if not isinstance(value, dict):
        raise ProtocolError("payload must be a JSON object")
    return value


def _exact_keys(obj: Dict[str, Any], expected: frozenset, label: str) -> None:
    keys = set(obj)
    unknown = sorted(keys - expected)
    if unknown:
        raise ProtocolError(f"{label} has unknown key(s): {', '.join(unknown)}")
    missing = sorted(expected - keys)
    if missing:
        raise ProtocolError(f"{label} is missing key(s): {', '.join(missing)}")


def _strict_version(obj: Dict[str, Any]) -> None:
    version = obj["v"]
    if type(version) is not int:
        raise ProtocolError("v must be an integer (bool is not accepted)")
    if version != 1:
        raise ProtocolError(f"v must be 1 (got {version})")


def _strict_str(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ProtocolError(f"{label} must be a string")
    return value


def _require_true_flag(value: Any, label: str) -> None:
    if value is not True:
        raise ProtocolError(f"{label} must be exactly true")


@dataclass(frozen=True)
class MetaExpectation:
    """Immutable expectation the reply metadata must satisfy exactly."""

    python_executable: str
    sys_path_profile: str
    integrity_rid: Optional[int]

    def __post_init__(self) -> None:
        if self.sys_path_profile not in SYS_PATH_PROFILES:
            raise ProtocolError(
                f"sys_path_profile must be one of {SYS_PATH_PROFILES}"
            )
        rid = self.integrity_rid
        if rid is not None and type(rid) is not int:
            raise ProtocolError("integrity_rid must be an integer or None")
        if not isinstance(self.python_executable, str) or not self.python_executable:
            raise ProtocolError("python_executable must be a non-empty string")
        if not os.path.isabs(self.python_executable):
            raise ProtocolError("python_executable must be an absolute path")
        canonical_path(self.python_executable)


@dataclass(frozen=True)
class Request:
    v: int
    request_id: str
    png_bytes: bytes


@dataclass(frozen=True)
class Reply:
    status: str
    request_id: str
    raw_body: Optional[str] = None
    code: Optional[str] = None
    message: Optional[str] = None


def dumps_payload(obj: Dict[str, Any], max_payload: int) -> bytes:
    try:
        text = json.dumps(
            obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"cannot serialize payload: {exc}") from None
    try:
        data = text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ProtocolError(f"payload is not valid UTF-8 text: {exc}") from None
    if len(data) > max_payload:
        raise ProtocolError(f"payload exceeds {max_payload} bytes (got {len(data)})")
    return data


def build_request(png_bytes: bytes, request_id: str) -> bytes:
    """Build one framed request; the caller must supply a valid ASCII id."""
    validate_request_id(request_id)
    data = validate_png_bytes(png_bytes)
    obj = {
        "v": 1,
        "request_id": request_id,
        "png_b64": base64.b64encode(data).decode("ascii"),
    }
    payload = dumps_payload(obj, MAX_REQUEST_PAYLOAD_BYTES)
    return encode_frame(payload, MAX_REQUEST_PAYLOAD_BYTES)


def parse_request(payload: bytes) -> Request:
    obj = _load_object(payload)
    _exact_keys(obj, REQUEST_KEYS, "request")
    _strict_version(obj)
    request_id = validate_request_id(obj["request_id"])
    png_bytes = validate_png_b64(obj["png_b64"])
    return Request(v=1, request_id=request_id, png_bytes=png_bytes)


def _validate_meta(meta: Any, expected: MetaExpectation) -> None:
    if not isinstance(meta, dict):
        raise ProtocolError("meta must be a JSON object")
    _exact_keys(meta, META_KEYS, "meta")
    executable = _strict_str(meta["python_executable"], "meta.python_executable")
    if not os.path.isabs(executable):
        raise ProtocolError("meta.python_executable must be an absolute path")
    if not _paths_match(executable, expected.python_executable):
        raise ProtocolError(
            "meta.python_executable does not match the immutable launch expectation"
        )
    _require_true_flag(meta["isolated"], "meta.isolated")
    _require_true_flag(meta["no_site"], "meta.no_site")
    _require_true_flag(meta["no_bytecode"], "meta.no_bytecode")
    _require_true_flag(meta["utf8"], "meta.utf8")
    profile = meta["sys_path_profile"]
    if not isinstance(profile, str) or profile not in SYS_PATH_PROFILES:
        raise ProtocolError(f"meta.sys_path_profile must be one of {SYS_PATH_PROFILES}")
    if profile != expected.sys_path_profile:
        raise ProtocolError(
            "meta.sys_path_profile does not match the immutable launch expectation"
        )
    rid = meta["integrity_rid"]
    if rid is not None and type(rid) is not int:
        raise ProtocolError("meta.integrity_rid must be an integer or null")
    if rid != expected.integrity_rid:
        raise ProtocolError(
            "meta.integrity_rid does not match the immutable launch expectation"
        )


def build_success(request_id: str, raw_body: str, meta: Dict[str, Any]) -> bytes:
    """Build one framed success reply (fixture/child side)."""
    validate_request_id(request_id)
    if not isinstance(raw_body, str):
        raise ProtocolError("raw_body must be a string")
    obj = {
        "v": 1,
        "request_id": request_id,
        "status": STATUS_OK,
        "raw_body": raw_body,
        "meta": meta,
    }
    return encode_frame(dumps_payload(obj, MAX_REPLY_PAYLOAD_BYTES), MAX_REPLY_PAYLOAD_BYTES)


def build_error(request_id: str, code: str, message: str) -> bytes:
    """Build one framed error reply (fixture/child side)."""
    validate_request_id(request_id)
    if code not in ERROR_CODES:
        raise ProtocolError(f"code must be one of {ERROR_CODES}")
    if not isinstance(message, str):
        raise ProtocolError("message must be a string")
    if "\x00" in message:
        raise ProtocolError("message must not contain NUL")
    if utf8_size(message) > MAX_ERROR_MESSAGE_BYTES:
        raise ProtocolError(
            f"message exceeds {MAX_ERROR_MESSAGE_BYTES} UTF-8 bytes"
        )
    obj = {
        "v": 1,
        "request_id": request_id,
        "status": STATUS_ERROR,
        "code": code,
        "message": message,
    }
    return encode_frame(dumps_payload(obj, MAX_REPLY_PAYLOAD_BYTES), MAX_REPLY_PAYLOAD_BYTES)


def parse_reply(
    payload: bytes, request_id: str, expected_meta: MetaExpectation
) -> Reply:
    """Validate one reply payload against *request_id* and *expected_meta*."""
    validate_request_id(request_id)
    obj = _load_object(payload)
    status = obj.get("status") if isinstance(obj, dict) else None
    if not isinstance(status, str):
        raise ProtocolError("status must be a string")
    if status == STATUS_OK:
        _exact_keys(obj, SUCCESS_KEYS, "success reply")
        _strict_version(obj)
        if validate_request_id(obj["request_id"]) != request_id:
            raise ProtocolError("success reply request_id does not match the request")
        raw_body = _strict_str(obj["raw_body"], "raw_body")
        if "\x00" in raw_body:
            raise ProtocolError("raw_body must not contain NUL")
        if not raw_body.strip():
            raise ProtocolError("raw_body must not be empty or whitespace-only")
        size = utf8_size(raw_body)
        if size > MAX_RAW_BODY_BYTES:
            raise ProtocolError(
                f"raw_body exceeds {MAX_RAW_BODY_BYTES} UTF-8 bytes (got {size})"
            )
        _validate_meta(obj["meta"], expected_meta)
        return Reply(status=STATUS_OK, request_id=request_id, raw_body=raw_body)
    if status == STATUS_ERROR:
        _exact_keys(obj, ERROR_KEYS, "error reply")
        _strict_version(obj)
        if validate_request_id(obj["request_id"]) != request_id:
            raise ProtocolError("error reply request_id does not match the request")
        code = obj["code"]
        if not isinstance(code, str) or code not in ERROR_CODES:
            raise ProtocolError(f"code must be one of {ERROR_CODES}")
        message = _strict_str(obj["message"], "message")
        if "\x00" in message:
            raise ProtocolError("message must not contain NUL")
        if utf8_size(message) > MAX_ERROR_MESSAGE_BYTES:
            raise ProtocolError(
                f"message exceeds {MAX_ERROR_MESSAGE_BYTES} UTF-8 bytes"
            )
        return Reply(
            status=STATUS_ERROR,
            request_id=request_id,
            code=code,
            message=message,
        )
    raise ProtocolError(f"status must be 'ok' or 'error' (got {status!r})")
