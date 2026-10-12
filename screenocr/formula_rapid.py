"""V31-M1b-C staged lossless Rapid tokenizer adapter (STAGE ONLY, not integrated).

Public API
----------
``install_raw_decoding(engine, *, max_tokens=512) -> engine``

Installs, on ONE engine instance only, a lossless token-id -> raw-body decoder
that replaces the upstream destructive ``TokenizerCls.token2str`` and neutralises
the upstream normalising ``LaTeXOCR.post_process`` with an identity function.

Design constraints enforced here
--------------------------------
* ``tokenizers`` is imported **lazily inside the installer**; importing this
  module must never pull in ``tokenizers``, numpy, cv2, onnxruntime,
  ``rapid_latex_ocr``, Paddle or Tk.
* No class-level or vendor mutation: only the three instance attributes
  ``engine.tokenizer.tokenizer.decoder``, ``engine.tokenizer.token2str`` and
  ``engine.post_process`` are written.  No upstream method is copied.
* Installation is **transactional**.  Writability is proven by inspecting
  storage, never by assigning a value to the instance (a "no-op" assignment of
  a class method is not a no-op: it creates an instance shadow).  The three
  overrides plus the private marker are committed together; any failure
  restores the exact original decoder, attribute storage (own vs inherited),
  values and marker state, or raises ``rollback_failed`` rather than leaving a
  half-installed instance.
* Special-token removal is **ID driven** via
  ``native.decode(ids, skip_special_tokens=True)``; never a text-substring
  replacement, so ordinary token IDs that spell ``[EOS]`` survive verbatim.
* The stock model cap (``config.yaml`` ``max_seq_len: 512`` /
  ``models.py`` ``Decoder.max_seq_len = 512``) is preserved; no 8192-token or
  unlimited decode path is invented.
* Missing EOS, tokens after the first EOS, oversized/ill-typed/unknown IDs and
  unsupported decoded bodies raise a typed :class:`RawDecodeError` instead of
  being repaired, stripped, bracketed or otherwise guessed.

Repeat policy (documented choice)
---------------------------------
Repeat installation on the same engine is **explicitly idempotent**: a second
call with the same ``max_tokens`` re-verifies the wiring and returns the same
engine unchanged.  A second call with a *different* ``max_tokens``, or with the
previously installed wiring externally altered, raises
``RawDecodeError("repeat_conflict")`` / ``RawDecodeError("repeat_state_drift")``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

__all__ = [
    "RawDecodeError",
    "install_raw_decoding",
    "MAX_TOKENS_CAP",
    "PAD_ID",
    "BOS_ID",
    "EOS_ID",
]

#: Stock Rapid cap: config.yaml `max_seq_len: 512`, models.py `Decoder.max_seq_len = 512`.
MAX_TOKENS_CAP = 512

#: Pinned tokenizer JSON `added_tokens`: id 0 [PAD], id 1 [BOS], id 2 [EOS].
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2

_SPECIAL_IDS = frozenset((PAD_ID, BOS_ID, EOS_ID))
_EXPECTED_SPECIAL_CONTENT = {
    PAD_ID: "[PAD]",
    BOS_ID: "[BOS]",
    EOS_ID: "[EOS]",
}

#: Marker written on the engine instance recording this exact installation.
_MARKER_ATTR = "_v31m1b_raw_decoding"

#: Sentinel for "the instance dictionary has no such entry".
_MISSING = object()


class RawDecodeError(Exception):
    """Typed failure raised by the staged raw-decoding adapter.

    ``code`` is a stable machine-readable classifier; ``message`` is human
    readable; ``detail`` carries optional structured context.
    """

    def __init__(self, code: str, message: str, *, detail: Any = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.detail = detail

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"RawDecodeError(code={self.code!r}, message={self.message!r})"


def _identity_post_process(value: str) -> str:
    """Byte-exact replacement for ``LaTeXOCR.post_process`` (identity)."""
    return value


# --------------------------------------------------------------------------- #
# Installation-state bookkeeping
# --------------------------------------------------------------------------- #
def _read_marker(engine: Any) -> Dict[str, Any] | None:
    try:
        record = getattr(engine, _MARKER_ATTR, None)
    except Exception:  # noqa: BLE001 - a hostile engine must not mask the real failure
        return None
    return record if isinstance(record, dict) else None


def _write_marker(engine: Any, record: Dict[str, Any]) -> None:
    try:
        setattr(engine, _MARKER_ATTR, record)
    except Exception as exc:  # noqa: BLE001
        raise RawDecodeError(
            "state_not_recordable",
            "cannot record installation state on this engine",
            detail=type(exc).__name__,
        ) from exc


# --------------------------------------------------------------------------- #
# Argument / object validation
# --------------------------------------------------------------------------- #
def _validate_max_tokens(max_tokens: Any) -> int:
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise RawDecodeError(
            "invalid_max_tokens",
            "max_tokens must be a plain int",
            detail=type(max_tokens).__name__,
        )
    if max_tokens < 1:
        raise RawDecodeError("invalid_max_tokens", "max_tokens must be >= 1", detail=max_tokens)
    if max_tokens > MAX_TOKENS_CAP:
        raise RawDecodeError(
            "invalid_max_tokens",
            f"max_tokens must not exceed the stock cap {MAX_TOKENS_CAP}",
            detail=max_tokens,
        )
    return max_tokens


def _validate_engine_shape(engine: Any) -> tuple[Any, Any]:
    if engine is None:
        raise RawDecodeError("unsupported_engine", "engine must not be None")

    wrapper = getattr(engine, "tokenizer", None)
    if wrapper is None:
        raise RawDecodeError("unsupported_engine", "engine.tokenizer is missing")
    native = getattr(wrapper, "tokenizer", None)
    if native is None:
        raise RawDecodeError("unsupported_engine", "engine.tokenizer.tokenizer is missing")

    for name in ("post_process",):
        if not callable(getattr(engine, name, None)):
            raise RawDecodeError("unsupported_engine", f"engine.{name} is not callable")
    if not callable(getattr(wrapper, "token2str", None)):
        raise RawDecodeError("unsupported_engine", "engine.tokenizer.token2str is not callable")

    for name in ("to_str", "decode", "id_to_token"):
        if not callable(getattr(native, name, None)):
            raise RawDecodeError(
                "unsupported_native_tokenizer",
                f"native tokenizer has no callable {name}()",
            )
    if not hasattr(native, "decoder"):
        raise RawDecodeError(
            "unsupported_native_tokenizer",
            "native tokenizer exposes no settable `decoder` attribute",
        )
    return wrapper, native


def _validate_native_schema(native: Any) -> Dict[str, Any]:
    """Reject any native tokenizer whose schema is not the pinned ByteLevel one."""
    try:
        meta = json.loads(native.to_str())
    except Exception as exc:  # noqa: BLE001
        raise RawDecodeError(
            "unsupported_native_tokenizer",
            "native tokenizer to_str() did not yield JSON metadata",
            detail=type(exc).__name__,
        ) from exc
    if not isinstance(meta, dict):
        raise RawDecodeError("unsupported_native_tokenizer", "native metadata is not an object")

    pre_tokenizer = meta.get("pre_tokenizer")
    if not isinstance(pre_tokenizer, dict) or pre_tokenizer.get("type") != "ByteLevel":
        raise RawDecodeError(
            "unsupported_schema",
            "pre-tokenizer is not ByteLevel",
            detail=pre_tokenizer,
        )
    if meta.get("decoder") is not None:
        raise RawDecodeError(
            "unsupported_schema",
            "native decoder is not NULL in the original tokenizer JSON",
            detail=meta.get("decoder"),
        )

    added_tokens = meta.get("added_tokens")
    if not isinstance(added_tokens, list):
        raise RawDecodeError("unsupported_schema", "added_tokens is not a list")
    added_by_id: Dict[int, Dict[str, Any]] = {}
    for row in added_tokens:
        if isinstance(row, dict) and isinstance(row.get("id"), int) and not isinstance(row.get("id"), bool):
            added_by_id[row["id"]] = row

    for token_id, content in _EXPECTED_SPECIAL_CONTENT.items():
        row = added_by_id.get(token_id)
        if row is None:
            raise RawDecodeError(
                "unsupported_schema",
                f"added_tokens has no id {token_id}",
            )
        if row.get("special") is not True:
            raise RawDecodeError(
                "unsupported_schema",
                f"id {token_id} is not marked special",
                detail=row,
            )
        if row.get("content") != content:
            raise RawDecodeError(
                "unsupported_schema",
                f"id {token_id} content is not {content!r}",
                detail=row.get("content"),
            )

    # Cross-check the same claim through the public added-token API.
    public_added = getattr(native, "get_added_tokens_decoder", None)
    if callable(public_added):
        try:
            decoded = public_added()
        except Exception as exc:  # noqa: BLE001
            raise RawDecodeError(
                "unsupported_native_tokenizer",
                "get_added_tokens_decoder() failed",
                detail=type(exc).__name__,
            ) from exc
        for token_id, content in _EXPECTED_SPECIAL_CONTENT.items():
            entry = decoded.get(token_id)
            if entry is None or entry.content != content or not bool(entry.special):
                raise RawDecodeError(
                    "unsupported_schema",
                    f"public added-token API disagrees for id {token_id}",
                    detail=None if entry is None else (entry.content, bool(entry.special)),
                )

    # Behavioural proof that the three IDs really are special to the native codec.
    for token_id, content in _EXPECTED_SPECIAL_CONTENT.items():
        if native.id_to_token(token_id) != content:
            raise RawDecodeError(
                "unsupported_schema",
                f"id_to_token({token_id}) is not {content!r}",
            )
    probe = sorted(_SPECIAL_IDS)
    try:
        skipped = native.decode(list(probe), skip_special_tokens=True)
        raw = native.decode(list(probe), skip_special_tokens=False)
    except Exception as exc:  # noqa: BLE001
        raise RawDecodeError(
            "unsupported_native_tokenizer",
            "native decode() failed on the special-token probe",
            detail=type(exc).__name__,
        ) from exc
    if skipped != "":
        raise RawDecodeError(
            "unsupported_schema",
            "special IDs are not skipped by ID when skip_special_tokens=True",
            detail=skipped,
        )
    # With the pinned NULL decoder the raw form is the space-joined contents;
    # assert containment rather than an exact join, which is decoder-dependent.
    missing = [c for c in _EXPECTED_SPECIAL_CONTENT.values() if c not in raw]
    if missing:
        raise RawDecodeError(
            "unsupported_schema",
            "special IDs do not decode to their registered contents",
            detail={"missing": missing, "raw": raw},
        )

    return meta


# --------------------------------------------------------------------------- #
# Replacement token2str
# --------------------------------------------------------------------------- #
def _extract_single_row(tokens: Any, cap: int) -> List[int]:
    tolist = getattr(tokens, "tolist", None)
    if not callable(tolist):
        raise RawDecodeError(
            "invalid_input_type",
            "tokens must be ndarray-like and provide .tolist()",
            detail=type(tokens).__name__,
        )
    try:
        data = tolist()
    except Exception as exc:  # noqa: BLE001
        raise RawDecodeError(
            "invalid_input_type",
            ".tolist() failed on the supplied tokens",
            detail=type(exc).__name__,
        ) from exc

    if not isinstance(data, list):
        raise RawDecodeError(
            "invalid_shape",
            "tolist() must yield a 1D list or a single-row 2D list",
            detail=type(data).__name__,
        )
    if data and all(isinstance(row, list) for row in data):
        if len(data) != 1:
            raise RawDecodeError(
                "invalid_shape",
                "only a single sequence may be decoded (single-image decoder)",
                detail=len(data),
            )
        row = data[0]
    else:
        row = data

    if not isinstance(row, list):
        raise RawDecodeError("invalid_shape", "token row is not a list", detail=type(row).__name__)
    if not row:
        raise RawDecodeError("empty_token_sequence", "token sequence is empty")
    if len(row) > cap:
        raise RawDecodeError(
            "sequence_too_long",
            f"token sequence length {len(row)} exceeds max_tokens={cap}",
            detail=len(row),
        )
    return row


def _validate_token_ids(native: Any, ids: List[int]) -> None:
    for index, token_id in enumerate(ids):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise RawDecodeError(
                "invalid_token_type",
                f"token at index {index} is not a plain int",
                detail=type(token_id).__name__,
            )
        if token_id < 0:
            raise RawDecodeError(
                "invalid_token_id",
                f"token at index {index} is negative",
                detail=token_id,
            )
    for index, token_id in enumerate(ids):
        try:
            known = native.id_to_token(token_id)
        except Exception:  # noqa: BLE001 - out-of-range ids must not escape as raw errors
            known = None
        if known is None:
            raise RawDecodeError(
                "unknown_token_id",
                f"token at index {index} is outside the vocabulary",
                detail=token_id,
            )


def _validate_body(body: Any) -> str:
    if not isinstance(body, str):
        raise RawDecodeError("unsupported_body", "decoded body is not text", detail=type(body).__name__)
    if body == "":
        raise RawDecodeError("empty_body", "decoded body is empty")
    if body.strip() == "":
        raise RawDecodeError("blank_body", "decoded body is whitespace only")
    if "\x00" in body:
        raise RawDecodeError("nul_body", "decoded body contains NUL")
    if "\ufffd" in body:
        raise RawDecodeError(
            "replacement_character_body",
            "decoded body contains U+FFFD (unsupported byte-level decoding)",
        )
    try:
        body.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise RawDecodeError(
            "invalid_utf8_body",
            "decoded body is not encodable as UTF-8",
            detail=str(exc),
        ) from exc
    return body


def _make_token2str(native: Any, cap: int):
    def token2str(tokens: Any) -> List[str]:
        """Lossless id -> raw body; returns exactly one element."""
        ids = _extract_single_row(tokens, cap)
        _validate_token_ids(native, ids)

        try:
            eos_index = ids.index(EOS_ID)
        except ValueError:
            raise RawDecodeError(
                "missing_eos",
                f"token sequence has no EOS id {EOS_ID}; generation looks truncated",
            ) from None

        for token_id in ids[eos_index + 1 :]:
            if token_id not in _SPECIAL_IDS:
                raise RawDecodeError(
                    "tokens_after_eos",
                    "a non-special token follows the first EOS",
                    detail=token_id,
                )

        body_ids = ids[:eos_index]
        try:
            body = native.decode(body_ids, skip_special_tokens=True)
        except Exception as exc:  # noqa: BLE001
            raise RawDecodeError(
                "native_decode_failed",
                "native decode() raised on the validated body ids",
                detail=type(exc).__name__,
            ) from exc

        return [_validate_body(body)]

    token2str.__name__ = "token2str"
    token2str.__qualname__ = "install_raw_decoding.<locals>.token2str"
    return token2str


# --------------------------------------------------------------------------- #
# Public installer
# --------------------------------------------------------------------------- #
def install_raw_decoding(engine: Any, *, max_tokens: int = MAX_TOKENS_CAP) -> Any:
    """Install lossless raw decoding on ONE engine instance and return it.

    Raises :class:`RawDecodeError` for any unsupported engine/tokenizer schema,
    invalid ``max_tokens``, or conflicting repeat installation.

    Either every instance override is installed and the installation marker is
    recorded, or the engine is left exactly as it was found.  A failure that
    cannot be fully undone raises ``RawDecodeError("rollback_failed")`` instead
    of returning a half-installed engine.
    """
    cap = _validate_max_tokens(max_tokens)

    existing = _read_marker(engine)
    if existing is not None:
        if existing.get("max_tokens") != cap:
            raise RawDecodeError(
                "repeat_conflict",
                "engine already has raw decoding installed with a different max_tokens",
                detail={"installed": existing.get("max_tokens"), "requested": cap},
            )
        wrapper = getattr(engine, "tokenizer", None)
        native = getattr(wrapper, "tokenizer", None)
        decoder = getattr(native, "decoder", None)
        if (
            native is not existing.get("native")
            or getattr(wrapper, "token2str", None) is not existing.get("token2str")
            or getattr(engine, "post_process", None) is not _identity_post_process
            or decoder is None
            or type(decoder).__name__ != "ByteLevel"
        ):
            raise RawDecodeError(
                "repeat_state_drift",
                "previously installed raw decoding was altered; refusing to re-install",
            )
        return engine

    wrapper, native = _validate_engine_shape(engine)
    _validate_native_schema(native)

    # Lazy import: never executed at module import time.
    from tokenizers import decoders

    replacement = _make_token2str(native, cap)
    byte_level = decoders.ByteLevel()

    # Storage we can prove is unwritable is refused here, before anything is
    # mutated.  No attribute is assigned to find this out.
    _reject_read_only_targets(engine, wrapper)

    # Exact snapshot of the three write targets plus the private marker, so a
    # failed commit can restore the original decoder, attribute STORAGE
    # (own-vs-inherited), values and marker state.
    snapshots = _snapshot_install_targets(engine, wrapper, native)
    marker_state = _snapshot_marker(engine)

    record = {
        "max_tokens": cap,
        "native": native,
        "wrapper": wrapper,
        "token2str": replacement,
        "post_process": _identity_post_process,
    }
    # First mutation: on a slotted engine this fails with state_not_recordable
    # while the instance is still byte-for-byte as it was found.
    _write_marker(engine, record)

    try:
        # Only these three instance-level writes happen; no class is touched.
        native.decoder = byte_level
        wrapper.token2str = replacement
        engine.post_process = _identity_post_process
    except Exception as exc:  # noqa: BLE001
        rollback_failures = _rollback_install(engine, snapshots, marker_state)
        if rollback_failures:
            raise RawDecodeError(
                "rollback_failed",
                "install failed and the original instance state could not be restored",
                detail={
                    "install_error": type(exc).__name__,
                    "rollback_failures": rollback_failures,
                },
            ) from exc
        raise RawDecodeError(
            "unsupported_native_tokenizer",
            "could not install the raw-decoding instance overrides",
            detail=type(exc).__name__,
        ) from exc
    return engine


def _read_only_property(target: Any, attribute: str) -> Any:
    """Return the class-level ``property`` when it is read-only, else ``None``.

    Only a plain read-only ``property`` is detected; a C-extension getset
    descriptor (the real native ``Tokenizer.decoder``) is deliberately NOT
    treated as unwritable, because it is the supported storage.
    """
    for klass in type(target).__mro__:
        if attribute in klass.__dict__:
            descriptor = klass.__dict__[attribute]
            if isinstance(descriptor, property) and descriptor.fset is None:
                return descriptor
            return None
    return None


def _reject_read_only_targets(engine: Any, wrapper: Any) -> None:
    for label, target, attribute in (
        ("wrapper.token2str", wrapper, "token2str"),
        ("engine.post_process", engine, "post_process"),
    ):
        if _read_only_property(target, attribute) is not None:
            raise RawDecodeError(
                "unsupported_engine",
                f"{label} is a read-only property on this instance",
            )


def _instance_dict(target: Any) -> Dict[str, Any] | None:
    """The target's real instance ``__dict__``, or ``None`` (slots/C object)."""
    try:
        storage = object.__getattribute__(target, "__dict__")
    except Exception:  # noqa: BLE001
        return None
    return storage if isinstance(storage, dict) else None


def _snapshot_install_targets(engine: Any, wrapper: Any, native: Any) -> List[Dict[str, Any]]:
    snapshots: List[Dict[str, Any]] = []
    for label, target, attribute in (
        ("native.decoder", native, "decoder"),
        ("wrapper.token2str", wrapper, "token2str"),
        ("engine.post_process", engine, "post_process"),
    ):
        try:
            original = getattr(target, attribute)
        except Exception as exc:  # noqa: BLE001
            raise RawDecodeError(
                "unsupported_engine",
                f"{label} cannot be read on this instance",
                detail=type(exc).__name__,
            ) from exc
        storage = _instance_dict(target)
        snapshots.append(
            {
                "label": label,
                "target": target,
                "attribute": attribute,
                "original": original,
                "storage": storage,
                "own_storage": storage is not None and attribute in storage,
            }
        )
    return snapshots


def _snapshot_marker(engine: Any) -> tuple[bool, Any]:
    try:
        return True, getattr(engine, _MARKER_ATTR)
    except Exception:  # noqa: BLE001 - absence is the normal case
        return False, None


def _same_value(current: Any, original: Any) -> bool:
    """Identity-first equality that also recognises re-created bound methods.

    ``getattr`` returns a *fresh* bound-method object on every access, so an
    ``is`` test would report a change that never happened and provoke exactly
    the shadowing write this repair removes.
    """
    if current is original:
        return True
    current_func = getattr(current, "__func__", None)
    original_func = getattr(original, "__func__", None)
    if current_func is not None and original_func is not None:
        return (
            current_func is original_func
            and getattr(current, "__self__", None) is getattr(original, "__self__", None)
        )
    if isinstance(original, (bool, int, float, str, bytes, type(None))):
        return type(current) is type(original) and current == original
    return False


def _restore_target(snapshot: Dict[str, Any]) -> None:
    """Restore one target to its exact original storage and value."""
    target = snapshot["target"]
    attribute = snapshot["attribute"]
    original = snapshot["original"]
    storage = snapshot["storage"]
    if snapshot["own_storage"]:
        if not _same_value(storage.get(attribute, _MISSING), original):
            setattr(target, attribute, original)
        return
    if storage is not None and attribute in storage:
        # The commit created a NEW instance-dictionary entry (a shadow).  Drop
        # it: assigning the original value back would leave an equivalent
        # bound callable in the instance dictionary instead of the inherited
        # class attribute the instance really had.
        del storage[attribute]
        return
    if not _same_value(getattr(target, attribute), original):
        setattr(target, attribute, original)


def _restore_marker(engine: Any, marker_state: tuple[bool, Any]) -> None:
    present, original = marker_state
    if present:
        setattr(engine, _MARKER_ATTR, original)
        return
    try:
        delattr(engine, _MARKER_ATTR)
    except AttributeError:
        pass


def _rollback_install(
    engine: Any, snapshots: List[Dict[str, Any]], marker_state: tuple[bool, Any]
) -> List[str]:
    """Undo the commit in reverse order; return labels that could not be undone.

    A non-empty result is a hard error (``rollback_failed``), never a silently
    accepted best-effort rollback.
    """
    failures: List[str] = []
    for snapshot in reversed(snapshots):
        try:
            _restore_target(snapshot)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{snapshot['label']}: {type(exc).__name__}")
    try:
        _restore_marker(engine, marker_state)
    except Exception as exc:  # noqa: BLE001
        failures.append(f"marker: {type(exc).__name__}")
    return failures
