"""V31-M1B runtime: the pure ``FormulaService`` front end for the formula worker.

Importing this module is cheap and side-effect free: no filesystem access, no SDK
or ML import, no native job, no ``Popen``, no Tk.  ``FormulaService(project_root)``
stores one plain trusted local root string and nothing else; every filesystem,
environment, integrity and process decision happens per request inside
``recognize_payload``, before any child is launched.

The service owns *coordination only*.  It composes the three accepted stage
modules (assets, codec constants, ownership), the accepted engine-neutral
process client and the child script :mod:`screenocr.formula_worker`.  It never
repairs, rewrites or post-processes a raw body, and it never claims
mathematical accuracy: a returned string is the exact character sequence the
engine produced.

Contract references are to ``artifacts/v3.1/contract-sdk-runtime.md``
(revision2, frozen): L6 service API, L7 deadline/slot, L8 cancellation, L9
per-request asset verification, L12 launch/environment, L13 ownership guard,
L18 pinned 5th config, L20 inline 28-pin table, L21 providers/raw seam.
"""
from __future__ import annotations

import hashlib
import math
import numbers
import ntpath
import os
import re
import stat
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .formula_assets import (
    AssetPaths,
    AssetsCancelled,
    AssetsTimedOut,
    AssetsUnavailable,
    verify_assets,
)
from .formula_ownership import OwnershipUnavailable, ensure_formula_process_ownership
from .formula_process import (
    BusyError,
    CancelledError,
    FormulaProcessClient,
    LaunchSpec,
    ProcessFailureError,
    ProtocolFailureError,
    TimedOutError,
    UnavailableError,
    resolve_base_interpreter,
)
from .formula_protocol import (
    MAX_PNG_BYTES,
    MAX_RAW_BODY_BYTES,
    MetaExpectation,
    ProtocolError,
    validate_png_bytes,
    validate_request_id,
)

__all__ = [
    "FormulaService",
    "SERVICE_WINDOW_SECONDS",
    "CHILD_PROFILE",
    "EXPECTED_INTEGRITY_RID",
    "EXPECTED_PINS",
    "PINNED_PACKAGE_CONFIG_SHA256",
    "PINNED_CONFIG_VALUES",
]

# --------------------------------------------------------------------------- #
# frozen constants (contract L7, L12, L18, L20)
# --------------------------------------------------------------------------- #
SERVICE_WINDOW_SECONDS = 30.0
CHILD_PROFILE = "formula_site_only"
EXPECTED_INTEGRITY_RID = 8192
EXPECTED_PYTHON_VERSION = "3.11"
PE_MACHINE_AMD64 = 0x8664

FORMULA_SITE_RELATIVE = (".venv-formula", "Lib", "site-packages")
MODEL_DIR_RELATIVE = ("models", "formula", "rapid-latex-ocr-0.0.9")
WORKER_RELATIVE = ("screenocr", "formula_worker.py")
PINNED_PACKAGE_CONFIG_RELATIVE = ("rapid_latex_ocr", "config.yaml")
PINNED_PACKAGE_CONFIG_MAX_BYTES = 4096
PINNED_PACKAGE_CONFIG_SHA256 = (
    "4D85C7BF8AB5740F5FB031474997BC61ED3DBD143593A599D118A22CC9A2D57B"
)
METADATA_READ_LIMIT = 8192
PE_READ_LIMIT = 4096

PINNED_CONFIG_KEYS = (
    "max_width", "max_height", "min_height", "min_width",
    "bos_token", "max_seq_len", "eos_token", "temperature",
)
PINNED_CONFIG_VALUES: Mapping[str, Any] = {
    "max_width": 672, "max_height": 192, "min_height": 32, "min_width": 32,
    "bos_token": 1, "max_seq_len": 512, "eos_token": 2, "temperature": 0.00001,
}

#: Immutable copy of the frozen 28-pin lock (setup-plan/requirements-formula-lock.txt,
#: canonical-LF SHA E7529DBE004492CDC6887FA756280CB2C590CCAEFF51D36CD65352DCB3291E00).
#: Stored inline on purpose: the runtime never reads artifacts/setup-plan.
EXPECTED_PINS: Tuple[Tuple[str, str], ...] = (
    ("anyio", "4.15.1"),
    ("certifi", "2026.7.22"),
    ("chardet", "5.2.0"),
    ("charset-normalizer", "3.5.2"),
    ("click", "8.5.0"),
    ("colorama", "0.4.6"),
    ("filelock", "4.0.12"),
    ("flatbuffers", "25.12.19"),
    ("fsspec", "2026.9.0"),
    ("h11", "0.16.0"),
    ("hf-xet", "1.7.0"),
    ("httpcore", "1.0.9"),
    ("httpx", "0.28.1"),
    ("huggingface_hub", "1.33.0"),
    ("idna", "3.20"),
    ("numpy", "1.26.4"),
    ("onnxruntime", "1.30.0"),
    ("opencv-python", "4.11.0.86"),
    ("packaging", "26.3"),
    ("pillow", "12.3.0"),
    ("protobuf", "7.36.2"),
    ("PyYAML", "6.0.3"),
    ("rapid-latex-ocr", "0.0.9"),
    ("requests", "2.32.5"),
    ("tokenizers", "0.23.2"),
    ("tqdm", "4.70.1"),
    ("typing_extensions", "4.16.0"),
    ("urllib3", "2.8.0"),
)
#: Bootstrap tooling the isolated environment may carry without a pin (contract L20).
ALLOWED_EXTRA_PINS = frozenset({"pip", "setuptools"})

MANUAL_SETUP_HINT = (
    "run the manual setup-formula.ps1 to verify the four pinned assets and write "
    "formula-assets.json"
)
_CLOSED_MESSAGE = "the service is closed"

_CPU_PROVIDER = "CPUExecutionProvider"
_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_NAME_SPLIT = re.compile(r"[-_.]+")
_DEVICES = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)

# --------------------------------------------------------------------------- #
# patchable seams (private; production uses the real implementations)
# --------------------------------------------------------------------------- #
_monotonic = time.monotonic
_ensure_ownership = ensure_formula_process_ownership
_make_client = FormulaProcessClient
_verify_assets = verify_assets


class _OwnershipFailure(UnavailableError):
    """Terminal: this process could not establish formula-child ownership."""


def _canonical_name(name: str) -> str:
    """PEP 503 canonical form, so ``PyYAML``/``pyyaml`` compare equal."""
    return _NAME_SPLIT.sub("-", name).strip().lower()


# --------------------------------------------------------------------------- #
# shared read-only checks (used by the parent service and by the child worker)
# --------------------------------------------------------------------------- #
def _read_bounded_regular(path: Any, limit: int, label: str) -> bytes:
    """Read a bounded regular non-link file; typed ``UnavailableError`` otherwise."""
    if not isinstance(path, str) or not path or "\x00" in path:
        raise UnavailableError(f"{label} must be a non-empty path")
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise UnavailableError(
            f"{label} is missing or unreadable ({exc.strerror or exc}); {MANUAL_SETUP_HINT}"
        ) from None
    if stat.S_ISLNK(info.st_mode) or (getattr(info, "st_file_attributes", 0) & _REPARSE):
        raise UnavailableError(f"{label} is a reparse point or link; refusing to follow it")
    if not stat.S_ISREG(info.st_mode):
        raise UnavailableError(f"{label} is not a regular file")
    if info.st_size > limit:
        raise UnavailableError(f"{label} exceeds {limit} bytes")
    try:
        with open(path, "rb") as handle:
            blob = handle.read(limit + 1)
    except OSError as exc:
        raise UnavailableError(f"{label} is unreadable ({exc.strerror or exc})") from None
    if len(blob) > limit:
        raise UnavailableError(f"{label} exceeds {limit} bytes")
    return blob


def _parse_config_value(key: str, value: str) -> Any:
    if key == "temperature":
        try:
            return float(value)
        except ValueError:
            raise ProcessFailureError(
                f"pinned config key {key!r} must be a decimal number", code="load_error"
            ) from None
    if not value.isdigit():
        raise ProcessFailureError(
            f"pinned config key {key!r} must be a non-negative integer", code="load_error"
        )
    return int(value)


def parse_pinned_config(raw: Any) -> Dict[str, Any]:
    """Strictly parse the 8-key pinned package config.

    Rejects non-ASCII text, NUL, a wrong line count, comments, duplicate keys,
    unknown keys, missing keys, and any value/type other than the frozen eight
    (contract L18).  Raises the accepted ``ProcessFailureError`` with code
    ``load_error``.
    """
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise ProcessFailureError("pinned config must be bytes", code="load_error")
    try:
        text = bytes(raw).decode("ascii")
    except UnicodeDecodeError:
        raise ProcessFailureError("pinned config is not ASCII text", code="load_error") from None
    if "\x00" in text:
        raise ProcessFailureError("pinned config contains a NUL byte", code="load_error")
    if "\r" in text:
        raise ProcessFailureError(
            "pinned config must use LF line endings", code="load_error"
        )
    body = text[:-1] if text.endswith("\n") else text
    lines = body.split("\n")
    if len(lines) != len(PINNED_CONFIG_KEYS):
        raise ProcessFailureError(
            f"pinned config must have exactly {len(PINNED_CONFIG_KEYS)} lines", code="load_error"
        )
    values: Dict[str, Any] = {}
    for line in lines:
        if line.count(":") != 1:
            raise ProcessFailureError(
                "pinned config lines must be exactly 'key: value'", code="load_error"
            )
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key in values:
            raise ProcessFailureError(
                f"pinned config has a duplicate key {key!r}", code="load_error"
            )
        if key not in PINNED_CONFIG_VALUES:
            raise ProcessFailureError(
                f"pinned config has an unexpected key {key!r}", code="load_error"
            )
        values[key] = _parse_config_value(key, value)
    for key, expected in PINNED_CONFIG_VALUES.items():
        if key not in values:
            raise ProcessFailureError(f"pinned config is missing key {key!r}", code="load_error")
        actual = values[key]
        if key == "temperature":
            if not isinstance(actual, float) or actual != expected:
                raise ProcessFailureError(
                    f"pinned config key {key!r} is {actual!r}, expected {expected!r}",
                    code="load_error",
                )
        elif type(actual) is not int or actual != expected:
            raise ProcessFailureError(
                f"pinned config key {key!r} is {actual!r}, expected {expected!r}",
                code="load_error",
            )
    return {key: values[key] for key in PINNED_CONFIG_KEYS}


def read_pinned_config(path: Any) -> Dict[str, Any]:
    """Read, digest-check and strictly parse the pinned package ``config.yaml``."""
    raw = _read_bounded_regular(path, PINNED_PACKAGE_CONFIG_MAX_BYTES, "the pinned package config")
    digest = hashlib.sha256(raw).hexdigest().upper()
    if digest != PINNED_PACKAGE_CONFIG_SHA256:
        raise ProcessFailureError(
            f"the pinned package config digest is {digest}, expected "
            f"{PINNED_PACKAGE_CONFIG_SHA256}",
            code="load_error",
        )
    return parse_pinned_config(raw)


def _metadata_identity(blob: bytes) -> Tuple[Optional[str], Optional[str]]:
    text = blob.decode("utf-8", "replace")
    name = version = None
    for line in text.splitlines():
        if not line.strip():
            break
        if name is None and line.startswith("Name:"):
            name = line[5:].strip()
        elif version is None and line.startswith("Version:"):
            version = line[8:].strip()
        if name and version:
            break
    return name, version


def read_installed_pins(site_dir: Any) -> Dict[str, str]:
    """Read ``Name``/``Version`` from every ``*.dist-info/METADATA`` under the site.

    Standard-metadata inspection only: nothing is imported, executed or modified.
    """
    inventory: Dict[str, str] = {}
    try:
        with os.scandir(site_dir) as entries:
            for entry in entries:
                if not entry.name.endswith(".dist-info"):
                    continue
                try:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                except OSError:
                    continue
                try:
                    with open(os.path.join(entry.path, "METADATA"), "rb") as handle:
                        blob = handle.read(METADATA_READ_LIMIT)
                except OSError:
                    continue
                name, version = _metadata_identity(blob)
                if name and version:
                    inventory[_canonical_name(name)] = version
    except OSError as exc:
        raise UnavailableError(
            f"the formula site is unreadable ({exc.strerror or exc}); {MANUAL_SETUP_HINT}"
        ) from None
    return inventory


def verify_installed_pins(installed: Mapping[str, str]) -> None:
    """Require all 28 frozen versions; allow only unpinned pip/setuptools extras."""
    problems: List[str] = []
    for name, version in EXPECTED_PINS:
        actual = installed.get(_canonical_name(name))
        if actual != version:
            problems.append(f"{name}=={version} is installed as {actual!r}")
    pinned = {_canonical_name(name) for name, _ in EXPECTED_PINS}
    unexpected = sorted(
        name for name in set(installed) - pinned if name not in ALLOWED_EXTRA_PINS
    )
    if unexpected:
        problems.append("unexpected packages: " + ", ".join(unexpected))
    if problems:
        raise ProcessFailureError(
            "the formula environment does not match the frozen 28-pin inventory: "
            + "; ".join(problems)[:512],
            code="load_error",
        )


def assert_cpu_providers(engine: Any) -> Tuple[str, ...]:
    """Require exactly CPU execution on all three pinned sessions (contract L21)."""
    try:
        sessions = (
            ("image_resizer", engine.image_resizer.session),
            ("encoder", engine.encoder_decoder.encoder.session),
            ("decoder", engine.encoder_decoder.decoder.session.session),
        )
    except AttributeError as exc:
        raise ProcessFailureError(
            f"the constructed engine has an unexpected session layout ({exc})", code="load_error"
        ) from None
    labels: List[str] = []
    for label, session in sessions:
        getter = getattr(session, "get_providers", None)
        if not callable(getter):
            raise ProcessFailureError(
                f"{label} session does not expose get_providers()", code="load_error"
            )
        try:
            providers = tuple(getter())
        except Exception as exc:  # noqa: BLE001 - reported as a typed load failure
            raise ProcessFailureError(
                f"cannot read the {label} providers ({type(exc).__name__})", code="load_error"
            ) from None
        if providers != (_CPU_PROVIDER,):
            raise ProcessFailureError(
                f"{label} providers are {providers!r}, expected only {_CPU_PROVIDER!r}",
                code="load_error",
            )
        labels.append(f"{label}:{_CPU_PROVIDER}")
    return tuple(labels)


def assert_sdk_shape(engine: Any, expected: Mapping[str, Any]) -> None:
    """Confirm the constructed engine really carries the 8 pinned values."""
    encoder_decoder = getattr(engine, "encoder_decoder", None)
    checks = (
        ("encoder_decoder.max_seq_len", getattr(encoder_decoder, "max_seq_len", None),
         expected["max_seq_len"], "int"),
        ("encoder_decoder.bos_token", getattr(encoder_decoder, "bos_token", None),
         expected["bos_token"], "int"),
        ("encoder_decoder.eos_token", getattr(encoder_decoder, "eos_token", None),
         expected["eos_token"], "int"),
        ("max_dims", getattr(engine, "max_dims", None),
         [expected["max_width"], expected["max_height"]], "dims"),
        ("min_dims", getattr(engine, "min_dims", None),
         [expected["min_width"], expected["min_height"]], "dims"),
        ("temperature", getattr(engine, "temperature", None), expected["temperature"], "float"),
    )
    problems: List[str] = []
    for label, actual, want, kind in checks:
        if kind == "int":
            if type(actual) is not int or actual != want:
                problems.append(f"{label}={actual!r} (expected {want!r})")
        elif kind == "float":
            if isinstance(actual, bool) or not isinstance(actual, (int, float)):
                problems.append(f"{label}={actual!r} (expected {want!r})")
            elif float(actual) != float(want):
                problems.append(f"{label}={actual!r} (expected {want!r})")
        else:
            # Consume at most three values: dimensions must be exactly two
            # integral scalars, without coercing floats, strings, or booleans.
            # numbers.Integral accepts NumPy integer scalars without importing
            # NumPy into the parent process.
            try:
                iterator = iter(actual)
                got = []
                for _ in range(3):
                    try:
                        item = next(iterator)
                    except StopIteration:
                        break
                    if isinstance(item, bool) or not isinstance(item, numbers.Integral):
                        raise TypeError
                    got.append(int(item))
            except BaseException:  # noqa: BLE001 - untrusted shape object
                problems.append(f"{label} must contain exactly two integer dimensions")
                continue
            if len(got) != 2 or got != want:
                problems.append(f"{label} must contain exactly two pinned integer dimensions")
    if problems:
        raise ProcessFailureError(
            "the constructed engine does not match the pinned configuration: "
            + "; ".join(problems)[:512],
            code="load_error",
        )


def validate_png(png_bytes: Any) -> Tuple[int, int]:
    """Verify PNG bytes locally with Pillow and return ``(width, height)``.

    The SDK's bytes branch does not catch corrupt input, so this must run before
    the engine sees the bytes.  Pillow is imported lazily and only here; the
    module itself stays import-pure.  Every rejection is the accepted
    ``ProtocolFailureError`` with wire code ``invalid_input``.
    """
    if not isinstance(png_bytes, (bytes, bytearray, memoryview)):
        raise ProtocolFailureError("png_bytes must be bytes", code="invalid_input")
    data = bytes(png_bytes)
    if not data:
        raise ProtocolFailureError("png_bytes must not be empty", code="invalid_input")
    if len(data) > MAX_PNG_BYTES:
        raise ProtocolFailureError(
            f"png_bytes exceeds {MAX_PNG_BYTES} bytes (got {len(data)})", code="invalid_input"
        )
    try:
        import warnings
        from io import BytesIO

        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        raise ProcessFailureError(
            f"Pillow is not importable in the formula environment ({exc})", code="load_error"
        ) from None
    try:
        with warnings.catch_warnings():
            # A decompression bomb raises a *Warning* subclass, not an error, so it
            # must be promoted explicitly before any pixel is touched.
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                image_format = image.format
                width, height = image.size
                image.verify()
    except (UnidentifiedImageError, Image.DecompressionBombError,
            Image.DecompressionBombWarning, OSError, ValueError, SyntaxError) as exc:
        raise ProtocolFailureError(
            f"the payload is not a usable PNG image ({type(exc).__name__})", code="invalid_input"
        ) from None
    if image_format != "PNG":
        raise ProtocolFailureError(
            f"the payload is {image_format!r}, not PNG", code="invalid_input"
        )
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ProtocolFailureError("the PNG has non-positive dimensions", code="invalid_input")
    return width, height


def validate_raw_body(pred: Any) -> str:
    """Return the engine body unchanged after strict non-blank/NUL/48KiB checks."""
    if not isinstance(pred, str):
        raise ProcessFailureError("the engine did not return a text body", code="inference_error")
    if "\x00" in pred:
        raise ProcessFailureError("the raw body contains a NUL character", code="inference_error")
    if not pred.strip():
        raise ProcessFailureError(
            "the raw body is empty or whitespace-only", code="inference_error"
        )
    try:
        size = len(pred.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ProcessFailureError(
            f"the raw body is not encodable as UTF-8 ({exc})", code="inference_error"
        ) from None
    if size > MAX_RAW_BODY_BYTES:
        raise ProcessFailureError(
            f"the raw body exceeds {MAX_RAW_BODY_BYTES} UTF-8 bytes (got {size})",
            code="inference_error",
        )
    return pred


def _process_integrity_rid() -> Optional[int]:
    """Read this process's own token integrity RID (read-only, no elevation)."""
    if os.name != "nt":
        return None
    try:
        from .permissions import get_integrity_level
    except Exception:  # noqa: BLE001 - a missing helper is an honest unknown
        return None
    try:
        return get_integrity_level()
    except Exception:  # noqa: BLE001
        return None


def _read_pyvenv_version(config_path: str) -> str:
    blob = _read_bounded_regular(config_path, 8192, "the base interpreter configuration")
    try:
        text = blob.decode("utf-8", "replace")
    except Exception:  # pragma: no cover - decode is already lenient
        raise UnavailableError("the base interpreter configuration is unreadable") from None
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip().lower() == "version":
            version = value.strip().strip('"')
            if version:
                return version
    raise UnavailableError("the base interpreter configuration records no version")


def _pe_machine(path: str) -> int:
    """Return the PE ``Machine`` field of an executable, bounded and read-only."""
    try:
        with open(path, "rb") as handle:
            header = handle.read(PE_READ_LIMIT)
            if len(header) < 0x40 or header[:2] != b"MZ":
                raise UnavailableError("the base interpreter is not a PE image")
            offset = int.from_bytes(header[0x3C:0x40], "little")
            if offset < 0x40 or offset + 6 > len(header):
                handle.seek(offset)
                tail = handle.read(6)
            else:
                tail = header[offset:offset + 6]
    except OSError as exc:
        raise UnavailableError(
            f"the base interpreter is unreadable ({exc.strerror or exc})"
        ) from None
    if len(tail) < 6 or tail[:4] != b"PE\x00\x00":
        raise UnavailableError("the base interpreter is not a PE image")
    return int.from_bytes(tail[4:6], "little")


def _lexical_absolute(path: Any, label: str) -> str:
    """Reject any spelling that is not a drive-qualified absolute local path."""
    if not isinstance(path, str) or not path:
        raise UnavailableError(f"{label} must be a non-empty absolute local path")
    if "\x00" in path:
        raise UnavailableError(f"{label} must not contain a NUL character")
    drive = ntpath.splitdrive(path)[0]
    if path.startswith(("\\\\", "//")) or drive.startswith(("\\\\", "//")):
        raise UnavailableError(f"{label} must not be a UNC/network path")
    if (len(drive) != 2 or not drive[0].isascii() or not drive[0].isalpha()
            or drive[1] != ":" or not ntpath.isabs(path)):
        raise UnavailableError(f"{label} must be an absolute path with a drive letter")
    for part in path.replace("/", "\\").split("\\"):
        if part.split(".")[0].rstrip(" .").upper() in _DEVICES:
            raise UnavailableError(f"{label} must not name a reserved device")
    return path


def _require_real_directory(path: Any, label: str) -> str:
    _lexical_absolute(path, label)
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise UnavailableError(
            f"{label} is not readable ({exc.strerror or exc}); {MANUAL_SETUP_HINT}"
        ) from None
    if stat.S_ISLNK(info.st_mode) or (getattr(info, "st_file_attributes", 0) & _REPARSE):
        raise UnavailableError(f"{label} must be a real directory, not a link or reparse point")
    if not stat.S_ISDIR(info.st_mode):
        raise UnavailableError(f"{label} is not a directory")
    return path


def _child_env() -> Dict[str, str]:
    """The exact three-key child environment (contract L12); nothing else is passed."""
    system_root = os.environ.get("SystemRoot")
    if (not isinstance(system_root, str) or not system_root or "\x00" in system_root
            or not os.path.isabs(system_root)):
        raise UnavailableError(
            "SystemRoot is not a valid inherited environment value; refusing to launch "
            "the formula child"
        )
    return {
        "SystemRoot": system_root,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUTF8": "1",
    }


def _resolve_base(root: str) -> str:
    """Resolve the original base interpreter and prove it is 3.11 x64."""
    config_path = os.path.join(root, ".venv", "pyvenv.cfg")
    base = resolve_base_interpreter(root)
    version = _read_pyvenv_version(config_path)
    if not version.startswith(EXPECTED_PYTHON_VERSION + "."):
        raise UnavailableError(
            f"the base interpreter records Python {version}, expected "
            f"{EXPECTED_PYTHON_VERSION}.x"
        )
    if _pe_machine(base) != PE_MACHINE_AMD64:
        raise UnavailableError("the base interpreter is not a 64-bit x86 image")
    return base


# --------------------------------------------------------------------------- #
# per-request preparation stages (module-level so tests can intercept them)
# --------------------------------------------------------------------------- #
def _prep_site(root: str, ctx: "_RequestContext") -> str:
    _require_real_directory(root, "project_root")
    site = os.path.join(root, *FORMULA_SITE_RELATIVE)
    if not os.path.isdir(site):
        raise UnavailableError(
            f"the formula environment is not installed at {site}; {MANUAL_SETUP_HINT}"
        )
    if not os.path.isdir(os.path.join(site, "rapid_latex_ocr")):
        raise UnavailableError(
            f"the formula package is not installed in {site}; {MANUAL_SETUP_HINT}"
        )
    return site


def _prep_metadata(site: str, ctx: "_RequestContext") -> None:
    verify_installed_pins(read_installed_pins(site))


def _prep_config(site: str, ctx: "_RequestContext") -> str:
    path = os.path.join(site, *PINNED_PACKAGE_CONFIG_RELATIVE)
    read_pinned_config(path)
    return path


def _prep_assets(model_dir: str, ctx: "_RequestContext") -> AssetPaths:
    try:
        return _verify_assets(
            model_dir,
            require_manifest=True,
            cancel_event=ctx.view,
            absolute_deadline=ctx.deadline,
        )
    except AssetsCancelled as exc:
        raise CancelledError(str(exc)) from None
    except AssetsTimedOut as exc:
        raise TimedOutError(str(exc)) from None
    except AssetsUnavailable as exc:
        raise UnavailableError(str(exc)) from None


def _prep_spec(root: str, site: str, model_dir: str, config_path: str,
               ctx: "_RequestContext") -> LaunchSpec:
    rid = _process_integrity_rid()
    if rid != EXPECTED_INTEGRITY_RID:
        raise UnavailableError(
            f"this process integrity RID is {rid!r}, expected {EXPECTED_INTEGRITY_RID}; "
            "refusing to launch the formula child"
        )
    base = _resolve_base(root)
    script = os.path.join(root, *WORKER_RELATIVE)
    if not os.path.isfile(script):
        raise UnavailableError(f"the formula worker script is missing at {script}")
    try:
        return LaunchSpec(
            python_executable=base,
            script_path=script,
            expected_meta=MetaExpectation(
                python_executable=base,
                sys_path_profile=CHILD_PROFILE,
                integrity_rid=EXPECTED_INTEGRITY_RID,
            ),
            argv=(root, site, model_dir, config_path),
            env=_child_env(),
            working_directory=root,
        )
    except ProtocolError as exc:
        raise ProcessFailureError(str(exc), code="load_error") from None


def _prep_guard(ctx: "_RequestContext") -> None:
    try:
        _ensure_ownership()
    except OwnershipUnavailable as exc:
        raise _OwnershipFailure(str(exc)) from None


# --------------------------------------------------------------------------- #
# service internals
# --------------------------------------------------------------------------- #
class _CancellationView:
    """Read-only combined view: this request's own flag OR the caller's event.

    The caller's event object is only ever asked ``is_set()``; it is never set or
    cleared by this runtime (contract L8).
    """

    __slots__ = ("_own", "_caller")

    def __init__(self, own: threading.Event, caller: Any) -> None:
        self._own = own
        self._caller = caller

    def is_set(self) -> bool:
        if self._own.is_set():
            return True
        return bool(self._caller.is_set())


@dataclass
class _RequestContext:
    event: threading.Event
    view: _CancellationView
    started: float
    deadline: float
    stage: str = "entry"
    field_lock: Any = field(default=None, repr=False)


def _entry_deadline(value: Any, started: float) -> float:
    """``D = min(callerD, t0 + 30)`` or ``t0 + 30``; typed before any filesystem use."""
    cap = started + SERVICE_WINDOW_SECONDS
    if value is None:
        return cap
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolFailureError(
            "absolute_deadline must be a finite monotonic timestamp or None"
        )
    try:
        number = float(value)
    except OverflowError:
        # A huge int is not a usable deadline and must never be interpolated:
        # Python refuses to render ints past 4300 digits.
        raise ProtocolFailureError(
            "absolute_deadline must be a finite monotonic timestamp within float range or None"
        ) from None
    if math.isnan(number) or math.isinf(number):
        raise ProtocolFailureError(
            "absolute_deadline must be a finite monotonic timestamp or None"
        )
    return min(number, cap)


def _entry_validate(png_bytes: Any, request_id: Any, cancel_event: Any) -> None:
    try:
        validate_request_id(request_id)
        validate_png_bytes(png_bytes)
        if not callable(getattr(cancel_event, "is_set", None)):
            raise ProtocolError("cancel_event must provide is_set()")
    except ProtocolError as exc:
        raise ProtocolFailureError(str(exc)) from None


class FormulaService:
    """One-request-at-a-time coordinator for the formula child process.

    ``FormulaService(project_root)`` is pure: it stores the caller's trusted local
    root string and touches nothing.  Every filesystem, environment, integrity,
    ownership and process decision is made per request inside
    :meth:`recognize_payload`.
    """

    def __init__(self, project_root: str) -> None:
        if not isinstance(project_root, str) or not project_root:
            raise ProtocolFailureError("project_root must be a non-empty absolute local path")
        if "\x00" in project_root:
            raise ProtocolFailureError("project_root must not contain a NUL character")
        self._project_root = project_root
        self._state = threading.Lock()
        self._slot = threading.Lock()
        self._closed = False
        self._poison_reason: Optional[str] = None
        self._client: Optional[FormulaProcessClient] = None
        self._ctx: Optional[_RequestContext] = None
        self._prep: Optional[Dict[str, Any]] = None

    # -- read-only introspection ------------------------------------------- #
    @property
    def closed(self) -> bool:
        with self._state:
            return self._closed

    @property
    def poisoned(self) -> bool:
        with self._state:
            reason = self._poison_reason
            client = self._client
        if reason is not None:
            return True
        return bool(client is not None and client.poisoned)

    @property
    def active(self) -> bool:
        with self._state:
            ctx = self._ctx
            client = self._client
        if ctx is not None:
            return True
        return bool(client is not None and client.active)

    def diagnostics(self) -> Dict[str, Any]:
        """Plain copied diagnostics: no image bytes, no body, no private history."""
        with self._state:
            closed = self._closed
            poison = self._poison_reason
            prep = dict(self._prep) if self._prep is not None else None
            client = self._client
        return {
            "closed": bool(closed),
            "poisoned": poison,
            "service_prep": prep,
            "client": client.diagnostics() if client is not None else None,
        }

    # -- control ------------------------------------------------------------ #
    def close(self) -> None:
        """Permanently refuse new work; signal the active request outside the lock."""
        with self._state:
            self._closed = True
            ctx = self._ctx
            client = self._client
        if ctx is not None:
            ctx.event.set()
        if client is not None:
            client.close()

    def cancel_active(self) -> None:
        """Signal the active request; never the caller's event, never the OS."""
        with self._state:
            ctx = self._ctx
            client = self._client
        if ctx is not None:
            ctx.event.set()
        if client is not None:
            client.cancel_active()

    # -- public API --------------------------------------------------------- #
    def recognize_payload(
        self,
        png_bytes: bytes,
        request_id: str,
        cancel_event: Any,
        absolute_deadline: Optional[float] = None,
    ) -> str:
        """Run one formula request; return the exact raw body or raise typed."""
        started = _monotonic()
        deadline = _entry_deadline(absolute_deadline, started)
        _entry_validate(png_bytes, request_id, cancel_event)

        with self._state:
            closed = self._closed
            poison = self._poison_reason
            client = self._client
        if closed:
            raise UnavailableError(_CLOSED_MESSAGE)
        if poison is not None:
            raise UnavailableError(
                f"the formula service is unavailable ({poison}); {MANUAL_SETUP_HINT}"
            )
        if client is not None and client.poisoned:
            raise UnavailableError(
                f"the formula client is poisoned; {MANUAL_SETUP_HINT}"
            )

        if not self._slot.acquire(blocking=False):
            # A busy request mutates nothing, releases nothing and launches nothing.
            raise BusyError("another formula request is already active")
        acquired = True
        ctx: Optional[_RequestContext] = None
        context_cleared = False
        try:
            own_event = threading.Event()
            ctx = _RequestContext(
                event=own_event,
                view=_CancellationView(own_event, cancel_event),
                started=started,
                deadline=deadline,
            )
            with self._state:
                if self._closed:
                    raise UnavailableError(_CLOSED_MESSAGE)
                if self._poison_reason is not None:
                    raise UnavailableError(
                        f"the formula service is unavailable ({self._poison_reason}); "
                        f"{MANUAL_SETUP_HINT}"
                    )
                self._ctx = ctx
                self._prep = {
                    "started": started,
                    "deadline": deadline,
                    "elapsed": 0.0,
                    "stage": "entry",
                }
            run_error: Optional[BaseException] = None
            body: Optional[str] = None
            try:
                body = self._run(ctx, png_bytes, request_id)
            except BaseException as exc:  # preserve the exact delegated failure
                run_error = exc

            # Final observations happen before the owner-only clear, while the
            # slot remains held. Never call caller/client code or the clock under
            # _state: cancellation events and fake clients are external objects.
            final_elapsed = max(0.0, _monotonic() - started)
            caller_cancelled = ctx.view.is_set()
            with self._state:
                final_client = self._client
            client_poisoned = bool(final_client is not None and final_client.poisoned)
            terminal_error: Optional[BaseException] = None
            with self._state:
                closed = self._closed
                poison = self._poison_reason
                # Adjudicate success while ownership is still published. The
                # context clear below is the final shared-state transition.
                if run_error is None:
                    if closed:
                        terminal_error = UnavailableError(_CLOSED_MESSAGE)
                    elif poison is not None or client_poisoned:
                        terminal_error = UnavailableError("the formula client became unavailable")
                    # This is the request-owned standard Event; sample it here
                    # so cancel_active() cannot latch between a stale read and clear.
                    elif ctx.event.is_set() or caller_cancelled:
                        terminal_error = CancelledError("the formula request was cancelled")
                    elif final_elapsed >= deadline - started:
                        terminal_error = TimedOutError("the formula request deadline expired")
                if self._prep is not None:
                    self._prep["elapsed"] = final_elapsed
                    self._prep["stage"] = ctx.stage
                # Clear only this request's context, exactly once, as the last
                # state transition before slot release.
                if self._ctx is ctx:
                    self._ctx = None
                    context_cleared = True

            if run_error is not None:
                raise run_error
            if terminal_error is not None:
                raise terminal_error
            return body
        finally:
            try:
                if ctx is not None and not context_cleared:
                    # Defensive cleanup if an external finalization probe raised;
                    # normal completion clears above while holding _state.
                    with self._state:
                        if self._ctx is ctx:
                            self._ctx = None
                            context_cleared = True
            finally:
                if acquired:
                    self._slot.release()

    # -- internals ---------------------------------------------------------- #
    def _set_stage(self, stage: str) -> None:
        now = _monotonic()
        with self._state:
            if self._prep is not None:
                self._prep["stage"] = stage
                self._prep["elapsed"] = max(0.0, now - self._prep["started"])

    def _checkpoint(self, ctx: _RequestContext, stage: str) -> None:
        """Stop check used before and after every preparation stage."""
        ctx.stage = stage
        self._set_stage(stage)
        with self._state:
            closed = self._closed
        if closed:
            raise UnavailableError(_CLOSED_MESSAGE)
        if ctx.view.is_set():
            raise CancelledError("the formula request was cancelled during preparation")
        if _monotonic() >= ctx.deadline:
            raise TimedOutError("the formula request deadline expired during preparation")

    def _mark_poisoned(self, reason: str) -> None:
        with self._state:
            if self._poison_reason is None:
                self._poison_reason = reason

    def _publish_client(self, spec: LaunchSpec, ctx: _RequestContext) -> FormulaProcessClient:
        with self._state:
            if self._closed:
                raise UnavailableError(_CLOSED_MESSAGE)
            existing = self._client
            if existing is not None:
                return existing
            if self._poison_reason is not None:
                raise UnavailableError(
                    f"the formula service is unavailable ({self._poison_reason}); "
                    f"{MANUAL_SETUP_HINT}"
                )
        client = _make_client(spec)
        with self._state:
            if self._closed:
                publish = False
            else:
                publish = True
                if self._client is None:
                    self._client = client
                else:
                    client = self._client
        if not publish:
            # A close that landed during construction is never delegated to.
            try:
                client.close()
            except Exception:  # noqa: BLE001 - the refusal below is the real signal
                pass
            raise UnavailableError(_CLOSED_MESSAGE)
        return client

    def _run(self, ctx: _RequestContext, png_bytes: bytes, request_id: str) -> str:
        root = self._project_root
        with self._state:
            client = self._client

        self._checkpoint(ctx, "site")
        site = _prep_site(root, ctx)
        self._checkpoint(ctx, "site")

        self._checkpoint(ctx, "metadata")
        _prep_metadata(site, ctx)
        self._checkpoint(ctx, "metadata")

        self._checkpoint(ctx, "config")
        config_path = _prep_config(site, ctx)
        self._checkpoint(ctx, "config")

        model_dir = os.path.join(root, *MODEL_DIR_RELATIVE)
        self._checkpoint(ctx, "assets")
        _prep_assets(model_dir, ctx)
        self._checkpoint(ctx, "assets")

        if client is None:
            self._checkpoint(ctx, "spec")
            spec = _prep_spec(root, site, model_dir, config_path, ctx)
            self._checkpoint(ctx, "spec")
            self._checkpoint(ctx, "guard")
            try:
                _prep_guard(ctx)
            except _OwnershipFailure as exc:
                self._mark_poisoned("ownership_unavailable")
                raise UnavailableError(str(exc)) from None
            self._checkpoint(ctx, "guard")
            client = self._publish_client(spec, ctx)

        self._checkpoint(ctx, "delegate")
        body = client.recognize_payload(png_bytes, request_id, ctx.view, ctx.deadline)
        self._checkpoint(ctx, "return")
        return body
