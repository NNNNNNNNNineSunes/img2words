"""Explicit, offline-only materialization of the already pinned formula assets.

This is a bounded setup stage, not a package installer or cold-install workflow.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import stat
import sys
import tempfile
import time
import ntpath
from dataclasses import dataclass
from typing import Optional


class SetupError(RuntimeError):
    """A safe, actionable setup failure; no completion manifest was committed."""


@dataclass(frozen=True)
class SetupResult:
    status: str
    source_dir: str
    target_dir: str
    manifest_path: str
    note: str = "Experimental local asset availability only; verify all mathematics manually."


# The accepted public verifier is imported only after adding the known project root.
# Import machinery and transitive stdlib reads are not installer operations.
def _api(project_root: str):
    root = os.path.abspath(project_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from screenocr import formula_assets
    return formula_assets


def _checked_verify_assets(api, model_dir: str, *, context: str, **kwargs):
    """Normalize ordinary public verifier failures without swallowing interrupts."""
    try:
        return api.verify_assets(model_dir, **kwargs)
    except Exception as exc:
        raise SetupError(
            f"{context} verification failed; preserve local files and follow the documented isolated manual setup."
        ) from exc


def _stop(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise SetupError("setup deadline expired before manifest commit; rerun the explicit setup command")


def _validate_timeout(timeout: object) -> float:
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise SetupError("timeout must be a finite positive number of seconds")
    try:
        value = float(timeout)
    except (OverflowError, ValueError):
        raise SetupError("timeout must be a finite positive number of seconds") from None
    if not math.isfinite(value) or value <= 0:
        raise SetupError("timeout must be a finite positive number of seconds")
    return value


def _local_root(path: object) -> str:
    if not isinstance(path, str) or not path or "\x00" in path:
        raise SetupError("project root must be a trusted local absolute path")
    drive, _ = ntpath.splitdrive(path)
    if (not ntpath.isabs(path) or len(drive) != 2 or drive[1] != ":"
            or path.startswith(("\\\\", "//"))):
        raise SetupError("project root must be a trusted drive-qualified local path (not UNC)")
    devices = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    devices.update(f"{stem}{n}" for stem in ("COM", "LPT") for n in range(1, 10))
    devices.update(f"{stem}{sup}" for stem in ("COM", "LPT") for sup in ("¹", "²", "³"))
    for part in path.replace("/", "\\").split("\\"):
        if part.split(".")[0].rstrip(" .").upper() in devices:
            raise SetupError("project root must not contain a reserved device component")
    return os.path.abspath(path)


def _pin_table(api) -> dict:
    try:
        doc = json.loads(api.manifest_bytes())
        files = doc["files"]
        if set(files) != {"image_resizer.onnx", "encoder.onnx", "decoder.onnx", "tokenizer.json"}:
            raise ValueError("unexpected manifest inventory")
        table = {}
        for name, record in files.items():
            size, digest = record["bytes"], record["sha256"]
            if (isinstance(size, bool) or not isinstance(size, int) or size <= 0
                    or not isinstance(digest, str) or len(digest) != 64
                    or any(ch not in "0123456789ABCDEF" for ch in digest)):
                raise ValueError("invalid public manifest record")
            table[name] = (size, digest)
        return table
    except Exception as exc:
        raise SetupError(f"accepted public asset manifest is invalid: {exc}") from None


def _metadata_check(root: str, lock_path: str, deadline: float) -> None:
    _stop(deadline)
    cfg = os.path.join(root, ".venv", "pyvenv.cfg")
    try:
        with open(cfg, "r", encoding="utf-8") as stream:
            text = stream.read(16385)
        _stop(deadline)
        if len(text) > 16384:
            raise ValueError("pyvenv.cfg too large")
        values = {}
        for line in text.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key.strip().lower()] = value.strip()
        base = values.get("executable") or values.get("home")
        if not base:
            raise ValueError("no base executable/home recorded")
        candidate = base if base.lower().endswith(".exe") else os.path.join(base, "python.exe")
        if os.path.normcase(os.path.abspath(candidate)) != os.path.normcase(os.path.abspath(sys.executable)):
            raise ValueError("helper is not running with the original base interpreter recorded by .venv")
    except Exception as exc:
        raise SetupError(f"original base interpreter check failed ({exc}); use the documented isolated manual setup") from None
    if sys.version_info[:2] != (3, 11) or sys.maxsize <= 2**32:
        raise SetupError("setup requires the original 64-bit Python 3.11 base interpreter")
    site = os.path.join(root, ".venv-formula", "Lib", "site-packages")
    try:
        expected = {}
        _stop(deadline)
        with open(lock_path, "rb") as stream:
            lock_bytes = stream.read(32769)
        _stop(deadline)
        if len(lock_bytes) > 32768 or hashlib.sha256(lock_bytes).hexdigest().upper() != "E7529DBE004492CDC6887FA756280CB2C590CCAEFF51D36CD65352DCB3291E00":
            raise ValueError("stage lockfile differs from the accepted exact 28-pin bytes")
        for line in lock_bytes.decode("utf-8").splitlines():
            _stop(deadline)
            line = line.strip()
            if line:
                name, version = line.split("==", 1)
                normalized = name.lower().replace("_", "-")
                if normalized in expected:
                    raise ValueError(f"duplicate lock entry: {normalized}")
                expected[normalized] = version
        if len(expected) != 28:
            raise ValueError("canonical lock must contain exactly 28 package pins")
        found = {}
        _stop(deadline)
        for dist in importlib.metadata.distributions(path=[site]):
            _stop(deadline)
            name, version = dist.metadata.get("Name"), dist.version
            if name:
                normalized = name.lower().replace("_", "-")
                found[normalized] = version
        missing = sorted(name for name, version in expected.items() if found.get(name) != version)
        extras = sorted(set(found) - set(expected) - {"pip", "setuptools"})
        if missing or extras:
            raise ValueError(f"missing/version-mismatched={missing}; disallowed extras={extras}")
    except Exception as exc:
        raise SetupError(f"isolated formula environment metadata check failed ({exc}); install the exact lock manually into .venv-formula only") from None


def _lstat(path: str):
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None


def _regular(path: str, expected: tuple, deadline: float) -> bool:
    _stop(deadline)
    info = _lstat(path)
    if info is None:
        return False
    if (stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400
            or not stat.S_ISREG(info.st_mode)):
        raise SetupError(f"refusing to reuse non-regular or reparse target: {path}")
    if info.st_size != expected[0]:
        raise SetupError(f"conflicting existing target size; preserve and inspect manually: {path}")
    digest = hashlib.sha256()
    count = 0
    with open(path, "rb") as stream:
        while count <= expected[0]:
            _stop(deadline)
            block = stream.read(min(1 << 20, expected[0] + 1 - count))
            if not block:
                break
            count += len(block)
            digest.update(block)
            _stop(deadline)
    if count != expected[0] or digest.hexdigest().upper() != expected[1]:
        raise SetupError(f"conflicting existing target content; preserve and inspect manually: {path}")
    return True


def _manifest_state(api, path: str, expected_manifest: bytes, deadline: float) -> bool:
    _stop(deadline)
    info = _lstat(path)
    if info is None:
        return False
    if (stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400
            or not stat.S_ISREG(info.st_mode)):
        raise SetupError("existing completion manifest is not a regular file; preserve and inspect manually")
    if info.st_size > api.MANIFEST_MAX_BYTES:
        raise SetupError("existing completion manifest is malformed; preserve and inspect manually")
    with open(path, "rb") as stream:
        _stop(deadline)
        data = stream.read(api.MANIFEST_MAX_BYTES + 1)
        _stop(deadline)
    if data != expected_manifest:
        raise SetupError("existing completion manifest is foreign or malformed; preserve and inspect manually")
    return True


def _mkdir_tracked(path: str, created: list, deadline: float) -> None:
    drive, tail = ntpath.splitdrive(os.path.abspath(path))
    parts = [p for p in tail.replace("/", "\\").split("\\") if p]
    current = drive + "\\"
    for part in parts:
        _stop(deadline)
        current = os.path.join(current, part)
        info = _lstat(current)
        if info is None:
            try:
                os.mkdir(current)
                info = os.lstat(current)
                created.append((current, info.st_dev, info.st_ino))
            except FileExistsError:
                info = os.lstat(current)
        if (stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400
                or not stat.S_ISDIR(info.st_mode)):
            raise SetupError(f"destination path component is not a safe directory: {current}")
        _stop(deadline)


def _record_owned_stage_file(stream, path: str, stage_files: list) -> None:
    """Record only an exclusive file whose open-handle identity is stable at its path."""
    try:
        opened = os.fstat(stream.fileno())
    except (OSError, ValueError) as exc:
        raise SetupError(f"cannot establish owned staging-file identity ({exc}); preserving uncertain stage entry") from None
    identity = (opened.st_dev, opened.st_ino)
    if not all(identity) or not stat.S_ISREG(opened.st_mode):
        raise SetupError("staging handle has no trustworthy nonzero regular-file identity; preserving uncertain entry")
    # Register the opened handle identity before another stop check or any content I/O.
    stage_files.append((path, identity[0], identity[1]))
    current = os.lstat(path)
    if (stat.S_ISLNK(current.st_mode) or getattr(current, "st_file_attributes", 0) & 0x400
            or not stat.S_ISREG(current.st_mode)
            or (current.st_dev, current.st_ino) != identity):
        raise SetupError("staging path no longer identifies the exclusive open file; preserving uncertain entry")


def _copy_bounded(source: str, stage: str, expected: tuple, deadline: float,
                  stage_files: list) -> None:
    digest, count = hashlib.sha256(), 0
    with open(source, "rb") as src, open(stage, "xb") as dst:
        _record_owned_stage_file(dst, stage, stage_files)
        while count < expected[0] + 1:
            _stop(deadline)
            data = src.read(min(1 << 20, expected[0] + 1 - count))
            _stop(deadline)
            if not data:
                break
            written = 0
            while written < len(data):
                _stop(deadline)
                n = dst.write(data[written:])
                if not n:
                    raise SetupError("short write while staging an asset")
                written += n
                _stop(deadline)
            count += len(data)
            digest.update(data)
        _stop(deadline)
        dst.flush()
        _stop(deadline)
        os.fsync(dst.fileno())
        _stop(deadline)
    if count != expected[0] or digest.hexdigest().upper() != expected[1]:
        raise SetupError("source changed or staged asset failed pinned size/hash verification")


def _clean_owned(stage_dir: Optional[str], stage_identity, stage_files: list, created: list) -> None:
    if stage_dir and stage_identity is not None:
        try:
            root_info = os.lstat(stage_dir)
            if ((root_info.st_dev, root_info.st_ino) == stage_identity
                    and stat.S_ISDIR(root_info.st_mode)
                    and not (getattr(root_info, "st_file_attributes", 0) & 0x400)):
                for path, dev, ino in reversed(stage_files):
                    if os.path.dirname(os.path.abspath(path)) != os.path.abspath(stage_dir):
                        continue
                    try:
                        info = os.lstat(path)
                        if ((info.st_dev, info.st_ino) == (dev, ino) and stat.S_ISREG(info.st_mode)
                                and not (getattr(info, "st_file_attributes", 0) & 0x400)):
                            os.unlink(path)
                    except OSError:
                        continue
                if not os.listdir(stage_dir):
                    current = os.lstat(stage_dir)
                    if (current.st_dev, current.st_ino) == stage_identity:
                        os.rmdir(stage_dir)
        except OSError:
            pass
    for path, dev, ino in reversed(created):
        try:
            info = os.lstat(path)
            if (info.st_dev, info.st_ino) == (dev, ino) and stat.S_ISDIR(info.st_mode) and not os.listdir(path):
                os.rmdir(path)
        except OSError:
            pass


def setup(project_root: str, source_dir: Optional[str] = None, timeout: float = 300.0) -> SetupResult:
    """Copy verified local G0 assets into models/formula and publish manifest last."""
    seconds = _validate_timeout(timeout)  # before path discovery, imports, or filesystem operations
    deadline = time.monotonic() + seconds
    root = _local_root(project_root)
    api = _api(root)
    table = _pin_table(api)
    root = os.path.abspath(root)
    if source_dir is None:
        candidates = (
            os.path.join(root, "artifacts", "v3.1", "models", "rapid-latex-ocr-0.0.9"),
            os.path.join(root, "models", "formula", "rapid-latex-ocr-0.0.9"),
        )
        source_dir = next((p for p in candidates if os.path.isdir(p)), None)
        if source_dir is None:
            raise SetupError("no known local G0 source found; pass -LocalModelsPath explicitly after preserving the offline manual instructions")
    source_dir = os.path.abspath(source_dir)
    _metadata_check(root, os.path.join(os.path.dirname(__file__), "requirements-formula-lock.txt"), deadline)
    # Validate all four source files through the public API before any destination changes.
    _stop(deadline)
    _checked_verify_assets(api, source_dir, context="Local source", require_manifest=False,
                           absolute_deadline=deadline)
    target = os.path.join(root, "models", "formula", "rapid-latex-ocr-0.0.9")
    manifest_path = os.path.join(target, api.MANIFEST_NAME)
    expected_manifest = api.manifest_bytes()
    parent = os.path.dirname(target)
    # Before parent creation: inspect every existing component and every target conflict.
    for directory in (os.path.join(root, "models"), parent, target):
        info = _lstat(directory)
        if info is not None and (stat.S_ISLNK(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & 0x400 or not stat.S_ISDIR(info.st_mode)):
            raise SetupError(f"destination component conflicts with a non-directory/reparse path: {directory}")
    target_exists = _lstat(target) is not None
    existing_manifest = _manifest_state(api, manifest_path, expected_manifest, deadline)
    existing = {}
    if target_exists:
        for name, pin in table.items():
            existing[name] = _regular(os.path.join(target, name), pin, deadline)
        if existing_manifest and not all(existing.values()):
            raise SetupError("completion manifest exists but one or more pinned assets are missing; preserve and inspect manually")
        if existing_manifest:
            _checked_verify_assets(api, target, context="Existing installation",
                                   absolute_deadline=deadline)
            return SetupResult("preserved", source_dir, target, manifest_path)
    else:
        existing = {name: False for name in table}
    _stop(deadline)
    created: list = []
    stage_dir: Optional[str] = None
    stage_identity = None
    stage_files: list = []
    committed = False
    try:
        # Parent creation is delayed until after source and initial conflict verification.
        _mkdir_tracked(parent, created, deadline)
        # Ensure the exact target leaf remains safe; create it if absent and track identity.
        info = _lstat(target)
        if info is None:
            try:
                os.mkdir(target)
                info = os.lstat(target)
                created.append((target, info.st_dev, info.st_ino))
            except FileExistsError:
                info = os.lstat(target)
        if (stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400
                or not stat.S_ISDIR(info.st_mode)):
            raise SetupError("destination target changed to a non-directory/reparse object")
        for name, pin in table.items():
            path = os.path.join(target, name)
            if _regular(path, pin, deadline):
                existing[name] = True
        # Manifest presence is rechecked before staging; nonce is exclusive and same-volume.
        _manifest_state(api, manifest_path, expected_manifest, deadline)
        stage_dir = tempfile.mkdtemp(prefix=".formula-setup-", dir=parent)
        stage_info = os.lstat(stage_dir)
        stage_identity = (stage_info.st_dev, stage_info.st_ino)
        if (not stat.S_ISDIR(stage_info.st_mode) or getattr(stage_info, "st_file_attributes", 0) & 0x400):
            raise SetupError("exclusive staging directory is not a regular directory")
        if stage_info.st_dev != os.stat(target).st_dev:
            raise SetupError("staging and model target are not on the same volume")
        for name, pin in table.items():
            if existing[name]:
                continue
            staged_path = os.path.join(stage_dir, name)
            _copy_bounded(os.path.join(source_dir, name), staged_path, pin, deadline, stage_files)
        # Per-file no-clobber promotion. On Windows os.rename fails if target exists.
        for name, pin in table.items():
            if existing[name]:
                continue
            _stop(deadline)
            destination = os.path.join(target, name)
            if _regular(destination, pin, deadline):
                os.unlink(os.path.join(stage_dir, name))
                existing[name] = True
                continue
            try:
                os.rename(os.path.join(stage_dir, name), destination)
            except FileExistsError:
                if not _regular(destination, pin, deadline):
                    raise SetupError(f"conflicting target appeared during promotion: {destination}")
                os.unlink(os.path.join(stage_dir, name))
            _stop(deadline)
            if not _regular(destination, pin, deadline):
                raise SetupError(f"promoted asset failed verification: {destination}")
        # Complete source/target public verification and every metadata check before commit.
        _stop(deadline)
        _checked_verify_assets(api, source_dir, context="Final source", require_manifest=False,
                               absolute_deadline=deadline)
        _checked_verify_assets(api, target, context="Final destination", require_manifest=False,
                               absolute_deadline=deadline)
        _metadata_check(root, os.path.join(os.path.dirname(__file__), "requirements-formula-lock.txt"), deadline)
        _stop(deadline)
        # Idempotence/race handling: if a valid manifest appeared, only accept after full API verification.
        if _manifest_state(api, manifest_path, expected_manifest, deadline):
            _checked_verify_assets(api, target, context="Competing completed installation",
                                   absolute_deadline=deadline)
            committed = True
            return SetupResult("preserved", source_dir, target, manifest_path)
        # Exclusive no-clobber publication; os.rename (never replace) is Windows semantics.
        manifest_stage = os.path.join(stage_dir, api.MANIFEST_NAME)
        with open(manifest_stage, "xb") as stream:
            _record_owned_stage_file(stream, manifest_stage, stage_files)
            view = memoryview(expected_manifest)
            offset = 0
            while offset < len(view):
                _stop(deadline)
                n = stream.write(view[offset:])
                if not n:
                    raise SetupError("short write while staging completion manifest")
                offset += n
                _stop(deadline)
            _stop(deadline)
            stream.flush()
            _stop(deadline)
            os.fsync(stream.fileno())
            _stop(deadline)
        _stop(deadline)
        try:
            os.rename(manifest_stage, manifest_path)
        except FileExistsError:
            if not _manifest_state(api, manifest_path, expected_manifest, deadline):
                raise SetupError("foreign completion manifest appeared during commit; preserved")
            _checked_verify_assets(api, target, context="Late winning installation",
                                   absolute_deadline=deadline)
            committed = True
            return SetupResult("preserved", source_dir, target, manifest_path)
        # Successful manifest rename is irreversible commit; never post-commit veto.
        committed = True
        return SetupResult("committed", source_dir, target, manifest_path)
    finally:
        _clean_owned(stage_dir, stage_identity, stage_files, created)


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Explicit offline formula asset setup (no package installation).")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--local-models-path")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args(argv)
    result = setup(args.project_root, args.local_models_path, args.timeout)
    print(json.dumps({"status": result.status, "source_dir": result.source_dir,
                      "target_dir": result.target_dir, "manifest_path": result.manifest_path,
                      "note": result.note}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SetupError, KeyboardInterrupt) as exc:
        print(f"Offline formula setup failed before completion: {exc}", file=sys.stderr)
        raise SystemExit(2)
