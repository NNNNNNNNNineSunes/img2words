"""Launch shim for the Screen OCR GUI.

The Windows workspace directory (and therefore ``.venv\\Scripts\\pythonw.exe``)
carries a mandatory *low* integrity label, so executing that image drops the
process to low integrity and breaks hotkeys / input.  This module never touches
the process token or its own filesystem ACLs.  Instead it is meant to be run by
the *original, unlabelled base interpreter* (``<base>\\pythonw.exe`` /
``<base>\\python.exe``, the interpreter recorded in ``.venv\\pyvenv.cfg``) and
only prepends the project's virtual-environment ``site-packages`` to
``sys.path``.

``sys.executable``, ``sys.prefix`` and ``sys.base_prefix`` are deliberately left
untouched: the running interpreter *is* the base interpreter, so nothing
re-executes ``.venv\\Scripts\\pythonw.exe``.

Only the standard library is imported at module import time; every heavy or
project import happens lazily inside :func:`main`.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

__all__ = ["LAUNCH_HELP", "configure_environment", "main", "report_error"]

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR_NAME = ".venv"
VENV_SITE_PACKAGES_PARTS = (VENV_DIR_NAME, "Lib", "site-packages")
SITE_PACKAGES_MARKER = os.path.join(VENV_DIR_NAME, "Lib", "site-packages")

LAUNCH_HELP = """Usage: bootstrap.py [--diagnose] [--smoke-seconds SECONDS]

Runs the Screen OCR GUI (arguments after -- are passed to screenocr.app.main).

  --diagnose    print a JSON environment/integrity report and exit (no GUI)
"""


class BootstrapError(RuntimeError):
    """Raised when the launching interpreter cannot run the project."""


def _norm(path: object) -> str:
    """Normalise a path for case-insensitive duplicate detection."""
    text = os.fspath(path)
    if not text:
        return ""
    absolute = os.path.abspath(text)
    return os.path.normcase(os.path.normpath(absolute)).rstrip("\\/")


def require_supported_runtime() -> Tuple[int, int]:
    """Validate 3.11 / 64-bit before anything heavy is imported."""
    if sys.version_info[:2] != (3, 11):
        raise BootstrapError(
            "Python 3.11 is required; this interpreter is "
            f"{sys.version_info.major}.{sys.version_info.minor}"
            f".{sys.version_info.micro} ({sys.executable})."
        )
    if sys.maxsize <= 2 ** 32:
        raise BootstrapError(
            "A 64-bit Python 3.11 is required; this interpreter is 32-bit "
            f"({sys.executable})."
        )
    return sys.version_info[:2]


def project_site_packages(project_root: Optional[os.PathLike] = None) -> Path:
    """Return the project virtual environment's ``site-packages`` directory."""
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    return root.joinpath(*VENV_SITE_PACKAGES_PARTS)


def _path_entry_key(entry: object) -> str:
    """Return a normalised key for a ``sys.path`` entry, or ``''`` for non-paths.

    ``os.PathLike`` entries are accepted because Python's import system accepts
    them; the original entry object is preserved in ``sys.path``.
    """
    if isinstance(entry, (str, os.PathLike)):
        return _norm(entry)
    return ""


def configure_environment(project_root: Optional[os.PathLike] = None) -> Path:
    """Make the project's virtual environment importable from the base interpreter.

    Returns the ``site-packages`` directory that was prepended.  Ordering is
    ``project root`` first, then ``.venv/Lib/site-packages``, then whatever the
    base interpreter already had -- including the base interpreter's own
    ``site-packages`` and the per-user site directory.  Entries are de-duplicated
    case-insensitively, so repeated calls are idempotent.
    """
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    site_packages = root.joinpath(*VENV_SITE_PACKAGES_PARTS)
    if not site_packages.is_dir():
        raise BootstrapError(
            "The project environment is missing: expected "
            f"{site_packages}. Run setup.ps1 / start.cmd again to create it."
        )

    wanted = (str(root), str(site_packages))
    wanted_keys = {_norm(entry) for entry in wanted}

    ordered = list(wanted)
    seen = set(wanted_keys)
    for entry in list(sys.path):
        # sys.path entries must stay plain strings: pathlib objects are not
        # resolved reliably by the base interpreter's FileFinder on Windows.
        candidate = entry if isinstance(entry, str) else _path_entry_key(entry)
        if not candidate:
            if entry not in ordered:
                ordered.append(entry)
            continue
        key = _path_entry_key(candidate)
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        if candidate not in ordered:
            ordered.append(candidate)

    sys.path[:] = ordered
    return site_packages


def get_integrity_level():
    """Return the current process integrity RID via screenocr.permissions.

    Returns ``None`` when that module (or its function) is not available.  The
    module is imported lazily so this launch shim never depends on it.
    """
    try:
        from screenocr.permissions import get_integrity_level as _get_integrity_level
    except Exception:
        return None
    if not callable(_get_integrity_level):
        return None
    try:
        value = _get_integrity_level()
    except Exception:
        return None
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


DIAGNOSTIC_MODULES = ("PIL", "mss", "rapidocr_onnxruntime")


def _module_report(name: str) -> dict:
    """Import *name* and report where it actually resolved from."""
    import importlib

    try:
        module = importlib.import_module(name)
    except BaseException as exc:  # pragma: no cover - exercised via the real env
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    path = getattr(module, "__file__", None)
    return {"ok": True, "path": path, "name": getattr(module, "__name__", name)}


def build_diagnostic(project_root: Optional[os.PathLike] = None,
                     module_names: Sequence[str] = DIAGNOSTIC_MODULES,
                     integrity_getter=None) -> dict:
    """Collect the JSON diagnostic payload (no GUI, no Tk, no user input)."""
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    getter = get_integrity_level if integrity_getter is None else integrity_getter
    try:
        require_supported_runtime()
        error = None
    except BootstrapError as exc:
        error = str(exc)
    return {
        "sys.executable": sys.executable,
        "sys._base_executable": getattr(sys, "_base_executable", None),
        "sys.prefix": sys.prefix,
        "sys.base_prefix": sys.base_prefix,
        "python_version": sys.version,
        "python_version_info": list(sys.version_info[:3]),
        "is_64bit": sys.maxsize > 2 ** 32,
        "project_root": str(root),
        "venv_site_packages": str(root.joinpath(*VENV_SITE_PACKAGES_PARTS)),
        "dont_write_bytecode": bool(sys.dont_write_bytecode),
        "environment": {
            "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE"),
        },
        "sys.path": [str(entry) for entry in sys.path],
        "modules": {name: _module_report(name) for name in module_names},
        "integrity_level": getter(),
        "error": error,
    }


def _project_log(message: str) -> None:
    """Best-effort audit line in the project's own log directory.

    This is the only file bootstrap.py may write, and only when stderr is
    unusable (``pythonw.exe`` has no console stream).
    """
    try:
        logdir = PROJECT_ROOT / "logs"
        logdir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with (logdir / "bootstrap.log").open("a", encoding="utf-8") as handle:
            handle.write(f"{stamp} bootstrap: {message}\n")
    except Exception:
        pass


def _show_messagebox(message: str) -> None:
    """Last-resort user-visible error under ``pythonw.exe`` (no console)."""
    try:
        import tkinter
        from tkinter import messagebox

        root = tkinter.Tk()
        root.withdraw()
        messagebox.showerror("Screen OCR 启动失败", message, parent=root)
        root.destroy()
    except Exception:
        pass


def report_error(message: str) -> None:
    """Report a startup failure without ever raising.

    ``sys.stderr`` is ``None`` under ``pythonw.exe``, so writing to it blindly
    raises ``AttributeError`` and hides the real error.  Order: stderr when it
    exists, otherwise a message box, always one line in the project log.
    """
    _project_log(message)
    if sys.stderr is not None:
        try:
            sys.stderr.write(f"bootstrap: {message}\n")
            sys.stderr.flush()
        except Exception:
            pass
        return
    _show_messagebox(message)


def _print_diagnostic(project_root: Optional[os.PathLike] = None,
                      module_names: Optional[Sequence[str]] = None) -> int:
    import json

    if module_names is None:
        module_names = DIAGNOSTIC_MODULES
    report = build_diagnostic(project_root, module_names=module_names)
    try:
        json.dump(report, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        sys.stdout.flush()
    except Exception as exc:
        _project_log(f"cannot write the diagnostic report: {exc}")
        return 2
    failed = sorted(name for name, info in report["modules"].items() if not info.get("ok"))
    if failed:
        return 1
    return 1 if report["error"] else 0


def _parse_launch_args(argv: Sequence[str]) -> Tuple[bool, Optional[list]]:
    """Split ``--diagnose`` out of *argv*; return ``(diagnose, passthrough)``."""
    passthrough = list(argv)
    if "--diagnose" not in passthrough:
        return False, passthrough
    passthrough.remove("--diagnose")
    return True, passthrough


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Configure the environment, then diagnose or delegate to the GUI app."""
    args = list(sys.argv[1:] if argv is None else argv)
    diagnose, passthrough = _parse_launch_args(args)

    if passthrough and passthrough[0] in ("-h", "--help"):
        if sys.stdout is not None:
            sys.stdout.write(LAUNCH_HELP)
        return 0

    try:
        require_supported_runtime()
    except BootstrapError as exc:
        report_error(str(exc))
        return 2

    try:
        configure_environment(PROJECT_ROOT)
    except BootstrapError as exc:
        report_error(str(exc))
        return 2

    if diagnose:
        if passthrough:
            report_error("--diagnose does not accept extra arguments: " + " ".join(passthrough))
            return 2
        return _print_diagnostic(PROJECT_ROOT, module_names=DIAGNOSTIC_MODULES)

    try:
        from screenocr.app import main as app_main
    except Exception as exc:
        report_error(
            f"cannot import screenocr.app ({type(exc).__name__}: {exc}). Run setup.ps1 first."
        )
        return 2
    return app_main(passthrough)


if __name__ == "__main__":
    raise SystemExit(main())
