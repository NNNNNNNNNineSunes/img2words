"""V31-M1B formula child worker: one request in, one bounded frame out.

The accepted engine-neutral client launches this script as::

    <base python> -I -S -B -X utf8 screenocr/formula_worker.py \
        <project_root> <formula_site> <model_dir> <config_path>

``-I`` ignores ``PYTHONPATH``/user site, so this process adds the project
standard modules and the formula site itself; it never touches the original
``.venv`` site or any global/per-user ML site.

Before *any* optional SDK import the worker redirects stdout to stderr:

1. ``sys.stdout.flush()``
2. ``saved = os.dup(1)`` -- the only handle to the real stdout pipe
3. ``os.dup2(2, 1)`` -- fd 1 now points at the stderr pipe
4. typed ``GetStdHandle``/``SetStdHandle(STD_OUTPUT_HANDLE -> STD_ERROR)``
5. legacy ``msvcrt._dup2(2, 1)`` -- the legacy CRT keeps its **own** descriptor
   table, so steps 3-4 alone leave ``msvcrt.printf`` output uncaptured

so every Python, CRT and native stdout write (``print``, ``msvcrt.printf``)
is captured by the bounded stderr diagnostic instead of reaching the frame.
The single reply frame is then written with ``os.write(saved, ...)`` and nobody
else ever receives ``saved``.  A redirect failure exits nonzero with no partial
frame.  There is no result-file IPC.

This worker never rewrites the engine body, never repairs math, and makes no
accuracy claim: the returned characters are exactly what the engine produced.
"""
from __future__ import annotations

import ctypes
import json
import os
import struct
import sys
from ctypes import wintypes
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# exit codes (no frame is written for any of these)
# --------------------------------------------------------------------------- #
EXIT_REDIRECT_FAILED = 70
EXIT_REQUEST_FAILED = 71
EXIT_INTERNAL = 72

STD_INPUT_HANDLE = 0xFFFFFFF6
STD_OUTPUT_HANDLE = 0xFFFFFFF5
STD_ERROR_HANDLE = 0xFFFFFFF4
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

DIAGNOSTIC_LIMIT = 8192
HEAVY_MODULES = ("numpy", "cv2", "onnxruntime", "tokenizers", "PIL", "yaml")


# --------------------------------------------------------------------------- #
# stdout isolation (must run before any optional import)
# --------------------------------------------------------------------------- #
def _redirect_stdout_to_stderr() -> int:
    """Route fd 1 and the Win32 stdout handle to stderr; return the saved fd.

    Three separate stdout sinks have to be redirected, and all three are
    verified by the G3 oracle:

    1. the UCRT descriptor table (what ``print``/``os.write(1)`` use) via
       ``os.dup2(2, 1)``;
    2. the Win32 ``STD_OUTPUT_HANDLE`` via typed ``GetStdHandle``/``SetStdHandle``,
       which is what ``WriteFile(GetStdHandle(...))`` uses;
    3. the **legacy** ``msvcrt.dll`` descriptor table, which is distinct from the
       UCRT table: ``os.dup2`` does *not* move ``msvcrt.printf`` output, and its
       ``fflush(NULL)`` then fails, so real CRT noise is silently **dropped**
       rather than routed.  Re-pointing legacy fd 1 at fd 2 is what actually
       captures it, as contract L17 requires ("ALL Python/native CRT/GetStdHandle
       stdout noises routed stderr").

    The measured consequence of omitting step 3 is a loss of diagnostic
    completeness, not frame corruption: the oracle records both behaviours
    rather than asserting the stronger claim.
    """
    try:
        sys.stdout.flush()
    except Exception:  # noqa: BLE001 - flushing is best effort before the dup
        pass
    saved = os.dup(1)
    os.dup2(2, 1)
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetStdHandle.argtypes = [wintypes.DWORD]
        kernel32.GetStdHandle.restype = wintypes.HANDLE
        kernel32.SetStdHandle.argtypes = [wintypes.DWORD, wintypes.HANDLE]
        kernel32.SetStdHandle.restype = wintypes.BOOL
        handle = kernel32.GetStdHandle(wintypes.DWORD(STD_ERROR_HANDLE))
        if not handle or handle == INVALID_HANDLE_VALUE:
            raise OSError("GetStdHandle(STD_ERROR_HANDLE) returned no usable handle")
        if not kernel32.SetStdHandle(wintypes.DWORD(STD_OUTPUT_HANDLE), handle):
            raise OSError(ctypes.get_last_error(), "SetStdHandle(STD_OUTPUT_HANDLE) failed")
        legacy = ctypes.CDLL("msvcrt")
        legacy._dup2.argtypes = [ctypes.c_int, ctypes.c_int]
        legacy._dup2.restype = ctypes.c_int
        # Legacy _dup2 follows the POSIX convention: 0 means success.
        if legacy._dup2(2, 1) != 0:
            raise OSError("the legacy CRT could not redirect its stdout descriptor")
    # From here on Python-level stdout noise is stderr noise as well.
    sys.stdout = sys.stderr
    return saved


def _write_frame(saved: int, frame: bytes) -> None:
    """Write the whole frame to the saved private fd, handling short writes."""
    view = memoryview(frame)
    while view:
        written = os.write(saved, view)
        if written <= 0:
            raise OSError("short write to the saved stdout descriptor")
        view = view[written:]


# --------------------------------------------------------------------------- #
# bounded diagnostics
# --------------------------------------------------------------------------- #
def _redact(path: str) -> str:
    home = os.path.expanduser("~")
    if home and os.path.normcase(path).startswith(os.path.normcase(home)):
        return "<home>" + path[len(home):]
    return path


def _exception_diagnostic(stage: str, exc: BaseException) -> Dict[str, str]:
    """Build a bounded error label without rendering exception-controlled text."""
    return {"error": stage, "type": type(exc).__name__}


def _emit(diag: Dict[str, Any]) -> None:
    """Write one bounded diagnostic block to stderr; never raw math or images."""
    try:
        text = json.dumps(diag, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    except Exception:  # noqa: BLE001 - diagnostics must never break the reply
        text = '{"diag":"unserializable"}'
    if len(text) > DIAGNOSTIC_LIMIT:
        text = text[:DIAGNOSTIC_LIMIT]
    try:
        sys.stderr.write("formula-worker " + text + "\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass


def _integrity_rid() -> Optional[int]:
    """Read this child's own token integrity RID (read-only, no elevation)."""
    if os.name != "nt":
        return None
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        advapi32.OpenProcessToken.argtypes = [
            wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
        advapi32.GetTokenInformation.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD)]
        advapi32.GetSidSubAuthorityCount.argtypes = [ctypes.c_void_p]
        advapi32.GetSidSubAuthorityCount.restype = ctypes.POINTER(wintypes.BYTE)
        advapi32.GetSidSubAuthority.argtypes = [ctypes.c_void_p, wintypes.DWORD]
        advapi32.GetSidSubAuthority.restype = ctypes.POINTER(wintypes.DWORD)
        token = wintypes.HANDLE()
        if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), 0x0008,
                                         ctypes.byref(token)):
            return None
        try:
            size = wintypes.DWORD()
            advapi32.GetTokenInformation(token, 25, None, 0, ctypes.byref(size))
            if not size.value:
                return None
            buffer = ctypes.create_string_buffer(size.value)
            if not advapi32.GetTokenInformation(token, 25, buffer, size.value,
                                                ctypes.byref(size)):
                return None
            sid = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_void_p))[0]
            count = advapi32.GetSidSubAuthorityCount(sid)[0]
            if not count:
                return None
            return int(advapi32.GetSidSubAuthority(sid, count - 1)[0])
        finally:
            kernel32.CloseHandle(token)
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# framing
# --------------------------------------------------------------------------- #
def _read_exact(stream: Any, count: int) -> bytes:
    chunks: List[bytes] = []
    remaining = count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("the request stream ended before the frame was complete")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_request(max_payload: int) -> bytes:
    header = _read_exact(sys.stdin.buffer, 4)
    (length,) = struct.unpack(">I", header)
    if length == 0 or length > max_payload:
        raise ValueError(f"declared request length {length} is out of range")
    return _read_exact(sys.stdin.buffer, length)


# --------------------------------------------------------------------------- #
# worker body
# --------------------------------------------------------------------------- #
def main(argv: List[str]) -> int:
    try:
        saved = _redirect_stdout_to_stderr()
    except BaseException as exc:  # noqa: BLE001 - no partial frame is permitted
        try:
            sys.stderr.write(
                "formula-worker stdout_redirect_failed: "
                + type(exc).__name__ + "\n"
            )
        except Exception:  # noqa: BLE001
            pass
        return EXIT_REDIRECT_FAILED

    if len(argv) != 4:
        _emit({"error": "usage", "argv_count": len(argv)})
        return EXIT_REQUEST_FAILED
    root, site, model_dir, config_path = argv

    # The project standard modules must be importable before the formula site is
    # added, so the protocol/validation helpers resolve to the frozen sources.
    if root not in sys.path:
        sys.path.insert(0, root)
    if site not in sys.path:
        sys.path.insert(0, site)

    try:
        from screenocr.formula_backend import (
            CHILD_PROFILE,
            EXPECTED_INTEGRITY_RID,
            assert_cpu_providers,
            assert_sdk_shape,
            read_installed_pins,
            read_pinned_config,
            validate_png,
            validate_raw_body,
            verify_installed_pins,
        )
        from screenocr.formula_protocol import (
            MAX_REQUEST_PAYLOAD_BYTES,
            build_error,
            build_success,
            parse_request,
        )
    except BaseException as exc:  # noqa: BLE001
        _emit(_exception_diagnostic("import", exc))
        return EXIT_REQUEST_FAILED

    request_id: Optional[str] = None
    try:
        try:
            request = parse_request(_read_request(MAX_REQUEST_PAYLOAD_BYTES))
        except BaseException as exc:  # noqa: BLE001
            _emit(_exception_diagnostic("request", exc))
            return EXIT_REQUEST_FAILED
        request_id = request.request_id

        # Revalidate the four pinned assets before any SDK constructor.  The
        # parent's absolute deadline is what bounds this work; no child deadline
        # key is added to the wire.
        from screenocr.formula_assets import verify_assets

        try:
            verify_assets(model_dir, require_manifest=True)
        except BaseException as exc:  # noqa: BLE001
            _emit(_exception_diagnostic("assets", exc))
            _write_frame(saved, build_error(request_id, "load_error",
                                            "the pinned formula assets are not usable"))
            return 0

        try:
            config = read_pinned_config(config_path)
            verify_installed_pins(read_installed_pins(site))
        except BaseException as exc:  # noqa: BLE001
            _emit(_exception_diagnostic("config", exc))
            _write_frame(saved, build_error(request_id, "load_error",
                                            "the pinned formula configuration is not usable"))
            return 0

        try:
            dims = validate_png(request.png_bytes)
        except BaseException as exc:  # noqa: BLE001
            _emit({"error": "png", "type": type(exc).__name__})
            _write_frame(saved, build_error(request_id, "invalid_input",
                                            "the request payload is not a usable PNG image"))
            return 0

        # --- optional SDK import happens only now -------------------------- #
        try:
            from rapid_latex_ocr import LaTeXOCR

            from screenocr.formula_rapid import MAX_TOKENS_CAP, install_raw_decoding
        except BaseException as exc:  # noqa: BLE001
            _emit(_exception_diagnostic("sdk_import", exc))
            _write_frame(saved, build_error(request_id, "load_error",
                                            "the formula SDK could not be imported"))
            return 0

        try:
            engine = LaTeXOCR(
                config_path=config_path,
                image_resizer_path=os.path.join(model_dir, "image_resizer.onnx"),
                encoder_path=os.path.join(model_dir, "encoder.onnx"),
                decoder_path=os.path.join(model_dir, "decoder.onnx"),
                tokenizer_json=os.path.join(model_dir, "tokenizer.json"),
            )
        except BaseException as exc:  # noqa: BLE001
            _emit(_exception_diagnostic("sdk_construct", exc))
            _write_frame(saved, build_error(request_id, "load_error",
                                            "the formula engine could not be constructed"))
            return 0

        try:
            providers = assert_cpu_providers(engine)
            assert_sdk_shape(engine, config)
            install_raw_decoding(engine, max_tokens=MAX_TOKENS_CAP)
        except BaseException as exc:  # noqa: BLE001
            _emit(_exception_diagnostic("sdk_shape", exc))
            _write_frame(saved, build_error(request_id, "load_error",
                                            "the formula engine does not match the pinned shape"))
            return 0

        try:
            pred, elapsed = engine(request.png_bytes)
        except BaseException as exc:  # noqa: BLE001
            _emit(_exception_diagnostic("infer", exc))
            _write_frame(saved, build_error(request_id, "inference_error",
                                            "the formula engine failed on this image"))
            return 0

        try:
            body = validate_raw_body(pred)
        except BaseException as exc:  # noqa: BLE001
            _emit(_exception_diagnostic("body", exc))
            _write_frame(saved, build_error(request_id, "inference_error",
                                            "the engine body failed validation"))
            return 0

        rid = _integrity_rid()
        meta = {
            "python_executable": sys.executable,
            "isolated": bool(sys.flags.isolated),
            "no_site": bool(sys.flags.no_site),
            "no_bytecode": bool(sys.flags.dont_write_bytecode),
            "utf8": bool(sys.flags.utf8_mode),
            "sys_path_profile": CHILD_PROFILE,
            "integrity_rid": rid,
        }
        heavy = {}
        for name in HEAVY_MODULES:
            module = sys.modules.get(name)
            path = getattr(module, "__file__", None) if module is not None else None
            if path:
                heavy[name] = _redact(path)
        _emit({
            "sys_path": [_redact(entry) for entry in sys.path],
            "heavy": heavy,
            "providers": list(providers),
            "dims": list(dims),
            "elapsed_s": round(float(elapsed), 6),
            "body_bytes": len(body.encode("utf-8")),
            "integrity_rid": rid,
            "expected_rid": EXPECTED_INTEGRITY_RID,
        })
        frame = build_success(request_id, body, meta)
        _write_frame(saved, frame)
        return 0
    except BaseException as exc:  # noqa: BLE001 - last-resort typed worker error
        _emit(_exception_diagnostic("unexpected", exc))
        if request_id is not None:
            try:
                _write_frame(saved, build_error(request_id, "worker_error",
                                                "the formula worker failed unexpectedly"))
                return 0
            except BaseException:  # noqa: BLE001
                pass
        return EXIT_INTERNAL
    finally:
        try:
            os.close(saved)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
