"""Engine-neutral bounded process client for the M1a formula worker.

This module owns *process mechanics* only: it launches one child interpreter,
writes one framed request to its anonymous stdin pipe, drains its stdout/stderr
with bounded buffers, enforces one monotonic deadline plus one total cleanup
grace, and returns the validated raw body.  It never imports an optional ML
dependency, Tk, or the application, performs no network/GPU work and has no
clipboard/history side effects.

Windows notes
-------------
* The child is launched with ``-I -S -B -X utf8`` so its ``sys.path`` profile is
  honestly ``stdlib_only`` (no ``.venv`` site-packages) and its stdout is a
  plain anonymous pipe, never a file or named pipe.
* ``CREATE_NO_WINDOW`` keeps the console hidden.  ``bufsize=0`` keeps the pipe
  objects raw (``FileIO``), so a pump thread never blocks inside a
  ``BufferedReader`` lock that another thread would need to close.
* ``subprocess.Popen`` on Windows can sit inside an uninterruptible
  ``CreateProcess`` call.  The launch therefore runs on its own owned daemon
  thread: if the deadline expires before the OS returns, the caller reports
  ``pending_spawn`` and poisons the client, and the launcher's continuation
  terminates and reaps whatever child eventually appears.
"""
from __future__ import annotations

import math
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .formula_protocol import (
    HEADER_SIZE,
    MAX_PNG_BYTES,
    MAX_REPLY_PAYLOAD_BYTES,
    MetaExpectation,
    ProtocolError,
    build_request,
    canonical_path,
    decode_frame,
    parse_reply,
    validate_png_bytes,
    validate_request_id,
)

__all__ = [
    "DEFAULT_SUCCESS_WINDOW_SECONDS",
    "TOTAL_CLEANUP_GRACE_SECONDS",
    "POLL_INTERVAL_SECONDS",
    "STDERR_RETAIN_BYTES",
    "STREAM_CHUNK_BYTES",
    "CHILD_FLAGS",
    "THREAD_PREFIX",
    "LaunchSpec",
    "Diagnostics",
    "FormulaProcessError",
    "UnavailableError",
    "BusyError",
    "CancelledError",
    "TimedOutError",
    "ProtocolFailureError",
    "ProcessFailureError",
    "CleanupFailureError",
    "FormulaProcessClient",
    "child_argv",
    "child_popen_kwargs",
    "success_deadline",
    "resolve_base_interpreter",
]

DEFAULT_SUCCESS_WINDOW_SECONDS = 30.0
TOTAL_CLEANUP_GRACE_SECONDS = 2.0
POLL_INTERVAL_SECONDS = 0.02
STDERR_RETAIN_BYTES = 64 * 1024
STREAM_CHUNK_BYTES = 8 * 1024
STDOUT_LIMIT_BYTES = MAX_REPLY_PAYLOAD_BYTES + HEADER_SIZE + 1
THREAD_PREFIX = "screenocr-formula-"
CHILD_FLAGS = ("-I", "-S", "-B", "-X", "utf8")


# --------------------------------------------------------------------------- #
# patchable seams (tests replace these; production uses the real implementations)
# --------------------------------------------------------------------------- #
_popen: Callable[..., Any] = subprocess.Popen


def _now() -> float:
    """Monotonic clock used for every deadline/grace decision."""
    return time.monotonic()


def _poll(proc: Any) -> Optional[int]:
    """Non-blocking child exit check."""
    return proc.poll()


def _read_chunk(stream: Any, size: int) -> bytes:
    """One bounded read from a raw pipe."""
    return stream.read(size)


def _write_all(stream: Any, data: bytes) -> None:
    """Write every byte of *data*, looping over short writes."""
    view = memoryview(data)
    while view:
        written = stream.write(view)
        if not written:
            raise OSError("short write to the child stdin pipe")
        view = view[written:]
    flush = getattr(stream, "flush", None)
    if callable(flush):
        flush()


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #
class FormulaProcessError(RuntimeError):
    """Base class for every typed client failure."""

    kind = "process"

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        diagnostics: Optional["Diagnostics"] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.diagnostics = diagnostics

    def diagnostics_dict(self) -> Dict[str, Any]:
        return self.diagnostics.as_dict() if self.diagnostics is not None else {}


class UnavailableError(FormulaProcessError):
    kind = "unavailable"


class BusyError(FormulaProcessError):
    kind = "busy"


class CancelledError(FormulaProcessError):
    kind = "cancelled"


class TimedOutError(FormulaProcessError):
    kind = "timed_out"


class ProtocolFailureError(FormulaProcessError):
    kind = "protocol"


class ProcessFailureError(FormulaProcessError):
    kind = "process"


class CleanupFailureError(ProcessFailureError):
    kind = "cleanup_failed"


_ERROR_CLASSES: Dict[str, type] = {
    "unavailable": UnavailableError,
    "invalid_input": ProtocolFailureError,
    "load_error": ProcessFailureError,
    "inference_error": ProcessFailureError,
    "worker_error": ProcessFailureError,
}


# --------------------------------------------------------------------------- #
# launch specification
# --------------------------------------------------------------------------- #
def _require_absolute(path: Any, label: str) -> str:
    if not isinstance(path, str) or not path:
        raise ProtocolError(f"{label} must be a non-empty string")
    if "\x00" in path:
        raise ProtocolError(f"{label} must not contain NUL")
    if not os.path.isabs(path):
        raise ProtocolError(f"{label} must be an absolute path")
    return os.path.abspath(path)


@dataclass(frozen=True, eq=False)
class LaunchSpec:
    """Immutable launch description: interpreter, child script, argv, env, meta.

    The expected metadata is part of the spec, so a reply can only be accepted
    if the child reports exactly the launch facts this spec promised.
    """

    python_executable: str
    script_path: str
    expected_meta: MetaExpectation
    argv: Tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    working_directory: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "python_executable", _require_absolute(self.python_executable, "python_executable")
        )
        object.__setattr__(
            self, "script_path", _require_absolute(self.script_path, "script_path")
        )
        if not isinstance(self.argv, tuple):
            object.__setattr__(self, "argv", tuple(self.argv))
        for index, item in enumerate(self.argv):
            if not isinstance(item, str):
                raise ProtocolError(f"argv[{index}] must be a string")
            if "\x00" in item:
                raise ProtocolError(f"argv[{index}] must not contain NUL")
        if not isinstance(self.env, Mapping):
            raise ProtocolError("env must be a mapping of str to str")
        env: Dict[str, str] = {}
        for key, value in self.env.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ProtocolError("env must map str to str")
            if "\x00" in key or "\x00" in value:
                raise ProtocolError("env entries must not contain NUL")
            env[key] = value
        object.__setattr__(self, "env", MappingProxyType(env))
        if self.working_directory is not None:
            object.__setattr__(
                self,
                "working_directory",
                _require_absolute(self.working_directory, "working_directory"),
            )
        if not isinstance(self.expected_meta, MetaExpectation):
            raise ProtocolError("expected_meta must be a MetaExpectation")
        if canonical_path(self.python_executable) != canonical_path(
            self.expected_meta.python_executable
        ):
            raise ProtocolError(
                "expected_meta.python_executable must match python_executable"
            )


def child_argv(spec: LaunchSpec) -> List[str]:
    """The exact child argv: base interpreter, isolation flags, script, extras."""
    return [spec.python_executable, *CHILD_FLAGS, spec.script_path, *spec.argv]


def child_popen_kwargs(spec: LaunchSpec) -> Dict[str, Any]:
    """The exact Popen keyword arguments (anonymous pipes, hidden console)."""
    kwargs: Dict[str, Any] = {
        "stdin": subprocess.PIPE,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "bufsize": 0,
        "env": dict(spec.env),
    }
    if spec.working_directory is not None:
        kwargs["cwd"] = spec.working_directory
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return kwargs


def success_deadline(start: Optional[float] = None, window: float = DEFAULT_SUCCESS_WINDOW_SECONDS) -> float:
    """Default application window: request start + 30s (never reset later)."""
    return (_now() if start is None else start) + window


# --------------------------------------------------------------------------- #
# base interpreter resolution
# --------------------------------------------------------------------------- #
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_DIR_NAME = ".venv"
PYVENV_CFG = "pyvenv.cfg"


def _inside(path: str, directory: str) -> bool:
    key = os.path.normcase(os.path.normpath(path))
    root = os.path.normcase(os.path.normpath(directory)).rstrip("\\/")
    return key == root or key.startswith(root + os.sep)


def resolve_base_interpreter(
    project_root: Optional[str] = None, venv_dir: Optional[str] = None
) -> str:
    """Discover the original unlabelled base interpreter from ``pyvenv.cfg``.

    The workspace ``.venv\\Scripts\\python.exe`` launcher carries a low
    integrity label, so launching through it would silently downgrade the child.
    This mirrors ``launch.ps1``: read the recorded base interpreter, then refuse
    any answer that lives inside the virtual environment.  Nothing here imports
    an optional dependency or executes the interpreter.
    """
    root = os.path.abspath(project_root or PROJECT_ROOT)
    venv = os.path.abspath(venv_dir or os.path.join(root, VENV_DIR_NAME))
    config_path = os.path.join(venv, PYVENV_CFG)
    if not os.path.isfile(config_path):
        raise UnavailableError(
            f"cannot resolve the base interpreter: {config_path} is missing"
        )
    recorded: List[str] = []
    try:
        with open(config_path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                value = value.strip().strip('"')
                if not value:
                    continue
                if key.strip().lower() in ("executable", "home"):
                    recorded.append(value)
    except OSError as exc:
        raise UnavailableError(
            f"cannot read {config_path}: {type(exc).__name__}: {exc}"
        ) from None

    candidates: List[str] = []
    for value in recorded:
        if os.path.basename(value).lower().endswith(".exe"):
            candidates.append(value)
        else:
            candidates.append(os.path.join(value, "python.exe"))
    for candidate in candidates:
        absolute = os.path.abspath(candidate)
        if not os.path.isfile(absolute):
            continue
        if _inside(absolute, venv):
            continue
        return absolute
    raise UnavailableError(
        "cannot resolve an original base interpreter outside "
        f"{venv} from {config_path}"
    )


# --------------------------------------------------------------------------- #
# bounded buffers / threads
# --------------------------------------------------------------------------- #
class _Sink:
    """Bounded byte buffer; optionally discards (but counts) excess."""

    def __init__(self, limit: int, discard: bool) -> None:
        self.limit = limit
        self.discard = discard
        self._buffer = bytearray()
        self._total = 0
        self._discarded = 0
        self._overflow = False
        self._lock = threading.Lock()

    def append(self, data: bytes) -> None:
        with self._lock:
            self._total += len(data)
            room = self.limit - len(self._buffer)
            if room > 0:
                self._buffer.extend(data[:room])
            excess = len(data) - max(room, 0)
            if excess > 0:
                if self.discard:
                    self._discarded += excess
                else:
                    self._overflow = True

    def snapshot(self) -> bytes:
        with self._lock:
            return bytes(self._buffer)

    @property
    def total(self) -> int:
        with self._lock:
            return self._total

    @property
    def discarded(self) -> int:
        with self._lock:
            return self._discarded

    @property
    def overflow(self) -> bool:
        with self._lock:
            return self._overflow


class _Pump(threading.Thread):
    """Daemon reader that drains one pipe into a bounded sink."""

    def __init__(self, stream: Any, sink: _Sink, label: str) -> None:
        super().__init__(name=f"{THREAD_PREFIX}pump-{label}", daemon=True)
        self.stream = stream
        self.sink = sink
        self.label = label
        self.error: Optional[BaseException] = None
        self.done = threading.Event()

    def run(self) -> None:
        try:
            while True:
                chunk = _read_chunk(self.stream, STREAM_CHUNK_BYTES)
                if not chunk:
                    break
                self.sink.append(chunk)
        except BaseException as exc:  # noqa: BLE001 - reported, never raised here
            self.error = exc
        finally:
            self.done.set()
            try:
                self.stream.close()
            except Exception:
                pass


class _Writer(threading.Thread):
    """Daemon writer: sends the whole frame, then closes stdin (EOF)."""

    def __init__(self, stream: Any, data: bytes) -> None:
        super().__init__(name=f"{THREAD_PREFIX}writer", daemon=True)
        self.stream = stream
        self.data = data
        self.error: Optional[BaseException] = None
        self.done = threading.Event()

    def run(self) -> None:
        try:
            _write_all(self.stream, self.data)
        except BaseException as exc:  # noqa: BLE001 - reported, never raised here
            self.error = exc
        finally:
            try:
                self.stream.close()
            except Exception:
                pass
            self.done.set()


class _Active:
    """Per-request cancellation/launch state shared with the launcher thread."""

    __slots__ = ("caller_cancel", "cancel", "lock", "spawn_done", "spawn_error", "proc", "abandoned", "launch_committed")

    def __init__(self, caller_cancel: Any) -> None:
        self.caller_cancel = caller_cancel
        self.cancel = threading.Event()
        self.lock = threading.Lock()
        self.spawn_done = threading.Event()
        self.spawn_error: Optional[BaseException] = None
        self.proc: Any = None
        self.abandoned = False
        self.launch_committed = False

    def cancelled(self) -> bool:
        return self.cancel.is_set() or self.caller_cancel.is_set()

    def abandon(self) -> bool:
        """Claim the abandon right; False means the launcher already published."""
        with self.lock:
            if self.spawn_done.is_set():
                return False
            self.abandoned = True
            return True


class _Launcher(threading.Thread):
    """Owned daemon that performs the possibly uninterruptible OS spawn."""

    def __init__(self, client: "FormulaProcessClient", active: _Active) -> None:
        super().__init__(name=f"{THREAD_PREFIX}launcher", daemon=True)
        self.client = client
        self.active = active

    def run(self) -> None:
        active = self.active
        try:
            proc = _popen(self.client.child_argv(), **self.client.child_popen_kwargs())
        except BaseException as exc:  # noqa: BLE001 - handed to the caller
            with active.lock:
                active.spawn_error = exc
                active.spawn_done.set()
            return
        with active.lock:
            active.proc = proc
            abandoned = active.abandoned
            active.spawn_done.set()
        if abandoned:
            # The caller already gave up: own this late child and clean it up.
            self.client._late_cleanup(proc)


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #
@dataclass
class Diagnostics:
    """Bounded, truthful record of one request attempt."""

    stage: str = "init"
    started: float = 0.0
    elapsed: float = 0.0
    request_bytes: int = 0
    png_bytes: int = 0
    reply_bytes: int = 0
    stderr_retained: int = 0
    stderr_discarded: int = 0
    stdout_overflow: bool = False
    spawned: bool = False
    pid: Optional[int] = None
    exited: Optional[int] = None
    reaped: bool = False
    threads_joined: bool = False
    pending_spawn: bool = False
    cleanup_ms: float = 0.0
    terminate_error: Optional[str] = None
    kill_error: Optional[str] = None
    wait_error: Optional[str] = None
    writer_error: Optional[str] = None
    reader_error: Optional[str] = None
    deadline_hit: bool = False
    cancel_hit: bool = False
    poison_reason: Optional[str] = None
    handles_closed: bool = False
    notes: List[str] = field(default_factory=list)

    def note(self, text: str) -> None:
        if len(self.notes) < 16:
            self.notes.append(text)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "elapsed_s": round(self.elapsed, 6),
            "request_bytes": self.request_bytes,
            "png_bytes": self.png_bytes,
            "reply_bytes": self.reply_bytes,
            "stderr_retained": self.stderr_retained,
            "stderr_discarded": self.stderr_discarded,
            "stdout_overflow": self.stdout_overflow,
            "spawned": self.spawned,
            "pid": self.pid,
            "exited": self.exited,
            "reaped": self.reaped,
            "threads_joined": self.threads_joined,
            "pending_spawn": self.pending_spawn,
            "cleanup_ms": round(self.cleanup_ms, 3),
            "terminate_error": self.terminate_error,
            "kill_error": self.kill_error,
            "wait_error": self.wait_error,
            "writer_error": self.writer_error,
            "reader_error": self.reader_error,
            "deadline_hit": self.deadline_hit,
            "cancel_hit": self.cancel_hit,
            "poison_reason": self.poison_reason,
            "handles_closed": self.handles_closed,
            "notes": list(self.notes),
        }


def _describe(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".replace("\x00", "\\0")
    return text[:256]


# --------------------------------------------------------------------------- #
# client
# --------------------------------------------------------------------------- #
class FormulaProcessClient:
    """One-request-at-a-time bounded client for an engine-neutral child process."""

    def __init__(
        self,
        launch_spec: LaunchSpec,
        *,
        success_window: float = DEFAULT_SUCCESS_WINDOW_SECONDS,
        cleanup_grace: float = TOTAL_CLEANUP_GRACE_SECONDS,
        poll_interval: float = POLL_INTERVAL_SECONDS,
    ) -> None:
        if not isinstance(launch_spec, LaunchSpec):
            raise ProtocolError("launch_spec must be a LaunchSpec")
        if not (0 < float(cleanup_grace) <= TOTAL_CLEANUP_GRACE_SECONDS):
            raise ProtocolError(
                f"cleanup_grace must be in (0, {TOTAL_CLEANUP_GRACE_SECONDS}]"
            )
        if not (0 < float(poll_interval) <= 0.05):
            raise ProtocolError("poll_interval must be in (0, 0.05]")
        if not (0 < float(success_window)):
            raise ProtocolError("success_window must be positive")
        self._spec = launch_spec
        self._success_window = float(success_window)
        self._cleanup_grace = float(cleanup_grace)
        self._poll_interval = float(poll_interval)
        self._slot = threading.Lock()
        self._state = threading.Lock()
        self._closed = False
        self._poison: Optional[str] = None
        self._active: Optional[_Active] = None
        self._last_diagnostics: Optional[Diagnostics] = None
        self._late = {"count": 0, "reaped": 0, "failed": 0, "last_error": None}

    # -- introspection ----------------------------------------------------- #
    @property
    def spec(self) -> LaunchSpec:
        return self._spec

    @property
    def closed(self) -> bool:
        with self._state:
            return self._closed

    @property
    def poisoned(self) -> bool:
        with self._state:
            return self._poison is not None

    @property
    def active(self) -> bool:
        with self._state:
            return self._active is not None

    @property
    def last_diagnostics(self) -> Optional[Diagnostics]:
        return self._last_diagnostics

    def diagnostics(self) -> Dict[str, Any]:
        diag = self._last_diagnostics
        with self._state:
            snapshot = {
                "closed": self._closed,
                "poisoned": self._poison,
                "late_cleanups": dict(self._late),
            }
        snapshot["last"] = diag.as_dict() if diag is not None else None
        return snapshot

    # -- child configuration ----------------------------------------------- #
    def child_argv(self) -> List[str]:
        return child_argv(self._spec)

    def child_popen_kwargs(self) -> Dict[str, Any]:
        return child_popen_kwargs(self._spec)

    # -- control ----------------------------------------------------------- #
    def cancel_active(self) -> None:
        """Signal the active request to stop.  Never terminates anything here."""
        with self._state:
            active = self._active
        if active is not None:
            active.cancel.set()

    def close(self) -> None:
        """Permanently refuse new work and signal the active request."""
        with self._state:
            self._closed = True
            active = self._active
        if active is not None:
            active.cancel.set()

    # -- public API -------------------------------------------------------- #
    def recognize_payload(
        self,
        png_bytes: bytes,
        request_id: str,
        cancel_event: Any,
        absolute_deadline: Optional[float] = None,
    ) -> str:
        """Run one request; return the validated raw body or raise a typed error."""
        diag = Diagnostics(started=_now())
        self._last_diagnostics = diag
        request_entered = False
        slot_acquired = False
        try:
            try:
                # The default window includes all preparation time from call entry.
                deadline = self._resolve_deadline(absolute_deadline, diag.started)
                request_id = validate_request_id(request_id)
                data = validate_png_bytes(png_bytes)
                cancel_event = self._coerce_event(cancel_event)
            except ProtocolError as exc:
                diag.stage = "preflight"
                raise ProtocolFailureError(str(exc), diagnostics=diag) from None
            diag.png_bytes = len(data)

            with self._state:
                closed = self._closed
                poison = self._poison
            if closed:
                diag.stage = "preflight"
                raise UnavailableError("the client is closed", diagnostics=diag)
            if poison is not None:
                diag.stage = "preflight"
                diag.poison_reason = poison
                raise UnavailableError(
                    f"the client is poisoned: {poison}", diagnostics=diag
                )
            reason = self._stop_reason(cancel_event, deadline, None)
            if reason is not None:
                diag.stage = "preflight"
                self._raise_stop(reason, diag)

            if not self._slot.acquire(blocking=False):
                diag.stage = "preflight"
                raise BusyError("another request is already active", diagnostics=diag)
            slot_acquired = True
            request_entered = True
            return self._recognize(request_id, data, cancel_event, deadline, diag)
        finally:
            try:
                # Keep this caller's slot until diagnostics and the final success
                # guard have completed. Otherwise a later request can publish its
                # active handle and have this finalizer mistake it for its own.
                diag.elapsed = max(0.0, _now() - diag.started)
                self._last_diagnostics = diag
                with self._state:
                    active = self._active if request_entered else None
                try:
                    if diag.stage == "finalize" and diag.reaped and diag.threads_joined:
                        reason = self._stop_reason(cancel_event, deadline, active)
                        if reason is not None:
                            if reason == "cancel":
                                diag.cancel_hit = True
                            else:
                                diag.deadline_hit = True
                            diag.elapsed = max(0.0, _now() - diag.started)
                            self._last_diagnostics = diag
                            self._raise_stop(reason, diag, launched=True)
                finally:
                    # Clear only the active object observed for this request, and
                    # complete every shared-state operation before slot release.
                    with self._state:
                        if self._active is active:
                            self._active = None
            finally:
                if slot_acquired:
                    self._slot.release()

    # -- internals --------------------------------------------------------- #
    def _resolve_deadline(self, value: Optional[float], start: Optional[float] = None) -> float:
        if value is None:
            origin = _now() if start is None else start
            return origin + self._success_window
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProtocolError("absolute_deadline must be a finite number or None")
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            raise ProtocolError("absolute_deadline must be finite")
        return number

    @staticmethod
    def _coerce_event(cancel_event: Any) -> Any:
        if not callable(getattr(cancel_event, "is_set", None)):
            raise ProtocolError("cancel_event must provide is_set()")
        return cancel_event

    def _stop_reason(
        self, cancel_event: Any, deadline: float, active: Optional[_Active]
    ) -> Optional[str]:
        if active is not None and active.cancelled():
            return "cancel"
        if cancel_event is not None and cancel_event.is_set():
            return "cancel"
        with self._state:
            closed = self._closed
        if closed:
            return "cancel"
        if _now() >= deadline:
            return "deadline"
        return None

    def _raise_stop(self, reason: str, diag: Diagnostics, launched: bool = False) -> None:
        if reason == "cancel":
            diag.cancel_hit = True
            raise CancelledError(
                "the request was cancelled after the child was launched"
                if launched
                else "the request was cancelled",
                diagnostics=diag,
            )
        diag.deadline_hit = True
        raise TimedOutError(
            "the deadline expired after the child was launched but before any reply"
            if launched
            else "the request deadline expired before any child was launched",
            diagnostics=diag,
        )

    def _mark_poisoned(self, reason: str) -> None:
        with self._state:
            if self._poison is None:
                self._poison = reason

    def _recognize(
        self,
        request_id: str,
        data: bytes,
        cancel_event: Any,
        deadline: float,
        diag: Diagnostics,
    ) -> str:
        diag.stage = "serialize"
        try:
            frame = build_request(data, request_id)
        except ProtocolError as exc:
            raise ProtocolFailureError(str(exc), diagnostics=diag) from None
        diag.request_bytes = len(frame)
        if _now() >= deadline:
            diag.deadline_hit = True
            raise TimedOutError(
                "the deadline expired while serializing the request", diagnostics=diag
            )

        active = _Active(cancel_event)
        with self._state:
            self._active = active
        proc: Any = None
        pumps: List[_Pump] = []
        writer: Optional[_Writer] = None
        out_sink: Optional[_Sink] = None
        err_sink: Optional[_Sink] = None
        try:
            diag.stage = "spawn"
            proc, pending, stop, cleanup_cutoff = self._spawn_bounded(
                active, deadline, diag
            )
            if pending:
                diag.pending_spawn = True
                self._mark_poisoned("pending_spawn: the child could not be confirmed launched")
                diag.poison_reason = "pending_spawn"
                diag.note("the launcher continuation owns any eventual late child")
                if stop == "deadline":
                    diag.deadline_hit = True
                    raise TimedOutError(
                        "the child launch did not complete before the deadline "
                        "(pending_spawn); the client is poisoned",
                        diagnostics=diag,
                    )
                diag.cancel_hit = True
                raise CancelledError(
                    "the request was cancelled while the child launch was pending; "
                    "the client is poisoned",
                    diagnostics=diag,
                )
            # Popen has returned an owned process, even if a stop raced its
            # committed launch. Publish those facts before any cleanup/errors.
            diag.spawned = proc is not None
            diag.pid = getattr(proc, "pid", None) if proc is not None else None
            if stop is not None:
                if proc is not None:
                    self._cleanup(proc, [], None, diag, stop, cleanup_cutoff)
                    self._raise_stop(stop, diag, launched=True)
                self._raise_stop(stop, diag)

            diag.stage = "exchange"
            out_sink = _Sink(STDOUT_LIMIT_BYTES, discard=False)
            err_sink = _Sink(STDERR_RETAIN_BYTES, discard=True)
            pumps = [
                _Pump(proc.stdout, out_sink, "stdout"),
                _Pump(proc.stderr, err_sink, "stderr"),
            ]
            writer = _Writer(proc.stdin, frame)
            for thread in (*pumps, writer):
                thread.start()

            stop = self._await(active, proc, writer, pumps, out_sink, deadline, diag)
            if stop is not None:
                self._cleanup(proc, pumps, writer, diag, stop)
                if stop == "deadline":
                    diag.deadline_hit = True
                    raise TimedOutError(
                        "the request deadline expired before a complete reply "
                        "and process exit",
                        diagnostics=diag,
                    )
                if stop == "cancel":
                    diag.cancel_hit = True
                    raise CancelledError(
                        "the request was cancelled before a complete reply "
                        "and process exit",
                        diagnostics=diag,
                    )
                self._raise_io_failure(stop, diag)

            diag.stage = "verify"
            if _now() >= deadline:
                self._cleanup(proc, pumps, writer, diag, "deadline")
                diag.deadline_hit = True
                raise TimedOutError(
                    "the deadline expired after the child exited", diagnostics=diag
                )
            try:
                exit_code = _poll(proc)
            except Exception as exc:  # noqa: BLE001
                diag.wait_error = _describe(exc)
                self._cleanup(proc, pumps, writer, diag, "wait_error")
                raise ProcessFailureError(
                    f"failed to wait for the child: {diag.wait_error}",
                    diagnostics=diag,
                ) from None
            diag.exited = exit_code
            if exit_code is None:
                self._cleanup(proc, pumps, writer, diag, "no_exit")
                raise ProcessFailureError(
                    "the child output ended but the process had not exited",
                    diagnostics=diag,
                )
            if out_sink.overflow:
                self._cleanup(proc, pumps, writer, diag, "overflow")
                raise ProtocolFailureError(
                    "the child wrote more than the reply limit to stdout",
                    diagnostics=diag,
                )
            if exit_code != 0:
                self._cleanup(proc, pumps, writer, diag, "exit")
                raise ProcessFailureError(
                    f"the child exited with code {exit_code}; a nonzero exit can "
                    "never be a successful result",
                    diagnostics=diag,
                )

            diag.stage = "parse"
            try:
                payload = decode_frame(out_sink.snapshot(), MAX_REPLY_PAYLOAD_BYTES)
                reply = parse_reply(payload, request_id, self._spec.expected_meta)
            except ProtocolError as exc:
                self._cleanup(proc, pumps, writer, diag, "protocol")
                raise ProtocolFailureError(str(exc), diagnostics=diag) from None
            diag.reply_bytes = len(payload)

            diag.stage = "finalize"
            if _now() >= deadline:
                self._cleanup(proc, pumps, writer, diag, "deadline")
                diag.deadline_hit = True
                raise TimedOutError(
                    "the deadline expired while validating the reply", diagnostics=diag
                )
            if reply.status == "error":
                self._cleanup(proc, pumps, writer, diag, "error_frame")
                self._raise_worker_error(reply.code or "worker_error", reply.message or "", diag)

            if not self._finalize_success(proc, pumps, writer, diag):
                self._mark_poisoned("cleanup_failed: the child or its IO threads were not reaped")
                diag.poison_reason = "cleanup_failed"
                raise CleanupFailureError(
                    "the reply was valid but the child could not be fully reaped; "
                    "the client is poisoned",
                    diagnostics=diag,
                )
            # Parsing/finalization may itself be slow. Success is eligible only
            # after all reaping and diagnostics are final, against the original
            # deadline and current caller/client stop signals.
            stop = self._stop_reason(cancel_event, deadline, active)
            if stop is not None:
                if stop == "cancel":
                    diag.cancel_hit = True
                else:
                    diag.deadline_hit = True
                self._raise_stop(stop, diag, launched=True)
            return reply.raw_body or ""
        finally:
            if out_sink is not None:
                diag.stdout_overflow = out_sink.overflow
            if err_sink is not None:
                diag.stderr_retained = len(err_sink.snapshot())
                diag.stderr_discarded = err_sink.discarded

    # -- launch ------------------------------------------------------------ #
    def _spawn_bounded(
        self, active: _Active, deadline: float, diag: Diagnostics
    ) -> Tuple[Any, bool, Optional[str], Optional[float]]:
        """Wait for the OS spawn; return process, pending, stop and grace cutoff.

        A cancel/deadline that lands *during* the spawn does not immediately
        abandon the child: ``CreateProcess`` cannot be interrupted, so the
        caller waits one bounded cleanup grace for the OS call to return.  Only
        if that grace expires is the spawn abandoned (and the client poisoned),
        because at that point the caller can no longer confirm ownership.
        """
        # Commit under the client state lock. close() either wins before this
        # point (zero Popen) or signals a committed launch without blocking.
        with self._state:
            if self._closed:
                reason = "cancel"
            elif active.cancelled():
                reason = "cancel"
            elif _now() >= deadline:
                reason = "deadline"
            else:
                reason = None
                active.launch_committed = True
        if reason is not None:
            return None, False, reason, None

        launcher = _Launcher(self, active)
        launcher.start()
        stop: Optional[str] = None
        give_up_at: Optional[float] = None
        while True:
            with active.lock:
                spawn_done = active.spawn_done.is_set()
            reason = self._stop_reason(active.caller_cancel, deadline, active)
            if reason is not None and stop is None:
                stop = reason
                give_up_at = _now() + self._cleanup_grace
            if spawn_done:
                break
            if stop is not None and give_up_at is not None and _now() >= give_up_at:
                if active.abandon():
                    return None, True, stop, give_up_at
                break
            time.sleep(self._poll_interval)
        error = active.spawn_error
        if error is not None:
            raise UnavailableError(
                f"cannot launch the child process: {_describe(error)}",
                diagnostics=diag,
            )
        return active.proc, False, stop, give_up_at

    def _late_cleanup(self, proc: Any) -> None:
        """Continuation for a child that appeared after the caller gave up."""
        reaped = False
        error: Optional[str] = None
        deadline = _now() + self._cleanup_grace
        try:
            try:
                if _poll(proc) is None:
                    proc.terminate()
            except Exception as exc:  # noqa: BLE001
                error = _describe(exc)
            while _now() < deadline:
                try:
                    if _poll(proc) is not None:
                        reaped = True
                        break
                except Exception as exc:  # noqa: BLE001
                    error = _describe(exc)
                    break
                time.sleep(self._poll_interval)
            if not reaped:
                try:
                    proc.kill()
                except Exception as exc:  # noqa: BLE001
                    error = _describe(exc)
                try:
                    proc.wait(timeout=max(0.0, deadline - _now()))
                    reaped = True
                except Exception as exc:  # noqa: BLE001
                    error = _describe(exc)
        finally:
            for stream_name in ("stdin", "stdout", "stderr"):
                stream = getattr(proc, stream_name, None)
                close = getattr(stream, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
        with self._state:
            self._late["count"] += 1
            if reaped:
                self._late["reaped"] += 1
            else:
                self._late["failed"] += 1
            self._late["last_error"] = error

    # -- exchange ---------------------------------------------------------- #
    def _await(
        self,
        active: _Active,
        proc: Any,
        writer: _Writer,
        pumps: Sequence[_Pump],
        out_sink: _Sink,
        deadline: float,
        diag: Diagnostics,
    ) -> Optional[str]:
        while True:
            reason = self._stop_reason(active.caller_cancel, deadline, active)
            if reason is not None:
                return reason
            if out_sink.overflow:
                return "overflow"
            if writer.error is not None:
                diag.writer_error = _describe(writer.error)
                return "writer_error"
            for pump in pumps:
                if pump.error is not None:
                    diag.reader_error = _describe(pump.error)
                    return "reader_error"
            if writer.done and all(pump.done for pump in pumps):
                try:
                    if _poll(proc) is not None:
                        return None
                except Exception as exc:  # noqa: BLE001
                    diag.wait_error = _describe(exc)
                    return "wait_error"
            time.sleep(self._poll_interval)

    def _exit_code(self, proc: Any, diag: Diagnostics) -> Optional[int]:
        try:
            code = _poll(proc)
        except Exception as exc:  # noqa: BLE001
            diag.wait_error = _describe(exc)
            return None
        diag.exited = code
        return code

    def _raise_io_failure(self, stop: str, diag: Diagnostics) -> None:
        if stop == "overflow":
            raise ProtocolFailureError(
                "the child wrote more than the reply limit to stdout",
                diagnostics=diag,
            )
        if stop == "writer_error":
            raise ProcessFailureError(
                f"failed to send the request: {diag.writer_error}", diagnostics=diag
            )
        if stop == "reader_error":
            raise ProcessFailureError(
                f"failed to read the child output: {diag.reader_error}", diagnostics=diag
            )
        raise ProcessFailureError(
            f"failed to wait for the child: {diag.wait_error}", diagnostics=diag
        )

    def _raise_worker_error(self, code: str, message: str, diag: Diagnostics) -> None:
        cls = _ERROR_CLASSES.get(code, ProcessFailureError)
        raise cls(
            f"the child reported {code}: {message}"[:512], code=code, diagnostics=diag
        )

    # -- cleanup ----------------------------------------------------------- #
    def _cleanup(
        self,
        proc: Any,
        pumps: Sequence[_Pump],
        writer: Optional[_Writer],
        diag: Diagnostics,
        reason: str,
        cleanup_cutoff: Optional[float] = None,
    ) -> None:
        """Terminate, reap and join within ONE total monotonic grace budget."""
        started = _now()
        deadline = cleanup_cutoff if cleanup_cutoff is not None else started + self._cleanup_grace
        threads: List[threading.Thread] = list(pumps)
        if writer is not None:
            threads.append(writer)
        diag.stage = f"cleanup:{reason}"
        exited: Optional[int] = None
        try:
            try:
                if _poll(proc) is None:
                    proc.terminate()
            except Exception as exc:  # noqa: BLE001
                diag.terminate_error = _describe(exc)
            while _now() < deadline:
                try:
                    code = _poll(proc)
                except Exception as exc:  # noqa: BLE001
                    diag.wait_error = _describe(exc)
                    break
                if code is not None:
                    exited = code
                    break
                time.sleep(min(self._poll_interval, max(0.0, deadline - _now())))
            if exited is None:
                try:
                    proc.kill()
                except Exception as exc:  # noqa: BLE001
                    diag.kill_error = _describe(exc)
                try:
                    exited = proc.wait(timeout=max(0.0, deadline - _now()))
                except Exception as exc:  # noqa: BLE001
                    diag.wait_error = _describe(exc)
        finally:
            joined = True
            for thread in threads:
                remaining = deadline - _now()
                if remaining > 0:
                    thread.join(remaining)
                if thread.is_alive():
                    joined = False
            diag.threads_joined = joined
            diag.exited = exited if exited is not None else diag.exited
            diag.reaped = diag.exited is not None
            diag.handles_closed = self._close_streams(proc, threads)
            diag.cleanup_ms = max(0.0, _now() - started) * 1000.0
        if not diag.reaped or not diag.threads_joined:
            self._mark_poisoned(
                "cleanup_failed: child reaped=%s threads_joined=%s"
                % (diag.reaped, diag.threads_joined)
            )
            diag.poison_reason = "cleanup_failed"

    def _finalize_success(
        self,
        proc: Any,
        pumps: Sequence[_Pump],
        writer: _Writer,
        diag: Diagnostics,
    ) -> bool:
        started = _now()
        deadline = started + self._cleanup_grace
        threads: List[threading.Thread] = [*pumps, writer]
        diag.stage = "finalize"
        diag.exited = self._exit_code(proc, diag)
        joined = True
        for thread in threads:
            remaining = deadline - _now()
            if thread.is_alive() and remaining > 0:
                thread.join(remaining)
            if thread.is_alive():
                joined = False
        diag.threads_joined = joined
        diag.reaped = diag.exited is not None
        diag.handles_closed = self._close_streams(proc, threads)
        diag.cleanup_ms = max(0.0, _now() - started) * 1000.0
        return bool(diag.reaped and diag.threads_joined)

    @staticmethod
    def _close_streams(proc: Any, threads: Sequence[threading.Thread]) -> bool:
        """Close pipes only when no owned thread can still be using them."""
        names = {thread.name for thread in threads}
        ok = True
        pairs = (
            ("stdin", f"{THREAD_PREFIX}writer"),
            ("stdout", f"{THREAD_PREFIX}pump-stdout"),
            ("stderr", f"{THREAD_PREFIX}pump-stderr"),
        )
        for attribute, owner in pairs:
            if owner in names:
                thread = next(t for t in threads if t.name == owner)
                if thread.is_alive():
                    ok = False
                    continue
            stream = getattr(proc, attribute, None)
            close = getattr(stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    ok = False
        return ok
