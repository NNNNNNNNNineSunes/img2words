"""Task-scoped, passive Escape observer for formula capture (Windows only).

Import and construction are inert. The observer owns only plain Python state and
its native monitor thread; GUI integration is deliberately outside this module.
"""
from __future__ import annotations

import ctypes
import threading
from typing import Any, Optional


WH_KEYBOARD_LL = 13
WM_KEYDOWN = 0x0100
WM_SYSKEYDOWN = 0x0104
WM_QUIT = 0x0012
PM_NOREMOVE = 0x0000
VK_ESCAPE = 0x1B


class EscapeUnavailable(RuntimeError):
    """The passive formula Escape observer could not be safely armed."""


# A strong, module-owned lifetime fence for uncertain native callback ownership.
_registry_lock = threading.Lock()
_retained_observers: set["FormulaEscapeObserver"] = set()
_permanent_fault = threading.Event()


def _reserve_for_start(observer: "FormulaEscapeObserver") -> str:
    """Atomically enforce one uncertain/in-flight observer per module."""
    with _registry_lock:
        for old in tuple(_retained_observers):
            thread = old._thread
            if thread is not None and not thread.is_alive() and old.clean:
                _retained_observers.discard(old)
        if _permanent_fault.is_set():
            return "observer_permanently_unavailable"
        if any(not old.clean for old in _retained_observers):
            return "observer_previous_cleanup_pending"
        with observer._lock:
            if observer._state != "NEW" or observer._stop_requested.is_set():
                return "observer_single_use"
            observer._state = "STARTING"
            observer._diagnostic = "starting"
            _retained_observers.add(observer)
        return ""


def _release_if_clean(observer: "FormulaEscapeObserver") -> None:
    if not observer.clean:
        return
    with _registry_lock:
        _retained_observers.discard(observer)


def _native_api() -> "_NativeApi":
    """Load and bind Win32 APIs lazily; never called by import or construction."""
    return _NativeApi()


class _KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", ctypes.c_uint32),
        ("scanCode", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("time", ctypes.c_uint32),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _NativeApi:
    """Private Win32 ABI adapter. Tests replace _native_api with a fake."""

    def __init__(self) -> None:
        if not hasattr(ctypes, "WinDLL") or not hasattr(ctypes, "WINFUNCTYPE"):
            raise EscapeUnavailable("windows_native_api_unavailable")
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._lresult = ctypes.c_ssize_t
        self._wparam = ctypes.c_size_t
        self._lparam = ctypes.c_ssize_t
        self._hook_proc_type = ctypes.WINFUNCTYPE(
            self._lresult, ctypes.c_int, self._wparam, self._lparam
        )
        self._user32.SetWindowsHookExW.argtypes = [
            ctypes.c_int, self._hook_proc_type, ctypes.c_void_p, ctypes.c_uint32
        ]
        self._user32.SetWindowsHookExW.restype = ctypes.c_void_p
        self._user32.UnhookWindowsHookEx.argtypes = [ctypes.c_void_p]
        self._user32.UnhookWindowsHookEx.restype = ctypes.c_int
        self._user32.CallNextHookEx.argtypes = [
            ctypes.c_void_p, ctypes.c_int, self._wparam, self._lparam
        ]
        self._user32.CallNextHookEx.restype = self._lresult
        self._user32.PeekMessageW.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32
        ]
        self._user32.PeekMessageW.restype = ctypes.c_int
        self._user32.GetMessageW.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32
        ]
        self._user32.GetMessageW.restype = ctypes.c_int
        self._user32.PostThreadMessageW.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32, self._wparam, self._lparam
        ]
        self._user32.PostThreadMessageW.restype = ctypes.c_int
        self._kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
        self._kernel32.GetModuleHandleW.restype = ctypes.c_void_p

    def make_callback(self, callback: Any) -> Any:
        return self._hook_proc_type(callback)

    def ensure_queue(self) -> None:
        # A non-null MSG-sized aligned buffer is sufficient for Peek/GetMessage.
        self._msg = ctypes.create_string_buffer(64)
        self._user32.PeekMessageW(self._msg, None, 0, 0, PM_NOREMOVE)

    def install(self, callback: Any, _thread_id: int) -> Any:
        module = self._kernel32.GetModuleHandleW(None)
        hook = self._user32.SetWindowsHookExW(WH_KEYBOARD_LL, callback, module, 0)
        if not hook:
            raise EscapeUnavailable("hook_registration_failed")
        return hook

    def get_message(self) -> int:
        return int(self._user32.GetMessageW(self._msg, None, 0, 0))

    def post_quit(self, thread_id: int) -> bool:
        return bool(self._user32.PostThreadMessageW(thread_id, WM_QUIT, 0, 0))

    def unhook(self, handle: Any) -> bool:
        return bool(self._user32.UnhookWindowsHookEx(handle))

    def call_next(self, handle: Any, code: int, wparam: int, lparam: int) -> int:
        return int(self._user32.CallNextHookEx(handle, code, wparam, lparam))


def _read_vk_code(lparam: int) -> int:
    data = ctypes.cast(ctypes.c_void_p(lparam), ctypes.POINTER(_KBDLLHOOKSTRUCT)).contents
    return int(data.vkCode)


class FormulaEscapeObserver:
    """Single-use observer that latches Escape-down into a caller-owned Event.

    No native APIs, threads, GUI imports, or registry operations occur until
    ``start``. ``stop`` is nonblocking and permanently deactivates this run.
    """

    def __init__(self, cancel_event: threading.Event) -> None:
        if not isinstance(cancel_event, threading.Event):
            raise TypeError("cancel_event must be a threading.Event")
        self._cancel_event = cancel_event
        self._lock = threading.Lock()
        self._state = "NEW"
        self._stop_requested = threading.Event()
        self._queue_ready = threading.Event()
        self._startup_result = threading.Event()
        self._thread_done = threading.Event()
        self._fault = threading.Event()
        self._clean_unhook = False
        self._active_event: Optional[threading.Event] = None
        self._native_handle: Any = None
        self._native_api_obj: Any = None
        self._callback: Any = None
        self._thread: Optional[threading.Thread] = None
        self._thread_id: Optional[int] = None
        self._startup_ok = False
        self._diagnostic = "idle"

    @property
    def healthy(self) -> bool:
        with self._lock:
            return self._state == "ARMED" and not self._fault.is_set() and not self._stop_requested.is_set()

    @property
    def faulted(self) -> bool:
        return self._fault.is_set()

    @property
    def done(self) -> bool:
        thread = self._thread
        return self._thread_done.is_set() and (thread is None or not thread.is_alive())

    @property
    def clean(self) -> bool:
        with self._lock:
            no_hook = self._native_handle is None
            thread = self._thread
            exited = thread is None or not thread.is_alive()
            return self._thread_done.is_set() and exited and (self._clean_unhook or no_hook) and self._state == "DONE_CLEAN"

    @property
    def diagnostics(self) -> dict[str, str]:
        with self._lock:
            return {"state": self._state, "status": self._diagnostic}

    def start(self, ready_timeout: float = 1.0) -> None:
        if not isinstance(ready_timeout, (int, float)) or ready_timeout < 0 or ready_timeout > 1.0:
            raise ValueError("ready_timeout must be between 0 and 1 second")
        rejection = _reserve_for_start(self)
        if rejection:
            raise EscapeUnavailable(rejection)
        try:
            thread = threading.Thread(target=self._run, name="FormulaEscapeObserver", daemon=True)
            self._thread = thread
            thread.start()
        except BaseException:
            self._set_fault("thread_start_failed")
            self._startup_result.set()
            self._thread_done.set()
            with self._lock:
                self._state = "DONE_CLEAN"
                self._diagnostic = "thread_start_failed"
            _release_if_clean(self)
            raise EscapeUnavailable("thread_start_failed")
        if not self._startup_result.wait(float(ready_timeout)):
            self._set_fault("startup_timeout")
            self.stop()
            raise EscapeUnavailable("startup_timeout")
        if not self._startup_ok:
            raise EscapeUnavailable(self._diagnostic)

    def stop(self) -> None:
        thread_id: Optional[int]
        queue_ready: bool
        with self._lock:
            if self._state == "NEW":
                self._stop_requested.set()
                self._active_event = None
                self._state = "DONE_CLEAN"
                self._diagnostic = "stopped_before_start"
                self._thread_done.set()
                return
            finished = self._thread_done.is_set()
            if finished:
                thread_id = None
                queue_ready = False
            else:
                self._stop_requested.set()
                self._active_event = None
                if self._state not in ("DONE_CLEAN", "FAULTED_UNCLEAN", "FAULTED_PENDING"):
                    self._state = "STOPPING"
                    self._diagnostic = "stopping"
                queue_ready = self._queue_ready.is_set()
                thread_id = self._thread_id
        if finished:
            _release_if_clean(self)
            return
        if queue_ready and thread_id is not None:
            try:
                ok = bool(self._native_api_obj.post_quit(thread_id))
            except BaseException:
                ok = False
            if not ok:
                self._set_fault("quit_post_failed")
        # Deliberately no join: GUI caller must remain nonblocking.

    def _set_fault(self, diagnostic: str) -> None:
        self._fault.set()
        try:
            self._cancel_event.set()
        except BaseException:
            # A broken caller Event must not escape a native callback boundary.
            pass
        with self._lock:
            self._active_event = None
            if self._state != "DONE_CLEAN":
                self._state = "FAULTED_PENDING"
                self._diagnostic = diagnostic

    def _native_callback(self, code: int, message: int, lparam: int) -> int:
        try:
            if code >= 0 and message in (WM_KEYDOWN, WM_SYSKEYDOWN):
                if _read_vk_code(lparam) == VK_ESCAPE:
                    with self._lock:
                        event = self._active_event
                    if event is not None:
                        event.set()
        except BaseException:
            self._set_fault("callback_processing_failed")
        try:
            api = self._native_api_obj
            handle = self._native_handle
            if api is None:
                return 0
            return int(api.call_next(handle, int(code), int(message), int(lparam)))
        except BaseException:
            self._set_fault("call_next_failed")
            return 0

    def _run(self) -> None:
        api = None
        acquired = None
        try:
            api = _native_api()
            self._native_api_obj = api
            api.ensure_queue()
            self._thread_id = threading.get_ident()
            self._queue_ready.set()
            if self._stop_requested.is_set():
                self._startup_ok = False
                self._diagnostic = "stopped_before_registration"
                self._startup_result.set()
                return
            callback = api.make_callback(self._native_callback)
            self._callback = callback
            acquired = api.install(callback, self._thread_id)
            # Own the returned handle before checking the permanent stop latch.
            with self._lock:
                self._native_handle = acquired
            if not acquired:
                raise EscapeUnavailable("hook_registration_failed")
            with self._lock:
                if self._stop_requested.is_set() or self._fault.is_set():
                    self._state = "STOPPING"
                    self._diagnostic = "stopped_during_registration"
                else:
                    self._active_event = self._cancel_event
                    self._state = "ARMED"
                    self._diagnostic = "armed"
                    self._startup_ok = True
            self._startup_result.set()
            if self._stop_requested.is_set() or self._fault.is_set():
                return
            while True:
                result = int(api.get_message())
                if result == 0:
                    break
                if result < 0:
                    raise EscapeUnavailable("message_loop_failed")
            if not self._stop_requested.is_set():
                self._set_fault("unexpected_thread_exit")
        except BaseException as exc:
            # Only fixed labels are exposed; exception text/native details are discarded.
            if not self._startup_result.is_set():
                label = "native_api_unavailable" if isinstance(exc, EscapeUnavailable) else "monitor_failed"
                with self._lock:
                    self._diagnostic = label
                self._startup_ok = False
                self._startup_result.set()
            self._set_fault("monitor_failed" if not isinstance(exc, EscapeUnavailable) else "native_operation_failed")
        finally:
            unhook_ok = acquired is None
            if acquired is not None:
                try:
                    unhook_ok = bool(api.unhook(acquired))
                except BaseException:
                    unhook_ok = False
            if acquired is not None and not unhook_ok:
                _permanent_fault.set()
                self._set_fault("unhook_failed")
            with self._lock:
                self._active_event = None
                self._clean_unhook = bool(unhook_ok)
                self._thread_done.set()
                if unhook_ok:
                    self._state = "DONE_CLEAN"
                    if self._diagnostic in ("armed", "stopping", "stopped_during_registration"):
                        self._diagnostic = "stopped_cleanly"
                else:
                    self._state = "FAULTED_UNCLEAN"
                    self._diagnostic = "unhook_failed"
            self._startup_result.set()
            _release_if_clean(self)
