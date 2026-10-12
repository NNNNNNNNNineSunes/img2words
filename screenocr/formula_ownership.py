"""Lazy, irreversible Windows job ownership for an explicitly invoked formula child.

This stage-only module is intentionally not imported by the product yet.  Importing
it performs no native load or process/resource operation.  The successful unnamed
job handle is retained for this module's entire process lifetime.
"""
from __future__ import annotations

import ctypes
import os
import threading
from ctypes import wintypes

__all__ = ["OwnershipUnavailable", "ensure_formula_process_ownership"]

_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_HANDLE_FLAG_INHERIT = 0x00000001

_lock = threading.Lock()
_state = "new"  # new -> ready | failed; failures are terminal
_job_handle = None  # strong module-level lifetime; intentionally never closed
_failure = None


class OwnershipUnavailable(RuntimeError):
    """The process cannot establish its required formula-child ownership."""


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _BASIC_LIMIT(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _EXTENDED_LIMIT(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _BASIC_LIMIT),
                ("IoInfo", _IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)]


def _load_native_api():
    """Load and fully type the Win32 API only at explicit initialization time."""
    if os.name != "nt":
        raise OSError("formula process ownership is supported only on Windows")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    HANDLE = ctypes.c_void_p
    DWORD = wintypes.DWORD
    BOOL = wintypes.BOOL
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = HANDLE
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = HANDLE
    kernel32.SetInformationJobObject.argtypes = [HANDLE, ctypes.c_int, ctypes.c_void_p, DWORD]
    kernel32.SetInformationJobObject.restype = BOOL
    kernel32.QueryInformationJobObject.argtypes = [HANDLE, ctypes.c_int, ctypes.c_void_p, DWORD,
                                                    ctypes.POINTER(DWORD)]
    kernel32.QueryInformationJobObject.restype = BOOL
    kernel32.GetHandleInformation.argtypes = [HANDLE, ctypes.POINTER(DWORD)]
    kernel32.GetHandleInformation.restype = BOOL
    kernel32.AssignProcessToJobObject.argtypes = [HANDLE, HANDLE]
    kernel32.AssignProcessToJobObject.restype = BOOL
    kernel32.CloseHandle.argtypes = [HANDLE]
    kernel32.CloseHandle.restype = BOOL
    return kernel32


def _last_error(api):
    getter = getattr(api, "last_error", None)
    if getter:
        return getter()
    return ctypes.get_last_error()


def _check(ok, api, operation):
    if not ok:
        raise OSError(_last_error(api), f"{operation} failed")


def _initialize(api):
    """Perform one attempt. Test seam accepts a fake WinAPI-shaped object."""
    global _job_handle, _state, _failure
    handle = api.CreateJobObjectW(None, None)
    if not handle:
        raise OSError(_last_error(api), "CreateJobObjectW failed")
    assigned = False
    try:
        limits = _EXTENDED_LIMIT()
        limits.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        size = ctypes.sizeof(limits)
        _check(api.SetInformationJobObject(handle, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                                           ctypes.byref(limits), size), api,
               "SetInformationJobObject")
        queried = _EXTENDED_LIMIT()
        returned = wintypes.DWORD(0)
        _check(api.QueryInformationJobObject(handle, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                                             ctypes.byref(queried), ctypes.sizeof(queried),
                                             ctypes.byref(returned)), api,
               "QueryInformationJobObject")
        if queried.BasicLimitInformation.LimitFlags != _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE:
            raise OSError("QueryInformationJobObject returned unexpected LimitFlags")
        if returned.value != ctypes.sizeof(queried):
            raise OSError("QueryInformationJobObject returned an unexpected byte count")
        flags = wintypes.DWORD(0)
        _check(api.GetHandleInformation(handle, ctypes.byref(flags)), api,
               "GetHandleInformation")
        if flags.value & _HANDLE_FLAG_INHERIT:
            raise OSError("job handle is inheritable; refusing self-assignment")
        _check(api.AssignProcessToJobObject(handle, api.GetCurrentProcess()), api,
               "AssignProcessToJobObject(current process)")
        assigned = True
        # Publish immediately; no later error path may close an assigned job.
        _job_handle = handle
        _state = "ready"
        return
    finally:
        if not assigned:
            # This is the one exact close path for an unassigned owned handle.
            api.CloseHandle(handle)


def _ensure_with_api(api):
    """Private deterministic fake seam; public API always supplies native API."""
    global _state, _failure
    with _lock:
        if _state == "ready":
            return None
        if _state == "failed":
            raise OwnershipUnavailable(f"formula process ownership unavailable: {_failure}")
        try:
            _initialize(api)
        except BaseException as exc:
            _state = "failed"
            _failure = f"{type(exc).__name__}: {exc}"
            raise OwnershipUnavailable(
                "formula process ownership could not be established; formula launch is disabled: "
                + _failure
            ) from None
    return None


def ensure_formula_process_ownership() -> None:
    """Establish this process's permanent child-kill job before formula Popen.

    Call only immediately before the first explicit/manual formula launch. A
    failure permanently poisons initialization and must prevent formula Popen.
    """
    global _state, _failure
    if _state == "ready":
        return None
    if _state == "failed":
        raise OwnershipUnavailable(f"formula process ownership unavailable: {_failure}")
    with _lock:
        if _state == "ready":
            return None
        if _state == "failed":
            raise OwnershipUnavailable(f"formula process ownership unavailable: {_failure}")
        try:
            _initialize(_load_native_api())
        except BaseException as exc:
            _state = "failed"
            _failure = f"{type(exc).__name__}: {exc}"
            raise OwnershipUnavailable(
                "formula process ownership could not be established; formula launch is disabled: "
                + _failure
            ) from None
    return None
