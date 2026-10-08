"""Read the current Windows integrity level without changing permissions.

A Low-integrity interpreter cannot reliably monitor a normal desktop's input.
This is distinct from whether the user account is an administrator.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import sys

LOW_INTEGRITY = 0x1000
MEDIUM_INTEGRITY = 0x2000


def get_integrity_level() -> int | None:
    """Return the token's integrity RID (Low=0x1000, Medium=0x2000), or None."""
    if sys.platform != "win32":
        return None
    token = wintypes.HANDLE()
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        advapi32.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                              ctypes.POINTER(wintypes.HANDLE)]
        advapi32.OpenProcessToken.restype = wintypes.BOOL
        advapi32.GetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                 ctypes.c_void_p, wintypes.DWORD,
                                                 ctypes.POINTER(wintypes.DWORD)]
        advapi32.GetTokenInformation.restype = wintypes.BOOL
        advapi32.GetSidSubAuthorityCount.argtypes = [ctypes.c_void_p]
        advapi32.GetSidSubAuthorityCount.restype = ctypes.POINTER(wintypes.BYTE)
        advapi32.GetSidSubAuthority.argtypes = [ctypes.c_void_p, wintypes.DWORD]
        advapi32.GetSidSubAuthority.restype = ctypes.POINTER(wintypes.DWORD)
        if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), 0x0008, ctypes.byref(token)):
            return None
        size = wintypes.DWORD()
        advapi32.GetTokenInformation(token, 25, None, 0, ctypes.byref(size))
        if not size.value:
            return None
        buffer = ctypes.create_string_buffer(size.value)
        if not advapi32.GetTokenInformation(token, 25, buffer, size.value, ctypes.byref(size)):
            return None
        # TOKEN_MANDATORY_LABEL begins with SID_AND_ATTRIBUTES.Sid.
        sid = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_void_p))[0]
        count = advapi32.GetSidSubAuthorityCount(sid)[0]
        if not count:
            return None
        return advapi32.GetSidSubAuthority(sid, count - 1)[0]
    except (OSError, AttributeError, ValueError):
        return None
    finally:
        if token:
            kernel32.CloseHandle(token)


def integrity_name(level: int | None) -> str:
    if level is None:
        return "Unknown"
    if level < MEDIUM_INTEGRITY:
        return "Low"
    if level < 0x3000:
        return "Medium"
    return "High"
