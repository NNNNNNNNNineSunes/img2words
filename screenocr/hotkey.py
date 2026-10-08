"""Windows Ctrl+left-drag listener. Hook callbacks never perform OCR or Tk work."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import queue
import sys
import threading


class MouseHook:
    """Emit (start|move|end, (physical_x, physical_y)) to a thread-safe queue.

    The activating mouse-down/up are swallowed so underlying applications do
    not receive an unmatched click or change their selection. Normal clicks
    and movement pass through. A released Ctrl does not cancel an active drag.
    """

    def __init__(self, events: queue.Queue):
        self.events = events
        self.enabled = threading.Event()
        self.enabled.set()
        self._ready = threading.Event()
        self._thread = None
        self._thread_id = 0
        self._dragging = False
        self._cancelled = False
        self._error = None
        self._callback = None  # Keep the ctypes callback alive for the hook.
        self._callback_count = 0
        self._left_down_count = 0
        self._activation_count = 0

    def diagnostics(self):
        """Counts only: no key contents, screenshots or cursor coordinates recorded."""
        return {"callbacks": self._callback_count, "left_down": self._left_down_count,
                "activations": self._activation_count,
                "thread_alive": bool(self._thread and self._thread.is_alive())}

    def _handle_mouse(self, message, point, ctrl_down=False):
        """Process one mouse notification; return whether it must be swallowed."""
        self._callback_count += 1
        if message == 0x0201:  # WM_LBUTTONDOWN
            self._left_down_count += 1
            if not self._dragging and self.enabled.is_set() and ctrl_down:
                self.enabled.clear()
                self._dragging = True
                self._cancelled = False
                self._activation_count += 1
                self.events.put(("start", point))
                return True
        elif message == 0x0200 and self._dragging:  # WM_MOUSEMOVE
            if not self._cancelled:
                self.events.put(("move", point))
        elif message == 0x0202 and self._dragging:  # WM_LBUTTONUP
            self._dragging = False
            if not self._cancelled:
                self.events.put(("end", point))
            return True
        return False

    def start(self):
        if sys.platform != "win32":
            raise RuntimeError("全局鼠标快捷键仅支持 Windows。")
        self._thread = threading.Thread(target=self._run, name="mouse-hook", daemon=True)
        self._thread.start()
        if not self._ready.wait(5):
            raise RuntimeError("鼠标快捷键初始化超时。")
        if self._error:
            raise RuntimeError(f"无法注册鼠标快捷键：{self._error}")

    def cancel_drag(self):
        # Still swallow the corresponding button-up after cancellation.
        self._cancelled = True

    def stop(self):
        self.enabled.clear()
        if self._thread_id and self._thread and self._thread.is_alive():
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            user32.PostThreadMessageW.argtypes = [wintypes.DWORD, wintypes.UINT,
                                                  wintypes.WPARAM, wintypes.LPARAM]
            user32.PostThreadMessageW.restype = wintypes.BOOL
            user32.PostThreadMessageW(self._thread_id, 0x0012, 0, 0)  # WM_QUIT
            self._thread.join(timeout=2)

    def _run(self):
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        hook = None
        LRESULT = ctypes.c_ssize_t
        HOOKPROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)

        class MSLLHOOKSTRUCT(ctypes.Structure):
            _fields_ = [("pt", wintypes.POINT), ("mouseData", wintypes.DWORD),
                        ("flags", wintypes.DWORD), ("time", wintypes.DWORD),
                        ("dwExtraInfo", ctypes.c_size_t)]

        user32.SetWindowsHookExW.argtypes = [ctypes.c_int, HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD]
        user32.SetWindowsHookExW.restype = wintypes.HANDLE
        user32.CallNextHookEx.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
        user32.CallNextHookEx.restype = LRESULT
        user32.UnhookWindowsHookEx.argtypes = [wintypes.HANDLE]
        user32.UnhookWindowsHookEx.restype = wintypes.BOOL
        user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
        user32.GetAsyncKeyState.restype = wintypes.SHORT
        user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
        user32.GetMessageW.restype = wintypes.BOOL
        kernel32.GetCurrentThreadId.restype = wintypes.DWORD
        kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        kernel32.GetModuleHandleW.restype = wintypes.HMODULE

        def callback(code, message, data):
            try:
                if code >= 0:
                    info = ctypes.cast(data, ctypes.POINTER(MSLLHOOKSTRUCT)).contents
                    point = (info.pt.x, info.pt.y)
                    ctrl_down = message == 0x0201 and bool(user32.GetAsyncKeyState(0x11) & 0x8000)
                    if self._handle_mouse(message, point, ctrl_down):
                        return 1
            except Exception as exc:
                # Exceptions must never escape a native callback.
                self.events.put(("hook_error", str(exc)))
            return user32.CallNextHookEx(hook, code, message, data)

        try:
            self._thread_id = kernel32.GetCurrentThreadId()
            self._callback = HOOKPROC(callback)
            hook = user32.SetWindowsHookExW(14, self._callback, kernel32.GetModuleHandleW(None), 0)
            if not hook:
                raise ctypes.WinError(ctypes.get_last_error())
            # Create the message queue before announcing readiness.
            msg = wintypes.MSG()
            user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 0)
            self._ready.set()
            while True:
                result = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if result == 0:
                    break
                if result == -1:
                    raise ctypes.WinError(ctypes.get_last_error())
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
        except Exception as exc:
            self._error = exc
            if self._ready.is_set():
                self.events.put(("hook_error", str(exc)))
        finally:
            self._ready.set()
            if hook:
                user32.UnhookWindowsHookEx(hook)
            self._thread_id = 0
