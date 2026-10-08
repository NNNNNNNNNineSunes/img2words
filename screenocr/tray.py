"""Windows notification-area ("tray") icon for the screen OCR tool.

Only the Win32 API is used, through :mod:`ctypes`, so no extra dependency is
needed.  A dedicated daemon thread owns a hidden top-level window and pumps its
message queue; that window is the one the shell notifies about the icon (double
click) and it owns the right-click menu.

The Tk thread is never touched from here.  Every user action is published to the
application's :class:`queue.Queue` as a single tuple::

    ("tray_show", None)   double click on the icon
    ("tray_exit", None)   "退出" in the icon menu

Importing this module is safe everywhere: the Windows DLLs and
``ctypes.WINFUNCTYPE`` are touched only by :func:`_load_api`.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import logging
import sys
import threading
from types import SimpleNamespace

LOG = logging.getLogger(__name__)

__all__ = ["TrayIcon", "MENU_ITEMS"]

# --- window and icon messages ------------------------------------------------
WM_NULL = 0x0000
WM_DESTROY = 0x0002
WM_CLOSE = 0x0010
WM_COMMAND = 0x0111
WM_LBUTTONDBLCLK = 0x0203
WM_RBUTTONUP = 0x0205
WM_APP = 0x8000
#: Legacy (pre-Vista) shell callback message: the mouse message arrives in the
#: low word of ``lParam``.  Chosen deliberately instead of NOTIFYICON_VERSION_4.
WM_TRAYICON = WM_APP + 1

# --- Shell_NotifyIconW -------------------------------------------------------
NIM_ADD = 0x00000000
NIM_DELETE = 0x00000002
NIF_MESSAGE = 0x00000001
NIF_ICON = 0x00000002
NIF_TIP = 0x00000004
#: ``IDI_APPLICATION`` as MAKEINTRESOURCE: a shared system icon, never destroyed.
IDI_APPLICATION = 32512

# --- popup menu --------------------------------------------------------------
MF_STRING = 0x00000000
TPM_RETURNCMD = 0x0100
TPM_NONOTIFY = 0x0080
TPM_RIGHTBUTTON = 0x0002
IDM_SHOW = 1
IDM_EXIT = 2
MENU_ITEMS = ((IDM_SHOW, "显示主窗口"), (IDM_EXIT, "退出"))

ERROR_CLASS_ALREADY_EXISTS = 1410
TRAY_TIP = "屏幕框选 OCR：双击显示主窗口，右键打开菜单"
#: Message the shell broadcasts after Explorer (re)starts.
TASKBAR_CREATED = "TaskbarCreated"

#: Application events.  These names are part of the app <-> tray contract.
EVENT_SHOW = "tray_show"
EVENT_EXIT = "tray_exit"

_API = None

#: Backends whose window class could not be released.  Keeping the instance alive
#: keeps its ctypes WNDPROC alive: native code still pointing at a released
#: callback would crash the process.  An entry is dropped once a later cleanup
#: releases everything.
_UNREGISTERED_BACKENDS = []


def _message_event(message) -> "tuple | None":
    """Map a shell callback mouse message to an application event tuple."""
    if message == WM_LBUTTONDBLCLK:
        return (EVENT_SHOW, None)
    return None


def _menu_event(command) -> "tuple | None":
    """Map a menu command id to an application event tuple."""
    if command == IDM_SHOW:
        return (EVENT_SHOW, None)
    if command == IDM_EXIT:
        return (EVENT_EXIT, None)
    return None


def _load_api():
    """Load and cache the Win32 entry points and structures used by the icon.

    Every pointer-returning function gets an explicit ``restype`` (a 64-bit handle
    must never be truncated) and every argument list is declared explicitly.
    """
    global _API
    if _API is not None:
        return _API
    if sys.platform != "win32":
        raise RuntimeError("通知区域图标仅支持 Windows。")
    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        shell32 = ctypes.WinDLL("shell32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    except OSError as exc:  # pragma: no cover - depends on the host system
        raise RuntimeError(f"无法加载 Windows 通知区域 API：{exc}") from exc

    WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT,
                                 wintypes.WPARAM, wintypes.LPARAM)

    class GUID(ctypes.Structure):
        _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD),
                    ("Data3", wintypes.WORD), ("Data4", ctypes.c_byte * 8)]

    class NOTIFYICONDATAW(ctypes.Structure):
        """Full Windows 11 layout; ``cbSize`` is set from this structure."""

        _fields_ = [("cbSize", wintypes.DWORD), ("hWnd", wintypes.HWND),
                    ("uID", wintypes.UINT), ("uFlags", wintypes.UINT),
                    ("uCallbackMessage", wintypes.UINT), ("hIcon", wintypes.HICON),
                    ("szTip", wintypes.WCHAR * 128), ("dwState", wintypes.DWORD),
                    ("dwStateMask", wintypes.DWORD), ("szInfo", wintypes.WCHAR * 256),
                    ("uVersion", wintypes.UINT), ("szInfoTitle", wintypes.WCHAR * 64),
                    ("dwInfoFlags", wintypes.DWORD), ("guidItem", GUID),
                    ("hBalloonIcon", wintypes.HICON)]

    class WNDCLASSW(ctypes.Structure):
        _fields_ = [("style", wintypes.UINT), ("lpfnWndProc", WNDPROC),
                    ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int),
                    ("hInstance", wintypes.HINSTANCE), ("hIcon", wintypes.HICON),
                    ("hCursor", wintypes.HANDLE), ("hbrBackground", wintypes.HBRUSH),
                    ("lpszMenuName", wintypes.LPCWSTR), ("lpszClassName", wintypes.LPCWSTR)]

    user32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASSW)]
    user32.RegisterClassW.restype = wintypes.ATOM
    user32.UnregisterClassW.argtypes = [wintypes.LPCWSTR, wintypes.HINSTANCE]
    user32.UnregisterClassW.restype = wintypes.BOOL
    user32.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR,
                                       wintypes.DWORD, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_int, ctypes.c_int, wintypes.HWND,
                                       wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID]
    user32.CreateWindowExW.restype = wintypes.HWND
    user32.DestroyWindow.argtypes = [wintypes.HWND]
    user32.DestroyWindow.restype = wintypes.BOOL
    user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM,
                                      wintypes.LPARAM]
    user32.DefWindowProcW.restype = ctypes.c_ssize_t
    user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND,
                                   wintypes.UINT, wintypes.UINT]
    user32.GetMessageW.restype = wintypes.BOOL
    user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
    user32.TranslateMessage.restype = wintypes.BOOL
    user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
    user32.DispatchMessageW.restype = ctypes.c_ssize_t
    user32.PostQuitMessage.argtypes = [ctypes.c_int]
    user32.PostQuitMessage.restype = None
    user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM,
                                    wintypes.LPARAM]
    user32.PostMessageW.restype = wintypes.BOOL
    user32.LoadIconW.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR]
    user32.LoadIconW.restype = wintypes.HICON
    user32.CreatePopupMenu.argtypes = []
    user32.CreatePopupMenu.restype = wintypes.HMENU
    user32.AppendMenuW.argtypes = [wintypes.HMENU, wintypes.UINT, ctypes.c_size_t,
                                   wintypes.LPCWSTR]
    user32.AppendMenuW.restype = wintypes.BOOL
    user32.TrackPopupMenu.argtypes = [wintypes.HMENU, wintypes.UINT, ctypes.c_int,
                                      ctypes.c_int, ctypes.c_int, wintypes.HWND,
                                      ctypes.c_void_p]
    user32.TrackPopupMenu.restype = ctypes.c_int
    user32.DestroyMenu.argtypes = [wintypes.HMENU]
    user32.DestroyMenu.restype = wintypes.BOOL
    user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
    user32.GetCursorPos.restype = wintypes.BOOL
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.SetForegroundWindow.restype = wintypes.BOOL
    user32.RegisterWindowMessageW.argtypes = [wintypes.LPCWSTR]
    user32.RegisterWindowMessageW.restype = wintypes.UINT
    kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    kernel32.GetModuleHandleW.restype = wintypes.HMODULE
    shell32.Shell_NotifyIconW.argtypes = [wintypes.DWORD, ctypes.POINTER(NOTIFYICONDATAW)]
    shell32.Shell_NotifyIconW.restype = wintypes.BOOL

    _API = SimpleNamespace(user32=user32, shell32=shell32, kernel32=kernel32,
                           WNDPROC=WNDPROC, WNDCLASSW=WNDCLASSW,
                           NOTIFYICONDATAW=NOTIFYICONDATAW, GUID=GUID)
    return _API


class _Win32TrayBackend:
    """Native half of the tray icon; lives entirely on the icon thread.

    *emit* receives one of the two application event tuples, *should_stop* reports
    whether a stop was requested during startup, and *api* is a test seam: unit
    tests inject a fake API namespace so no real window or icon is created.
    """

    def __init__(self, emit, should_stop, api=None):
        self.emit = emit
        self._should_stop = should_stop
        self._api = api if api is not None else _load_api()
        self.hwnd = None
        self._class_name = None
        self._class_registered = False
        self._wnd_class = None
        self._wnd_proc_ref = None  # Keep the ctypes callback alive while the class lives.
        self._data = None
        self._icon_added = False
        self._taskbar_message = 0

    # -- startup -----------------------------------------------------------
    def start(self):
        """Register the class, create the hidden window and add the icon.

        Any failure releases whatever was already created before it propagates,
        so a failed start never leaves a window class, window or thread behind.
        """
        try:
            self._create_window()
            self._add_icon()
        except BaseException:
            self.close()
            raise
        LOG.info("notification area icon registered: hwnd=%s", self.hwnd)

    def _create_window(self):
        api = self._api
        instance = api.kernel32.GetModuleHandleW(None)
        if not instance:
            raise ctypes.WinError(ctypes.get_last_error())
        self._class_name = f"ScreenOCR.TrayIcon.{id(self):x}"
        # Explorer broadcasts this after a shell restart; only then must the icon
        # be registered again (see ``_wnd_proc``).
        self._taskbar_message = api.user32.RegisterWindowMessageW(TASKBAR_CREATED)
        callback = api.WNDPROC(self._wnd_proc)
        wnd_class = api.WNDCLASSW()
        wnd_class.lpfnWndProc = callback
        wnd_class.hInstance = instance
        wnd_class.lpszClassName = self._class_name
        self._wnd_class = wnd_class
        self._wnd_proc_ref = callback
        if api.user32.RegisterClassW(ctypes.byref(wnd_class)):
            self._class_registered = True
        elif ctypes.get_last_error() != ERROR_CLASS_ALREADY_EXISTS:
            raise ctypes.WinError(ctypes.get_last_error())
        hwnd = api.user32.CreateWindowExW(0, self._class_name, self._class_name, 0,
                                          0, 0, 0, 0, None, None, instance, None)
        if not hwnd:
            raise ctypes.WinError(ctypes.get_last_error())
        self.hwnd = hwnd

    def _add_icon(self):
        api = self._api
        if self._should_stop():
            raise RuntimeError("stop was requested before the icon was registered")
        data = api.NOTIFYICONDATAW()
        data.cbSize = ctypes.sizeof(api.NOTIFYICONDATAW)
        data.hWnd = self.hwnd
        data.uID = 1
        data.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP
        data.uCallbackMessage = WM_TRAYICON
        # MAKEINTRESOURCE(IDI_APPLICATION): a shared system icon, so it must not
        # be passed to DestroyIcon by us.
        data.hIcon = api.user32.LoadIconW(
            None, ctypes.cast(ctypes.c_void_p(IDI_APPLICATION), wintypes.LPCWSTR))
        if not data.hIcon:
            raise ctypes.WinError(ctypes.get_last_error())
        data.szTip = TRAY_TIP
        self._data = data  # Must outlive the icon: the shell keeps the hWnd/uID.
        if not api.shell32.Shell_NotifyIconW(NIM_ADD, ctypes.byref(data)):
            self._data = None
            raise ctypes.WinError(ctypes.get_last_error())
        self._icon_added = True

    def _readd_icon(self):
        """Register the icon again after Explorer restarted."""
        if self._data is None:
            return
        try:
            readded = bool(self._api.shell32.Shell_NotifyIconW(NIM_ADD, ctypes.byref(self._data)))
        except Exception:
            readded = False
        if readded:
            self._icon_added = True
            LOG.info("notification area icon re-registered after the shell restarted")
            return
        LOG.warning("Shell_NotifyIconW(NIM_ADD) failed after the shell restarted")
        # Without an icon the window would be unreachable, so end the message loop:
        # the app notices the dead icon thread and restores its main window.
        self._icon_added = False
        if not self._destroy_window():
            # The window survived, so no WM_DESTROY will end the loop: post the quit
            # here; the thread cleanup then retries the destroy and retains the callback.
            self._api.user32.PostQuitMessage(0)

    # -- message loop ------------------------------------------------------
    def pump(self):
        """Dispatch messages until ``WM_QUIT`` (posted by ``WM_DESTROY``)."""
        api = self._api
        message = wintypes.MSG()
        while True:
            result = api.user32.GetMessageW(ctypes.byref(message), None, 0, 0)
            if result == 0:
                return
            if result == -1:
                raise ctypes.WinError(ctypes.get_last_error())
            api.user32.TranslateMessage(ctypes.byref(message))
            api.user32.DispatchMessageW(ctypes.byref(message))

    def request_stop(self):
        """Ask the message loop to end, from any thread.

        Raises :class:`RuntimeError` when the request could not be delivered, so
        the caller never reports a stop that was not actually requested.
        """
        hwnd = self.hwnd
        if not hwnd:
            return
        try:
            if not self._api.user32.PostMessageW(hwnd, WM_CLOSE, 0, 0):
                raise ctypes.WinError(ctypes.get_last_error())
        except Exception as exc:
            raise RuntimeError(f"无法请求通知区域图标退出：{exc}") from exc

    def _wnd_proc(self, hwnd, message, wparam, lparam):
        """Window procedure for the hidden tray window.  Never raises."""
        try:
            if message == WM_TRAYICON:
                mouse_message = lparam & 0xFFFF
                if mouse_message == WM_RBUTTONUP:
                    self._show_menu()
                else:
                    event = _message_event(mouse_message)
                    if event is not None:
                        self.emit(event)
                return 0
            if message == WM_COMMAND:
                event = _menu_event(wparam & 0xFFFF)
                if event is not None:
                    self.emit(event)
                return 0
            if self._taskbar_message and message == self._taskbar_message:
                self._readd_icon()
                return 0
            if message == WM_CLOSE:
                # The icon goes first: NIM_DELETE identifies it by the owner window
                # handle, which DestroyWindow would invalidate.  Neither step may
                # skip the other, whatever the native call returns.
                try:
                    self._remove_icon()
                finally:
                    if not self._destroy_window(hwnd):
                        # No WM_DESTROY will follow, so the loop must be ended here;
                        # the thread cleanup retries the destroy and retains the callback.
                        self._api.user32.PostQuitMessage(0)
                return 0
            if message == WM_DESTROY:
                if hwnd == self.hwnd:
                    self.hwnd = None  # Destroyed elsewhere: nothing left to retry.
                try:
                    self._remove_icon()
                finally:
                    # Must always run, or the message loop would never end.
                    self._api.user32.PostQuitMessage(0)
                return 0
        except Exception:  # An exception must never escape a native callback.
            LOG.exception("notification area callback failed")
        return self._api.user32.DefWindowProcW(hwnd, message, wparam, lparam)

    def _show_menu(self):
        """Show the right-click menu and publish the chosen command."""
        api = self._api
        hwnd = self.hwnd
        if not hwnd:
            return
        menu = api.user32.CreatePopupMenu()
        if not menu:
            return
        try:
            for command, label in MENU_ITEMS:
                api.user32.AppendMenuW(menu, MF_STRING, command, label)
            point = wintypes.POINT()
            if not api.user32.GetCursorPos(ctypes.byref(point)):
                return
            # Documented shell rule for tray menus: the owner window must be
            # foreground, and a null message after TrackPopupMenu closes the menu
            # when the user clicks elsewhere.
            api.user32.SetForegroundWindow(hwnd)
            chosen = api.user32.TrackPopupMenu(
                menu, TPM_RETURNCMD | TPM_NONOTIFY | TPM_RIGHTBUTTON,
                point.x, point.y, 0, hwnd, None)
            api.user32.PostMessageW(hwnd, WM_NULL, 0, 0)
            event = _menu_event(chosen)
            if event is not None:
                self.emit(event)
        finally:
            api.user32.DestroyMenu(menu)

    # -- shutdown ----------------------------------------------------------
    def _destroy_window(self, hwnd=None):
        """Destroy the hidden window; the handle is kept when the call fails."""
        hwnd = hwnd or self.hwnd
        if not hwnd:
            return True
        try:
            destroyed = bool(self._api.user32.DestroyWindow(hwnd))
        except Exception:
            destroyed = False
        if not destroyed:
            LOG.warning("DestroyWindow failed for hwnd=%s; the handle is kept", hwnd)
            return False
        if hwnd == self.hwnd:
            self.hwnd = None
        return True

    def _remove_icon(self):
        """``NIM_DELETE`` once; the icon state is kept when the call fails."""
        if not self._icon_added or self._data is None:
            return True
        data = self._data
        try:
            removed = bool(self._api.shell32.Shell_NotifyIconW(NIM_DELETE, ctypes.byref(data)))
        except Exception:
            removed = False
        if not removed:
            LOG.warning("Shell_NotifyIconW(NIM_DELETE) failed; the icon state is kept")
            return False
        self._icon_added = False
        self._data = None
        return True

    def close(self):
        """Release icon, window and class; safe to call more than once.

        A failed native call keeps whatever it could not release (icon data,
        window handle, callback) for a later retry: releasing a callback whose
        class is still registered would crash the process.
        """
        self._remove_icon()
        if not self._destroy_window():
            if self not in _UNREGISTERED_BACKENDS:
                _UNREGISTERED_BACKENDS.append(self)
            return
        if self._class_registered:
            try:
                released = self._api.user32.UnregisterClassW(
                    self._class_name, self._api.kernel32.GetModuleHandleW(None))
            except Exception:
                released = False
            if not released:
                LOG.warning("UnregisterClassW failed; the window procedure is retained")
                if self not in _UNREGISTERED_BACKENDS:
                    _UNREGISTERED_BACKENDS.append(self)
                return
            self._class_registered = False
        if self in _UNREGISTERED_BACKENDS:
            _UNREGISTERED_BACKENDS.remove(self)
        # Nothing is registered any more, so the retained callback can go.
        self._wnd_class = None
        self._wnd_proc_ref = None


class TrayIcon:
    """Windows notification-area icon owned by one daemon thread.

    ``events`` is the application queue; only ``("tray_show", None)`` and
    ``("tray_exit", None)`` are ever put on it.
    """

    STOP_TIMEOUT = 2.5

    def __init__(self, events: "queue.Queue", *, backend_factory=None, start_timeout: float = 5.0):
        self.events = events
        self._factory = backend_factory or _Win32TrayBackend
        self.start_timeout = max(0.1, float(start_timeout))
        self._thread = None
        self._backend = None
        self._started = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._error = None
        self._running = False

    @property
    def running(self) -> bool:
        """``True`` while the icon is live and no stop has been requested."""
        return bool(self._running and not self._stop.is_set()
                    and self._thread is not None and self._thread.is_alive())

    def start(self):
        """Start the icon thread and wait until the icon is registered.

        Raises :class:`RuntimeError` when a previous icon thread is still alive,
        when registration fails, or when it exceeds ``start_timeout``; the failure
        paths request a stop first.  Starting a running icon is a no-op.
        """
        with self._lock:
            if self.running:
                return
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("上一次的通知区域图标线程仍在运行，无法再次启动。")
        self._stop.clear()
        self._error = None
        self._started.clear()
        thread = threading.Thread(target=self._run, name="tray-icon", daemon=True)
        with self._lock:
            self._thread = thread
        thread.start()
        if not self._started.wait(self.start_timeout):
            self._request_cleanup_stop()
            raise RuntimeError("通知区域图标初始化超时。")
        if self._error is not None:
            error = self._error
            self._request_cleanup_stop()
            raise RuntimeError(f"无法注册通知区域图标：{error}") from error
        LOG.info("TRAY_READY: notification area icon available")

    def _request_cleanup_stop(self):
        """Best-effort stop after a failed start; never mask the start error."""
        try:
            self.stop()
        except RuntimeError as exc:
            LOG.warning("notification area icon did not stop after a failed start: %s", exc)

    def stop(self):
        """Request shutdown and wait for the icon thread.

        Returns ``True`` once the thread ended.  Raises :class:`RuntimeError` when
        it is still alive after the bounded wait: the thread and backend references
        are then kept, so ``start()`` cannot spawn a second thread and ``stop()``
        can be retried.  Stopping a stopped icon is a no-op.
        """
        self._stop.set()
        with self._lock:
            thread, backend = self._thread, self._backend
        request_error = None
        if backend is not None:
            try:
                backend.request_stop()
            except Exception as exc:
                request_error = exc
                LOG.warning("notification area stop request failed: %s", exc)
        if thread is None:
            self._running = False
            return True
        if thread is not threading.current_thread():
            if thread.is_alive():
                thread.join(timeout=self.STOP_TIMEOUT)
            if thread.is_alive():
                raise RuntimeError(
                    f"通知区域图标线程未能在 {self.STOP_TIMEOUT:g} 秒内退出"
                    + (f"（停止请求失败：{request_error}）" if request_error else "。"))
            with self._lock:
                self._thread = None
                self._backend = None
        if request_error is not None:
            raise RuntimeError(f"无法请求通知区域图标退出：{request_error}") from request_error
        self._running = False
        return True

    def _emit(self, event):
        """Publish one ``(name, value)`` tuple; never raises into a native callback."""
        try:
            self.events.put(event)
        except Exception:
            LOG.warning("notification area event could not be queued")

    def _cleanup(self, backend):
        if backend is None:
            return
        try:
            backend.close()
        except Exception:
            LOG.exception("notification area cleanup failed")

    def _run(self):
        backend = None
        started = False
        try:
            backend = self._backend = self._factory(self._emit, self._stop.is_set)
            backend.start()
            started = True
        except BaseException as exc:
            self._error = exc
            self._cleanup(backend)
        else:
            # A stop requested while starting must never be reported as running.
            self._running = not self._stop.is_set()
        finally:
            self._started.set()
        if not started:
            return
        # Pump even after an early stop request: this thread registered the icon,
        # so this thread must also release it.
        try:
            backend.pump()
        except BaseException:
            LOG.exception("notification area message loop failed")
        finally:
            self._running = False
            self._backend = None
            self._cleanup(backend)
