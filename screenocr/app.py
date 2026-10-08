"""Small Tk status window and frozen-desktop selection overlay."""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import datetime
import logging
import os
from pathlib import Path
import queue
import sys
import threading
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox, ttk

from PIL import ImageEnhance, ImageTk

from .capture import capture_screen, enable_dpi_awareness
from .clipboard import write_text
from .geometry import normalize_rect
from .history import HistoryStore
from .hotkey import MouseHook
from .ocr import OCREngine
from .pipeline import NoTextError, recognize_and_copy
from .permissions import MEDIUM_INTEGRITY, get_integrity_level, integrity_name
from .tray import TrayIcon

LOG = logging.getLogger(__name__)


class ScreenOCRApp:
    #: Layout sizes are 96-DPI design values; every one is scaled by the real Tk
    #: scaling so no window is laid out with hardcoded physical pixels.
    BASE_WRAP = 430
    MIN_MAIN_WIDTH = 340
    MIN_MAIN_HEIGHT = 220
    MAIN_BUTTON_MIN = 260
    MAIN_PADDING = 18              # raw ttk padding per side: ttk never scales padding
    HISTORY_LIST_CHARS = 36        # initial visible width, so the preview is readable
    HISTORY_LIST_CHARS_MIN = 16    # the Listbox's own minimum: the pane can shrink
    HISTORY_TEXT_CHARS = 20        # the Text's own minimum; it expands into the pane
    HISTORY_LIST_ROWS = 12
    HISTORY_TEXT_ROWS = 14
    HISTORY_COLUMN_TARGET = 430
    HISTORY_WINDOW_BASE = (600, 300)   # design px: ~900x450 on a 150% display
    HISTORY_SASH_BASE = 270            # design px for the initial list pane
    HISTORY_PANE_LEFT_MIN = 120        # design px the list pane must keep
    HISTORY_PANE_RIGHT_MIN = 150       # design px the text pane must keep
    HISTORY_PREVIEW_MIN = 8
    HISTORY_PREVIEW_MAX = 16
    HISTORY_HIGHLIGHT_TAG = "history_record_selected"
    HISTORY_HIGHLIGHT_COLORS = {"background": "#fde68a", "foreground": "#7c2d12"}

    def __init__(self, root: tk.Tk, *, smoke_seconds: float = 0, history_store=None):
        self.root = root
        self.events = queue.Queue()
        self.hook = MouseHook(self.events)
        self.engine = None
        self.overlay = None
        self.toast = None
        self.snapshot = None
        self.bounds = None
        self.start_point = None
        self.worker = None
        self.closing = False
        self.closed = False
        self.ready = False
        self.smoke_seconds = smoke_seconds
        self.tray = None
        self.tray_available = False
        # The constructor keyword is ``history_store``; ``self.history`` is the live
        # store used by every recording/viewer call site.
        self.history = history_store if history_store is not None else HistoryStore()
        self.history_view = None
        self.history_list = None
        self.history_text = None
        self.history_status_var = None
        self.history_status_label = None
        self.history_splitter = None
        self.history_left = None
        self.history_right = None
        self.history_items = []
        self.history_row_ids = []
        self.history_selected_id = None
        self.history_spans = {}
        self.history_summaries = []
        self.history_summary_width = 0
        self.background_timer = None
        self.poll_timer = None
        self.smoke_timer = None
        self.main_layout_timer = None
        self._status_trace = None
        self.history_sash_pending = False
        self.title_label = None
        self.info_label = None
        self.main_frame = None
        self.buttons_frame = None
        self._wrap_width = None
        self._history_status_wrap = None
        self.integrity_level = get_integrity_level()
        self.input_restricted = self.integrity_level is not None and self.integrity_level < MEDIUM_INTEGRITY
        LOG.info("PROCESS pid=%s executable=%s integrity=%s", os.getpid(), sys.executable,
                 integrity_name(self.integrity_level))
        root.title("屏幕框选 OCR")
        root.resizable(True, True)
        root.protocol("WM_DELETE_WINDOW", self.request_window_close)
        root.bind("<Unmap>", self._on_root_unmap, add="+")
        self._wrap = self._scaled(self.BASE_WRAP)
        self.main_frame = frame = ttk.Frame(root, padding=self.MAIN_PADDING)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)   # extra height pushes the buttons to the bottom
        self.title_label = ttk.Label(frame, text="Ctrl + 鼠标左键拖动框选",
                                     font=("Microsoft YaHei UI", 14))
        self.title_label.grid(row=0, column=0, sticky="w")
        self.info_label = ttk.Label(
            frame, text="松开鼠标识别并复制；Esc 取消选区。\nV2：最小化或关闭（X）会隐藏到通知区域托盘；"
                        "双击托盘图标重开，退出请点『退出』或托盘右键菜单。",
            justify="left", wraplength=self._wrap)
        self.info_label.grid(row=1, column=0, sticky="we", pady=10)
        self.status = tk.StringVar(value="正在加载离线中英文模型……")
        self._status_trace = self.status.trace_add("write", self._on_status_change)
        self.status_label = ttk.Label(frame, textvariable=self.status, wraplength=self._wrap)
        self.status_label.grid(row=2, column=0, sticky="nwe", pady=5)
        self.buttons_frame = buttons = ttk.Frame(frame)
        buttons.grid(row=3, column=0, sticky="sew", pady=(8, 0))
        # Two uniform columns: the four main buttons stay inside the window at every
        # DPI and share the width when it grows.
        for index, (text, command) in enumerate((("历史记录", self.open_history),
                                                 ("后台运行", self._background_clicked),
                                                 ("权限诊断", self._show_diagnostics),
                                                 ("退出", self.close))):
            row, column = divmod(index, 2)
            ttk.Button(buttons, text=text, command=command).grid(
                row=row, column=column, sticky="we", padx=(0, 8), pady=(0, 6))
        for column in (0, 1):
            buttons.columnconfigure(column, weight=1, uniform="main_buttons")
        frame.bind("<Configure>", self._on_main_configure)
        self._fit_main_window()
        self._start_tray()
        self.worker = threading.Thread(target=self._load_engine, name="ocr-initialize", daemon=True)
        self.worker.start()
        self.poll_timer = root.after(10, self._poll)
        if smoke_seconds:
            self.smoke_timer = root.after(int(smoke_seconds * 1000), self._finish_smoke)

    def _scaled(self, size):
        """Convert a 96-DPI design size to the screen's pixels via Tk scaling."""
        try:
            scaling = float(self.root.tk.call("tk", "scaling"))
        except Exception:
            scaling = 96.0 / 72.0   # no Tk interpreter (unit tests): use the design DPI
        if scaling <= 0:
            scaling = 96.0 / 72.0
        return int(round(size * scaling / (96.0 / 72.0)))

    def _set_wrap(self, wrap):
        """Apply one wrap width to both wrapping labels."""
        self._wrap = int(wrap)
        for label in (self.info_label, self.status_label):
            if label is not None:
                label.configure(wraplength=self._wrap)

    def _on_main_configure(self, event):
        """Re-wrap the labels from the real content width (width changes only).

        The frame's own padding is excluded so a wrapped line can never be wider
        than the label it is drawn in; the wrap is derived from the width alone, so
        the resulting height change cannot feed back into another width change.
        """
        width = int(getattr(event, "width", 0) or 0)
        if width <= 1 or width == self._wrap_width:
            return
        self._wrap_width = width
        self._set_wrap(max(1, width - 2 * self.MAIN_PADDING - 4))
        self._schedule_main_layout()

    def _on_status_change(self, *_args):
        """A longer status text can need a taller window, so re-fit after Tk settles."""
        self._schedule_main_layout()

    def _schedule_main_layout(self, stage=0):
        """One deduplicated idle re-fit of the main window height (two bounded stages)."""
        if getattr(self, "closing", False) or getattr(self, "closed", False):
            return
        if self.main_layout_timer is not None:
            return
        try:
            self.main_layout_timer = self.root.after_idle(
                lambda: self._apply_main_min_height(stage))
        except Exception:
            LOG.warning("WINDOW_FIT_FAILED: the layout refresh could not be scheduled")
            self.main_layout_timer = None

    def _apply_main_min_height(self, stage=0):
        """Grow (or relax) ``minsize`` to the wrapped content height.

        The first idle pass can run before Tk has propagated the new label size up
        to the root, because that parent/frame geometry is queued as separate idle
        work; stage 0 therefore reads nothing and only arms stage 1, which applies
        the settled request.  Two passes are the bound, the timer id stays tracked,
        and no ``update`` pump is used, so this cannot re-enter the idle queue.
        """
        self.main_layout_timer = None
        if getattr(self, "closing", False) or getattr(self, "closed", False):
            return
        if stage < 1:
            self._schedule_main_layout(stage + 1)
            return
        try:
            current_width, current_height = self.root.minsize()
            current_width, current_height = int(current_width), int(current_height)
            required = int(self.root.winfo_reqheight())
            limit = max(self._scaled(self.MIN_MAIN_HEIGHT),
                        self.root.winfo_screenheight() - self._scaled(80))
        except Exception:
            return
        height = min(max(self._scaled(self.MIN_MAIN_HEIGHT), required), limit)
        if (current_width, current_height) == (current_width, height):
            return
        if height < current_height:
            # Pinning the client size (no offsets: x/y stay) keeps the window still
            # while the floor relaxes; a manual resize down to the new floor stays valid.
            try:
                self.root.geometry(
                    f"{int(self.root.winfo_width())}x{int(self.root.winfo_height())}")
            except Exception:
                LOG.warning("WINDOW_FIT_FAILED: the visible size could not be pinned")
        try:
            self.root.minsize(current_width, height)
        except Exception:
            LOG.warning("WINDOW_FIT_FAILED: the minimum height could not be applied")

    def _fit_main_window(self):
        """Size the main window from its content, clamped to the usable screen.

        The minimum is only as wide as the two-column button grid needs, so the
        window can still be narrowed (the labels re-wrap) instead of clipping.
        """
        self._set_wrap(self._scaled(self.BASE_WRAP))
        self.root.update_idletasks()
        width = max(self.root.winfo_reqwidth(), self._scaled(self.MIN_MAIN_WIDTH))
        height = max(self.root.winfo_reqheight(), self._scaled(self.MIN_MAIN_HEIGHT))
        limit_width = max(self._scaled(self.MIN_MAIN_WIDTH),
                          self.root.winfo_screenwidth() - self._scaled(80))
        limit_height = max(self._scaled(self.MIN_MAIN_HEIGHT),
                           self.root.winfo_screenheight() - self._scaled(120))
        buttons_width = 0
        if self.buttons_frame is not None:
            buttons_width = int(self.buttons_frame.winfo_reqwidth()) + self._scaled(36)
        min_width = max(buttons_width, self._scaled(self.MAIN_BUTTON_MIN))
        self.root.geometry(f"{min(width, limit_width)}x{min(height, limit_height)}")
        self.root.minsize(min_width, self._scaled(self.MIN_MAIN_HEIGHT))
        self._wrap_width = None     # let the first Configure event set the real wrap

    def _load_engine(self):
        try:
            engine = OCREngine()
            # A small blank image triggers real model loading before enabling clicks.
            from PIL import Image
            engine.recognize(Image.new("RGB", (100, 50), "white"))
            self.events.put(("loaded", engine))
        except Exception as exc:
            LOG.exception("OCR initialization failed")
            self.events.put(("load_error", str(exc)))

    def _poll(self):
        self.poll_timer = None      # this firing consumed the tracked id
        if self.closing:
            return
        try:
            # Bound processing so a stream of mouse movement cannot starve Tk.
            for _ in range(200):
                try:
                    event, value = self.events.get_nowait()
                except queue.Empty:
                    break
                if event == "loaded":
                    self.engine = value
                    self.hook.start()
                    self.ready = True
                    if self.input_restricted:
                        self.status.set("输入监听受限（Low 权限）。请退出，在 Windows 资源管理器双击 start.cmd；不要直接启动 .venv 内的 Python。")
                        LOG.warning("INPUT_RESTRICTED_LOW: global Ctrl+mouse input is not verified; launch start.cmd from Explorer")
                    else:
                        if self._tray_is_available():
                            self.status.set("已就绪：Ctrl + 左键拖动，识别后直接 Ctrl+V。")
                        else:
                            self.status.set("已就绪：Ctrl + 左键拖动识别；通知区域图标不可用，"
                                            "最小化或关闭将退回任务栏。")
                        LOG.info("READY: OCR models and global mouse hook loaded")
                        if not self.smoke_seconds:
                            self.background_timer = self.root.after(1200, self._background_once)
                elif event == "load_error":
                    self.status.set("模型加载失败，请检查依赖后重新启动。")
                    self._notify(f"OCR 模型加载失败：{value}", error=True)
                elif event == "start":
                    LOG.info("SELECTION_START: Ctrl+left-button activation received")
                    self._begin(value)
                elif event == "move" and self.overlay:
                    self._draw(value)
                elif event == "end" and self.overlay:
                    self._finish(value)
                elif event == "result":
                    self.hook.enabled.set()
                    self.status.set(f"已复制 {len(value)} 个字符，可 Ctrl+V 粘贴。")
                    self._notify(f"识别完成，已复制 {len(value)} 个字符。")
                    LOG.info("OCR completed: %s characters copied", len(value))
                    # Records are stored before this event is published.
                    self._refresh_history_if_open()
                elif event == "history_error":
                    # The clipboard copy already succeeded; only the recording failed.
                    self.status.set("已复制文字，但历史记录保存失败。")
                    self._notify(str(value), error=True)
                    LOG.warning("HISTORY_SAVE_FAILED: the copied text is still on the clipboard")
                elif event == "error":
                    self.hook.enabled.set()
                    self.status.set(str(value))
                    self._notify(str(value), error=True)
                elif event == "hook_error":
                    self.ready = False
                    self.status.set(f"鼠标快捷键异常：{value}；请重新启动。")
                    self._remove_overlay()
                    self._notify(self.status.get(), error=True)
                elif event == "tray_show":
                    LOG.info("TRAY_SHOW: restore requested from the notification area")
                    self.show_main()
                elif event == "tray_exit":
                    LOG.info("TRAY_EXIT: exit requested from the notification area")
                    self.close()
                    return
            if getattr(self, "tray_available", False) and not self._tray_is_available():
                self._tray_lost()
            if self.overlay and ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000:
                self._cancel()
        except Exception as exc:
            LOG.exception("GUI event failed")
            self._remove_overlay()
            self.hook.cancel_drag()
            if self.ready:
                self.hook.enabled.set()
            self.status.set(f"操作失败：{exc}")
            self._notify(self.status.get(), error=True)
        finally:
            if not self.closing:
                self.poll_timer = self.root.after(10, self._poll)

    def _show_diagnostics(self):
        counters = self.hook.diagnostics()
        text = (f"进程：{os.getpid()}\n解释器：{sys.executable}\n"
                f"Windows 完整性级别：{integrity_name(self.integrity_level)}\n"
                f"鼠标回调：{counters['callbacks']}\n左键按下：{counters['left_down']}\n"
                f"检测到 Ctrl+左键：{counters['activations']}\n")
        if self.input_restricted:
            text += "\nLow 权限限制全局输入。请退出并从资源管理器双击 start.cmd。无需管理员权限。"
        else:
            text += "\n先尝试 Ctrl+左键拖动，再打开此诊断查看计数。"
        LOG.info("INPUT_DIAGNOSTICS: integrity=%s counts=%s", integrity_name(self.integrity_level), counters)
        messagebox.showinfo("输入监听诊断", text, parent=self.root)

    def _start_tray(self):
        """Register the notification area icon once; never block the window."""
        try:
            tray = TrayIcon(self.events)
            tray.start()
        except Exception as exc:
            self.tray = None
            self.tray_available = False
            LOG.warning("TRAY_UNAVAILABLE: %s; the window stays accessible", exc)
            self.status.set("通知区域图标不可用，窗口将保持显示；可继续使用框选识别。")
            return
        self.tray = tray
        self.tray_available = True

    def _tray_is_available(self):
        """True only while the notification area icon thread is really alive."""
        tray = getattr(self, "tray", None)
        return bool(getattr(self, "tray_available", False) and tray is not None and tray.running)

    def _tray_lost(self):
        """The icon thread died although the app relied on it: show the window."""
        self.tray_available = False
        LOG.warning("TRAY_LOST: the notification area icon thread stopped; restoring the window")
        self.show_main()
        self.status.set("通知区域图标已停止，窗口已恢复到任务栏。")
        self._notify("通知区域图标已停止，窗口已恢复到任务栏。", error=True)

    def _stop_tray(self):
        tray = getattr(self, "tray", None)
        self.tray = None
        self.tray_available = False
        if tray is not None:
            tray.stop()

    def hide_main(self, *, warn_fallback=False):
        """Hide the status window, preferring the notification area."""
        if getattr(self, "closing", False) or getattr(self, "closed", False):
            return False
        if self._tray_is_available():
            try:
                self.root.withdraw()
                return True
            except Exception as exc:
                LOG.warning("TRAY_HIDE_FAILED: %s", exc)
        try:
            self.root.iconify()
        except Exception as exc:
            LOG.warning("MINIMIZE_FAILED: %s", exc)
            return False
        if warn_fallback:
            self._notify("通知区域图标不可用，已最小化到任务栏。")
        return False

    def show_main(self):
        """Restore the status window (tray double click); never touches the overlay."""
        if getattr(self, "closing", False) or getattr(self, "closed", False):
            return
        timer, self.background_timer = getattr(self, "background_timer", None), None
        if timer is not None:
            # An explicit restore must not be undone by the pending start-up hide.
            try:
                self.root.after_cancel(timer)
            except Exception as exc:
                LOG.warning("AUTO_HIDE_CANCEL_FAILED: %s", exc)
        try:
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
        except Exception as exc:
            LOG.warning("RESTORE_FAILED: %s", exc)

    def _on_root_unmap(self, event):
        """Hide into the tray when only the main window is minimized/closed."""
        if getattr(self, "closing", False) or getattr(self, "closed", False):
            return
        if getattr(event, "widget", None) is not self.root:
            return  # An overlay, toast or dialog closed: never hide the window.
        try:
            if self.root.state() != "iconic":
                return
        except Exception:
            return
        if self._tray_is_available():
            try:
                self.root.withdraw()
            except Exception as exc:
                LOG.warning("TRAY_HIDE_FAILED: %s", exc)

    def request_window_close(self):
        """Window X: hide into the notification area when the icon is available."""
        if getattr(self, "closing", False) or getattr(self, "closed", False):
            return
        if self._tray_is_available():
            self.hide_main()
            self.status.set("已移入通知区域；双击托盘图标可重新打开，右键菜单可退出。")
            return
        self.close()

    def _background_clicked(self):
        """后台运行 button: prefer the tray, otherwise keep the task-bar warning."""
        self.hide_main(warn_fallback=True)

    def _background_once(self):
        self.background_timer = None
        if self.closing:
            return
        if self.overlay:
            # Keep the re-scheduled id so an explicit restore can still cancel it.
            self.background_timer = self.root.after(500, self._background_once)
        else:
            self.hide_main()

    def _begin(self, point):
        self._dismiss_toast()
        self.start_point = point
        try:
            # Freeze before showing our overlay, so the mask is never OCR'd.
            self.snapshot, self.bounds = capture_screen()
        except Exception as exc:
            LOG.exception("Screen capture failed")
            self.hook.cancel_drag()
            self.hook.enabled.set()
            self.status.set(f"截图失败：{exc}")
            self._notify(self.status.get(), error=True)
            return
        window = self.overlay = tk.Toplevel(self.root)
        window.withdraw()
        window.overrideredirect(True)
        window.attributes("-topmost", True)
        window.geometry(f"{self.bounds.width}x{self.bounds.height}+0+0")
        self.canvas = tk.Canvas(window, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.backdrop = ImageTk.PhotoImage(ImageEnhance.Brightness(self.snapshot).enhance(0.55))
        self.canvas.create_image(0, 0, image=self.backdrop, anchor="nw")
        self.outline = self.canvas.create_rectangle(0, 0, 0, 0, outline="#38bdf8", width=3)
        self.caption = self.canvas.create_text(20, 20, anchor="nw", fill="white",
                                              font=("Microsoft YaHei UI", 12),
                                              text="拖动框选 · 松开识别 · Esc 取消")
        window.bind("<Escape>", lambda _event: self._cancel())
        window.update_idletasks()
        window.deiconify()
        window.update_idletasks()
        # Tk's negative geometry offsets mean 'relative to right edge'. Use native
        # physical coordinates instead, which also handles left/top monitors.
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.GetParent.argtypes = [wintypes.HWND]
        user32.GetParent.restype = wintypes.HWND
        user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int,
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT]
        user32.SetWindowPos.restype = wintypes.BOOL
        hwnd = user32.GetParent(window.winfo_id()) or window.winfo_id()
        if not user32.SetWindowPos(hwnd, ctypes.c_void_p(-1), self.bounds.left,
                                    self.bounds.top, self.bounds.width, self.bounds.height, 0x0040):
            raise ctypes.WinError(ctypes.get_last_error())
        window.focus_force()
        self._draw(point)
        self.status.set("正在框选……")

    def _draw(self, point):
        x1, y1 = self.start_point
        x2, y2 = point
        left, top = self.bounds.left, self.bounds.top
        self.canvas.coords(self.outline, min(x1, x2) - left, min(y1, y2) - top,
                           max(x1, x2) - left, max(y1, y2) - top)
        self.canvas.itemconfigure(self.caption, text=f"{abs(x2-x1)} × {abs(y2-y1)} 像素 · 松开识别 · Esc 取消")
        # Keep help near the active monitor rather than an arbitrary virtual edge.
        self.canvas.coords(self.caption, max(12, min(x1-left, self.bounds.width-360)),
                           max(12, min(y1-top-32, self.bounds.height-35)))

    def _finish(self, point):
        rect = normalize_rect(self.start_point, point, self.bounds)
        if rect is None:
            self._remove_overlay()
            self.hook.enabled.set()
            self.status.set("选区太小：请至少框选 5 × 5 像素。")
            self._notify(self.status.get(), error=True)
            return
        image = self.snapshot.crop((rect.left-self.bounds.left, rect.top-self.bounds.top,
                                    rect.right-self.bounds.left, rect.bottom-self.bounds.top))
        self._remove_overlay()
        self.status.set("正在识别……请稍候")
        self.worker = threading.Thread(target=self._recognize, args=(image,), name="ocr-worker", daemon=True)
        self.worker.start()

    def _recognize(self, image):
        try:
            text = recognize_and_copy(image, self.engine)
        except NoTextError:
            self.events.put(("error", "未识别到文字；剪切板原内容未修改。"))
        except Exception as exc:
            LOG.exception("OCR or clipboard failed")
            self.events.put(("error", f"识别或复制失败：{exc}"))
        else:
            # The clipboard already holds the text: recording it is best effort and
            # must never turn a successful copy into a reported OCR/clipboard failure.
            # The store is written before ("result", text) is published, so anything
            # that reacts to that event can already reload the stored history.
            failed = self._record_history(text) is False
            self.events.put(("result", text))
            if failed:
                self.events.put(("history_error", "历史记录保存失败；本次文字已复制到剪贴板。"))

    def _record_history(self, text):
        """Append one recognized result; ``None`` means no store, ``False`` a failure."""
        store = getattr(self, "history", None)
        if store is None:
            return None
        try:
            return bool(store.append(text))
        except Exception as exc:
            # Never log the recognized text itself, only the failure class.
            LOG.warning("HISTORY_APPEND_FAILED: %s", type(exc).__name__)
            return False

    def _cancel(self):
        if self.overlay:
            self.hook.cancel_drag()
            self._remove_overlay()
            self.hook.enabled.set()
            self.status.set("已取消；剪切板原内容未修改。")

    # -- history viewer ----------------------------------------------------
    def open_history(self):
        """Open or reuse the single history window.

        The list shows one row per record (newest first) and the read-only text
        pane always shows *all* records, oldest first, separated by blank lines.
        """
        window = self._history_window()
        if window is not None:
            self._refresh_history()
            window.deiconify()
            window.lift()
            window.focus_force()
            return window
        window = self.history_view = tk.Toplevel(self.root)
        window.title("识别历史记录")
        window.resizable(True, True)
        # Transient keeps the viewer with the main window: no extra task-bar entry
        # and no orphan window when the main window hides into the tray.
        window.transient(self.root)
        window.protocol("WM_DELETE_WINDOW", self._close_history)
        frame = ttk.Frame(window, padding=10)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)
        toolbar = ttk.Frame(frame)
        toolbar.grid(row=0, column=0, sticky="we")
        actions = (("刷新", self._refresh_history),
                   ("复制所选记录", self._copy_history),
                   ("删除所选记录", self._delete_history),
                   ("复制全部", self._copy_all_history),
                   ("复制选中文字", self._copy_history_selection),
                   ("清空全部", self._clear_history),
                   ("关闭", self._close_history))
        buttons = [ttk.Button(toolbar, text=label, command=command) for label, command in actions]
        columns = self._history_columns(buttons)
        for index, (button, (_label, _command)) in enumerate(zip(buttons, actions)):
            row, column = divmod(index, columns)
            button.grid(row=row, column=column, sticky="we", padx=(0, 6), pady=(0, 4))
        for column in range(columns):
            toolbar.columnconfigure(column, weight=1)
        # A native horizontal sash: the list keeps a usable width when the pane is
        # squeezed, and the user can drag it without any custom handler.
        self.history_splitter = splitter = ttk.Panedwindow(frame, orient="horizontal")
        splitter.grid(row=1, column=0, sticky="nsew", pady=(8, 6))
        self.history_left = left = ttk.Frame(splitter)
        self.history_right = right = ttk.Frame(splitter)
        splitter.add(left, weight=1)
        splitter.add(right, weight=2)
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        self.history_list = tk.Listbox(left, width=self.HISTORY_LIST_CHARS_MIN,
                                       height=self.HISTORY_LIST_ROWS, exportselection=False)
        self.history_list.grid(row=0, column=0, sticky="nsew")
        list_scroll = ttk.Scrollbar(left, orient="vertical", command=self.history_list.yview)
        list_scroll.grid(row=0, column=1, sticky="ns")
        self.history_list.configure(yscrollcommand=list_scroll.set)
        self.history_list.bind("<<ListboxSelect>>", self._on_history_select)
        self.history_list.bind("<Configure>", self._on_history_configure)
        self.history_text = tk.Text(right, wrap="word", width=self.HISTORY_TEXT_CHARS,
                                    height=self.HISTORY_TEXT_ROWS, state="disabled")
        self.history_text.grid(row=0, column=0, sticky="nsew")
        text_scroll = ttk.Scrollbar(right, orient="vertical", command=self.history_text.yview)
        text_scroll.grid(row=0, column=1, sticky="ns")
        self.history_text.configure(yscrollcommand=text_scroll.set)
        self.history_text.bind("<Control-c>", self._copy_history_selection)
        try:
            self.history_text.tag_configure(self.HISTORY_HIGHLIGHT_TAG,
                                            **self.HISTORY_HIGHLIGHT_COLORS)
            # Below 'sel': the user's drag selection must stay on top of the
            # record highlight so copy-selection keeps working.
            self.history_text.tag_lower(self.HISTORY_HIGHLIGHT_TAG)
        except Exception:
            LOG.warning("HISTORY_VIEW_FAILED: the record highlight could not be configured")
        splitter.bind("<Configure>", self._on_history_configure)
        self.history_sash_pending = True   # applied on the first realised configure
        self.history_status_var = tk.StringVar(value="")
        self.history_status_label = ttk.Label(frame, textvariable=self.history_status_var,
                                             wraplength=self._wrap)
        self.history_status_label.grid(row=2, column=0, sticky="we")
        self._fit_history_window(window, toolbar, splitter)
        self._refresh_history()
        return window

    def _history_columns(self, buttons):
        """Pick 2-4 toolbar columns whose widest row fits the scaled target width."""
        target = self._scaled(self.HISTORY_COLUMN_TARGET)
        widths = [max(1, button.winfo_reqwidth()) for button in buttons]
        for columns in (4, 3, 2):
            rows = [widths[index:index + columns] for index in range(0, len(widths), columns)]
            needed = max(sum(row) + 6 * (len(row) - 1) for row in rows)
            if needed <= target or columns == 2:
                return columns
        return 2

    def _fit_history_window(self, window, toolbar, splitter):
        """Give the viewer a roomy natural size, clamped to the screen.

        The minimum only has to keep the toolbar and both panes usable, so the
        sash can still be dragged to a 20/80 or 80/20 split.
        """
        window.update_idletasks()
        base_width, base_height = self.HISTORY_WINDOW_BASE
        width = max(window.winfo_reqwidth(), self._scaled(base_width))
        height = max(window.winfo_reqheight(), self._scaled(base_height))
        limit_width = max(self._scaled(240), self.root.winfo_screenwidth() - self._scaled(80))
        limit_height = max(self._scaled(240), self.root.winfo_screenheight() - self._scaled(120))
        width, height = min(width, limit_width), min(height, limit_height)
        window.geometry(f"{width}x{height}")
        panes_min = (self._scaled(self.HISTORY_LIST_CHARS_MIN * 8)
                     + self._scaled(self.HISTORY_TEXT_CHARS * 8) + self._scaled(60))
        window.minsize(max(toolbar.winfo_reqwidth() + self._scaled(24), panes_min),
                       min(height, self._scaled(base_height)))

    def _set_history_sash(self, splitter=None):
        """Apply the initial sash once the paned window really has a width.

        Asking for a sash position before the toplevel is realised is clamped to 0,
        which leaves the list pane invisible, so this only arms the one-shot
        initialization: the first realised ``<Configure>`` applies it and it is
        never touched again (a later user drag must survive).
        """
        if not getattr(self, "history_sash_pending", False):
            return False
        splitter = splitter if splitter is not None else getattr(self, "history_splitter", None)
        if splitter is None:
            self.history_sash_pending = False
            return False
        try:
            width = int(splitter.winfo_width())
        except Exception:
            self.history_sash_pending = False
            return False
        if width <= 1:
            return False                    # not realised yet: stay armed
        desired = self._scaled(self.HISTORY_SASH_BASE)
        keep_right = self._scaled(self.HISTORY_PANE_RIGHT_MIN)
        position = max(self._scaled(self.HISTORY_PANE_LEFT_MIN), min(desired, width - keep_right))
        try:
            splitter.sashpos(0, position)
        except Exception:
            LOG.warning("HISTORY_VIEW_FAILED: the splitter position could not be set")
            self.history_sash_pending = False
            return False
        self.history_sash_pending = False    # one-shot: a later drag is never overridden
        return True

    def _on_history_configure(self, _event=None):
        """Apply the pending sash, then re-render the cached rows for the real width."""
        self._set_history_sash()
        width = self._history_list_width()
        if width is None or width <= 1:
            return
        self._resize_history_status()
        if abs(width - (getattr(self, "history_summary_width", 0) or 0)) <= 2:
            return
        self._rebuild_history_rows(width)

    def _resize_history_status(self):
        label = getattr(self, "history_status_label", None)
        splitter = getattr(self, "history_splitter", None)
        if label is None or splitter is None:
            return
        try:
            width = int(splitter.winfo_width())
        except Exception:
            return
        wrap = max(self._scaled(200), width - 4)
        if width > 1 and wrap != self._history_status_wrap:
            self._history_status_wrap = wrap
            label.configure(wraplength=wrap)

    def _rebuild_history_rows(self, width):
        """Re-clip the cached summaries for *width* pixels: no store access."""
        listbox = getattr(self, "history_list", None)
        font = self._history_font()
        items = getattr(self, "history_items", None) or []
        if listbox is None:
            self.history_summary_width = width
            return
        rows = [self._history_summary(record, index, width_px=width, font=font)
                for index, record in enumerate(reversed(items))]
        try:
            listbox.delete(0, "end")
            for row in rows:
                listbox.insert("end", row)
        except Exception:
            LOG.warning("HISTORY_VIEW_FAILED: the record rows could not be resized")
            return
        self.history_summaries = rows
        self.history_summary_width = width
        self._sync_history_selection()

    def _history_window(self):
        window = getattr(self, "history_view", None)
        if window is None:
            return None
        try:
            if window.winfo_exists():
                return window
        except Exception:
            pass
        self.history_view = None
        return None

    @staticmethod
    def _record_id(record):
        value = getattr(record, "id", None)
        return None if value is None else str(value)

    @staticmethod
    def _record_text(record):
        return str(getattr(record, "text", "") or "")

    @staticmethod
    def _format_time(created_at, compact=False):
        """Local ``YYYY-MM-DD HH:MM:SS`` (or ``MM-DD HH:MM``); legacy shows 时间未知."""
        if not created_at:
            return "时间未知"
        try:
            parsed = datetime.datetime.fromisoformat(str(created_at))
        except ValueError:
            return str(created_at)
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone()
        return parsed.strftime("%m-%d %H:%M" if compact else "%Y-%m-%d %H:%M:%S")

    def _history_all_text(self):
        """Every record's text, oldest first, separated by exactly one blank line."""
        return "\n\n".join(self._record_text(record)
                           for record in getattr(self, "history_items", None) or [])

    def _refresh_history(self, *, select_newest=True):
        """Reload records (oldest first) and rebuild the list plus the aggregate text.

        The selected record is resolved through the per-refresh id cache, so a new
        OCR append can never make the selection point at another record.
        """
        items = []
        store = getattr(self, "history", None)
        if store is not None:
            try:
                items = list(store.load_records())
            except Exception as exc:
                LOG.warning("HISTORY_LOAD_FAILED: %s", type(exc).__name__)
                self._history_status("无法读取历史记录；本次识别仍可正常复制。")
                items = []
        self.history_items = items
        self.history_row_ids = [self._record_id(record) for record in reversed(items)]
        if self.history_selected_id not in self.history_row_ids:
            # An id that no longer exists must never select (or delete) another record.
            self.history_selected_id = None
        listbox = getattr(self, "history_list", None)
        width = self._history_list_width()
        font = self._history_font()
        rows = [self._history_summary(record, index, width_px=width, font=font)
                for index, record in enumerate(reversed(items))]
        if listbox is not None:
            try:
                listbox.delete(0, "end")
                for row in rows:
                    listbox.insert("end", row)
            except Exception:
                LOG.warning("HISTORY_VIEW_FAILED: the record list could not be updated")
        self.history_summaries = rows
        self.history_summary_width = width or 0
        if self.history_selected_id is None and self.history_row_ids and select_newest:
            # Only a viewer without a selection follows the newest record.
            self.history_selected_id = self.history_row_ids[0]
        self._render_history_text()
        self._sync_history_selection()
        return items

    def _refresh_history_if_open(self):
        """Refresh an open viewer after a new result; never steal an existing selection."""
        if self._history_window() is None:
            return False
        self._refresh_history()
        return True

    def _history_list_width(self):
        """Interior pixel width of the list pane, or ``None`` before it is mapped.

        Borders and the focus highlight are subtracted: a row may use the pixels the
        text can actually occupy, not the widget's outer width.
        """
        listbox = getattr(self, "history_list", None)
        if listbox is None:
            return None
        try:
            width = int(listbox.winfo_width())
            if width <= 1:
                width = int(listbox.winfo_reqwidth())
            if width <= 1:
                return None
            inset = 0
            for option in ("borderwidth", "highlightthickness"):
                try:
                    inset += int(listbox.cget(option))
                except Exception:
                    pass
            return max(1, width - 2 * inset)
        except Exception:
            return None

    def _history_font(self):
        """The list's own font, or ``None`` when no Tk font is available (unit tests)."""
        try:
            name = "TkDefaultFont"
            listbox = getattr(self, "history_list", None)
            if listbox is not None:
                name = str(listbox.cget("font") or name)
            return tkfont.nametofont(name)
        except Exception:
            return None

    def _history_preview(self, text, room_px, font):
        """First few characters that fit *room_px*, ellipsis-clipped; '' if none fit.

        The returned string is always measured, so a candidate carrying the ellipsis
        can never be wider than the room (narrow glyphs like 'i' would otherwise
        slip through the 17-character probe).
        """
        text = text[:self.HISTORY_PREVIEW_MAX + 1]
        if not text or room_px <= 0:
            return ""
        candidate = (text if len(text) <= self.HISTORY_PREVIEW_MAX
                     else text[:self.HISTORY_PREVIEW_MAX] + "…")
        if font.measure(candidate) <= room_px:
            return candidate
        budget = room_px - font.measure("…")
        if budget <= 0:
            return ""
        trimmed = text
        while trimmed and font.measure(trimmed) > budget:
            trimmed = trimmed[:-1]
        return trimmed + "…" if trimmed else ""

    def _history_summary(self, record, index, width_px=None, font=None):
        """One row: index, local time, then as much text preview as really fits.

        The preview is clipped from the measured pixel width, so a narrow pane
        compacts the date instead of degrading into a timestamp-only row.
        """
        stamp = self._format_time(getattr(record, "created_at", None))
        text = " ".join(self._record_text(record).split())
        prefix = f"{index + 1:>2}. "
        font = font if font is not None else self._history_font()
        if width_px is None:
            width_px = self._history_list_width()
        if font is None or not width_px:
            preview = text[:self.HISTORY_PREVIEW_MAX]
            return f"{prefix}{stamp}{' ' + preview if preview else ''}"
        room = width_px - font.measure(prefix) - font.measure(stamp) - font.measure(" ")
        if room < font.measure("0" * self.HISTORY_PREVIEW_MIN) and len(stamp) > 11:
            stamp = self._format_time(getattr(record, "created_at", None), compact=True)
            room = width_px - font.measure(prefix) - font.measure(stamp) - font.measure(" ")
        preview = self._history_preview(text, room, font)
        return f"{prefix}{stamp}{' ' + preview if preview else ''}"

    def _render_history_text(self):
        """Show the whole aggregate in the read-only pane, preserving a live selection.

        Records are inserted one by one and their Tcl start/end indices are taken
        from the widget itself, so no Python character offsets (which differ from
        Tcl's UTF-16 indices around emoji) are ever computed.
        """
        widget = getattr(self, "history_text", None)
        if widget is None:
            return
        try:
            previous = widget.get("1.0", "end-1c")
            selected = tuple(str(index) for index in widget.tag_ranges("sel"))
            parts = [self._record_text(record) for record in
                     getattr(self, "history_items", None) or []]
            aggregate = "\n\n".join(parts)
            widget.configure(state="normal")
            widget.delete("1.0", "end")
            spans = {}
            for position, record in enumerate(getattr(self, "history_items", None) or []):
                if position:
                    widget.insert("end", "\n\n")
                start = str(widget.index("end-1c"))
                widget.insert("end", self._record_text(record))
                entry_id = self._record_id(record)
                if entry_id is not None:
                    spans[entry_id] = (start, str(widget.index("end-1c")))
            self.history_spans = spans
            # Restore the selection only while the aggregate merely grew at the end;
            # losing the oldest record (100-entry cap) changes the prefix -> clear it.
            if len(selected) >= 2 and aggregate.startswith(previous):
                widget.tag_add("sel", selected[0], selected[1])
            widget.configure(state="disabled")
            self._apply_history_highlight(widget)
        except Exception:
            LOG.warning("HISTORY_VIEW_FAILED: the aggregate text could not be displayed")

    def _sync_history_selection(self):
        """Highlight the selected id in the list; the text pane is left untouched."""
        listbox = getattr(self, "history_list", None)
        if listbox is None:
            return
        selected = getattr(self, "history_selected_id", None)
        try:
            listbox.selection_clear(0, "end")
            if selected is not None and selected in (self.history_row_ids or []):
                index = self.history_row_ids.index(selected)
                listbox.selection_set(index)
                listbox.see(index)
        except Exception:
            LOG.warning("HISTORY_VIEW_FAILED: the selection could not be restored")

    def _selected_record(self):
        """Record for the current list row, resolved by the cached stable id."""
        listbox = getattr(self, "history_list", None)
        rows = getattr(self, "history_row_ids", None) or []
        if listbox is None:
            return None
        try:
            selection = listbox.curselection()
        except Exception:
            return None
        if not selection:
            return None
        try:
            index = int(selection[0])
        except (TypeError, ValueError, IndexError):
            return None
        if index < 0 or index >= len(rows) or rows[index] is None:
            return None
        wanted = rows[index]
        for record in getattr(self, "history_items", None) or []:
            if self._record_id(record) == wanted:
                return record
        return None

    def _on_history_select(self, _event=None):
        """Selection feedback only: the aggregate text pane keeps every record."""
        record = self._selected_record()
        if record is None:
            return
        self.history_selected_id = self._record_id(record)
        self._apply_history_highlight()
        self._history_status(
            f"已选择 {self._format_time(getattr(record, 'created_at', None))} · "
            f"{len(self._record_text(record))} 个字符；右侧始终显示全部记录。")

    def _apply_history_highlight(self, widget=None):
        """Tag only the selected record's span; the user's drag selection is left alone."""
        widget = widget if widget is not None else getattr(self, "history_text", None)
        if widget is None:
            return
        try:
            widget.tag_remove(self.HISTORY_HIGHLIGHT_TAG, "1.0", "end")
            span = (getattr(self, "history_spans", None) or {}).get(
                getattr(self, "history_selected_id", None))
            if span is None:
                return
            widget.tag_add(self.HISTORY_HIGHLIGHT_TAG, span[0], span[1])
            widget.see(span[0])
        except Exception:
            LOG.warning("HISTORY_VIEW_FAILED: the selected record could not be highlighted")

    def _copy_text(self, text, empty_message):
        """Single clipboard path for every history copy action; never appends history."""
        text = text or ""
        if not text:
            self._history_status(empty_message)
            return False
        try:
            write_text(text)
        except Exception as exc:
            LOG.warning("HISTORY_COPY_FAILED: %s", type(exc).__name__)
            # write_text can empty the clipboard before SetClipboardData fails.
            self._history_status("复制失败，请重试；剪贴板可能已改变。")
            return False
        self._history_status(f"已复制 {len(text)} 个字符（未新增历史记录）。")
        return True

    def _copy_history(self):
        """Copy the selected record's full text (by stable id), not the aggregate."""
        record = self._selected_record()
        if record is None:
            self._history_status("请先在列表中选择一条记录。")
            return False
        self.history_selected_id = self._record_id(record)
        return self._copy_text(self._record_text(record), "所选记录没有可复制的文字。")

    def _copy_all_history(self):
        """Copy every record, oldest first, without adding a history entry."""
        return self._copy_text(self._history_all_text(), "暂无历史记录。")

    def _copy_history_selection(self, event=None):
        """Copy the text selected in the pane; Ctrl+C returns ``break`` (single writer)."""
        widget = getattr(self, "history_text", None)
        selected = ""
        try:
            if widget is not None:
                selected = widget.get("sel.first", "sel.last")
        except Exception:
            selected = ""   # no selection: the clipboard must stay untouched
        result = self._copy_text(selected, "请先在右侧选择要复制的文字。")
        return "break" if event is not None else result

    def _delete_history(self):
        """Delete exactly the selected record, by the id taken before the dialog."""
        store = getattr(self, "history", None)
        record = self._selected_record()
        entry_id = self._record_id(record) if record is not None else None
        if store is None or entry_id is None:
            self._history_status("请先在列表中选择一条要删除的记录。")
            return False
        if not messagebox.askyesno("删除所选记录", "确定要删除所选的一条历史记录吗？剪贴板内容不会改变。",
                                   parent=getattr(self, "history_view", None)):
            self._history_status("已取消删除。")
            return False
        try:
            deleted = bool(store.delete(entry_id))
        except Exception as exc:
            LOG.warning("HISTORY_DELETE_FAILED: %s", type(exc).__name__)
            self._history_status("删除失败，历史记录未改变。")
            return False
        self.history_selected_id = None
        self._refresh_history(select_newest=False)
        if not deleted:
            self._history_status("该记录已不存在，列表已刷新（未删除其他记录）。")
            return False
        self._history_status("已删除所选记录（剪贴板内容未改变）。")
        return True

    def _clear_history(self):
        """Delete every stored record after confirmation; the clipboard is untouched."""
        store = getattr(self, "history", None)
        if store is None:
            self._history_status("历史记录不可用。")
            return False
        if not messagebox.askyesno("清空历史记录", "确定要删除全部历史记录吗？剪贴板内容不会改变。",
                                   parent=getattr(self, "history_view", None)):
            self._history_status("已取消清空。")
            return False
        try:
            store.clear()
        except Exception as exc:
            LOG.warning("HISTORY_CLEAR_FAILED: %s", type(exc).__name__)
            self._history_status("清空失败，历史记录未改变。")
            return False
        self.history_selected_id = None
        self._refresh_history(select_newest=False)
        self._history_status("历史记录已清空（剪贴板内容未改变）。")
        return True

    def _close_history(self):
        """Close only the history window; the app and the clipboard are untouched."""
        window = getattr(self, "history_view", None)
        self.history_view = None
        self.history_list = None
        self.history_text = None
        self.history_status_var = None
        self.history_status_label = None
        self.history_splitter = None
        self.history_left = None
        self.history_right = None
        self.history_items = []
        self.history_row_ids = []
        self.history_selected_id = None
        self.history_spans = {}
        self.history_summaries = []
        self.history_summary_width = 0
        self.history_sash_pending = False
        self._history_status_wrap = None
        if window is not None:
            try:
                window.destroy()
            except Exception:
                LOG.warning("HISTORY_VIEW_FAILED: the history window could not be closed")

    def _history_status(self, text):
        variable = getattr(self, "history_status_var", None)
        if variable is None:
            return
        try:
            variable.set(text)
        except Exception:
            LOG.warning("HISTORY_VIEW_FAILED: the status line could not be updated")

    def _remove_overlay(self):
        if self.overlay:
            self.overlay.destroy()
        self.overlay = None
        self.snapshot = None
        self.backdrop = None

    def _dismiss_toast(self):
        if self.toast:
            try:
                self.toast.destroy()
            except tk.TclError:
                pass
            self.toast = None

    def _notify(self, text, *, error=False):
        self._dismiss_toast()
        toast = self.toast = tk.Toplevel(self.root)
        toast.overrideredirect(True)
        toast.attributes("-topmost", True)
        color = "#7f1d1d" if error else "#164e63"
        tk.Label(toast, text=text, bg=color, fg="white", padx=18, pady=12,
                 wraplength=420, justify="left", font=("Microsoft YaHei UI", 10)).pack()
        toast.update_idletasks()
        toast.geometry(f"+{max(0, self.root.winfo_screenwidth()-toast.winfo_reqwidth()-30)}"
                       f"+{max(0, self.root.winfo_screenheight()-toast.winfo_reqheight()-70)}")
        toast.after(4500, self._dismiss_toast)

    def _finish_smoke(self):
        if not self.ready:
            self.smoke_failed = True
        else:
            self.smoke_failed = False
            LOG.info("GUI_SMOKE_PASS: Tk mainloop, OCR and hook registration passed; integrity=%s; physical input not tested",
                     integrity_name(self.integrity_level))
        self.close()

    def close(self):
        """Release timers, hook, tray icon, windows and Tk objects; safe to call twice.

        Pending timers are cancelled and every Tk reference is dropped here, on the
        GUI thread: a Tk variable or the interpreter itself freed later by another
        thread's garbage collector aborts the process with
        ``Tcl_AsyncDelete: async handler deleted by the wrong thread``.  Every step
        is attempted even when an earlier one fails (logged, never silent).
        """
        if getattr(self, "closed", False):
            return
        self.closing = True
        self.closed = True
        for name in ("poll_timer", "smoke_timer", "background_timer", "main_layout_timer"):
            timer = getattr(self, name, None)
            if timer is None:
                continue
            setattr(self, name, None)
            try:
                self.root.after_cancel(timer)
            except Exception as exc:
                LOG.warning("CLEANUP_FAILED (timer %s): %s", name, exc)
        trace = getattr(self, "_status_trace", None)
        self._status_trace = None
        if trace is not None:
            try:
                self.status.trace_remove("write", trace)
            except Exception as exc:
                LOG.warning("CLEANUP_FAILED (status trace): %s", exc)
        for label, action in (("mouse hook", self.hook.stop),
                              ("history viewer", self._close_history),
                              ("selection overlay", self._remove_overlay),
                              ("toast", self._dismiss_toast),
                              ("tray icon", self._stop_tray)):
            try:
                action()
            except Exception as exc:
                LOG.warning("CLEANUP_FAILED (%s): %s", label, exc)
        root = self.root
        for name in ("main_frame", "title_label", "info_label", "status", "status_label",
                     "buttons_frame", "history_view", "history_splitter", "history_left",
                     "history_right", "history_list", "history_text", "history_status_var",
                     "history_status_label", "canvas", "outline", "caption", "backdrop",
                     "overlay", "toast", "snapshot"):
            setattr(self, name, None)
        self.root = None
        if root is not None:
            try:
                root.destroy()
            except Exception as exc:
                LOG.warning("CLEANUP_FAILED (window): %s", exc)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ctrl+鼠标左键框选 OCR，自动复制中英文文字。")
    parser.add_argument("--smoke-seconds", type=float, default=0,
                        help="测试模式：指定秒数后退出，模型和钩子未就绪则返回失败")
    args = parser.parse_args(argv)
    if sys.platform != "win32":
        parser.error("此工具仅支持 Windows 11。")
    logdir = Path(__file__).resolve().parent.parent / "logs"
    logdir.mkdir(exist_ok=True)
    handlers = [logging.FileHandler(logdir / "screenocr.log", encoding="utf-8")]
    if sys.stderr is not None:  # pythonw.exe has no console stream.
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=handlers)
    enable_dpi_awareness()
    root = tk.Tk()
    try:
        app = ScreenOCRApp(root, smoke_seconds=args.smoke_seconds)
        root.mainloop()
        return 1 if getattr(app, "smoke_failed", False) else 0
    except Exception as exc:
        LOG.exception("Fatal startup error")
        messagebox.showerror("屏幕 OCR 启动失败", f"{exc}\n请重新安装 requirements.txt 中的依赖。", parent=root)
        root.destroy()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
