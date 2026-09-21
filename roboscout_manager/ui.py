"""Tiny optional tkinter UI: splash while the app boots, Setup dialog, message boxes.

Everything degrades to console/no-op when tkinter is unavailable or the caller
asks for ``silent`` so the same code paths run headless in CI and tests.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Any

from roboscout_manager import APP_NAME

log = logging.getLogger("roboscout.manager")


def _tk() -> Any:
    try:
        import tkinter  # type: ignore

        return tkinter
    except Exception:  # noqa: BLE001
        return None


class Splash:
    """Borderless "Starting RoboScoutAI…" card; call ``pump()`` while waiting."""

    def __init__(self, message: str = f"Starting {APP_NAME}…", *, enabled: bool = True) -> None:
        self._root: Any = None
        self._label: Any = None
        self._progress: Any = None
        if not enabled:
            return
        tk = _tk()
        if tk is None:
            return
        try:
            root = tk.Tk()
            root.title(APP_NAME)
            root.overrideredirect(True)
            root.attributes("-topmost", True)
            width, height = 360, 120
            sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
            root.geometry(f"{width}x{height}+{(sw - width) // 2}+{(sh - height) // 2}")
            frame = tk.Frame(root, bg="#0f172a", padx=24, pady=20, highlightthickness=1, highlightbackground="#334155")
            frame.pack(fill="both", expand=True)
            tk.Label(frame, text=APP_NAME, fg="#f8fafc", bg="#0f172a", font=("Segoe UI", 16, "bold"), anchor="w").pack(fill="x")
            self._label = tk.Label(frame, text=message, fg="#cbd5e1", bg="#0f172a", font=("Segoe UI", 10), anchor="w")
            self._label.pack(fill="x", pady=(6, 0))
            self._progress = tk.Label(frame, text="", fg="#38bdf8", bg="#0f172a", font=("Segoe UI", 9), anchor="w")
            self._progress.pack(fill="x")
            root.update()
            self._root = root
        except Exception as exc:  # noqa: BLE001
            log.debug("splash unavailable: %s", exc)
            self._root = None

    @property
    def active(self) -> bool:
        return self._root is not None

    def set_message(self, message: str, *, progress: int | None = None) -> None:
        if self._root is None:
            return
        try:
            self._label.configure(text=message)
            if progress is not None and self._progress is not None:
                bar = "█" * (progress // 5) + "░" * (20 - progress // 5)
                self._progress.configure(text=f"{bar} {progress}%")
            self._root.update()
        except Exception:  # noqa: BLE001
            self._root = None

    def pump(self) -> None:
        if self._root is None:
            return
        try:
            self._root.update()
        except Exception:  # noqa: BLE001
            self._root = None

    def close(self) -> None:
        if self._root is None:
            return
        try:
            self._root.destroy()
        except Exception:  # noqa: BLE001
            pass
        self._root = None


def message_box(title: str, text: str, *, kind: str = "info", silent: bool = False) -> None:
    if silent:
        return
    tk = _tk()
    if tk is not None:
        try:
            from tkinter import messagebox  # type: ignore

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            if kind == "error":
                messagebox.showerror(title, text, parent=root)
            elif kind == "warning":
                messagebox.showwarning(title, text, parent=root)
            else:
                messagebox.showinfo(title, text, parent=root)
            root.destroy()
            return
        except Exception as exc:  # noqa: BLE001
            log.debug("message box unavailable: %s", exc)
    stream = sys.stderr if kind == "error" else sys.stdout
    if stream is not None:
        print(f"{title}: {text}", file=stream)


def ask_yes_no(title: str, text: str, *, default: bool = True, silent: bool = False) -> bool:
    if silent:
        return default
    tk = _tk()
    if tk is not None:
        try:
            from tkinter import messagebox  # type: ignore

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            answer = bool(messagebox.askyesno(title, text, parent=root))
            root.destroy()
            return answer
        except Exception as exc:  # noqa: BLE001
            log.debug("yes/no box unavailable: %s", exc)
    return default


@dataclass
class SetupChoice:
    proceed: bool
    desktop_shortcut: bool = True
    mode: str = "install"  # install | upgrade | reinstall


def setup_dialog(
    install_root: str,
    *,
    version: str,
    existing_version: str = "",
    default_desktop: bool = True,
    silent: bool = False,
) -> SetupChoice:
    """Single-page installer dialog. Detects an existing install and offers Upgrade."""
    existing = (existing_version or "").strip()
    is_upgrade = bool(existing) and existing != version
    is_same = bool(existing) and existing == version
    if silent:
        return SetupChoice(
            proceed=True,
            desktop_shortcut=default_desktop,
            mode="reinstall" if is_same else ("upgrade" if existing else "install"),
        )
    tk = _tk()
    if tk is None:
        return SetupChoice(
            proceed=True,
            desktop_shortcut=default_desktop,
            mode="reinstall" if is_same else ("upgrade" if existing else "install"),
        )
    try:
        root = tk.Tk()
        title = f"{APP_NAME} Setup"
        if is_upgrade:
            title = f"{APP_NAME} Update"
        elif is_same:
            title = f"{APP_NAME} Repair"
        root.title(title)
        root.resizable(False, False)
        root.attributes("-topmost", True)
        width, height = 480, 300 if existing else 260
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{width}x{height}+{(sw - width) // 2}+{(sh - height) // 2}")
        frame = tk.Frame(root, padx=24, pady=20)
        frame.pack(fill="both", expand=True)

        if is_upgrade:
            headline = f"Update {APP_NAME}"
            body = (
                f"An existing install was found.\n\n"
                f"Current:  {existing}\n"
                f"New:      {version}\n\n"
                f"Location: {install_root}\n\n"
                "This upgrades in place (side-by-side). The previous version is kept for rollback."
            )
            action = "Update"
            mode = "upgrade"
        elif is_same:
            headline = f"Repair {APP_NAME} {version}"
            body = (
                f"{APP_NAME} {version} is already installed.\n\n"
                f"Location: {install_root}\n\n"
                "Setup will refresh the manager, shortcuts, and app files without removing your data."
            )
            action = "Repair"
            mode = "reinstall"
        else:
            headline = f"Install {APP_NAME} {version}"
            body = (
                "Installs for the current user only (no administrator rights needed).\n\n"
                f"Location: {install_root}\n\n"
                "Updates download side-by-side and switch atomically, so you can always roll back."
            )
            action = "Install"
            mode = "install"

        tk.Label(frame, text=headline, font=("Segoe UI", 14, "bold"), anchor="w").pack(fill="x")
        tk.Label(
            frame,
            text=body,
            justify="left",
            anchor="w",
            wraplength=430,
            font=("Segoe UI", 9),
        ).pack(fill="x", pady=(8, 8))
        desktop_var = tk.BooleanVar(value=default_desktop)
        tk.Checkbutton(frame, text="Create a Desktop shortcut", variable=desktop_var, anchor="w").pack(fill="x")
        result = SetupChoice(proceed=False, desktop_shortcut=default_desktop, mode=mode)

        def install() -> None:
            result.proceed = True
            result.desktop_shortcut = bool(desktop_var.get())
            result.mode = mode
            root.destroy()

        def cancel() -> None:
            root.destroy()

        buttons = tk.Frame(frame)
        buttons.pack(fill="x", pady=(16, 0))
        tk.Button(buttons, text="Cancel", width=12, command=cancel).pack(side="right", padx=(8, 0))
        tk.Button(buttons, text=action, width=12, command=install, default="active").pack(side="right")
        root.protocol("WM_DELETE_WINDOW", cancel)
        root.bind("<Return>", lambda _e: install())
        root.mainloop()
        return result
    except Exception as exc:  # noqa: BLE001
        log.debug("setup dialog unavailable: %s", exc)
        return SetupChoice(
            proceed=True,
            desktop_shortcut=default_desktop,
            mode="reinstall" if is_same else ("upgrade" if existing else "install"),
        )
