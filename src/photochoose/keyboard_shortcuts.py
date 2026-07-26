from __future__ import annotations

import sys
from typing import Callable, Mapping, Optional

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QApplication,
    QLineEdit,
    QPlainTextEdit,
    QTextEdit,
    QWidget,
)

WINDOWS_SCAN_CODE_SHORTCUTS = {
    0x30: "B",
    0x25: "K",
    0x13: "R",
    0x2D: "X",
    0x16: "U",
    0x18: "O",
}

# macOS reports the hardware-position key code as nativeVirtualKey(). These
# ANSI positions stay stable when the user switches to Cyrillic or another
# keyboard layout.
MACOS_VIRTUAL_KEY_SHORTCUTS = {
    0x0B: "B",
    0x28: "K",
    0x0F: "R",
    0x07: "X",
    0x20: "U",
    0x1F: "O",
}


def physical_shortcut_token(
    event: QKeyEvent, platform: Optional[str] = None
) -> Optional[str]:
    """Return a layout-independent culling key for a native key event."""

    if event.modifiers() & (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier):
        return None
    current_platform = platform or sys.platform
    if current_platform == "win32":
        return WINDOWS_SCAN_CODE_SHORTCUTS.get(event.nativeScanCode() & 0xFF)
    if current_platform == "darwin":
        return MACOS_VIRTUAL_KEY_SHORTCUTS.get(event.nativeVirtualKey() & 0xFF)
    return None


class LayoutIndependentShortcutFilter(QObject):
    """Dispatch physical Windows and macOS keys inside one top-level window."""

    def __init__(
        self,
        owner: QWidget,
        handlers: Mapping[str, Callable[[], None]],
    ) -> None:
        super().__init__(owner)
        self.owner = owner
        self.handlers = dict(handlers)
        application = QApplication.instance()
        if application is not None:
            application.installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() != QEvent.KeyPress or not isinstance(event, QKeyEvent):
            return False
        application = QApplication.instance()
        if application is None or application.activeWindow() is not self.owner:
            return False
        focus = application.focusWidget()
        if focus is not None:
            if focus.window() is not self.owner:
                return False
            if isinstance(focus, (QLineEdit, QPlainTextEdit, QTextEdit)):
                return False
        token = physical_shortcut_token(event)
        handler = self.handlers.get(token or "")
        if handler is None:
            return False
        handler()
        event.accept()
        return True
