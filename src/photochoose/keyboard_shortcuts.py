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


def physical_shortcut_token(
    event: QKeyEvent, platform: Optional[str] = None
) -> Optional[str]:
    """Return a layout-independent culling key for a native key event."""

    if event.modifiers() & (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier):
        return None
    if (platform or sys.platform) != "win32":
        return None
    return WINDOWS_SCAN_CODE_SHORTCUTS.get(event.nativeScanCode() & 0xFF)


class LayoutIndependentShortcutFilter(QObject):
    """Dispatch physical Windows culling keys inside one top-level window."""

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
