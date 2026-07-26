from __future__ import annotations

import unittest

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent

from photochoose.keyboard_shortcuts import (
    MACOS_VIRTUAL_KEY_SHORTCUTS,
    WINDOWS_SCAN_CODE_SHORTCUTS,
    physical_shortcut_token,
)


class KeyboardShortcutTests(unittest.TestCase):
    def test_windows_culling_shortcuts_use_physical_scan_codes(self) -> None:
        for scan_code, expected in WINDOWS_SCAN_CODE_SHORTCUTS.items():
            event = QKeyEvent(
                QEvent.KeyPress,
                ord("Ж"),
                Qt.NoModifier,
                scan_code,
                0,
                0,
                "ж",
            )
            self.assertEqual(expected, physical_shortcut_token(event, "win32"))

    def test_modified_or_non_windows_keys_are_not_intercepted(self) -> None:
        modified = QKeyEvent(
            QEvent.KeyPress,
            ord("Л"),
            Qt.ControlModifier,
            0x25,
            0,
            0,
            "л",
        )
        self.assertIsNone(physical_shortcut_token(modified, "win32"))
        self.assertIsNone(physical_shortcut_token(modified, "darwin"))

    def test_macos_culling_shortcuts_use_physical_virtual_keys(self) -> None:
        for virtual_key, expected in MACOS_VIRTUAL_KEY_SHORTCUTS.items():
            event = QKeyEvent(
                QEvent.KeyPress,
                ord("Ж"),
                Qt.NoModifier,
                0,
                virtual_key,
                0,
                "ж",
            )
            self.assertEqual(expected, physical_shortcut_token(event, "darwin"))


if __name__ == "__main__":
    unittest.main()
