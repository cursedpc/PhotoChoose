from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from photochoose.app import create_application  # noqa: E402
from photochoose.onboarding import OnboardingController  # noqa: E402


class OnboardingTests(unittest.TestCase):
    def test_guide_is_persistent_and_waits_for_relevant_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = create_application(["photochoose-hints-test"], root / "state.sqlite3")
            window = app.main_window
            window.show()
            app.processEvents()
            settings = QSettings(str(root / "settings.ini"), QSettings.IniFormat)
            guide = OnboardingController(
                window, settings=settings, auto_start=False
            )

            guide.restart()
            app.processEvents()
            self.assertIsNotNone(guide.overlay)
            self.assertEqual("welcome", guide.current_step.key)
            welcome = guide.overlay.body_label.text()
            self.assertIn("JPEG", welcome)
            self.assertIn("RAW", welcome)
            self.assertIn("original files are never changed", welcome)
            self.assertEqual(window.rect(), guide.overlay.geometry())

            QTest.mouseClick(guide.overlay, Qt.LeftButton)
            QTest.qWait(180)
            app.processEvents()
            self.assertEqual("folders", guide.current_step.key)
            self.assertTrue(
                settings.value(
                    guide.SETTINGS_PREFIX + "welcome", False, type=bool
                )
            )

            QTest.mouseClick(guide.overlay, Qt.LeftButton)
            QTest.qWait(180)
            app.processEvents()
            self.assertIsNone(guide.overlay)

            shoot = root / "Shoot"
            shoot.mkdir()
            success, encoded = cv2.imencode(
                ".jpg", np.full((80, 120, 3), 127, dtype=np.uint8)
            )
            self.assertTrue(success)
            (shoot / "frame.jpg").write_bytes(encoded.tobytes())
            project = window.database.add_project(shoot)
            window.database.scan_project(project.id)
            window._load_projects(project.id)
            guide.notify_context_changed()
            app.processEvents()
            self.assertEqual("workspace", guide.current_step.key)

            guide.restart()
            app.processEvents()
            self.assertEqual("welcome", guide.current_step.key)
            self.assertFalse(
                settings.value(
                    guide.SETTINGS_PREFIX + "folders", False, type=bool
                )
            )
            window.close()


if __name__ == "__main__":
    unittest.main()
