from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QRect, QSettings, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QWidget  # noqa: E402

from photochoose.app import create_application  # noqa: E402
from photochoose.onboarding import CoachMarkOverlay, OnboardingController  # noqa: E402


class OnboardingTests(unittest.TestCase):
    def test_target_highlight_tracks_widget_across_geometry_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = create_application(["photochoose-hint-position-test"], root / "state.sqlite3")
            host = QWidget()
            host.resize(920, 560)
            host.move(37, 51)
            host.show()
            target = QWidget(host)
            target.setGeometry(24, 18, 280, 34)
            target.show()
            overlay = CoachMarkOverlay(
                host,
                target,
                "Responsive target",
                "The highlight follows logical widget coordinates.",
                "GUIDE  8 / 8",
            )
            app.processEvents()

            for position, size in (
                (QPoint(37, 51), (920, 560)),
                (QPoint(213, 127), (640, 390)),
                (QPoint(5, 9), (1180, 720)),
            ):
                host.move(position)
                host.resize(*size)
                target.move(max(12, size[0] // 9), max(12, size[1] // 10))
                overlay._sync_geometry()
                expected_top_left = overlay.mapFrom(
                    host,
                    target.mapTo(host, target.rect().topLeft()),
                )
                expected_bottom_right = overlay.mapFrom(
                    host,
                    target.mapTo(host, target.rect().bottomRight()),
                )
                expected = (
                    QRect(expected_top_left, expected_bottom_right)
                    .normalized()
                    .adjusted(-6, -6, 6, 6)
                    .intersected(overlay.rect().adjusted(8, 8, -8, -8))
                )
                self.assertEqual(expected, overlay._target_rect)
                self.assertGreaterEqual(overlay._target_rect.height(), 8)

            overlay.detach()
            host.close()
            app.main_window.close()

    def test_hint_card_stays_inside_a_small_host_and_scrolls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = create_application(["photochoose-small-hint-test"], root / "state.sqlite3")
            host = QWidget()
            host.resize(320, 190)
            host.show()
            overlay = CoachMarkOverlay(
                host,
                None,
                "A responsive guide on every display",
                "This deliberately long explanation verifies that larger fonts and "
                "shorter screens never push the guide card outside the visible area. "
                "Every line remains reachable through the card's vertical scroll area.",
                "GUIDE  1 / 8",
            )
            app.processEvents()

            self.assertTrue(overlay.rect().contains(overlay.card.geometry()))
            self.assertGreater(
                overlay.card_scroll.verticalScrollBar().maximum(),
                0,
            )

            host.resize(760, 520)
            QTest.qWait(20)
            app.processEvents()
            self.assertTrue(overlay.rect().contains(overlay.card.geometry()))
            overlay.detach()
            host.close()
            app.main_window.close()

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
