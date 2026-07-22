from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QColor, QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from photochoose.analysis_overlay import render_analysis_overlay  # noqa: E402
from photochoose.app import repair_macos_qt_plugin_visibility  # noqa: E402
from photochoose.models import (  # noqa: E402
    Decision,
    EyeEvidence,
    FaceEvidence,
    ObjectEvidence,
    Photo,
)


class AnalysisOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        repair_macos_qt_plugin_visibility()
        cls.application = QApplication.instance() or QApplication(["overlay-test"])

    def test_overlay_draws_stored_face_and_eye_evidence_without_mutating_source(self) -> None:
        source = QImage(400, 300, QImage.Format_RGB32)
        source.fill(QColor("white"))
        eye = EyeEvidence(
            state="open",
            aperture=0.24,
            openness_score=1.0,
            focus_score=250.0,
            landmarks=(
                (0.25, 0.30),
                (0.27, 0.28),
                (0.31, 0.28),
                (0.33, 0.30),
                (0.31, 0.32),
                (0.27, 0.32),
            ),
        )
        photo = Photo(
            id=1,
            project_id=1,
            relative_path="portrait.jpg",
            absolute_path="/portrait.jpg",
            decision=Decision.KEEP,
            is_best=False,
            subject_regions=(
                FaceEvidence((0.2, 0.2, 0.3, 0.5), 0.98, 200.0, True, (eye,)),
            ),
        )

        output = render_analysis_overlay(source, photo)

        self.assertEqual(QColor("white"), QColor(source.pixel(80, 60)))
        self.assertNotEqual(QColor("white"), QColor(output.pixel(80, 60)))
        self.assertNotEqual(QColor("white"), QColor(output.pixel(100, 90)))

    def test_overlay_draws_object_evidence_without_faces(self) -> None:
        source = QImage(400, 300, QImage.Format_RGB32)
        source.fill(QColor("white"))
        photo = Photo(
            id=1,
            project_id=1,
            relative_path="still-life.jpg",
            absolute_path="/still-life.jpg",
            decision=Decision.KEEP,
            is_best=False,
            subject_objects=(
                ObjectEvidence(
                    "vase", (0.25, 0.2, 0.4, 0.6), 0.92, 180.0, "primary"
                ),
            ),
        )

        output = render_analysis_overlay(source, photo)

        self.assertEqual(QColor("white"), QColor(source.pixel(100, 60)))
        self.assertNotEqual(QColor("white"), QColor(output.pixel(100, 60)))


if __name__ == "__main__":
    unittest.main()
