from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QColor, QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from photochoose.analysis_overlay import (  # noqa: E402
    EYE_COLORS,
    AnalysisOverlayVisibility,
    EvidenceDisplayMode,
    EvidenceItemMode,
    _selected_evidence,
    _visible_evidence,
    render_analysis_overlay,
)
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

    def test_obscured_eye_has_distinct_overlay_color(self) -> None:
        self.assertIn("obscured", EYE_COLORS)
        self.assertNotEqual(EYE_COLORS["closed"], EYE_COLORS["obscured"])
        self.assertNotEqual(EYE_COLORS["uncertain"], EYE_COLORS["obscured"])

    def test_crowded_overlay_hides_redundant_and_low_priority_evidence(self) -> None:
        faces = tuple(
            FaceEvidence(
                (0.05 * index, 0.1, 0.12, 0.2),
                0.95 - index * 0.01,
                100.0,
                True,
            )
            for index in range(8)
        )
        objects = (
            ObjectEvidence("person", (0.0, 0.0, 0.8, 0.9), 0.95, 100.0, "primary"),
            ObjectEvidence("chair", (0.8, 0.8, 0.1, 0.1), 0.80, 80.0, "background"),
            *tuple(
                ObjectEvidence(
                    "dog",
                    (0.1 * index, 0.6, 0.12, 0.2),
                    0.90 - index * 0.02,
                    90.0,
                    "primary",
                )
                for index in range(6)
            ),
        )
        photo = Photo(
            id=1,
            project_id=1,
            relative_path="crowd.jpg",
            absolute_path="/crowd.jpg",
            decision=Decision.KEEP,
            is_best=False,
            subject_regions=faces,
            subject_objects=objects,
        )

        visible_objects, visible_faces, crowded = _visible_evidence(photo)

        self.assertTrue(crowded)
        self.assertEqual(4, len(visible_objects))
        self.assertEqual(6, len(visible_faces))
        self.assertNotIn("person", {item.label for item in visible_objects})
        self.assertNotIn("background", {item.importance for item in visible_objects})

    def test_all_mode_reveals_evidence_hidden_by_common_mode(self) -> None:
        eye = EyeEvidence(
            state="open",
            aperture=0.2,
            openness_score=0.9,
            focus_score=120.0,
            landmarks=((0.7, 0.2), (0.72, 0.2), (0.71, 0.22)),
        )
        faces = tuple(
            FaceEvidence(
                (0.05 * index, 0.1, 0.12, 0.2),
                0.95 - index * 0.01,
                100.0,
                True,
                (eye,) if index == 7 else (),
            )
            for index in range(8)
        )
        objects = (
            ObjectEvidence("person", (0.0, 0.0, 0.8, 0.9), 0.95, 100.0, "primary"),
            ObjectEvidence("chair", (0.8, 0.8, 0.1, 0.1), 0.80, 80.0, "background"),
            ObjectEvidence("dog", (0.1, 0.6, 0.12, 0.2), 0.90, 90.0, "primary"),
        )
        photo = Photo(
            id=1,
            project_id=1,
            relative_path="crowd.jpg",
            absolute_path="/crowd.jpg",
            decision=Decision.KEEP,
            is_best=False,
            subject_regions=faces,
            subject_objects=objects,
        )

        visible_objects, visible_faces, visible_eyes, crowded = _selected_evidence(
            photo,
            AnalysisOverlayVisibility(
                faces=EvidenceDisplayMode.ALL,
                eyes=EvidenceDisplayMode.ALL,
                objects=EvidenceDisplayMode.ALL,
            ),
        )

        self.assertTrue(crowded)
        self.assertEqual(objects, visible_objects)
        self.assertEqual(faces, visible_faces)
        self.assertEqual(((faces[7], eye),), visible_eyes)

    def test_individual_overrides_can_show_hidden_items_and_hide_visible_items(self) -> None:
        first_eye = EyeEvidence(
            state="closed",
            aperture=0.1,
            openness_score=0.2,
            focus_score=80.0,
            landmarks=((0.2, 0.2), (0.22, 0.2), (0.21, 0.22)),
        )
        second_eye = EyeEvidence(
            state="open",
            aperture=0.2,
            openness_score=0.9,
            focus_score=120.0,
            landmarks=((0.7, 0.2), (0.72, 0.2), (0.71, 0.22)),
        )
        faces = (
            FaceEvidence((0.1, 0.1, 0.2, 0.3), 0.9, 100.0, True, (first_eye,)),
            FaceEvidence((0.6, 0.1, 0.2, 0.3), 0.8, 90.0, False, (second_eye,)),
        )
        objects = (
            ObjectEvidence("person", (0.1, 0.1, 0.2, 0.3), 0.9, 100.0, "primary"),
            ObjectEvidence("vase", (0.7, 0.5, 0.1, 0.3), 0.8, 90.0, "primary"),
        )
        photo = Photo(
            id=2,
            project_id=1,
            relative_path="overrides.jpg",
            absolute_path="/overrides.jpg",
            decision=Decision.KEEP,
            is_best=False,
            subject_regions=faces,
            subject_objects=objects,
        )

        visible_objects, visible_faces, visible_eyes, _crowded = _selected_evidence(
            photo,
            AnalysisOverlayVisibility(
                faces=EvidenceDisplayMode.OFF,
                eyes=EvidenceDisplayMode.OFF,
                objects=EvidenceDisplayMode.COMMON,
                face_overrides=((1, EvidenceItemMode.SHOW),),
                eye_overrides=((1, 0, EvidenceItemMode.SHOW),),
                object_overrides=((0, EvidenceItemMode.SHOW), (1, EvidenceItemMode.HIDE)),
            ),
        )

        self.assertEqual((objects[0],), visible_objects)
        self.assertEqual((faces[1],), visible_faces)
        self.assertEqual(((faces[1], second_eye),), visible_eyes)


if __name__ == "__main__":
    unittest.main()
