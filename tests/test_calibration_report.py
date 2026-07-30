from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from photochoose.analysis_profiles import ANALYSIS_VERSION
from photochoose.calibration_report import (
    MIN_SUGGESTION_SAMPLES,
    build_calibration_report,
)
from photochoose.database import PhotoDatabase
from photochoose.models import Decision, PhotoView


class CalibrationReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.shoot = self.root / "Shoot"
        self.shoot.mkdir()
        self.database = PhotoDatabase(self.root / "photochoose.sqlite3")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_empty_report_requires_real_target_machine_data(self) -> None:
        project = self.database.add_project(self.shoot)

        report = build_calibration_report(self.database, project.id)

        self.assertEqual(0, report.total_actions)
        self.assertFalse(report.has_enough_decision_data)
        self.assertIn("early data", report.to_text())
        self.assertIn("target machine", report.to_text())
        self.assertIn("never changes profile thresholds", report.to_text())

    def test_report_groups_disagreements_and_waits_for_minimum_samples(self) -> None:
        for index in range(MIN_SUGGESTION_SAMPLES):
            (self.shoot / "soft-{0}.jpg".format(index)).write_bytes(b"placeholder")
        project = self.database.add_project(self.shoot)
        self.database.scan_project(project.id)
        photos = self.database.list_photos(project.id, PhotoView.ALL)

        for photo in photos:
            self.database.apply_analysis(
                photo.id,
                Decision.REVIEW,
                18.0,
                0.5,
                0.0,
                0.0,
                "hash-{0}".format(photo.id),
                ("Possibly soft or out of focus",),
                None,
            )
            self.database.set_decision([photo.id], Decision.KEEP)

        self.database.toggle_best([photos[0].id])
        report = build_calibration_report(self.database, project.id)

        self.assertEqual(MIN_SUGGESTION_SAMPLES + 1, report.total_actions)
        self.assertEqual(MIN_SUGGESTION_SAMPLES, report.decision_disagreements)
        self.assertEqual(0, report.best_corrections)
        self.assertEqual((("Review → Keep", MIN_SUGGESTION_SAMPLES),), report.directions)
        self.assertEqual("safe", report.profiles[0].profile)
        self.assertEqual(MIN_SUGGESTION_SAMPLES, report.profiles[0].rescued_to_keep)
        self.assertEqual("focus", report.signals[0].signal)
        self.assertEqual(MIN_SUGGESTION_SAMPLES, report.signals[0].disagreements)
        self.assertEqual(1, len(report.suggestions))
        self.assertIn("too cautious", report.suggestions[0].message)
        self.assertIn("target corpus", report.suggestions[0].message)

        text = report.to_text()
        self.assertIn("Safe / {0}".format(ANALYSIS_VERSION), text)
        self.assertIn("global focus", text)
        self.assertIn("soft-", text)


if __name__ == "__main__":
    unittest.main()
