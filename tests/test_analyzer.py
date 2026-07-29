from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from photochoose.analysis_profiles import (
    ANALYSIS_VERSION,
    BALANCED_PROFILE,
    SAFE_PROFILE,
    STRICT_PROFILE,
)
from photochoose.analyzer import AnalysisResult, BaselineAnalyzer
from photochoose.database import PhotoDatabase
from photochoose.models import Decision, PhotoView


class BaselineAnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.shoot = self.root / "Shoot"
        self.shoot.mkdir()
        self.database = PhotoDatabase(self.root / "photochoose.sqlite3")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def _jpeg_bytes(image: np.ndarray) -> bytes:
        success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success:
            raise AssertionError("OpenCV could not encode test image")
        return encoded.tobytes()

    def test_analysis_separates_baseline_failures_and_preserves_manual_choices(self) -> None:
        grid = np.indices((512, 512)).sum(axis=0)
        checkerboard = (((grid // 12) % 2) * 255).astype(np.uint8)
        sharp = cv2.cvtColor(checkerboard, cv2.COLOR_GRAY2BGR)
        sharp_bytes = self._jpeg_bytes(sharp)
        (self.shoot / "sharp.jpg").write_bytes(sharp_bytes)
        (self.shoot / "sharp-copy.jpg").write_bytes(sharp_bytes)
        (self.shoot / "03-flat.jpg").write_bytes(
            self._jpeg_bytes(np.full((512, 512, 3), 128, dtype=np.uint8))
        )
        (self.shoot / "04-black.jpg").write_bytes(
            self._jpeg_bytes(np.zeros((512, 512, 3), dtype=np.uint8))
        )

        project = self.database.add_project(self.shoot)
        self.database.scan_project(project.id)
        photos = self.database.list_photos(project.id, PhotoView.ALL)
        black = next(photo for photo in photos if photo.filename == "04-black.jpg")
        self.database.set_decision([black.id], Decision.KEEP)
        progress = []

        totals = BaselineAnalyzer(self.database).analyze_project(
            project.id, lambda complete, total: progress.append((complete, total))
        )
        analyzed = {
            photo.filename: photo for photo in self.database.list_photos(project.id, PhotoView.ALL)
        }

        self.assertEqual(Decision.KEEP, analyzed["sharp.jpg"].decision)
        self.assertEqual(Decision.REJECT, analyzed["sharp-copy.jpg"].decision)
        self.assertEqual(
            analyzed["sharp.jpg"].id,
            analyzed["sharp-copy.jpg"].duplicate_of_id,
        )
        self.assertIn("Exact duplicate", analyzed["sharp-copy.jpg"].auto_reasons[0])
        self.assertTrue(analyzed["sharp.jpg"].is_best)
        self.assertEqual("automatic", analyzed["sharp.jpg"].best_source)
        self.assertEqual(
            analyzed["sharp.jpg"].similarity_group,
            analyzed["sharp-copy.jpg"].similarity_group,
        )
        self.assertEqual(Decision.REVIEW, analyzed["03-flat.jpg"].decision)
        self.assertEqual(Decision.KEEP, analyzed["04-black.jpg"].decision)
        self.assertEqual("manual", analyzed["04-black.jpg"].decision_source)
        self.assertEqual((4, 4), progress[-1])
        self.assertEqual(2, totals[Decision.KEEP.value])
        self.assertEqual(1, totals[Decision.REVIEW.value])
        self.assertEqual(1, totals[Decision.REJECT.value])

        (self.shoot / "03-flat.jpg").write_bytes(sharp_bytes)
        (self.shoot / "04-black.jpg").write_bytes(sharp_bytes)
        self.database.scan_project(project.id)
        rescanned = {
            photo.filename: photo for photo in self.database.list_photos(project.id, PhotoView.ALL)
        }
        self.assertEqual("pending", rescanned["03-flat.jpg"].analysis_status)
        self.assertEqual(Decision.UNREVIEWED, rescanned["03-flat.jpg"].decision)
        self.assertEqual("none", rescanned["03-flat.jpg"].decision_source)
        self.assertEqual("pending", rescanned["04-black.jpg"].analysis_status)
        self.assertEqual(Decision.KEEP, rescanned["04-black.jpg"].decision)
        self.assertEqual("manual", rescanned["04-black.jpg"].decision_source)

    def test_pending_only_analysis_reuses_unchanged_results(self) -> None:
        first_path = self.shoot / "first.jpg"
        second_path = self.shoot / "second.jpg"
        first_path.write_bytes(
            self._jpeg_bytes(np.full((120, 160, 3), 70, dtype=np.uint8))
        )
        second_path.write_bytes(
            self._jpeg_bytes(np.full((120, 160, 3), 140, dtype=np.uint8))
        )
        project = self.database.add_project(self.shoot)
        self.database.scan_project(project.id)
        analyzer = BaselineAnalyzer(self.database)
        analyzer.analyze_project(project.id)

        second_path.write_bytes(
            self._jpeg_bytes(np.full((130, 170, 3), 210, dtype=np.uint8))
        )
        self.database.scan_project(project.id)
        self.assertEqual(1, self.database.count_pending_analysis(project.id))
        progress = []

        with patch.object(
            analyzer,
            "analyze_photo",
            wraps=analyzer.analyze_photo,
        ) as analyze_photo:
            analyzer.analyze_project(
                project.id,
                lambda complete, total: progress.append((complete, total)),
                pending_only=True,
            )

        self.assertEqual(1, analyze_photo.call_count)
        self.assertEqual((1, 1), progress[-1])
        photos = self.database.list_photos(project.id, PhotoView.ALL)
        self.assertTrue(all(photo.analysis_status != "pending" for photo in photos))
        self.assertTrue(all(photo.content_hash for photo in photos))

    def test_near_duplicate_group_selects_the_sharper_frame_as_best(self) -> None:
        scene = np.zeros((600, 900), dtype=np.uint8)
        cv2.rectangle(scene, (100, 100), (800, 500), 180, -1)
        cv2.putText(
            scene,
            "PHOTO",
            (180, 350),
            cv2.FONT_HERSHEY_SIMPLEX,
            4,
            255,
            12,
            cv2.LINE_AA,
        )
        sharp = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
        soft = cv2.GaussianBlur(sharp, (5, 5), 0)
        (self.shoot / "frame-001.jpg").write_bytes(self._jpeg_bytes(sharp))
        (self.shoot / "frame-002.jpg").write_bytes(self._jpeg_bytes(soft))
        project = self.database.add_project(self.shoot)
        self.database.scan_project(project.id)

        BaselineAnalyzer(self.database).analyze_project(project.id)

        analyzed = {
            photo.filename: photo for photo in self.database.list_photos(project.id, PhotoView.ALL)
        }
        sharp_photo = analyzed["frame-001.jpg"]
        soft_photo = analyzed["frame-002.jpg"]
        self.assertIsNotNone(sharp_photo.similarity_group)
        self.assertEqual(sharp_photo.similarity_group, soft_photo.similarity_group)
        self.assertGreater(sharp_photo.best_score, soft_photo.best_score)
        self.assertTrue(sharp_photo.is_best)
        self.assertEqual("automatic", sharp_photo.best_source)
        self.assertFalse(soft_photo.is_best)

        self.database.toggle_best([sharp_photo.id])
        BaselineAnalyzer(self.database).analyze_project(project.id)
        reranked = {
            photo.filename: photo for photo in self.database.list_photos(project.id, PhotoView.ALL)
        }
        self.assertFalse(reranked["frame-001.jpg"].is_best)
        self.assertEqual("manual", reranked["frame-001.jpg"].best_source)
        self.assertTrue(reranked["frame-002.jpg"].is_best)
        self.assertEqual("automatic", reranked["frame-002.jpg"].best_source)

    def test_uncertain_eyes_and_soft_face_are_sent_to_review(self) -> None:
        decision, reasons = BaselineAnalyzer._classify(
            focus_score=120.0,
            brightness=0.5,
            clipped_dark=0.0,
            clipped_bright=0.0,
            face_count=1,
            eye_count=0,
            subject_focus=10.0,
        )

        self.assertEqual(Decision.REVIEW, decision)
        self.assertTrue(any("face appears soft" in reason for reason in reasons))
        self.assertTrue(any("Eyes are not confidently visible" in reason for reason in reasons))

    def test_profiles_apply_increasingly_conservative_thresholds(self) -> None:
        common = dict(
            focus_score=35.0,
            brightness=0.5,
            clipped_dark=0.0,
            clipped_bright=0.0,
            face_count=0,
            eye_count=0,
            subject_focus=None,
        )

        safe, _safe_reasons = BaselineAnalyzer._classify(
            profile=SAFE_PROFILE, **common
        )
        balanced, _balanced_reasons = BaselineAnalyzer._classify(
            profile=BALANCED_PROFILE, **common
        )
        strict, _strict_reasons = BaselineAnalyzer._classify(
            profile=STRICT_PROFILE, **common
        )

        self.assertEqual(Decision.KEEP, safe)
        self.assertEqual(Decision.REVIEW, balanced)
        self.assertEqual(Decision.REVIEW, strict)

    def test_unique_extreme_exposure_is_reviewed_under_every_profile(self) -> None:
        for profile in (SAFE_PROFILE, BALANCED_PROFILE, STRICT_PROFILE):
            decision, reasons = BaselineAnalyzer._classify(
                focus_score=0.0,
                brightness=0.0,
                clipped_dark=1.0,
                clipped_bright=0.0,
                face_count=0,
                eye_count=0,
                subject_focus=None,
                profile=profile,
            )
            self.assertEqual(Decision.REVIEW, decision)
            self.assertIn("Almost entirely black", reasons)

    def test_analyzer_persists_profile_and_analysis_version(self) -> None:
        grid = np.indices((256, 256)).sum(axis=0)
        image = cv2.cvtColor((((grid // 8) % 2) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        (self.shoot / "one.jpg").write_bytes(self._jpeg_bytes(image))
        project = self.database.add_project(self.shoot)
        self.database.scan_project(project.id)

        BaselineAnalyzer(self.database, STRICT_PROFILE).analyze_project(project.id)

        photo = self.database.list_photos(project.id, PhotoView.ALL)[0]
        self.assertEqual("strict", photo.analysis_profile)
        self.assertEqual(ANALYSIS_VERSION, photo.analysis_version)

    def test_confidently_closed_eye_is_reviewed_and_lowers_best_score(self) -> None:
        decision, reasons = BaselineAnalyzer._classify(
            focus_score=120.0,
            brightness=0.5,
            clipped_dark=0.0,
            clipped_bright=0.0,
            face_count=1,
            eye_count=1,
            subject_focus=100.0,
            closed_eye_count=1,
        )
        open_result = AnalysisResult(
            decision=Decision.KEEP,
            focus_score=120.0,
            brightness=0.5,
            clipped_dark=0.0,
            clipped_bright=0.0,
            content_hash=None,
            perceptual_hash=None,
            face_count=1,
            eye_count=2,
            subject_focus_score=100.0,
            reasons=("Passed",),
        )
        closed_result = replace(
            open_result,
            decision=Decision.REVIEW,
            eye_count=1,
            closed_eye_count=1,
        )

        self.assertEqual(Decision.REVIEW, decision)
        self.assertTrue(any("closed" in reason for reason in reasons))
        self.assertGreater(
            BaselineAnalyzer._best_score(Decision.KEEP, open_result),
            BaselineAnalyzer._best_score(Decision.REVIEW, closed_result),
        )

    def test_burst_rank_uses_primary_eye_state_and_weakest_eye_focus(self) -> None:
        strong = AnalysisResult(
            decision=Decision.KEEP,
            focus_score=180.0,
            brightness=0.5,
            clipped_dark=0.0,
            clipped_bright=0.0,
            content_hash=None,
            perceptual_hash=None,
            face_count=2,
            eye_count=2,
            closed_eye_count=2,
            subject_focus_score=160.0,
            eye_focus_score=300.0,
            important_face_count=1,
            important_open_eye_count=2,
            important_closed_eye_count=0,
            reasons=("Passed",),
        )
        soft_primary_eyes = replace(strong, eye_focus_score=8.0)
        closed_primary_eye = replace(
            strong,
            important_open_eye_count=1,
            important_closed_eye_count=1,
        )
        obscured_primary_eye = replace(
            strong,
            important_open_eye_count=1,
            important_obscured_eye_count=1,
        )

        strong_score = BaselineAnalyzer._best_score(Decision.KEEP, strong)
        self.assertGreater(
            strong_score,
            BaselineAnalyzer._best_score(Decision.KEEP, soft_primary_eyes),
        )
        self.assertGreater(
            strong_score,
            BaselineAnalyzer._best_score(Decision.KEEP, obscured_primary_eye),
        )
        self.assertGreater(
            BaselineAnalyzer._best_score(Decision.KEEP, obscured_primary_eye),
            BaselineAnalyzer._best_score(Decision.KEEP, closed_primary_eye),
        )

    def test_obscured_eye_is_reviewed_without_a_closed_eye_claim(self) -> None:
        decision, reasons = BaselineAnalyzer._classify(
            focus_score=120.0,
            brightness=0.5,
            clipped_dark=0.0,
            clipped_bright=0.0,
            face_count=1,
            eye_count=1,
            subject_focus=100.0,
            obscured_eye_count=1,
        )

        self.assertEqual(Decision.REVIEW, decision)
        self.assertTrue(any("obscured or unresolved" in reason for reason in reasons))
        self.assertFalse(any("closed" in reason for reason in reasons))

    def test_general_subject_softness_is_review_only_for_detector_evidence(self) -> None:
        common = dict(
            focus_score=120.0,
            brightness=0.5,
            clipped_dark=0.0,
            clipped_bright=0.0,
            face_count=0,
            eye_count=0,
            subject_focus=None,
            primary_subject_focus=8.0,
        )

        detected, detected_reasons = BaselineAnalyzer._classify(
            object_model="nanodet-opencv", **common
        )
        fallback, fallback_reasons = BaselineAnalyzer._classify(
            object_model="spectral-saliency", **common
        )

        self.assertEqual(Decision.REVIEW, detected)
        self.assertTrue(any("primary detected subject" in item for item in detected_reasons))
        self.assertEqual(Decision.KEEP, fallback)
        self.assertFalse(any("primary detected subject" in item for item in fallback_reasons))

    def test_general_subject_focus_influences_best_ranking(self) -> None:
        common = AnalysisResult(
            decision=Decision.KEEP,
            focus_score=120.0,
            brightness=0.5,
            clipped_dark=0.0,
            clipped_bright=0.0,
            content_hash=None,
            perceptual_hash=None,
            face_count=0,
            eye_count=0,
            subject_focus_score=None,
            reasons=("Passed",),
        )
        sharp_subject = replace(common, primary_subject_focus_score=400.0)
        soft_subject = replace(common, primary_subject_focus_score=8.0)

        self.assertGreater(
            BaselineAnalyzer._best_score(Decision.KEEP, sharp_subject),
            BaselineAnalyzer._best_score(Decision.KEEP, soft_subject),
        )

    def test_capture_time_groups_a_visually_changing_burst(self) -> None:
        (self.shoot / "burst-1.jpg").write_bytes(
            self._jpeg_bytes(np.full((300, 400, 3), 100, dtype=np.uint8))
        )
        (self.shoot / "burst-2.jpg").write_bytes(
            self._jpeg_bytes(np.full((300, 400, 3), 120, dtype=np.uint8))
        )
        project = self.database.add_project(self.shoot)
        self.database.scan_project(project.id)
        photos = self.database.list_photos(project.id, PhotoView.ALL)
        first = replace(
            photos[0],
            captured_at="2026-07-22T14:15:16.000000",
            camera_model="EOS R6",
        )
        second = replace(
            photos[1],
            captured_at="2026-07-22T14:15:17.250000",
            camera_model="EOS R6",
        )
        common = dict(
            decision=Decision.KEEP,
            focus_score=100.0,
            brightness=0.5,
            clipped_dark=0.0,
            clipped_bright=0.0,
            content_hash=None,
            face_count=0,
            eye_count=0,
            subject_focus_score=None,
            reasons=("Passed",),
        )
        first_result = AnalysisResult(perceptual_hash="0000000000000000", **common)
        second_result = AnalysisResult(perceptual_hash="0000000000000fff", **common)

        BaselineAnalyzer(self.database)._group_and_select_best(
            project.id,
            [(first, first_result), (second, second_result)],
        )

        grouped = self.database.list_photos(project.id, PhotoView.ALL)
        self.assertTrue(all(photo.similarity_group.startswith("B") for photo in grouped))
        self.assertEqual(grouped[0].similarity_group, grouped[1].similarity_group)
        self.assertIn("1.25s", grouped[0].group_reason)

    def test_project_best_limit_is_used_after_group_ranking(self) -> None:
        for filename in ("burst-a.jpg", "burst-b.jpg", "burst-c.jpg"):
            (self.shoot / filename).write_bytes(
                self._jpeg_bytes(np.full((200, 300, 3), 120, dtype=np.uint8))
            )
        project = self.database.add_project(self.shoot)
        self.database.scan_project(project.id)
        self.database.set_project_best_count(project.id, 2)
        photos = self.database.list_photos(project.id, PhotoView.ALL)
        hashes = (
            "0000000000000000",
            "00000000000003ff",
            "0000000003ff0000",
        )
        focus_scores = (500.0, 300.0, 200.0)
        analyzed = []
        for index, (photo, perceptual_hash, focus_score) in enumerate(
            zip(photos, hashes, focus_scores)
        ):
            self.database.apply_analysis(
                photo.id,
                Decision.KEEP,
                focus_score,
                0.5,
                0.0,
                0.0,
                photo.filename,
                ("Passed",),
                None,
            )
            self.database.apply_subject_analysis(
                photo.id,
                perceptual_hash,
                0,
                0,
                0,
                0,
                None,
                "none",
                None,
                None,
            )
            timed = replace(
                photo,
                decision=Decision.KEEP,
                captured_at="2026-07-22T14:15:1{0}.000000".format(index),
                camera_model="EOS R6",
            )
            analyzed.append(
                (
                    timed,
                    AnalysisResult(
                        decision=Decision.KEEP,
                        focus_score=focus_score,
                        brightness=0.5,
                        clipped_dark=0.0,
                        clipped_bright=0.0,
                        content_hash=photo.filename,
                        perceptual_hash=perceptual_hash,
                        face_count=0,
                        eye_count=0,
                        subject_focus_score=None,
                        reasons=("Passed",),
                    ),
                )
            )

        BaselineAnalyzer(self.database)._group_and_select_best(project.id, analyzed)

        ranked = self.database.list_photos(project.id, PhotoView.ALL)
        self.assertEqual(
            {"burst-a.jpg", "burst-b.jpg"},
            {photo.filename for photo in ranked if photo.is_best},
        )

    def test_much_weaker_burst_frame_is_rejected_but_manual_review_is_safe(self) -> None:
        for filename in ("strong.jpg", "weak.jpg", "manual.jpg"):
            (self.shoot / filename).write_bytes(
                self._jpeg_bytes(np.full((300, 400, 3), 110, dtype=np.uint8))
            )
        project = self.database.add_project(self.shoot)
        self.database.scan_project(project.id)
        scanned = self.database.list_photos(project.id, PhotoView.ALL)
        by_name = {photo.filename: photo for photo in scanned}
        self.database.apply_analysis(
            by_name["strong.jpg"].id,
            Decision.KEEP,
            500.0,
            0.5,
            0.0,
            0.0,
            None,
            ("Passed",),
            None,
        )
        for filename in ("weak.jpg", "manual.jpg"):
            self.database.apply_analysis(
                by_name[filename].id,
                Decision.REVIEW,
                5.0,
                0.5,
                0.0,
                0.0,
                None,
                ("Very low detail",),
                None,
            )
        self.database.set_decision([by_name["manual.jpg"].id], Decision.REVIEW)
        current = {
            photo.filename: photo
            for photo in self.database.list_photos(project.id, PhotoView.ALL)
        }
        timestamped = {
            filename: replace(
                photo,
                captured_at="2026-07-22T14:15:{0}.000000".format(second),
                camera_model="EOS R6",
            )
            for (filename, photo), second in zip(current.items(), (16, 17, 17))
        }
        strong_result = AnalysisResult(
            decision=Decision.KEEP,
            focus_score=500.0,
            brightness=0.5,
            clipped_dark=0.0,
            clipped_bright=0.0,
            content_hash=None,
            perceptual_hash="0000000000000000",
            face_count=0,
            eye_count=0,
            subject_focus_score=None,
            reasons=("Passed",),
        )
        weak_result = replace(
            strong_result,
            decision=Decision.REVIEW,
            focus_score=5.0,
            perceptual_hash="0000000000000fff",
            reasons=("Very low detail",),
        )
        manual_result = replace(weak_result, perceptual_hash="00000000000007ff")

        BaselineAnalyzer(self.database)._group_and_select_best(
            project.id,
            [
                (timestamped["strong.jpg"], strong_result),
                (timestamped["weak.jpg"], weak_result),
                (timestamped["manual.jpg"], manual_result),
            ],
        )

        grouped = {
            photo.filename: photo
            for photo in self.database.list_photos(project.id, PhotoView.ALL)
        }
        self.assertEqual(Decision.REJECT, grouped["weak.jpg"].decision)
        self.assertTrue(any("ranked" in reason for reason in grouped["weak.jpg"].auto_reasons))
        self.assertEqual(Decision.REVIEW, grouped["manual.jpg"].decision)
        self.assertEqual("manual", grouped["manual.jpg"].decision_source)


if __name__ == "__main__":
    unittest.main()
