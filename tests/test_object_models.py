from __future__ import annotations

import hashlib
import unittest

import cv2
import numpy as np

from photochoose.object_models import OBJECT_MODEL_PATH, GeneralSubjectAnalyzer


class GeneralSubjectModelTests(unittest.TestCase):
    def test_bundled_model_matches_reviewed_hash_and_loads(self) -> None:
        expected = "4b82da9944b88577175ee23a459dce2e26e6e4be573def65b1055dc2d9720186"

        self.assertTrue(OBJECT_MODEL_PATH.is_file())
        self.assertEqual(expected, hashlib.sha256(OBJECT_MODEL_PATH.read_bytes()).hexdigest())
        self.assertTrue(GeneralSubjectAnalyzer().available)

    def test_blank_image_has_no_detected_or_salient_objects(self) -> None:
        result = GeneralSubjectAnalyzer().analyze(
            np.zeros((240, 320, 3), dtype=np.uint8)
        )

        self.assertEqual(0, result.object_count)
        self.assertEqual((), result.objects)

    def test_detection_maps_to_named_normalized_primary_evidence(self) -> None:
        image = np.indices((416, 416)).sum(axis=0).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        detections = np.asarray(
            [
                [80.0, 60.0, 330.0, 350.0, 0.93, 0.0],
                [15.0, 20.0, 70.0, 90.0, 0.65, 56.0],
            ],
            dtype=np.float32,
        )
        analyzer = GeneralSubjectAnalyzer(inference=lambda _blob: ())

        objects = analyzer._object_evidence(image, detections, (0, 0, 416, 416))

        self.assertEqual("person", objects[0].label)
        self.assertEqual("primary", objects[0].importance)
        self.assertEqual("chair", objects[1].label)
        self.assertTrue(
            all(0.0 <= coordinate <= 1.0 for item in objects for coordinate in item.bounds)
        )

    def test_region_focus_prefers_sharp_detail(self) -> None:
        grid = np.indices((256, 256)).sum(axis=0)
        checkerboard = (((grid // 4) % 2) * 255).astype(np.uint8)
        sharp = cv2.cvtColor(checkerboard, cv2.COLOR_GRAY2BGR)
        blurred = cv2.GaussianBlur(sharp, (21, 21), 0)

        sharp_score = GeneralSubjectAnalyzer._region_focus(sharp, (0.0, 0.0, 1.0, 1.0))
        blurred_score = GeneralSubjectAnalyzer._region_focus(
            blurred, (0.0, 0.0, 1.0, 1.0)
        )

        self.assertIsNotNone(sharp_score)
        self.assertIsNotNone(blurred_score)
        self.assertGreater(sharp_score, blurred_score)

    def test_tiny_moderate_confidence_object_is_not_promoted_to_primary(self) -> None:
        image = np.zeros((416, 416, 3), dtype=np.uint8)
        detections = np.asarray(
            [[10.0, 10.0, 25.0, 25.0, 0.43, 2.0]], dtype=np.float32
        )
        analyzer = GeneralSubjectAnalyzer(inference=lambda _blob: ())

        objects = analyzer._object_evidence(image, detections, (0, 0, 416, 416))

        self.assertEqual(1, len(objects))
        self.assertNotEqual("primary", objects[0].importance)

    def test_weak_non_person_guess_is_suppressed(self) -> None:
        image = np.zeros((416, 416, 3), dtype=np.uint8)
        detections = np.asarray(
            [[40.0, 40.0, 300.0, 300.0, 0.39, 56.0]], dtype=np.float32
        )
        analyzer = GeneralSubjectAnalyzer(inference=lambda _blob: ())

        objects = analyzer._object_evidence(image, detections, (0, 0, 416, 416))

        self.assertEqual((), objects)


if __name__ == "__main__":
    unittest.main()
