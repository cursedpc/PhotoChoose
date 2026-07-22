from __future__ import annotations

import hashlib
import unittest

import cv2
import numpy as np

from photochoose.vision_models import (
    FACE_LANDMARK_MODEL_PATH,
    FACE_MODEL_PATH,
    EyeLandmarkResult,
    FaceLandmarkEyeAnalyzer,
    ModelSubjectAnalyzer,
)


class _FakeLandmarkInference:
    def __init__(self, landmarks: np.ndarray) -> None:
        self.landmarks = landmarks
        self.input = None

    def __call__(self, tensor: np.ndarray) -> np.ndarray:
        self.input = tensor
        return self.landmarks.reshape(1, 1, 1, -1)


class _FakeFaceDetector:
    def __init__(self, faces: np.ndarray) -> None:
        self.faces = faces

    def setInputSize(self, _size) -> None:
        pass

    def detect(self, _image):
        return 1, self.faces


class _FakeEyeAnalyzer:
    available = True

    def analyze(self, _face_image):
        right = ((70.0, 100.0), (78.0, 94.0), (90.0, 94.0),
                 (98.0, 100.0), (90.0, 106.0), (78.0, 106.0))
        left = tuple((x + 80.0, y) for x, y in right)
        return (
            EyeLandmarkResult("open", 0.25, 1.0, right),
            EyeLandmarkResult("open", 0.25, 1.0, left),
        )


class VisionModelTests(unittest.TestCase):
    def test_bundled_model_files_match_reviewed_hashes(self) -> None:
        expected = {
            FACE_MODEL_PATH: "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
            FACE_LANDMARK_MODEL_PATH: (
                "c7d54204ce0448474c7f3fa9af494787c0965cbdd6f20fc72867e43046bd43d5"
            ),
        }
        for path, digest in expected.items():
            self.assertTrue(path.is_file(), path)
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_face_mesh_uses_rgb_nhwc_tensor_and_open_landmarks(self) -> None:
        landmarks = self._landmarks_with_eye_height(2.0)
        inference = _FakeLandmarkInference(landmarks)
        analyzer = FaceLandmarkEyeAnalyzer(inference=inference)

        eyes = analyzer.analyze(
            np.full((80, 100, 3), (10, 20, 30), dtype=np.uint8)
        )

        self.assertEqual(["open", "open"], [eye.state for eye in eyes])
        self.assertEqual((1, 256, 256, 3), inference.input.shape)
        self.assertAlmostEqual(30.0 / 255.0, float(inference.input[0, 0, 0, 0]))
        self.assertAlmostEqual(10.0 / 255.0, float(inference.input[0, 0, 0, 2]))

    def test_eye_aperture_has_conservative_closed_and_uncertain_bands(self) -> None:
        closed = FaceLandmarkEyeAnalyzer(
            inference=_FakeLandmarkInference(self._landmarks_with_eye_height(0.6))
        ).analyze(np.zeros((64, 64, 3), dtype=np.uint8))
        uncertain = FaceLandmarkEyeAnalyzer(
            inference=_FakeLandmarkInference(self._landmarks_with_eye_height(0.8))
        ).analyze(np.zeros((64, 64, 3), dtype=np.uint8))

        self.assertEqual(["closed", "closed"], [eye.state for eye in closed])
        self.assertEqual(["uncertain", "uncertain"], [eye.state for eye in uncertain])

    def test_real_bundled_models_load_and_blank_image_has_no_faces(self) -> None:
        analyzer = ModelSubjectAnalyzer()
        self.assertTrue(analyzer.available, analyzer.error)
        self.assertTrue(analyzer.eye_analyzer.available, analyzer.eye_analyzer.error)

        result = analyzer.analyze(np.zeros((240, 320, 3), dtype=np.uint8))

        self.assertIsNotNone(result)
        self.assertEqual(0, result.face_count)
        self.assertEqual("yunet+face-mesh-litert", result.model_name)

    def test_face_crop_stays_inside_image_bounds(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = ModelSubjectAnalyzer._face_crop(image, 0, 0, 40, 50)

        self.assertIsNotNone(crop)
        self.assertLessEqual(crop.shape[0], 68)
        self.assertLessEqual(crop.shape[1], 68)

    def test_subject_evidence_tracks_primary_faces_and_eye_focus(self) -> None:
        image = np.indices((300, 400)).sum(axis=0).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        faces = np.zeros((2, 15), dtype=np.float32)
        faces[0, :4] = (80, 70, 120, 120)
        faces[0, 14] = 0.98
        faces[1, :4] = (300, 30, 30, 30)
        faces[1, 14] = 0.90
        analyzer = ModelSubjectAnalyzer(eye_analyzer=_FakeEyeAnalyzer())
        analyzer.face_detector = _FakeFaceDetector(faces)

        result = analyzer.analyze(image)

        self.assertIsNotNone(result)
        self.assertEqual(2, result.face_count)
        self.assertEqual(1, result.important_face_count)
        self.assertTrue(result.regions[0].important)
        self.assertFalse(result.regions[1].important)
        self.assertEqual(2, result.important_open_eye_count)
        self.assertIsNotNone(result.eye_focus_score)
        self.assertEqual(6, len(result.regions[0].eyes[0].landmarks))
        self.assertTrue(
            all(
                0.0 <= coordinate <= 1.0
                for point in result.regions[0].eyes[0].landmarks
                for coordinate in point
            )
        )

    def test_face_box_touching_edge_is_clipped_before_normalizing(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        faces = np.zeros((1, 15), dtype=np.float32)
        faces[0, :4] = (-20, -10, 100, 80)
        faces[0, 14] = 0.95
        analyzer = ModelSubjectAnalyzer(eye_analyzer=_FakeEyeAnalyzer())
        analyzer.face_detector = _FakeFaceDetector(faces)

        result = analyzer.analyze(image)

        self.assertEqual((0.0, 0.0, 0.8, 0.7), result.regions[0].bounds)

    @staticmethod
    def _landmarks_with_eye_height(height: float) -> np.ndarray:
        landmarks = np.zeros((478, 3), dtype=np.float32)
        eye = np.array(
            [
                [0.0, 0.0],
                [2.0, -height],
                [8.0, -height],
                [10.0, 0.0],
                [8.0, height],
                [2.0, height],
            ],
            dtype=np.float32,
        )
        for indices in ((33, 160, 158, 133, 153, 144), (362, 385, 387, 263, 373, 380)):
            landmarks[np.asarray(indices), :2] = eye
        return landmarks


if __name__ == "__main__":
    unittest.main()
