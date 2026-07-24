from __future__ import annotations

import unittest

import cv2
import numpy as np

from photochoose.inference_backend import InferenceBackendError, OpenVINOCPUBackend
from photochoose.object_models import (
    OBJECT_MODEL_PATH,
    OBJECT_OUTPUT_NAMES,
    GeneralSubjectAnalyzer,
)
from photochoose.vision_models import (
    FACE_LANDMARK_MODEL_PATH,
    FACE_MODEL_PATH,
    YUNET_OUTPUT_NAMES,
    FaceLandmarkEyeAnalyzer,
    ModelSubjectAnalyzer,
)


class _FailingModel:
    def infer(self, _tensor, _output_names):
        raise InferenceBackendError("simulated OpenVINO inference failure")


class _FailingBackend:
    name = "openvino-cpu"

    def __init__(self, fail_during_compile: bool = False) -> None:
        self.fail_during_compile = fail_during_compile

    def compile_model(self, _model_path, _input_shape=None):
        if self.fail_during_compile:
            raise InferenceBackendError("simulated OpenVINO compile failure")
        return _FailingModel()


class _UnavailableEyeAnalyzer:
    available = False
    backend_name = "none"


class OpenVINOCPUBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = OpenVINOCPUBackend()

    def test_backend_is_explicitly_cpu_only(self) -> None:
        self.assertEqual("openvino-cpu", self.backend.name)
        self.assertEqual("CPU", self.backend.device_name)
        self.assertIn("CPU", self.backend.available_devices)

    def test_nanodet_outputs_match_legacy_opencv(self) -> None:
        random = np.random.default_rng(2026)
        image = random.integers(0, 256, (416, 416, 3), dtype=np.uint8)
        blob = GeneralSubjectAnalyzer._preprocess(image)

        network = cv2.dnn.readNet(str(OBJECT_MODEL_PATH))
        network.setInput(blob)
        legacy_outputs = network.forward(OBJECT_OUTPUT_NAMES)
        model = self.backend.compile_model(OBJECT_MODEL_PATH)
        openvino_outputs = model.infer(blob, OBJECT_OUTPUT_NAMES)

        for legacy, accelerated in zip(legacy_outputs, openvino_outputs):
            self.assertEqual(legacy.shape, accelerated.shape)
            np.testing.assert_allclose(legacy, accelerated, rtol=1e-4, atol=1e-4)

    def test_yunet_outputs_match_legacy_opencv(self) -> None:
        random = np.random.default_rng(550)
        image = random.integers(0, 256, (320, 480, 3), dtype=np.uint8)
        blob = cv2.dnn.blobFromImage(image)

        network = cv2.dnn.readNet(str(FACE_MODEL_PATH))
        network.setInput(blob)
        legacy_outputs = network.forward(YUNET_OUTPUT_NAMES)
        model = self.backend.compile_model(FACE_MODEL_PATH, (1, 3, -1, -1))
        openvino_outputs = model.infer(blob, YUNET_OUTPUT_NAMES)

        for legacy, accelerated in zip(legacy_outputs, openvino_outputs):
            self.assertEqual(legacy.shape, accelerated.shape)
            np.testing.assert_allclose(legacy, accelerated, rtol=1e-4, atol=1e-4)

    def test_face_mesh_output_matches_litert(self) -> None:
        random = np.random.default_rng(478)
        tensor = random.random((1, 256, 256, 3), dtype=np.float32)

        from ai_edge_litert.interpreter import Interpreter

        interpreter = Interpreter(model_path=str(FACE_LANDMARK_MODEL_PATH))
        interpreter.allocate_tensors()
        input_index = interpreter.get_input_details()[0]["index"]
        output_index = next(
            output["index"]
            for output in interpreter.get_output_details()
            if output["shape"][-1] == 1434
        )
        interpreter.set_tensor(input_index, tensor)
        interpreter.invoke()
        legacy = interpreter.get_tensor(output_index)

        model = self.backend.compile_model(
            FACE_LANDMARK_MODEL_PATH, (1, 256, 256, 3)
        )
        accelerated = model.infer(tensor, ("Identity",))[0]

        self.assertEqual(legacy.shape, accelerated.shape)
        np.testing.assert_allclose(legacy, accelerated, rtol=2e-4, atol=2e-4)

    def test_model_loading_falls_back_to_legacy_runtimes(self) -> None:
        backend = _FailingBackend(fail_during_compile=True)

        objects = GeneralSubjectAnalyzer(backend=backend)
        subjects = ModelSubjectAnalyzer(backend=backend)

        self.assertEqual("opencv", objects.backend_name)
        self.assertEqual("opencv", subjects.face_backend_name)
        self.assertEqual("litert", subjects.eye_analyzer.backend_name)
        self.assertIn("simulated OpenVINO compile failure", objects.fallback_reason)
        self.assertIn("simulated OpenVINO compile failure", subjects.fallback_reason)

    def test_object_inference_failure_retries_with_opencv(self) -> None:
        analyzer = GeneralSubjectAnalyzer(backend=_FailingBackend())

        result = analyzer.analyze(np.zeros((240, 320, 3), dtype=np.uint8))

        self.assertEqual("opencv", analyzer.backend_name)
        self.assertEqual("nanodet-opencv", result.model_name)
        self.assertIn("simulated OpenVINO inference failure", analyzer.fallback_reason)

    def test_yunet_inference_failure_retries_with_opencv(self) -> None:
        analyzer = ModelSubjectAnalyzer(
            eye_analyzer=_UnavailableEyeAnalyzer(), backend=_FailingBackend()
        )

        result = analyzer.analyze(np.zeros((240, 320, 3), dtype=np.uint8))

        self.assertEqual("opencv", analyzer.face_backend_name)
        self.assertEqual("yunet", result.model_name)
        self.assertIn("simulated OpenVINO inference failure", analyzer.fallback_reason)

    def test_face_mesh_inference_failure_retries_with_litert(self) -> None:
        analyzer = FaceLandmarkEyeAnalyzer(backend=_FailingBackend())

        analyzer.analyze(np.zeros((128, 128, 3), dtype=np.uint8))

        self.assertEqual("litert", analyzer.backend_name)
        self.assertTrue(analyzer.available)
        self.assertIn("simulated OpenVINO inference failure", analyzer.fallback_reason)


if __name__ == "__main__":
    unittest.main()
