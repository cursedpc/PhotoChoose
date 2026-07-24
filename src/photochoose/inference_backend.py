from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


class InferenceBackendError(RuntimeError):
    """A recoverable model-loading or inference failure."""


class OpenVINOCPUModel:
    """Compiled OpenVINO model restricted to the CPU device."""

    backend_name = "openvino-cpu"

    def __init__(self, compiled_model) -> None:
        self._compiled_model = compiled_model

    @property
    def output_names(self) -> Tuple[str, ...]:
        return tuple(output.any_name for output in self._compiled_model.outputs)

    def infer(
        self, tensor: np.ndarray, output_names: Sequence[str]
    ) -> Tuple[np.ndarray, ...]:
        try:
            results = self._compiled_model([np.ascontiguousarray(tensor)])
            return tuple(
                np.asarray(results[self._compiled_model.output(name)])
                for name in output_names
            )
        except Exception as exc:  # OpenVINO exposes backend-specific exception classes
            raise InferenceBackendError(str(exc)) from exc


class OpenVINOCPUBackend:
    """Load and compile local models with OpenVINO's CPU plugin only."""

    name = "openvino-cpu"
    device_name = "CPU"

    def __init__(self, core=None) -> None:
        try:
            if core is None:
                import openvino as ov

                core = ov.Core()
                self.version = ov.__version__
            else:
                self.version = "test"
            available_devices = tuple(core.available_devices)
        except Exception as exc:
            raise InferenceBackendError(str(exc)) from exc
        if self.device_name not in available_devices:
            raise InferenceBackendError("OpenVINO CPU device is unavailable")
        self._core = core
        self.available_devices = available_devices

    def compile_model(
        self,
        model_path: Path,
        input_shape: Optional[Sequence[int]] = None,
    ) -> OpenVINOCPUModel:
        try:
            model = self._core.read_model(str(model_path))
            if input_shape is not None:
                model.reshape({model.input(0): tuple(input_shape)})
            compiled_model = self._core.compile_model(model, self.device_name)
        except Exception as exc:
            raise InferenceBackendError(
                "Could not compile {0} for OpenVINO CPU: {1}".format(
                    model_path.name, exc
                )
            ) from exc
        return OpenVINOCPUModel(compiled_model)


_default_backend: Optional[OpenVINOCPUBackend] = None
_default_error: Optional[str] = None
_default_attempted = False


def default_openvino_cpu_backend() -> Optional[OpenVINOCPUBackend]:
    global _default_attempted, _default_backend, _default_error
    if not _default_attempted:
        _default_attempted = True
        try:
            _default_backend = OpenVINOCPUBackend()
        except InferenceBackendError as exc:
            _default_error = str(exc)
    return _default_backend


def openvino_backend_status() -> Dict[str, Optional[str]]:
    backend = default_openvino_cpu_backend()
    return {
        "name": backend.name if backend is not None else None,
        "device": backend.device_name if backend is not None else None,
        "version": backend.version if backend is not None else None,
        "error": _default_error,
    }
