from __future__ import annotations

import os
import stat
import sys
from pathlib import Path
from typing import Optional, Sequence

from PySide6.QtCore import QLibraryInfo, QLockFile, QStandardPaths, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from photochoose.database import PhotoDatabase
from photochoose.main_window import MainWindow
from photochoose.ui_theme import apply_application_theme

APP_ICON_PATH = Path(__file__).with_name("branding") / "photochoose-icon.png"
MACOS_INSTANCE_LOCK_NAME = "io.github.cursedpc.photochoose.lock"


def repair_macos_qt_plugin_visibility(plugin_root: Optional[Path] = None) -> int:
    """Clear UF_HIDDEN from Qt plugins before QApplication scans them.

    Some macOS development environments propagate the hidden flag from a
    dot-prefixed virtual environment to its contents. Qt excludes those files
    during plugin discovery, which makes the cocoa platform and JPEG handlers
    appear to be missing even though their dylibs are installed.
    """

    if sys.platform != "darwin":
        return 0

    root = plugin_root or Path(QLibraryInfo.path(QLibraryInfo.PluginsPath))
    if not root.is_dir():
        return 0

    repaired = 0
    for path in [root, *root.rglob("*")]:
        try:
            flags = path.stat().st_flags
            if flags & stat.UF_HIDDEN:
                os.chflags(path, flags & ~stat.UF_HIDDEN)
                repaired += 1
        except (AttributeError, OSError):
            # Installed application bundles may be read-only. In that case Qt
            # reports its normal plugin error and the bundle can be repaired.
            continue
    return repaired


def default_database_path() -> Path:
    data_directory = Path(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation))
    return data_directory / "photochoose.sqlite3"


def acquire_macos_instance_lock(lock_path: Optional[Path] = None) -> Optional[QLockFile]:
    """Prevent a frozen macOS bundle from creating duplicate main windows."""

    path = lock_path
    if path is None:
        temporary_directory = Path(
            QStandardPaths.writableLocation(QStandardPaths.TempLocation)
        )
        path = temporary_directory / MACOS_INSTANCE_LOCK_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = QLockFile(str(path))
    lock.setStaleLockTime(0)
    if not lock.tryLock(0):
        return None
    return lock


def create_application(
    arguments: Optional[Sequence[str]] = None,
    database_path: Optional[Path] = None,
) -> QApplication:
    repair_macos_qt_plugin_visibility()
    app = QApplication.instance()
    if app is None:
        app = QApplication(list(arguments) if arguments is not None else sys.argv)
    app.setOrganizationName("PhotoChoose")
    app.setApplicationName("PhotoChoose")
    app.setWindowIcon(QIcon(str(APP_ICON_PATH)))
    apply_application_theme(app)
    database = PhotoDatabase(database_path or default_database_path())
    window = MainWindow(database)
    window.show()
    app.main_window = window
    return app


def run_packaging_smoke_test() -> None:
    """Require the frozen build's bundled models to run on OpenVINO CPU."""

    import numpy as np

    from photochoose.inference_backend import openvino_backend_status
    from photochoose.object_models import GeneralSubjectAnalyzer
    from photochoose.vision_models import ModelSubjectAnalyzer

    status = openvino_backend_status()
    if status["device"] != "CPU":
        raise RuntimeError("OpenVINO CPU is unavailable: {0}".format(status["error"]))

    image = np.zeros((320, 416, 3), dtype=np.uint8)
    subject_analyzer = ModelSubjectAnalyzer()
    subject_result = subject_analyzer.analyze(image)
    if (
        subject_result is None
        or subject_analyzer.face_backend_name != "openvino-cpu"
        or subject_analyzer.eye_analyzer.backend_name != "openvino-cpu"
    ):
        raise RuntimeError("Bundled YuNet or face-mesh OpenVINO inference failed")
    subject_analyzer.eye_analyzer.analyze(image)

    object_analyzer = GeneralSubjectAnalyzer()
    object_result = object_analyzer.analyze(image)
    if (
        object_analyzer.backend_name != "openvino-cpu"
        or not object_result.model_name.startswith("nanodet-openvino-cpu")
    ):
        raise RuntimeError("Bundled NanoDet OpenVINO inference failed")


def _write_smoke_failure() -> None:
    smoke_log = os.environ.get("PHOTOCHOOSE_SMOKE_LOG")
    if not smoke_log:
        return
    import traceback

    try:
        Path(smoke_log).write_text(traceback.format_exc(), encoding="utf-8")
    except OSError:
        pass


def main() -> int:
    smoke_mode = "--smoke-test" in sys.argv
    instance_lock = None
    if sys.platform == "darwin" and getattr(sys, "frozen", False) and not smoke_mode:
        instance_lock = acquire_macos_instance_lock()
        if instance_lock is None:
            return 0
    if smoke_mode:
        try:
            temporary_directory = Path(
                QStandardPaths.writableLocation(QStandardPaths.TempLocation)
            )
            smoke_database = temporary_directory / "photochoose-packaging-smoke.sqlite3"
            app = create_application(database_path=smoke_database)
            run_packaging_smoke_test()
        except Exception:
            _write_smoke_failure()
            return 2
        QTimer.singleShot(0, app.quit)
    else:
        app = create_application()
    if instance_lock is not None:
        app.photochoose_instance_lock = instance_lock
    return app.exec()
