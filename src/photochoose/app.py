from __future__ import annotations

import os
import stat
import sys
from pathlib import Path
from typing import Optional, Sequence

from PySide6.QtCore import QLibraryInfo, QStandardPaths
from PySide6.QtWidgets import QApplication

from photochoose.database import PhotoDatabase
from photochoose.main_window import MainWindow
from photochoose.ui_theme import apply_application_theme


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
    apply_application_theme(app)
    database = PhotoDatabase(database_path or default_database_path())
    window = MainWindow(database)
    window.show()
    app.main_window = window
    return app


def main() -> int:
    app = create_application()
    return app.exec()
