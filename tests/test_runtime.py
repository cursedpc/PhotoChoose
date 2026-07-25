from __future__ import annotations

import sys
import tomllib
import unittest
from pathlib import Path

from PySide6.QtCore import QSize

from photochoose import __version__
from photochoose.fullscreen_viewer import FullscreenPhotoDialog


class RuntimeTests(unittest.TestCase):
    def test_project_uses_the_pinned_python_minor_version(self) -> None:
        self.assertEqual((3, 11), sys.version_info[:2])

    def test_release_versions_stay_aligned(self) -> None:
        project_root = Path(__file__).parents[1]
        metadata = tomllib.loads(
            (project_root / "pyproject.toml").read_text(encoding="utf-8")
        )
        project_version = metadata["project"]["version"]
        installer = (project_root / "packaging/windows/photochoose.iss").read_text(
            encoding="utf-8"
        )
        version_info = (
            project_root / "packaging/windows/version_info.txt"
        ).read_text(encoding="utf-8")

        self.assertEqual(project_version, __version__)
        self.assertIn('#define AppVersion "{0}"'.format(project_version), installer)
        self.assertIn(
            "StringStruct('ProductVersion', '{0}')".format(project_version),
            version_info,
        )

    def test_windows_installer_separates_program_and_user_data(self) -> None:
        project_root = Path(__file__).parents[1]
        installer = (project_root / "packaging/windows/photochoose.iss").read_text(
            encoding="utf-8"
        )

        self.assertIn(r"DefaultDirName={autopf}\PhotoChoose", installer)
        self.assertIn("PrivilegesRequired=admin", installer)
        self.assertIn("UsePreviousAppDir=yes", installer)
        self.assertIn("LegacyUninstallKey", installer)
        self.assertIn("HasLegacyPerUserInstall", installer)
        self.assertNotIn(r"DefaultDirName={localappdata}", installer)

    def test_fullscreen_targets_physical_display_pixels(self) -> None:
        self.assertEqual(
            QSize(1920, 1080),
            FullscreenPhotoDialog._physical_size(QSize(1920, 1080), 1.0),
        )
        self.assertEqual(
            QSize(2880, 1800),
            FullscreenPhotoDialog._physical_size(QSize(1440, 900), 2.0),
        )
        self.assertEqual(
            QSize(1600, 1066),
            FullscreenPhotoDialog._decode_size(
                QSize(6000, 4000), QSize(1600, 1200), False
            ),
        )
        self.assertEqual(
            QSize(1066, 1600),
            FullscreenPhotoDialog._decode_size(
                QSize(4000, 6000), QSize(1600, 1200), True
            ),
        )


if __name__ == "__main__":
    unittest.main()
