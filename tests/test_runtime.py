from __future__ import annotations

import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

from PySide6.QtCore import QSize

from photochoose import __version__
from photochoose.app import acquire_macos_instance_lock
from photochoose.fullscreen_viewer import FullscreenPhotoDialog


class RuntimeTests(unittest.TestCase):
    def test_instance_lock_rejects_duplicate_and_releases_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "photochoose.lock"
            first = acquire_macos_instance_lock(lock_path)
            self.assertIsNotNone(first)
            self.assertIsNone(acquire_macos_instance_lock(lock_path))
            first.unlock()
            replacement = acquire_macos_instance_lock(lock_path)
            self.assertIsNotNone(replacement)
            replacement.unlock()

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
        mac_spec = (project_root / "packaging/macos/photochoose.spec").read_text(
            encoding="utf-8"
        )
        release_workflow = (
            project_root / ".github/workflows/windows-installer.yml"
        ).read_text(encoding="utf-8")

        self.assertEqual(project_version, __version__)
        self.assertIn('#define AppVersion "{0}"'.format(project_version), installer)
        self.assertIn(
            "StringStruct('ProductVersion', '{0}')".format(project_version),
            version_info,
        )
        self.assertIn('target_arch = platform.machine()', mac_spec)
        self.assertIn('target_arch=target_arch', mac_spec)
        self.assertIn('("arm64", "x86_64")', mac_spec)
        self.assertNotIn("excluded_arm64_openvino_binaries", mac_spec)
        self.assertNotIn("libtbbbind_2_5.3.dylib", mac_spec)
        self.assertIn(
            "ai-edge-litert==1.0.1; sys_platform == 'darwin' and platform_machine == 'x86_64'",
            metadata["project"]["dependencies"],
        )
        self.assertIn(
            "ai-edge-litert==2.1.6; sys_platform != 'darwin' or platform_machine == 'arm64'",
            metadata["project"]["dependencies"],
        )
        self.assertIn(
            "openvino==2025.4.1; sys_platform != 'darwin' or platform_machine != 'arm64'",
            metadata["project"]["dependencies"],
        )
        self.assertIn(
            "openvino==2026.1.0; sys_platform == 'darwin' and platform_machine == 'arm64'",
            metadata["project"]["dependencies"],
        )
        self.assertIn("version=project_version", mac_spec)
        self.assertIn('"LSMultipleInstancesProhibited": True', mac_spec)
        self.assertIn("runner: macos-15-intel", release_workflow)
        self.assertIn("target_architecture: x86_64", release_workflow)
        self.assertIn("runner: macos-15", release_workflow)
        self.assertIn("target_architecture: arm64", release_workflow)
        self.assertIn("openvino_version: 2026.1.0", release_workflow)
        self.assertIn("litert_version: 2.1.6", release_workflow)
        self.assertIn("Confirm Windows dependency versions", release_workflow)
        self.assertIn("dmg_suffix: Apple-Silicon", release_workflow)
        self.assertIn(
            "python packaging/macos/normalize_openvino_arm64.py",
            release_workflow,
        )
        self.assertIn('lipo -verify_arch "${{ matrix.target_architecture }}"', release_workflow)
        self.assertIn("if: startsWith(github.ref, 'refs/tags/v')", release_workflow)
        self.assertNotIn(
            "startsWith(github.ref, 'refs/tags/v') || github.event_name",
            release_workflow,
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
