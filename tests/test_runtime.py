from __future__ import annotations

import sys
import unittest

from PySide6.QtCore import QSize

from photochoose.fullscreen_viewer import FullscreenPhotoDialog


class RuntimeTests(unittest.TestCase):
    def test_project_uses_the_pinned_python_minor_version(self) -> None:
        self.assertEqual((3, 11), sys.version_info[:2])

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
