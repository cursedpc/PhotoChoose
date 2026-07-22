from __future__ import annotations

import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

from photochoose.app import repair_macos_qt_plugin_visibility


@unittest.skipUnless(sys.platform == "darwin", "macOS-specific filesystem behavior")
class QtPluginVisibilityTests(unittest.TestCase):
    def test_hidden_plugin_files_are_made_discoverable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "plugins"
            platform_directory = root / "platforms"
            platform_directory.mkdir(parents=True)
            plugin = platform_directory / "libqcocoa.dylib"
            plugin.write_bytes(b"test placeholder")
            os.chflags(plugin, plugin.stat().st_flags | stat.UF_HIDDEN)
            self.assertTrue(plugin.stat().st_flags & stat.UF_HIDDEN)

            repaired = repair_macos_qt_plugin_visibility(root)

            self.assertEqual(1, repaired)
            self.assertFalse(plugin.stat().st_flags & stat.UF_HIDDEN)
