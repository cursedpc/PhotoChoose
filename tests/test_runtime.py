from __future__ import annotations

import sys
import unittest


class RuntimeTests(unittest.TestCase):
    def test_project_uses_the_pinned_python_minor_version(self) -> None:
        self.assertEqual((3, 11), sys.version_info[:2])


if __name__ == "__main__":
    unittest.main()
