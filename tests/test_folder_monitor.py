from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtTest import QTest  # noqa: E402

from photochoose.app import create_application  # noqa: E402
from photochoose.folder_monitor import FolderMonitor, snapshot_folder  # noqa: E402
from photochoose.models import Project  # noqa: E402
from photochoose.models import PhotoView  # noqa: E402


class FolderMonitorTests(unittest.TestCase):
    def test_snapshot_tracks_supported_files_and_recursive_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nested = root / "Camera A"
            nested.mkdir()
            (nested / "notes.txt").write_text("ignored", encoding="utf-8")

            directories, empty_signature, available = snapshot_folder(root)
            self.assertTrue(available)
            self.assertEqual(
                {str(root), str(nested)},
                set(directories),
            )

            photo = nested / "IMG_0001.JPG"
            photo.write_bytes(b"first")
            _, added_signature, _ = snapshot_folder(root)
            self.assertNotEqual(empty_signature, added_signature)

            photo.write_bytes(b"replacement contents")
            _, changed_signature, _ = snapshot_folder(root)
            self.assertNotEqual(added_signature, changed_signature)

            photo.unlink()
            _, removed_signature, _ = snapshot_folder(root)
            self.assertEqual(empty_signature, removed_signature)

    def test_settled_supported_change_emits_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = create_application(
                ["photochoose-folder-monitor-test"], root / "state.sqlite3"
            )
            shoot = root / "Shoot"
            shoot.mkdir()
            project = Project(7, "Shoot", str(shoot))
            monitor = FolderMonitor(
                debounce_ms=15,
                poll_interval_ms=600000,
            )
            changes = []
            monitor.folderChanged.connect(changes.append)
            monitor.set_projects([project])
            self._wait_for(app, lambda: project.id in monitor._signatures)

            (shoot / "notes.txt").write_text("not a photo", encoding="utf-8")
            monitor._directory_changed(str(shoot))
            QTest.qWait(80)
            app.processEvents()
            self.assertEqual([], changes)

            (shoot / "new-photo.png").write_bytes(b"image placeholder")
            monitor._directory_changed(str(shoot))
            self._wait_for(app, lambda: changes == [project.id])
            QTest.qWait(80)
            app.processEvents()
            self.assertEqual([project.id], changes)

            monitor.close()
            app.main_window.close()

    def test_changed_project_while_snapshot_runs_is_rescheduled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = create_application(
                ["photochoose-folder-monitor-replace-test"], root / "state.sqlite3"
            )
            first = root / "First"
            second = root / "Second"
            first.mkdir()
            second.mkdir()
            monitor = FolderMonitor(poll_interval_ms=600000)
            project = Project(4, "First", str(first))
            monitor.set_projects([project])
            monitor.set_projects([Project(4, "Second", str(second))])

            self._wait_for(
                app,
                lambda: monitor._projects.get(4) == second
                and 4 in monitor._signatures
                and 4 not in monitor._in_flight,
            )
            self.assertIn(str(second), monitor.watcher.directories())
            self.assertNotIn(str(first), monitor.watcher.directories())

            monitor.close()
            app.main_window.close()

    def test_open_folder_change_triggers_the_existing_scan_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = create_application(
                ["photochoose-auto-rescan-test"], root / "state.sqlite3"
            )
            window = app.main_window
            shoot = root / "Shoot"
            shoot.mkdir()
            (shoot / "first.jpg").write_bytes(b"first image placeholder")
            project = window.database.add_project(shoot)
            window.database.scan_project(project.id)
            window.folder_monitor._debounce_timer.setInterval(15)
            window._load_projects(project.id)
            self._wait_for(
                app,
                lambda: project.id in window.folder_monitor._signatures,
            )
            analyses = []
            window._start_analysis = (
                lambda selected, pending_only=False: analyses.append(
                    (selected.id, pending_only)
                )
            )

            (shoot / "second.jpg").write_bytes(b"second image placeholder")
            watched_root = window.folder_monitor.watcher.directories()[0]
            window.folder_monitor._directory_changed(watched_root)
            self._wait_for(
                app,
                lambda: analyses == [(project.id, True)],
                timeout=5.0,
            )

            self.assertEqual(
                2,
                len(window.database.list_photos(project.id, PhotoView.ALL)),
            )
            self.assertEqual(2, window.photo_model.rowCount())
            window._automatic_scan_projects.discard(project.id)
            window.close()

    def _wait_for(self, app, predicate, timeout: float = 3.0) -> None:
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            app.processEvents()
            QTest.qWait(10)
        app.processEvents()
        self.assertTrue(predicate(), "timed out waiting for folder monitor")


if __name__ == "__main__":
    unittest.main()
