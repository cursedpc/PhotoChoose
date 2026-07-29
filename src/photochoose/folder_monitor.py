from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

from PySide6.QtCore import (
    QFileSystemWatcher,
    QObject,
    QRunnable,
    QThreadPool,
    QTimer,
    Signal,
    Slot,
)

from photochoose.assets import SUPPORTED_ASSET_EXTENSIONS
from photochoose.models import Project

FolderSnapshot = Tuple[Tuple[str, ...], Optional[str], bool]


def _path_key(path: str) -> str:
    return os.path.normcase(os.path.normpath(path))


def snapshot_folder(root: Path) -> FolderSnapshot:
    """Describe supported files and discover watchable directories under root."""
    root_path = os.path.abspath(os.fspath(root))
    if not os.path.isdir(root_path):
        return (), None, False

    digest = hashlib.blake2b(digest_size=16)
    directories = []
    try:
        walker = os.walk(root_path, followlinks=False)
        for directory, child_directories, filenames in walker:
            child_directories[:] = sorted(
                name
                for name in child_directories
                if not os.path.islink(os.path.join(directory, name))
            )
            directories.append(os.path.normpath(directory))
            for filename in sorted(filenames):
                if Path(filename).suffix.casefold() not in SUPPORTED_ASSET_EXTENSIONS:
                    continue
                absolute_path = os.path.join(directory, filename)
                if os.path.islink(absolute_path):
                    continue
                try:
                    stat = os.stat(absolute_path)
                except OSError:
                    continue
                relative_path = os.path.relpath(absolute_path, root_path).replace(
                    os.sep, "/"
                )
                digest.update(relative_path.encode("utf-8", "surrogatepass"))
                digest.update(b"\0")
                digest.update(str(stat.st_size).encode("ascii"))
                digest.update(b"\0")
                digest.update(str(stat.st_mtime_ns).encode("ascii"))
                digest.update(b"\0")
    except OSError:
        return (), None, False
    return tuple(directories), digest.hexdigest(), True


class _SnapshotSignals(QObject):
    finished = Signal(int, int, object)


class _SnapshotTask(QRunnable):
    def __init__(self, project_id: int, token: int, root: Path) -> None:
        super().__init__()
        self.project_id = project_id
        self.token = token
        self.root = root
        self.signals = _SnapshotSignals()

    def run(self) -> None:
        self.signals.finished.emit(
            self.project_id,
            self.token,
            snapshot_folder(self.root),
        )


class FolderMonitor(QObject):
    """Watch open photo folders and report settled supported-file changes."""

    folderChanged = Signal(int)

    def __init__(
        self,
        parent: Optional[QObject] = None,
        debounce_ms: int = 1500,
        poll_interval_ms: int = 60000,
    ) -> None:
        super().__init__(parent)
        self._projects: dict[int, Path] = {}
        self._signatures: dict[int, str] = {}
        self._project_directories: dict[int, set[str]] = {}
        self._path_projects: dict[str, set[int]] = {}
        self._tokens: dict[int, int] = {}
        self._in_flight: set[int] = set()
        self._rerun: set[int] = set()
        self._dirty: set[int] = set()
        self._closed = False
        self._thread_pool = QThreadPool.globalInstance()

        self.watcher = QFileSystemWatcher(self)
        self.watcher.directoryChanged.connect(self._directory_changed)

        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(debounce_ms)
        self._debounce_timer.timeout.connect(self._snapshot_dirty_projects)

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(poll_interval_ms)
        self._poll_timer.timeout.connect(self._poll_projects)
        if os.environ.get("QT_QPA_PLATFORM", "").casefold() != "offscreen":
            self._poll_timer.start()

    def set_projects(self, projects: Iterable[Project]) -> None:
        next_projects = {
            project.id: Path(project.root_path).expanduser().absolute()
            for project in projects
        }
        removed_ids = set(self._projects) - set(next_projects)
        changed_ids = {
            project_id
            for project_id, root in next_projects.items()
            if self._projects.get(project_id) != root
        }
        for project_id in removed_ids | changed_ids:
            self._tokens[project_id] = self._tokens.get(project_id, 0) + 1
            self._signatures.pop(project_id, None)
            self._project_directories.pop(project_id, None)
            self._dirty.discard(project_id)
            self._rerun.discard(project_id)
        self._projects = next_projects

        for project_id in changed_ids:
            root = self._projects[project_id]
            self._project_directories[project_id] = (
                {os.path.normpath(os.fspath(root))} if root.is_dir() else set()
            )
        self._rebuild_watches()
        for project_id in changed_ids:
            self._request_snapshot(project_id)

    def refresh_project(self, project_id: int) -> None:
        if project_id in self._projects:
            self._request_snapshot(project_id)

    @Slot(str)
    def _directory_changed(self, path: str) -> None:
        project_ids = self._path_projects.get(_path_key(path), set())
        if not project_ids:
            return
        self._dirty.update(project_ids)
        self._debounce_timer.start()

    @Slot()
    def _snapshot_dirty_projects(self) -> None:
        dirty = tuple(self._dirty)
        self._dirty.clear()
        for project_id in dirty:
            self._request_snapshot(project_id)

    @Slot()
    def _poll_projects(self) -> None:
        for project_id in tuple(self._projects):
            self._request_snapshot(project_id)

    def _request_snapshot(self, project_id: int) -> None:
        if self._closed or project_id not in self._projects:
            return
        if project_id in self._in_flight:
            self._rerun.add(project_id)
            return
        token = self._tokens.get(project_id, 0) + 1
        self._tokens[project_id] = token
        self._in_flight.add(project_id)
        task = _SnapshotTask(project_id, token, self._projects[project_id])
        task.signals.finished.connect(self._snapshot_finished)
        self._thread_pool.start(task)

    @Slot(int, int, object)
    def _snapshot_finished(
        self,
        project_id: int,
        token: int,
        snapshot: FolderSnapshot,
    ) -> None:
        if token != self._tokens.get(project_id) or project_id not in self._projects:
            self._in_flight.discard(project_id)
            if project_id in self._rerun and project_id in self._projects:
                self._rerun.discard(project_id)
                self._request_snapshot(project_id)
            return
        self._in_flight.discard(project_id)
        directories, signature, available = snapshot
        if available and signature is not None:
            previous = self._signatures.get(project_id)
            self._signatures[project_id] = signature
            self._project_directories[project_id] = set(directories)
            self._rebuild_watches()
            if previous is not None and signature != previous:
                self.folderChanged.emit(project_id)
        if project_id in self._rerun:
            self._rerun.discard(project_id)
            self._request_snapshot(project_id)

    def _rebuild_watches(self) -> None:
        desired: dict[str, set[int]] = {}
        for project_id, directories in self._project_directories.items():
            for directory in directories:
                desired.setdefault(directory, set()).add(project_id)

        current = set(self.watcher.directories())
        wanted = set(desired)
        current_keys = {_path_key(path) for path in current}
        wanted_keys = {_path_key(path) for path in wanted}
        if current_keys != wanted_keys:
            # Some QFileSystemWatcher backends refuse to remove an active path.
            # Replacing the small watcher object releases obsolete native handles
            # consistently on macOS and Windows.
            previous_watcher = self.watcher
            previous_watcher.directoryChanged.disconnect(self._directory_changed)
            previous_watcher.deleteLater()
            self.watcher = QFileSystemWatcher(self)
            self.watcher.directoryChanged.connect(self._directory_changed)
            if wanted:
                self.watcher.addPaths(sorted(wanted))
        desired_by_key = {
            _path_key(path): project_ids for path, project_ids in desired.items()
        }
        self._path_projects = {
            key: desired_by_key[key]
            for path in self.watcher.directories()
            if (key := _path_key(path)) in desired_by_key
        }

    def close(self) -> None:
        self._closed = True
        self._debounce_timer.stop()
        self._poll_timer.stop()
        watched = self.watcher.directories()
        if watched:
            self.watcher.removePaths(watched)
        self._path_projects.clear()
        self._projects.clear()
        self._dirty.clear()
        self._rerun.clear()
        for project_id in tuple(self._tokens):
            self._tokens[project_id] += 1
