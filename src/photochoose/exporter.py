from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

from photochoose.database import PhotoDatabase
from photochoose.models import PhotoView


@dataclass(frozen=True)
class ExportResult:
    destination: str
    copied: int
    skipped: int
    failed: Tuple[str, ...]


class ProjectExporter:
    """Copy folder decisions without modifying source photos or existing exports."""

    def __init__(self, database: PhotoDatabase) -> None:
        self.database = database

    def export_keep(
        self,
        project_id: int,
        destination: Path,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> ExportResult:
        return self._export_view(project_id, PhotoView.KEEP, destination, progress)

    def export_best(
        self,
        project_id: int,
        destination: Path,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> ExportResult:
        return self._export_view(project_id, PhotoView.BEST, destination, progress)

    def _export_view(
        self,
        project_id: int,
        view: PhotoView,
        destination: Path,
        progress: Optional[Callable[[int, int], None]],
    ) -> ExportResult:
        photos = self.database.list_photos(project_id, view)
        entries = [
            (asset_path, Path(asset_path))
            for photo in photos
            for asset_path in (photo.asset_paths or (photo.relative_path,))
        ]
        return self._export_entries(project_id, destination, entries, progress)

    def export_selected(
        self,
        project_id: int,
        photo_ids: Sequence[int],
        destination: Path,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> ExportResult:
        """Copy exactly the selected photographs, independent of decisions/groups."""

        selected_ids = tuple(dict.fromkeys(int(photo_id) for photo_id in photo_ids))
        if not selected_ids:
            raise ValueError("Select at least one photograph to export")
        selected_id_set = set(selected_ids)
        photos = [
            photo
            for photo in self.database.list_photos(project_id, PhotoView.ALL)
            if photo.id in selected_id_set
        ]
        found_ids = {photo.id for photo in photos}
        if found_ids != selected_id_set:
            raise ValueError("One or more selected photographs are no longer in this folder")
        entries = [
            (asset_path, Path(asset_path))
            for photo in photos
            for asset_path in (photo.asset_paths or (photo.relative_path,))
        ]
        return self._export_entries(project_id, destination, entries, progress)

    def _export_entries(
        self,
        project_id: int,
        destination: Path,
        entries: Sequence[Tuple[str, Path]],
        progress: Optional[Callable[[int, int], None]],
    ) -> ExportResult:
        project = self.database.get_project(project_id)
        if project is None:
            raise ValueError("Unknown folder id: {0}".format(project_id))

        source_root = Path(project.root_path).resolve()
        destination = Path(destination).expanduser().resolve()
        if destination == source_root or source_root in destination.parents:
            raise ValueError("Choose an export folder outside the source shoot folder")

        total = len(entries)
        complete = 0
        copied = 0
        skipped = 0
        failed = []

        for asset_path, target_relative_path in entries:
            source = source_root / Path(asset_path)
            target = destination / target_relative_path
            try:
                if target.exists():
                    skipped += 1
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                    copied += 1
            except OSError as exc:
                failed.append("{0}: {1}".format(asset_path, exc))
            complete += 1
            if progress is not None:
                progress(complete, total)

        return ExportResult(
            destination=str(destination),
            copied=copied,
            skipped=skipped,
            failed=tuple(failed),
        )

    def export_count(self, project_id: int, view: PhotoView) -> int:
        photos = self.database.list_photos(project_id, view)
        return sum(
            len(photo.asset_paths or (photo.relative_path,)) for photo in photos
        )
