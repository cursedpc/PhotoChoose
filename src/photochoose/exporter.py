from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

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

    def export_keep_and_best(
        self,
        project_id: int,
        destination: Path,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> ExportResult:
        project = self.database.get_project(project_id)
        if project is None:
            raise ValueError("Unknown folder id: {0}".format(project_id))

        source_root = Path(project.root_path).resolve()
        destination = Path(destination).expanduser().resolve()
        if destination == source_root or source_root in destination.parents:
            raise ValueError("Choose an export folder outside the source shoot folder")

        categories = (
            ("Keep", self.database.list_photos(project_id, PhotoView.KEEP)),
            ("Best", self.database.list_photos(project_id, PhotoView.BEST)),
        )
        entries = [
            (category, photo, asset_path)
            for category, photos in categories
            for photo in photos
            for asset_path in (photo.asset_paths or (photo.relative_path,))
        ]
        total = len(entries)
        complete = 0
        copied = 0
        skipped = 0
        failed = []

        for category, photo, asset_path in entries:
            category_root = destination / category
            source = source_root / Path(asset_path)
            target = category_root / Path(asset_path)
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

    def export_count(self, project_id: int) -> int:
        keep = self.database.list_photos(project_id, PhotoView.KEEP)
        best = self.database.list_photos(project_id, PhotoView.BEST)
        return sum(
            len(photo.asset_paths or (photo.relative_path,)) for photo in keep + best
        )
