from __future__ import annotations

from collections import OrderedDict
from typing import List, Optional, Set, Tuple

from PySide6.QtCore import (
    QAbstractListModel,
    QModelIndex,
    QObject,
    QRunnable,
    QSize,
    Qt,
    QThreadPool,
    Signal,
    Slot,
)
from PySide6.QtGui import QColor, QImage, QPixmap

from photochoose.database import PhotoDatabase
from photochoose.image_loader import read_qimage
from photochoose.models import Photo, PhotoView

THUMBNAIL_SIZE = QSize(220, 160)
MAX_MEMORY_THUMBNAILS = 350


class _ThumbnailSignals(QObject):
    loaded = Signal(int, int, object)


class _ThumbnailTask(QRunnable):
    def __init__(self, photo_id: int, generation: int, path: str) -> None:
        super().__init__()
        self.photo_id = photo_id
        self.generation = generation
        self.path = path
        self.signals = _ThumbnailSignals()

    def run(self) -> None:
        image = read_qimage(self.path, THUMBNAIL_SIZE)
        self.signals.loaded.emit(self.photo_id, self.generation, image)


class PhotoListModel(QAbstractListModel):
    PhotoIdRole = Qt.UserRole + 1
    PathRole = Qt.UserRole + 2
    DecisionRole = Qt.UserRole + 3
    BestRole = Qt.UserRole + 4
    GroupRole = Qt.UserRole + 5
    AssetRole = Qt.UserRole + 6
    AnalysisEvidenceRole = Qt.UserRole + 7

    def __init__(self, database: PhotoDatabase, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.database = database
        self.photos: List[Photo] = []
        self._rows_by_id = {}
        self._cache = OrderedDict()
        self._thumbnail_generation = 0
        self._in_flight: Set[Tuple[int, int]] = set()
        self._thread_pool = QThreadPool.globalInstance()
        self._placeholder = QPixmap(THUMBNAIL_SIZE)
        self._placeholder.fill(QColor("#2b3038"))

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.photos)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self.photos):
            return None
        photo = self.photos[index.row()]

        if role == Qt.DisplayRole:
            return photo.filename
        if role == Qt.DecorationRole:
            pixmap = self._cache.get(photo.id)
            if pixmap is not None:
                self._cache.move_to_end(photo.id)
                return pixmap
            self._request_thumbnail(photo)
            return self._placeholder
        if role == Qt.ToolTipRole:
            best = "yes" if photo.is_best else "no"
            lines = [
                photo.absolute_path,
                "Files: {0}".format(photo.file_summary),
                "Decision: {0} ({1})".format(photo.decision.value, photo.decision_source),
                "Best: {0} ({1})".format(best, photo.best_source),
                "Analysis: {0} · {1} / {2}".format(
                    photo.analysis_status,
                    (photo.analysis_profile or "—").title(),
                    photo.analysis_version or "—",
                ),
            ]
            if photo.focus_score is not None:
                lines.append("Focus score: {0:.2f}".format(photo.focus_score))
            if photo.brightness is not None:
                lines.append("Brightness: {0:.1f}%".format(photo.brightness * 100.0))
            if photo.face_count:
                lines.append("Detected faces: {0}".format(photo.face_count))
                lines.append("Open eyes: {0}".format(photo.eye_count))
                lines.append("Closed eyes: {0}".format(photo.closed_eye_count))
                lines.append("Uncertain eyes: {0}".format(photo.uncertain_eye_count))
                lines.append("Obscured/unresolved eyes: {0}".format(photo.obscured_eye_count))
            if photo.subject_focus_score is not None:
                lines.append("Face-region focus: {0:.2f}".format(photo.subject_focus_score))
            if photo.subject_model != "none":
                lines.append("Subject model: {0}".format(photo.subject_model))
            if photo.face_confidence is not None:
                lines.append("Average face confidence: {0:.1f}%".format(
                    photo.face_confidence * 100.0
                ))
            if photo.eye_open_score is not None:
                lines.append("Average eye-open score: {0:.1f}%".format(
                    photo.eye_open_score * 100.0
                ))
            if photo.eye_focus_score is not None:
                lines.append("Weakest primary-eye focus: {0:.2f}".format(
                    photo.eye_focus_score
                ))
            if photo.eye_visibility_reasons:
                lines.append(
                    "Eye visibility: " + " · ".join(photo.eye_visibility_reasons)
                )
            if photo.object_count:
                lines.append(
                    "Object regions: {0} (class names are model guesses)".format(
                        photo.object_count
                    )
                )
                visible_guesses = [
                    item
                    for item in photo.subject_objects
                    if item.importance != "background"
                ][:8]
                for item in visible_guesses:
                    focus = "—" if item.focus_score is None else "{0:.1f}".format(
                        item.focus_score
                    )
                    lines.append(
                        "  Likely {0} ({1:.0f}%): {2}, focus {3}".format(
                            item.label,
                            item.confidence * 100.0,
                            item.importance,
                            focus,
                        )
                    )
                hidden_count = photo.object_count - len(visible_guesses)
                if hidden_count > 0:
                    lines.append(
                        "  +{0} lower-priority guess{1}".format(
                            hidden_count, "" if hidden_count == 1 else "es"
                        )
                    )
            if photo.object_model != "none":
                lines.append("Object model: {0}".format(photo.object_model))
            if photo.similarity_group:
                lines.append("Similar group: {0}".format(photo.similarity_group))
                if photo.group_reason:
                    lines.append(photo.group_reason)
            if photo.best_score is not None:
                lines.append("Group ranking score: {0:.2f}".format(photo.best_score))
            if photo.captured_at:
                lines.append("Captured: {0}".format(photo.captured_at.replace("T", " ")))
            camera = " ".join(
                value for value in (photo.camera_make, photo.camera_model) if value
            )
            if camera:
                lines.append("Camera: {0}".format(camera))
            if photo.auto_reasons:
                lines.append("Reasons:")
                lines.extend("• " + reason for reason in photo.auto_reasons)
            return "\n".join(lines)
        if role == self.PhotoIdRole:
            return photo.id
        if role == self.PathRole:
            return photo.display_path
        if role == self.DecisionRole:
            return photo.decision.value
        if role == self.BestRole:
            return photo.is_best
        if role == self.GroupRole:
            return photo.similarity_group
        if role == self.AssetRole:
            return photo.asset_badge
        if role == self.AnalysisEvidenceRole:
            return photo
        return None

    def refresh(
        self,
        project_id: int,
        view: PhotoView,
        invalidate_thumbnails: bool = False,
    ) -> None:
        self.set_photos(
            self.database.list_photos(project_id, view),
            invalidate_thumbnails=invalidate_thumbnails,
        )

    def refresh_person(
        self,
        person_id: Optional[int],
        project_id: Optional[int] = None,
        invalidate_thumbnails: bool = False,
    ) -> None:
        self.set_photos(
            self.database.list_photos_for_person(person_id, project_id),
            invalidate_thumbnails=invalidate_thumbnails,
        )

    def set_photos(
        self,
        photos: List[Photo],
        invalidate_thumbnails: bool = False,
    ) -> None:
        self.beginResetModel()
        if invalidate_thumbnails:
            # A rescan may replace a file while retaining its database id, and
            # SQLite may reuse a deleted row's id for a newly discovered file.
            # Do not let a pixmap (or late worker result) from the former scan
            # identify the new contents by id alone.
            self._thumbnail_generation += 1
            self._cache.clear()
            self._in_flight.clear()
        self.photos = list(photos)
        self._rows_by_id = {photo.id: row for row, photo in enumerate(self.photos)}
        self.endResetModel()

    def update_photo(self, row: int, photo: Photo) -> None:
        if not 0 <= row < len(self.photos):
            return
        previous = self.photos[row]
        self.photos[row] = photo
        if previous.id != photo.id:
            self._rows_by_id.pop(previous.id, None)
            self._rows_by_id[photo.id] = row
        index = self.index(self._model_row_for_photo_row(row), 0)
        self.dataChanged.emit(
            index,
            index,
            [Qt.DisplayRole, self.DecisionRole, self.BestRole, self.AnalysisEvidenceRole],
        )

    def _model_row_for_photo_row(self, row: int) -> int:
        return row

    def clear(self) -> None:
        self.beginResetModel()
        self._thumbnail_generation += 1
        self._cache.clear()
        self.photos = []
        self._rows_by_id = {}
        self._in_flight.clear()
        self.endResetModel()

    def photo_ids(self, indexes) -> List[int]:
        return [self.photos[index.row()].id for index in indexes if index.isValid()]

    def index_for_photo_id(self, photo_id: Optional[int]) -> QModelIndex:
        row = self._rows_by_id.get(photo_id)
        return QModelIndex() if row is None else self.index(self._model_row_for_photo_row(row), 0)

    def selected_photos(self, indexes) -> List[Photo]:
        return [self.photos[index.row()] for index in indexes if index.isValid()]

    def _request_thumbnail(self, photo: Photo) -> None:
        request_key = (self._thumbnail_generation, photo.id)
        if request_key in self._in_flight:
            return
        self._in_flight.add(request_key)
        task = _ThumbnailTask(
            photo.id,
            self._thumbnail_generation,
            photo.display_path,
        )
        task.signals.loaded.connect(self._thumbnail_loaded)
        self._thread_pool.start(task)

    @Slot(int, int, object)
    def _thumbnail_loaded(
        self,
        photo_id: int,
        generation: int,
        image: QImage,
    ) -> None:
        self._in_flight.discard((generation, photo_id))
        if generation != self._thumbnail_generation:
            return
        row = self._rows_by_id.get(photo_id)
        if row is None:
            return
        pixmap = self._placeholder if image.isNull() else QPixmap.fromImage(image)
        self._cache[photo_id] = pixmap
        self._cache.move_to_end(photo_id)
        while len(self._cache) > MAX_MEMORY_THUMBNAILS:
            self._cache.popitem(last=False)
        index = self.index(self._model_row_for_photo_row(row), 0)
        self.dataChanged.emit(index, index, [Qt.DecorationRole])
