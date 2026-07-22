from __future__ import annotations

from typing import List, Optional, Sequence

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QImage, QImageReader, QKeyEvent, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from photochoose.analysis_overlay import render_analysis_overlay
from photochoose.database import PhotoDatabase
from photochoose.models import Decision, Photo, PhotoView


class ComparisonPane(QFrame):
    decisionRequested = Signal(int, object)
    bestRequested = Signal(int)

    def __init__(self, side: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.photo: Optional[Photo] = None
        self.image = QImage()
        self.zoom_percent = 100
        self.overlay_enabled = False
        self.setObjectName("comparisonPane")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(11, 11, 11, 10)
        layout.setSpacing(8)
        self.heading = QLabel(side)
        self.heading.setObjectName("panelTitle")
        layout.addWidget(self.heading)

        self.scroll_area = QScrollArea()
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidgetResizable(False)
        self.image_label = QLabel()
        self.image_label.setObjectName("fullscreenCanvas")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidget(self.image_label)
        layout.addWidget(self.scroll_area, 1)

        self.details = QLabel()
        self.details.setObjectName("mutedLabel")
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.details)

        actions = QHBoxLayout()
        actions.setSpacing(6)
        self.decision_buttons = {}
        for text, decision in (
            ("Keep", Decision.KEEP),
            ("Review", Decision.REVIEW),
            ("Reject", Decision.REJECT),
            ("Clear", Decision.UNREVIEWED),
        ):
            button = QPushButton(text)
            if decision != Decision.UNREVIEWED:
                button.setCheckable(True)
                button.setProperty("kind", decision.value)
                self.decision_buttons[decision] = button
            else:
                button.setProperty("kind", "quiet")
            button.clicked.connect(
                lambda _checked=False, choice=decision: self._request_decision(choice)
            )
            actions.addWidget(button)
        self.best_button = QPushButton("☆ Best")
        self.best_button.setCheckable(True)
        self.best_button.setProperty("kind", "best")
        self.best_button.clicked.connect(self._request_best)
        actions.addWidget(self.best_button)
        layout.addLayout(actions)

    def set_photo(self, photo: Photo) -> None:
        self.photo = photo
        self.heading.setText(
            "{0}    ·    {1}{2}".format(
                photo.filename,
                photo.decision.value.title(),
                "    ·    ★ Best" if photo.is_best else "",
            )
        )
        for decision, button in self.decision_buttons.items():
            button.setChecked(photo.decision == decision)
        self.best_button.setChecked(photo.is_best)
        self.best_button.setText("★ Best" if photo.is_best else "☆ Best")
        metrics = [
            "Focus {0} · Face focus {1} · Eye focus {2} · Subject focus {3} · "
            "Faces {4} · Objects {5} · Eyes O/C/?/Obs {6}/{7}/{8}/{9} · Rank {10}".format(
                self._number(photo.focus_score),
                self._number(photo.subject_focus_score),
                self._number(photo.eye_focus_score),
                self._number(photo.primary_subject_focus_score),
                photo.face_count,
                photo.object_count,
                photo.eye_count,
                photo.closed_eye_count,
                photo.uncertain_eye_count,
                photo.obscured_eye_count,
                self._number(photo.best_score),
            )
        ]
        if photo.auto_reasons:
            metrics.append(" · ".join(photo.auto_reasons))
        self.details.setText("\n".join(metrics))
        self._load_image(photo.display_path)

    def set_zoom(self, zoom_percent: int) -> None:
        self.zoom_percent = zoom_percent
        self._render_image()

    def set_overlay_enabled(self, enabled: bool) -> None:
        self.overlay_enabled = enabled
        self._render_image()

    def _load_image(self, path: str) -> None:
        reader = QImageReader(path)
        reader.setAutoTransform(True)
        source_size = reader.size()
        if source_size.isValid():
            reader.setScaledSize(source_size.scaled(2200, 2200, Qt.KeepAspectRatio))
        self.image = reader.read()
        if self.image.isNull():
            self.image_label.setText("Could not load photograph")
            self.image_label.adjustSize()
            return
        QTimer.singleShot(0, self._render_image)

    def _render_image(self) -> None:
        if self.image.isNull():
            return
        viewport = self.scroll_area.viewport().size()
        viewport.setWidth(max(1, viewport.width() - 4))
        viewport.setHeight(max(1, viewport.height() - 4))
        fitted = self.image.size().scaled(viewport, Qt.KeepAspectRatio)
        factor = self.zoom_percent / 100.0
        target = fitted * factor
        target.setWidth(max(1, target.width()))
        target.setHeight(max(1, target.height()))
        display_image = self.image.scaled(
            target, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        if self.overlay_enabled and self.photo is not None:
            display_image = render_analysis_overlay(display_image, self.photo)
        pixmap = QPixmap.fromImage(display_image)
        self.image_label.setPixmap(pixmap)
        self.image_label.resize(pixmap.size())

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        QTimer.singleShot(0, self._render_image)

    def _request_decision(self, decision: Decision) -> None:
        if self.photo is not None:
            self.decisionRequested.emit(self.photo.id, decision)

    def _request_best(self) -> None:
        if self.photo is not None:
            self.bestRequested.emit(self.photo.id)

    @staticmethod
    def _number(value: Optional[float]) -> str:
        return "—" if value is None else "{0:.1f}".format(value)


class GroupComparisonDialog(QDialog):
    """Two-up burst comparison with synchronized zoom and pan."""

    photosChanged = Signal()

    def __init__(
        self,
        database: PhotoDatabase,
        photos: Sequence[Photo],
        anchor_photo_id: int,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.database = database
        self.photos: List[Photo] = list(photos)
        self.anchor_index = self._index_for_id(anchor_photo_id)
        self.candidate_index = self._next_distinct(self.anchor_index, 1)
        self._syncing_scroll = False
        group_id = self.photos[0].similarity_group if self.photos else "group"
        self.setWindowTitle("Compare {0} · PhotoChoose".format(group_id))
        self.setWindowFlags(Qt.Window)
        self.resize(1280, 820)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 10)
        layout.setSpacing(9)
        toolbar = QHBoxLayout()
        self.group_label = QLabel()
        self.group_label.setObjectName("panelTitle")
        toolbar.addWidget(self.group_label, 1)
        previous = QPushButton("← Previous")
        previous.setProperty("kind", "quiet")
        previous.clicked.connect(self.previous_candidate)
        toolbar.addWidget(previous)
        swap = QPushButton("Swap sides")
        swap.setProperty("kind", "quiet")
        swap.clicked.connect(self.swap_sides)
        toolbar.addWidget(swap)
        following = QPushButton("Next →")
        following.setProperty("kind", "quiet")
        following.clicked.connect(self.next_candidate)
        toolbar.addWidget(following)
        toolbar.addWidget(QLabel("Zoom"))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(50, 300)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setFixedWidth(150)
        self.zoom_slider.valueChanged.connect(self._zoom_changed)
        toolbar.addWidget(self.zoom_slider)
        self.zoom_label = QLabel("Fit")
        self.zoom_label.setMinimumWidth(42)
        toolbar.addWidget(self.zoom_label)
        self.overlay_button = QPushButton("Evidence")
        self.overlay_button.setCheckable(True)
        self.overlay_button.toggled.connect(self._overlay_changed)
        toolbar.addWidget(self.overlay_button)
        close = QPushButton("Close")
        close.setProperty("kind", "quiet")
        close.clicked.connect(self.close)
        toolbar.addWidget(close)
        layout.addLayout(toolbar)

        splitter = QSplitter(Qt.Horizontal)
        self.left_pane = ComparisonPane("Reference")
        self.right_pane = ComparisonPane("Candidate")
        for pane in (self.left_pane, self.right_pane):
            pane.decisionRequested.connect(self._set_decision)
            pane.bestRequested.connect(self._toggle_best)
            splitter.addWidget(pane)
        splitter.setSizes([640, 640])
        layout.addWidget(splitter, 1)
        self._connect_scrollbars()
        self._show_pair()

    def _index_for_id(self, photo_id: int) -> int:
        for index, photo in enumerate(self.photos):
            if photo.id == photo_id:
                return index
        return 0

    def _next_distinct(self, start: int, direction: int) -> int:
        if len(self.photos) < 2:
            return start
        candidate = (start + direction) % len(self.photos)
        return candidate if candidate != self.anchor_index else (candidate + direction) % len(
            self.photos
        )

    def _show_pair(self) -> None:
        if not self.photos:
            self.group_label.setText("No comparison group")
            return
        group = self.photos[0].similarity_group or "Group"
        reason = self.photos[0].group_reason or "Similar photographs"
        self.group_label.setText("{0} · {1}".format(group, reason))
        self.left_pane.set_photo(self.photos[self.anchor_index])
        self.right_pane.set_photo(self.photos[self.candidate_index])

    @Slot()
    def previous_candidate(self) -> None:
        self.candidate_index = self._next_distinct(self.candidate_index, -1)
        self._show_pair()

    @Slot()
    def next_candidate(self) -> None:
        self.candidate_index = self._next_distinct(self.candidate_index, 1)
        self._show_pair()

    @Slot()
    def swap_sides(self) -> None:
        self.anchor_index, self.candidate_index = self.candidate_index, self.anchor_index
        self._show_pair()

    @Slot(int)
    def _zoom_changed(self, value: int) -> None:
        self.zoom_label.setText("Fit" if value == 100 else "{0}%".format(value))
        self.left_pane.set_zoom(value)
        self.right_pane.set_zoom(value)

    @Slot(bool)
    def _overlay_changed(self, enabled: bool) -> None:
        self.left_pane.set_overlay_enabled(enabled)
        self.right_pane.set_overlay_enabled(enabled)

    @Slot(int, object)
    def _set_decision(self, photo_id: int, decision: Decision) -> None:
        self.database.set_decision([photo_id], decision)
        self._reload_photos()

    @Slot(int)
    def _toggle_best(self, photo_id: int) -> None:
        self.database.toggle_best([photo_id])
        self._reload_photos()

    def _reload_photos(self) -> None:
        if not self.photos:
            return
        anchor_id = self.photos[self.anchor_index].id
        candidate_id = self.photos[self.candidate_index].id
        group_id = self.photos[0].similarity_group
        all_photos = self.database.list_photos(self.photos[0].project_id, PhotoView.ALL)
        self.photos = [photo for photo in all_photos if photo.similarity_group == group_id]
        self.anchor_index = self._index_for_id(anchor_id)
        self.candidate_index = self._index_for_id(candidate_id)
        self.photosChanged.emit()
        self._show_pair()

    def _connect_scrollbars(self) -> None:
        for getter in (QScrollArea.horizontalScrollBar, QScrollArea.verticalScrollBar):
            left = getter(self.left_pane.scroll_area)
            right = getter(self.right_pane.scroll_area)
            left.valueChanged.connect(
                lambda value, source=left, target=right: self._sync_scroll(
                    source, target, value
                )
            )
            right.valueChanged.connect(
                lambda value, source=right, target=left: self._sync_scroll(
                    source, target, value
                )
            )

    def _sync_scroll(self, source, target, value: int) -> None:
        if self._syncing_scroll or source.maximum() <= 0:
            return
        self._syncing_scroll = True
        target.setValue(round(value / source.maximum() * target.maximum()))
        self._syncing_scroll = False

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Escape:
            self.close()
            return
        if event.key() == Qt.Key_Left:
            self.previous_candidate()
            return
        if event.key() == Qt.Key_Right:
            self.next_candidate()
            return
        if event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self.zoom_slider.setValue(min(300, self.zoom_slider.value() + 25))
            return
        if event.key() == Qt.Key_Minus:
            self.zoom_slider.setValue(max(50, self.zoom_slider.value() - 25))
            return
        if event.key() == Qt.Key_0:
            self.zoom_slider.setValue(100)
            return
        if event.key() == Qt.Key_O:
            self.overlay_button.setChecked(not self.overlay_button.isChecked())
            return
        super().keyPressEvent(event)
