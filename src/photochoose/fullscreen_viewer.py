from __future__ import annotations

import time
from typing import List, Optional, Sequence

from PySide6.QtCore import QEvent, QPoint, Qt, QTimer, Signal, Slot
from PySide6.QtGui import (
    QContextMenuEvent,
    QImage,
    QImageReader,
    QKeyEvent,
    QMouseEvent,
    QPixmap,
    QResizeEvent,
    QShowEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QGestureEvent,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSwipeGesture,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from photochoose.analysis_overlay import render_analysis_overlay
from photochoose.database import PhotoDatabase
from photochoose.models import Decision, Photo, PhotoView


class FullscreenPhotoDialog(QDialog):
    """Navigable fullscreen culling surface for the current photo view."""

    photosChanged = Signal()

    def __init__(
        self,
        database: PhotoDatabase,
        photos: Sequence[Photo],
        current_index: int = 0,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.database = database
        self.photos: List[Photo] = list(photos)
        self.current_index = min(max(0, current_index), max(0, len(self.photos) - 1))
        self._last_wheel_navigation = 0.0
        self._current_image = QImage()
        self.overlay_enabled = False
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setObjectName("fullscreenSurface")
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.grabGesture(Qt.SwipeGesture)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 10)
        layout.setSpacing(9)

        toolbar = QHBoxLayout()
        self.menu_button = QToolButton()
        self.menu_button.setText("More")
        self.menu_button.setProperty("kind", "quiet")
        self.menu_button.setToolTip("Photo and view menu")
        self.menu_button.clicked.connect(self._show_toolbar_menu)
        toolbar.addWidget(self.menu_button)
        self.previous_button = QPushButton("←")
        self.previous_button.setProperty("kind", "quiet")
        self.previous_button.setToolTip("Previous photo")
        self.previous_button.clicked.connect(self.previous_photo)
        toolbar.addWidget(self.previous_button)
        self.title_label = QLabel()
        self.title_label.setObjectName("panelTitle")
        self.title_label.setAlignment(Qt.AlignCenter)
        toolbar.addWidget(self.title_label, 1)
        self.next_button = QPushButton("→")
        self.next_button.setProperty("kind", "quiet")
        self.next_button.setToolTip("Next photo")
        self.next_button.clicked.connect(self.next_photo)
        toolbar.addWidget(self.next_button)
        close_button = QPushButton("Close")
        close_button.setProperty("kind", "quiet")
        close_button.clicked.connect(self.close)
        toolbar.addWidget(close_button)
        layout.addLayout(toolbar)

        self.keep_button = self._action_button(
            "Keep", lambda: self._set_decision(Decision.KEEP)
        )
        self.keep_button.setProperty("kind", "keep")
        self.review_button = self._action_button(
            "Review", lambda: self._set_decision(Decision.REVIEW)
        )
        self.review_button.setProperty("kind", "review")
        self.reject_button = self._action_button(
            "Reject", lambda: self._set_decision(Decision.REJECT)
        )
        self.reject_button.setProperty("kind", "reject")
        self.clear_button = self._action_button(
            "Clear", lambda: self._set_decision(Decision.UNREVIEWED)
        )
        self.clear_button.setProperty("kind", "quiet")
        self.best_button = self._action_button("Best", self._toggle_best)
        self.best_button.setProperty("kind", "best")
        self.best_button.setCheckable(True)
        self.overlay_button = self._action_button(
            "Evidence", self._toggle_analysis_overlay
        )
        self.overlay_button.setCheckable(True)
        for button in (self.keep_button, self.review_button, self.reject_button):
            button.setCheckable(True)

        self.image_label = QLabel()
        self.image_label.setObjectName("fullscreenCanvas")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(1, 1)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setContextMenuPolicy(Qt.CustomContextMenu)
        self.image_label.customContextMenuRequested.connect(self._show_image_context_menu)
        layout.addWidget(self.image_label, 1)

        self.footer_label = QLabel()
        self.footer_label.setAlignment(Qt.AlignCenter)
        self.footer_label.setObjectName("mutedLabel")
        layout.addWidget(self.footer_label)

        action_shelf = QFrame()
        action_shelf.setObjectName("actionShelf")
        action_layout = QHBoxLayout(action_shelf)
        action_layout.setContentsMargins(10, 7, 10, 7)
        action_layout.setSpacing(7)
        action_layout.addStretch(1)
        for button in (
            self.keep_button,
            self.review_button,
            self.reject_button,
            self.best_button,
            self.clear_button,
            self.overlay_button,
        ):
            action_layout.addWidget(button)
        action_layout.addStretch(1)
        layout.addWidget(action_shelf)
        self._show_current_photo()

    @property
    def current_photo(self) -> Optional[Photo]:
        if not self.photos:
            return None
        return self.photos[self.current_index]

    @staticmethod
    def _action_button(text: str, handler) -> QPushButton:
        button = QPushButton(text)
        button.clicked.connect(handler)
        return button

    def _show_current_photo(self) -> None:
        photo = self.current_photo
        if photo is None:
            self.setWindowTitle("PhotoChoose")
            self.title_label.setText("No photo")
            self.footer_label.setText("Esc to close")
            self.image_label.setText("No photograph to display")
            self._update_controls()
            return
        self.setWindowTitle(photo.filename)
        self.title_label.setText(
            "{0}    ·    {1}{2}".format(
                photo.filename,
                photo.decision.value.title(),
                "    ·    Best" if photo.is_best else "",
            )
        )
        self.footer_label.setText(
            "{0} / {1}   ·   ← → navigate   ·   K Keep   R Review   X Reject   "
            "B Best   O evidence   Esc close".format(
                self.current_index + 1, len(self.photos)
            )
        )
        self._load_image(photo.display_path)
        self._update_controls()
        self._update_action_state(photo)

    def _update_action_state(self, photo: Photo) -> None:
        self.keep_button.setChecked(photo.decision == Decision.KEEP)
        self.review_button.setChecked(photo.decision == Decision.REVIEW)
        self.reject_button.setChecked(photo.decision == Decision.REJECT)
        self.best_button.setChecked(photo.is_best)
        self.best_button.setText("Best selected" if photo.is_best else "Best")

    def _load_image(self, image_path: str) -> None:
        self.image_label.clear()
        self._current_image = QImage()
        reader = QImageReader(image_path)
        reader.setAutoTransform(True)
        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None:
            target = screen.availableGeometry().size()
            target.setWidth(max(1, target.width() - 48))
            target.setHeight(max(1, target.height() - 150))
            source_size = reader.size()
            if source_size.isValid():
                reader.setScaledSize(source_size.scaled(target, Qt.KeepAspectRatio))
        image = reader.read()
        if image.isNull():
            self.image_label.setText("Could not open this photograph")
            return
        self._current_image = image
        self._scale_current_image()

    def _scale_current_image(self) -> None:
        if self._current_image.isNull():
            return
        target = self.image_label.contentsRect().size()
        if target.width() < 2 or target.height() < 2:
            return
        display_image = self._current_image
        if self.overlay_enabled and self.current_photo is not None:
            display_image = render_analysis_overlay(display_image, self.current_photo)
        scaled = display_image.scaled(
            target,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(QPixmap.fromImage(scaled))

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._scale_current_image()

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self._scale_current_image)

    def _update_controls(self) -> None:
        has_photo = self.current_photo is not None
        self.previous_button.setEnabled(has_photo and self.current_index > 0)
        self.next_button.setEnabled(has_photo and self.current_index < len(self.photos) - 1)
        for button in (
            self.keep_button,
            self.review_button,
            self.reject_button,
            self.clear_button,
            self.best_button,
            self.overlay_button,
        ):
            button.setEnabled(has_photo)

    @Slot()
    def previous_photo(self) -> None:
        self._move(-1)

    @Slot()
    def next_photo(self) -> None:
        self._move(1)

    def _move(self, delta: int) -> None:
        target = self.current_index + delta
        if 0 <= target < len(self.photos):
            self.current_index = target
            self._show_current_photo()

    def _set_decision(self, decision: Decision) -> None:
        photo = self.current_photo
        if photo is None:
            return
        self.database.set_decision([photo.id], decision)
        self._reload_current_photo()

    def _toggle_best(self) -> None:
        photo = self.current_photo
        if photo is None:
            return
        self.database.toggle_best([photo.id])
        self._reload_current_photo()

    @Slot()
    def _toggle_analysis_overlay(self) -> None:
        self.overlay_enabled = self.overlay_button.isChecked()
        self._scale_current_image()

    def _reload_current_photo(self) -> None:
        photo = self.current_photo
        if photo is None:
            return
        refreshed = {
            item.id: item
            for item in self.database.list_photos(photo.project_id, PhotoView.ALL)
        }
        if photo.id in refreshed:
            self.photos[self.current_index] = refreshed[photo.id]
        self.photosChanged.emit()
        self._show_current_photo()

    def _context_menu(self) -> QMenu:
        menu = QMenu(self)
        previous = menu.addAction("Previous photo")
        previous.triggered.connect(self.previous_photo)
        next_action = menu.addAction("Next photo")
        next_action.triggered.connect(self.next_photo)
        previous.setEnabled(self.current_index > 0)
        next_action.setEnabled(self.current_index < len(self.photos) - 1)
        menu.addSeparator()
        keep = menu.addAction("Keep")
        keep.triggered.connect(lambda: self._set_decision(Decision.KEEP))
        review = menu.addAction("Review")
        review.triggered.connect(lambda: self._set_decision(Decision.REVIEW))
        reject = menu.addAction("Reject")
        reject.triggered.connect(lambda: self._set_decision(Decision.REJECT))
        clear = menu.addAction("Clear decision")
        clear.triggered.connect(lambda: self._set_decision(Decision.UNREVIEWED))
        best = menu.addAction("Toggle Best")
        best.triggered.connect(self._toggle_best)
        menu.addSeparator()
        overlay = menu.addAction("Analysis overlay")
        overlay.setCheckable(True)
        overlay.setChecked(self.overlay_enabled)
        overlay.toggled.connect(self._set_analysis_overlay)
        menu.addSeparator()
        close = menu.addAction("Close fullscreen")
        close.triggered.connect(self.close)
        enabled = self.current_photo is not None
        for action in (keep, review, reject, clear, best):
            action.setEnabled(enabled)
        return menu

    @Slot(bool)
    def _set_analysis_overlay(self, enabled: bool) -> None:
        self.overlay_enabled = enabled
        self.overlay_button.blockSignals(True)
        self.overlay_button.setChecked(enabled)
        self.overlay_button.blockSignals(False)
        self._scale_current_image()

    @Slot()
    def _show_toolbar_menu(self) -> None:
        menu = self._context_menu()
        menu.exec(self.menu_button.mapToGlobal(QPoint(0, self.menu_button.height())))

    @Slot(QPoint)
    def _show_context_menu(self, position: QPoint) -> None:
        self._context_menu().exec(self.mapToGlobal(position))

    @Slot(QPoint)
    def _show_image_context_menu(self, position: QPoint) -> None:
        self._context_menu().exec(self.image_label.mapToGlobal(position))

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        self._context_menu().exec(event.globalPos())
        event.accept()

    def event(self, event: QEvent) -> bool:
        if event.type() == QEvent.Gesture:
            gesture_event = event
            if isinstance(gesture_event, QGestureEvent):
                swipe = gesture_event.gesture(Qt.SwipeGesture)
                if isinstance(swipe, QSwipeGesture):
                    if swipe.horizontalDirection() == QSwipeGesture.Left:
                        self.next_photo()
                    elif swipe.horizontalDirection() == QSwipeGesture.Right:
                        self.previous_photo()
                    gesture_event.accept(swipe)
                    return True
        return super().event(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        horizontal = event.pixelDelta().x() or event.angleDelta().x()
        now = time.monotonic()
        if abs(horizontal) >= 20 and now - self._last_wheel_navigation >= 0.25:
            self._last_wheel_navigation = now
            if horizontal < 0:
                self.next_photo()
            else:
                self.previous_photo()
            event.accept()
            return
        super().wheelEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
            return
        if key in (Qt.Key_Right, Qt.Key_D, Qt.Key_Space):
            self.next_photo()
            return
        if key in (Qt.Key_Left, Qt.Key_A, Qt.Key_Backspace):
            self.previous_photo()
            return
        if key == Qt.Key_K:
            self._set_decision(Decision.KEEP)
            return
        if key == Qt.Key_R:
            self._set_decision(Decision.REVIEW)
            return
        if key == Qt.Key_X:
            self._set_decision(Decision.REJECT)
            return
        if key == Qt.Key_U:
            self._set_decision(Decision.UNREVIEWED)
            return
        if key == Qt.Key_B:
            self._toggle_best()
            return
        if key == Qt.Key_O:
            self.overlay_button.setChecked(not self.overlay_button.isChecked())
            return
        super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        self.close()
        event.accept()
