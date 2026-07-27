from __future__ import annotations

from typing import List, Optional, Sequence

from PySide6.QtCore import (
    QEvent,
    QItemSelectionModel,
    QModelIndex,
    QPoint,
    QRect,
    QSize,
    Qt,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QColor,
    QContextMenuEvent,
    QImage,
    QImageIOHandler,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QResizeEvent,
    QShortcut,
    QShowEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QGestureEvent,
    QHBoxLayout,
    QLabel,
    QListView,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QSwipeGesture,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from photochoose.analysis_overlay import render_analysis_overlay
from photochoose.database import PhotoDatabase
from photochoose.image_loader import read_qimage
from photochoose.keyboard_shortcuts import LayoutIndependentShortcutFilter
from photochoose.models import Decision, Photo, PhotoView
from photochoose.thumbnail_model import PhotoListModel
from photochoose.zoom_view import ZoomPanScrollArea


class FullscreenCarouselModel(PhotoListModel):
    """Photo model with blank edge slots so every active frame can be centered."""

    PaddingRole = Qt.UserRole + 40
    EDGE_SLOTS = 6

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid() or not self.photos:
            return 0
        return len(self.photos) + self.EDGE_SLOTS * 2

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        photo_row = self.photo_row(index)
        if photo_row is None:
            return role == self.PaddingRole
        if role == self.PaddingRole:
            return False
        return super().data(self.createIndex(photo_row, 0), role)

    def photo_row(self, index: QModelIndex) -> Optional[int]:
        if not index.isValid():
            return None
        photo_row = index.row() - self.EDGE_SLOTS
        return photo_row if 0 <= photo_row < len(self.photos) else None

    def index_for_photo_row(self, photo_row: int) -> QModelIndex:
        return self.index(photo_row + self.EDGE_SLOTS, 0)

    def _model_row_for_photo_row(self, row: int) -> int:
        return row + self.EDGE_SLOTS


class FullscreenCarouselDelegate(QStyledItemDelegate):
    """Compact filmstrip tile with decision color and an explicit active marker."""

    ITEM_SIZE = QSize(118, 82)
    DECISION_COLORS = {
        Decision.UNREVIEWED.value: QColor("#555860"),
        Decision.KEEP.value: QColor("#4f9c71"),
        Decision.REVIEW.value: QColor("#c4974d"),
        Decision.REJECT.value: QColor("#bd5b61"),
    }

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        return self.ITEM_SIZE

    def paint(
        self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex
    ) -> None:
        if index.data(FullscreenCarouselModel.PaddingRole):
            return
        painter.save()
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        tile = option.rect.adjusted(5, 7, -5, -7)
        image_box = tile.adjusted(3, 3, -3, -3)
        painter.fillRect(tile, QColor("#111214"))

        pixmap = index.data(Qt.DecorationRole)
        if pixmap is not None and not pixmap.isNull():
            scaled = pixmap.scaled(
                image_box.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            x = image_box.left() + (image_box.width() - scaled.width()) // 2
            y = image_box.top() + (image_box.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)

        decision = index.data(PhotoListModel.DecisionRole)
        color = self.DECISION_COLORS.get(decision, self.DECISION_COLORS["unreviewed"])
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(color, 2))
        painter.drawRect(tile.adjusted(1, 1, -1, -1))

        if index.data(PhotoListModel.BestRole):
            marker = QRect(tile.right() - 18, tile.top() + 2, 16, 15)
            painter.fillRect(marker, QColor("#cda552"))
            painter.setPen(QColor("#151619"))
            painter.drawText(marker, Qt.AlignCenter, "B")

        if option.state & QStyle.State_Selected:
            painter.setPen(QPen(QColor("#eef3fa"), 2))
            painter.drawRect(tile.adjusted(-2, -2, 2, 2))
            active_marker = QRect(option.rect.center().x() - 12, option.rect.top() + 1, 24, 3)
            painter.fillRect(active_marker, QColor("#6d9fe3"))
        painter.restore()


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
        self._current_image = QImage()
        self._interaction_pixmap = QPixmap()
        self._using_interaction_pixmap = False
        self._current_image_path = ""
        self._decoded_canvas_pixels = QSize()
        self._current_device_pixel_ratio = 1.0
        self.zoom_percent = 100
        self._center_after_render = False
        self._zoom_interaction_active = False
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
        self.decision_badge = QLabel("Unreviewed")
        self.decision_badge.setObjectName("decisionBadge")
        self.decision_badge.setAlignment(Qt.AlignCenter)
        title_area = QWidget()
        title_layout = QHBoxLayout(title_area)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(8)
        title_layout.addStretch(1)
        title_layout.addWidget(self.title_label)
        title_layout.addWidget(self.decision_badge)
        title_layout.addStretch(1)
        toolbar.addWidget(title_area, 1)
        self.next_button = QPushButton("→")
        self.next_button.setProperty("kind", "quiet")
        self.next_button.setToolTip("Next photo")
        self.next_button.clicked.connect(self.next_photo)
        toolbar.addWidget(self.next_button)
        toolbar.addWidget(QLabel("Zoom"))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(50, 300)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setFixedWidth(150)
        self.zoom_slider.valueChanged.connect(self._zoom_changed)
        self._continuous_zoom = 100.0
        self._gesture_zoom_active = False
        toolbar.addWidget(self.zoom_slider)
        self.zoom_label = QLabel("Fit")
        self.zoom_label.setMinimumWidth(42)
        toolbar.addWidget(self.zoom_label)
        close_button = QPushButton("Close")
        close_button.setProperty("kind", "quiet")
        close_button.clicked.connect(self.close)
        toolbar.addWidget(close_button)
        layout.addLayout(toolbar)

        self.image_scroll = ZoomPanScrollArea()
        self.image_scroll.setObjectName("fullscreenScrollArea")
        self.image_scroll.set_horizontal_navigation_enabled(True)
        self.image_scroll.zoomFactorRequested.connect(self._adjust_zoom)
        self.image_scroll.zoomInteractionStarted.connect(
            self._zoom_interaction_started
        )
        self.image_scroll.zoomInteractionFinished.connect(
            self._zoom_interaction_finished
        )
        self.image_scroll.horizontalNavigationRequested.connect(self._move)
        self.image_label = QLabel()
        self.image_label.setObjectName("fullscreenCanvas")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)
        self.image_label.setMinimumSize(1, 1)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setContextMenuPolicy(Qt.CustomContextMenu)
        self.image_label.customContextMenuRequested.connect(self._show_image_context_menu)
        self.image_scroll.setWidget(self.image_label)
        layout.addWidget(self.image_scroll, 1)

        self._resize_decode_timer = QTimer(self)
        self._resize_decode_timer.setSingleShot(True)
        self._resize_decode_timer.setInterval(180)
        self._resize_decode_timer.timeout.connect(self._decode_current_image)

        self.footer_label = QLabel()
        self.footer_label.setAlignment(Qt.AlignCenter)
        self.footer_label.setObjectName("mutedLabel")
        layout.addWidget(self.footer_label)

        filmstrip = QWidget()
        filmstrip.setObjectName("fullscreenFilmstrip")
        filmstrip_layout = QVBoxLayout(filmstrip)
        filmstrip_layout.setContentsMargins(0, 4, 0, 0)
        filmstrip_layout.setSpacing(3)
        self.carousel_position_label = QLabel()
        self.carousel_position_label.setObjectName("carouselPosition")
        self.carousel_position_label.setAlignment(Qt.AlignCenter)
        filmstrip_layout.addWidget(self.carousel_position_label)

        self.carousel_model = FullscreenCarouselModel(database, self)
        self.carousel_model.set_photos(self.photos)
        self.carousel = QListView()
        self.carousel.setObjectName("fullscreenCarousel")
        self.carousel.setModel(self.carousel_model)
        self.carousel.setItemDelegate(FullscreenCarouselDelegate(self.carousel))
        self.carousel.setViewMode(QListView.IconMode)
        self.carousel.setFlow(QListView.LeftToRight)
        self.carousel.setWrapping(False)
        self.carousel.setResizeMode(QListView.Fixed)
        self.carousel.setMovement(QListView.Static)
        self.carousel.setSelectionMode(QListView.SingleSelection)
        self.carousel.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.carousel.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.carousel.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.carousel.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.carousel.setUniformItemSizes(True)
        self.carousel.setSpacing(1)
        self.carousel.setFixedHeight(86)
        self.carousel.setFocusPolicy(Qt.NoFocus)
        self.carousel.clicked.connect(self._carousel_clicked)
        filmstrip_layout.addWidget(self.carousel)
        layout.addWidget(filmstrip)

        shortcut_specs = (
            (Qt.Key_Left, self.previous_photo),
            (Qt.Key_Right, self.next_photo),
            (Qt.Key_Escape, self.close),
            (Qt.Key_Space, self.next_photo),
            (Qt.Key_Backspace, self.previous_photo),
            (Qt.Key_K, lambda: self._set_decision(Decision.KEEP)),
            (Qt.Key_R, lambda: self._set_decision(Decision.REVIEW)),
            (Qt.Key_X, lambda: self._set_decision(Decision.REJECT)),
            (Qt.Key_U, lambda: self._set_decision(Decision.UNREVIEWED)),
            (Qt.Key_B, self._toggle_best),
            (Qt.Key_O, lambda: self._set_analysis_overlay(not self.overlay_enabled)),
        )
        self.window_shortcuts: List[QShortcut] = []
        for key, handler in shortcut_specs:
            shortcut = QShortcut(key, self)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.activated.connect(handler)
            self.window_shortcuts.append(shortcut)
        self.layout_independent_shortcuts = LayoutIndependentShortcutFilter(
            self,
            {
                "K": lambda: self._set_decision(Decision.KEEP),
                "R": lambda: self._set_decision(Decision.REVIEW),
                "X": lambda: self._set_decision(Decision.REJECT),
                "U": lambda: self._set_decision(Decision.UNREVIEWED),
                "B": self._toggle_best,
                "O": lambda: self._set_analysis_overlay(not self.overlay_enabled),
            },
        )
        self._show_current_photo()

    @property
    def current_photo(self) -> Optional[Photo]:
        if not self.photos:
            return None
        return self.photos[self.current_index]

    def _show_current_photo(self) -> None:
        photo = self.current_photo
        if photo is None:
            self.setWindowTitle("PhotoChoose")
            self.title_label.setText("No photo")
            self._update_decision_badge(None)
            self.footer_label.setText("Esc to close")
            self.image_label.setText("No photograph to display")
            self.carousel_position_label.setText("No photos")
            self._update_controls()
            return
        self.setWindowTitle(photo.filename)
        self.title_label.setText(photo.filename)
        self._update_decision_badge(photo)
        self.footer_label.setText(
            "← → navigate   ·   K Keep   R Review   X Reject   "
            "B Best   O evidence   Esc close"
        )
        self._load_image(photo.display_path)
        self._update_controls()
        self._sync_carousel()

    def _update_decision_badge(self, photo: Optional[Photo]) -> None:
        if photo is None:
            state = "unreviewed"
            text = "Unreviewed"
        else:
            state = photo.decision.value
            text = photo.decision.value.title()
            if photo.is_best:
                text += " · Best"
        self.decision_badge.setText(text)
        self.decision_badge.setProperty("state", state)
        self.decision_badge.style().unpolish(self.decision_badge)
        self.decision_badge.style().polish(self.decision_badge)

    def _load_image(self, image_path: str) -> None:
        self.image_label.clear()
        self._current_image = QImage()
        self._interaction_pixmap = QPixmap()
        self._using_interaction_pixmap = False
        self._current_image_path = image_path
        self._decoded_canvas_pixels = QSize()
        self._center_after_render = True
        self._decode_current_image(force=True)

    def _decode_current_image(self, force: bool = False) -> None:
        if not self._current_image_path:
            return
        canvas_pixels, device_pixel_ratio = self._physical_decode_target()
        if (
            not force
            and canvas_pixels == self._decoded_canvas_pixels
            and abs(device_pixel_ratio - self._current_device_pixel_ratio) < 0.01
        ):
            if self._using_interaction_pixmap:
                self._render_current_image()
            else:
                self._resize_current_image()
            return

        image = read_qimage(self._current_image_path, canvas_pixels)
        if image.isNull():
            self.image_label.setText("Could not open this photograph")
            return
        self._current_image = image
        self._decoded_canvas_pixels = canvas_pixels
        self._current_device_pixel_ratio = device_pixel_ratio
        self._render_current_image()

    def _render_current_image(self) -> None:
        if self._current_image.isNull():
            return
        display_image = self._current_image
        if self.overlay_enabled and self.current_photo is not None:
            display_image = render_analysis_overlay(display_image, self.current_photo)
        pixmap = QPixmap.fromImage(display_image)
        pixmap.setDevicePixelRatio(self._current_device_pixel_ratio)
        self.image_label.setPixmap(pixmap)
        self._interaction_pixmap = self._build_interaction_pixmap(display_image)
        self._using_interaction_pixmap = False
        self._resize_current_image()

    def _build_interaction_pixmap(self, image: QImage) -> QPixmap:
        base_pixels, device_pixel_ratio = self._physical_canvas_base()
        preview_limit = base_pixels * 1.25
        preview_size = image.size().scaled(preview_limit, Qt.KeepAspectRatio)
        preview = image
        if preview_size.width() < image.width() or preview_size.height() < image.height():
            preview = image.scaled(
                preview_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        pixmap = QPixmap.fromImage(preview)
        pixmap.setDevicePixelRatio(device_pixel_ratio)
        return pixmap

    def _resize_current_image(self) -> None:
        if self._current_image.isNull():
            return
        base_pixels, device_pixel_ratio = self._physical_canvas_base()
        fitted_size = self._current_image.size().scaled(base_pixels, Qt.KeepAspectRatio)
        fitted_size *= self.zoom_percent / 100.0
        fitted_size.setWidth(max(1, fitted_size.width()))
        fitted_size.setHeight(max(1, fitted_size.height()))
        self.image_label.resize(
            QSize(
                max(1, round(fitted_size.width() / device_pixel_ratio)),
                max(1, round(fitted_size.height() / device_pixel_ratio)),
            )
        )
        if self._center_after_render:
            self._center_after_render = False
            QTimer.singleShot(0, self.image_scroll.center_image)

    def _physical_canvas_base(self) -> tuple[QSize, float]:
        screen = self.screen() or QApplication.primaryScreen()
        device_pixel_ratio = screen.devicePixelRatio() if screen is not None else 1.0
        logical_size = self.image_scroll.viewport().contentsRect().size()
        if not self.isVisible() or logical_size.width() < 2 or logical_size.height() < 2:
            if screen is not None:
                logical_size = screen.availableGeometry().size()
                logical_size.setWidth(max(1, logical_size.width() - 48))
                logical_size.setHeight(max(1, logical_size.height() - 220))
            else:
                logical_size = QSize(1280, 720)
        return self._physical_size(logical_size, device_pixel_ratio), device_pixel_ratio

    def _physical_decode_target(self) -> tuple[QSize, float]:
        base, device_pixel_ratio = self._physical_canvas_base()
        factor = max(1.0, self.zoom_percent / 100.0)
        target = base * factor
        target.setWidth(max(1, target.width()))
        target.setHeight(max(1, target.height()))
        return target, device_pixel_ratio

    @staticmethod
    def _physical_size(logical_size: QSize, device_pixel_ratio: float) -> QSize:
        return QSize(
            max(1, round(logical_size.width() * device_pixel_ratio)),
            max(1, round(logical_size.height() * device_pixel_ratio)),
        )

    @staticmethod
    def _decode_size(source_size: QSize, target_size: QSize, rotated: bool) -> QSize:
        oriented_source = (
            QSize(source_size.height(), source_size.width()) if rotated else source_size
        )
        oriented_result = oriented_source.scaled(target_size, Qt.KeepAspectRatio)
        if rotated:
            return QSize(oriented_result.height(), oriented_result.width())
        return oriented_result

    @staticmethod
    def _transformation_rotates(
        transformation: QImageIOHandler.Transformation,
    ) -> bool:
        return transformation in {
            QImageIOHandler.TransformationRotate90,
            QImageIOHandler.TransformationMirrorAndRotate90,
            QImageIOHandler.TransformationFlipAndRotate90,
            QImageIOHandler.TransformationRotate270,
        }

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._resize_current_image()
        self._resize_decode_timer.start()

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, lambda: self._decode_current_image(force=True))
        QTimer.singleShot(0, self._center_carousel_current)

    def _update_controls(self) -> None:
        has_photo = self.current_photo is not None
        self.previous_button.setEnabled(has_photo and self.current_index > 0)
        self.next_button.setEnabled(has_photo and self.current_index < len(self.photos) - 1)

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

    @Slot(QModelIndex)
    def _carousel_clicked(self, index: QModelIndex) -> None:
        photo_row = self.carousel_model.photo_row(index)
        if photo_row is None:
            self._sync_carousel()
            return
        if photo_row == self.current_index:
            return
        self.current_index = photo_row
        self._show_current_photo()

    def _sync_carousel(self) -> None:
        if not self.photos:
            return
        index = self.carousel_model.index_for_photo_row(self.current_index)
        self.carousel.setCurrentIndex(index)
        self.carousel.selectionModel().select(index, QItemSelectionModel.ClearAndSelect)
        self.carousel_position_label.setText(
            "Current photo  {0} of {1}".format(self.current_index + 1, len(self.photos))
        )
        QTimer.singleShot(0, self._center_carousel_current)

    def _center_carousel_current(self) -> None:
        if not self.photos:
            return
        index = self.carousel_model.index_for_photo_row(self.current_index)
        self.carousel.scrollTo(index, QAbstractItemView.PositionAtCenter)

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
            self.carousel_model.update_photo(
                self.current_index, refreshed[photo.id]
            )
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
        reset_zoom = menu.addAction("Fit photograph to window")
        reset_zoom.setEnabled(self.zoom_percent != 100)
        reset_zoom.triggered.connect(lambda: self.zoom_slider.setValue(100))
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
        self._render_current_image()

    @Slot(int)
    def _zoom_changed(self, value: int) -> None:
        center = self.image_scroll.normalized_center()
        if not self._gesture_zoom_active:
            self._continuous_zoom = float(value)
        self.zoom_percent = value
        self.zoom_label.setText("Fit" if value == 100 else "{0}%".format(value))
        self.image_scroll.set_zoom_percent(value)
        self._resize_current_image()
        self.image_scroll.restore_normalized_center(center)
        if not self._zoom_interaction_active:
            self._resize_decode_timer.start()

    @Slot(float)
    def _adjust_zoom(self, factor: float) -> None:
        self._continuous_zoom = min(
            300.0, max(50.0, self._continuous_zoom * factor)
        )
        self._gesture_zoom_active = True
        try:
            self.zoom_slider.setValue(round(self._continuous_zoom))
        finally:
            self._gesture_zoom_active = False

    @Slot()
    def _zoom_interaction_started(self) -> None:
        self._zoom_interaction_active = True
        self._resize_decode_timer.stop()
        if not self._interaction_pixmap.isNull():
            self.image_label.setPixmap(self._interaction_pixmap)
            self._using_interaction_pixmap = True

    @Slot()
    def _zoom_interaction_finished(self) -> None:
        self._zoom_interaction_active = False
        self._resize_decode_timer.start()

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
        if event.type() == QEvent.ScreenChangeInternal:
            QTimer.singleShot(0, lambda: self._decode_current_image(force=True))
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
        if event.modifiers() & Qt.ControlModifier:
            delta = event.pixelDelta().y() or event.angleDelta().y()
            if delta:
                self._adjust_zoom(pow(1.0015, delta))
                event.accept()
                return
        horizontal = event.pixelDelta().x() or event.angleDelta().x()
        if abs(horizontal) >= 20 and self.zoom_percent <= 100:
            self._move(1 if horizontal < 0 else -1)
            event.accept()
            return
        super().wheelEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        self.close()
        event.accept()
