from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

from PySide6.QtCore import QEvent, QObject, QRect, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QFrame, QLabel, QScrollArea, QVBoxLayout, QWidget


@dataclass(frozen=True)
class HintStep:
    key: str
    title: str
    body: str
    is_available: Callable[[object], bool]
    host: Callable[[object], Optional[QWidget]]
    target: Callable[[object], Optional[QWidget]]


class CoachMarkOverlay(QWidget):
    dismissed = Signal()
    abandoned = Signal()

    def __init__(
        self,
        host: QWidget,
        target: Optional[QWidget],
        title: str,
        body: str,
        progress: str,
    ) -> None:
        super().__init__(host)
        self.host = host
        self.target = target
        self._target_rect = QRect()
        self._geometry_timer = QTimer(self)
        self._geometry_timer.setSingleShot(True)
        self._geometry_timer.timeout.connect(self._sync_geometry)
        self.setObjectName("coachMarkOverlay")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMouseTracking(True)

        self.card = QFrame(self)
        self.card.setObjectName("coachMarkCard")
        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(20, 18, 20, 16)
        card_layout.setSpacing(9)

        progress_label = QLabel(progress)
        progress_label.setObjectName("coachMarkProgress")
        card_layout.addWidget(progress_label)

        self.card_scroll = QScrollArea()
        self.card_scroll.setObjectName("coachMarkScroll")
        self.card_scroll.setWidgetResizable(True)
        self.card_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.card_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.card_scroll.setFrameShape(QFrame.NoFrame)

        self.card_content = QWidget()
        self.card_content.setObjectName("coachMarkContent")
        content_layout = QVBoxLayout(self.card_content)
        content_layout.setContentsMargins(0, 0, 4, 0)
        content_layout.setSpacing(9)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("coachMarkTitle")
        self.title_label.setWordWrap(True)
        content_layout.addWidget(self.title_label)

        self.body_label = QLabel(body)
        self.body_label.setObjectName("coachMarkBody")
        self.body_label.setWordWrap(True)
        self.body_label.setTextFormat(Qt.PlainText)
        content_layout.addWidget(self.body_label)
        self.card_scroll.setWidget(self.card_content)
        card_layout.addWidget(self.card_scroll, 1)

        continue_label = QLabel("Click anywhere to continue")
        continue_label.setObjectName("coachMarkContinue")
        card_layout.addWidget(continue_label)

        self.card.setStyleSheet(
            "QFrame#coachMarkCard {"
            " background: #25272c; border: 1px solid #565a63; border-radius: 4px;"
            "}"
            "QLabel#coachMarkProgress { color: #8faed8; font-size: 11px;"
            " font-weight: 600; }"
            "QLabel#coachMarkTitle { color: #f3f4f6; font-size: 17px;"
            " font-weight: 600; }"
            "QLabel#coachMarkBody { color: #d0d2d7; font-size: 13px; }"
            "QLabel#coachMarkContinue { color: #9498a1; font-size: 11px; }"
            "QScrollArea#coachMarkScroll, QWidget#coachMarkContent {"
            " background: transparent; border: 0;"
            "}"
        )
        self.card.setMinimumSize(1, 1)
        self._dismiss_widgets = (
            self.card,
            progress_label,
            self.card_scroll,
            self.card_scroll.viewport(),
            self.card_content,
            self.title_label,
            self.body_label,
            continue_label,
        )
        for widget in self._dismiss_widgets:
            widget.installEventFilter(self)
        self.host.installEventFilter(self)
        self._target_watchers = (self.target,) if self.target is not None else ()
        for widget in self._target_watchers:
            widget.installEventFilter(self)
        self._sync_geometry()
        self.show()
        self.raise_()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched in self._dismiss_widgets and event.type() == QEvent.MouseButtonPress:
            self.dismissed.emit()
            return True
        if watched is self.host:
            if event.type() in (
                QEvent.Resize,
                QEvent.Move,
                QEvent.Show,
                QEvent.LayoutRequest,
                QEvent.FontChange,
                QEvent.StyleChange,
            ):
                self._geometry_timer.start(0)
            elif event.type() in (QEvent.Close, QEvent.Hide):
                self.abandoned.emit()
        elif watched in self._target_watchers and event.type() in (
            QEvent.Resize,
            QEvent.Move,
            QEvent.Show,
            QEvent.Hide,
            QEvent.LayoutRequest,
            QEvent.ParentChange,
        ):
            self._geometry_timer.start(0)
        return super().eventFilter(watched, event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        event.accept()
        self.dismissed.emit()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        shade = QPainterPath()
        shade.setFillRule(Qt.OddEvenFill)
        shade.addRect(self.rect())
        if self._target_rect.isValid():
            shade.addRoundedRect(self._target_rect, 4, 4)
        painter.fillPath(shade, QColor(5, 6, 8, 150))
        if self._target_rect.isValid():
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor("#80a9df"), 2))
            painter.drawRoundedRect(self._target_rect, 4, 4)

    def _sync_geometry(self) -> None:
        if not self.host.isVisible():
            return
        self.setGeometry(self.host.rect())
        self._target_rect = self._mapped_target_rect()
        self.card.adjustSize()
        self._position_card()
        self.raise_()
        self.update()

    def _mapped_target_rect(self) -> QRect:
        if self.target is None or not self.target.isVisible():
            return QRect()
        # Keep this conversion entirely in Qt's logical widget coordinates.
        # A global-coordinate round trip mixes native pixels with logical pixels
        # on some Windows DPI configurations and can leave a displaced line.
        visible = self.target.visibleRegion().boundingRect()
        if not visible.isValid():
            return QRect()
        host_top_left = self.target.mapTo(self.host, visible.topLeft())
        host_bottom_right = self.target.mapTo(self.host, visible.bottomRight())
        top_left = self.mapFrom(self.host, host_top_left)
        bottom_right = self.mapFrom(self.host, host_bottom_right)
        target_rect = QRect(top_left, bottom_right).normalized().adjusted(-6, -6, 6, 6)
        clipped = target_rect.intersected(self.rect().adjusted(8, 8, -8, -8))
        # Do not paint accidental edge slivers when a target is outside a
        # scrolled or resized host while layouts are settling.
        if clipped.width() < 8 or clipped.height() < 8:
            return QRect()
        return clipped

    def _position_card(self) -> None:
        margin = min(22, max(6, min(self.width(), self.height()) // 14))
        gap = 16
        available_width = max(1, self.width() - margin * 2)
        available_height = max(1, self.height() - margin * 2)
        card_width = min(400, available_width)
        self.card.setFixedWidth(card_width)

        card_margins = self.card.layout().contentsMargins()
        scroll_bar_width = self.card_scroll.verticalScrollBar().sizeHint().width()
        content_width = max(
            1,
            card_width
            - card_margins.left()
            - card_margins.right()
            - scroll_bar_width,
        )
        self.card_content.setFixedWidth(content_width)
        self.card_content.adjustSize()

        fixed_height = (
            card_margins.top()
            + card_margins.bottom()
            + self.card.layout().spacing() * 2
            + self.card.layout().itemAt(0).widget().sizeHint().height()
            + self.card.layout().itemAt(2).widget().sizeHint().height()
        )
        scroll_height = min(
            self.card_content.sizeHint().height() + 2,
            max(1, available_height - fixed_height),
        )
        self.card_scroll.setFixedHeight(scroll_height)
        self.card.adjustSize()
        card_size = self.card.size()

        if not self._target_rect.isValid():
            x = (self.width() - card_size.width()) // 2
            y = (self.height() - card_size.height()) // 2
        else:
            room_right = self.width() - self._target_rect.right() - margin
            room_left = self._target_rect.left() - margin
            if room_right >= card_size.width() + gap:
                x = self._target_rect.right() + gap
                y = self._target_rect.center().y() - card_size.height() // 2
            elif room_left >= card_size.width() + gap:
                x = self._target_rect.left() - gap - card_size.width()
                y = self._target_rect.center().y() - card_size.height() // 2
            else:
                x = min(
                    max(margin, self._target_rect.center().x() - card_size.width() // 2),
                    max(margin, self.width() - margin - card_size.width()),
                )
                if self._target_rect.bottom() + gap + card_size.height() <= self.height():
                    y = self._target_rect.bottom() + gap
                else:
                    y = self._target_rect.top() - gap - card_size.height()
        x = min(max(margin, x), max(margin, self.width() - margin - card_size.width()))
        y = min(max(margin, y), max(margin, self.height() - margin - card_size.height()))
        self.card.move(x, y)

    def detach(self) -> None:
        self._geometry_timer.stop()
        self.host.removeEventFilter(self)
        for widget in self._target_watchers:
            widget.removeEventFilter(self)
        self.hide()
        self.deleteLater()


class OnboardingController(QObject):
    SETTINGS_PREFIX = "onboarding/workflow-v1/"

    def __init__(
        self,
        window: QWidget,
        settings: Optional[QSettings] = None,
        auto_start: bool = True,
    ) -> None:
        super().__init__(window)
        self.window = window
        self.settings = settings or QSettings()
        self.overlay: Optional[CoachMarkOverlay] = None
        self.current_step: Optional[HintStep] = None
        self.steps = self._build_steps()
        if auto_start and os.environ.get("QT_QPA_PLATFORM", "").lower() != "offscreen":
            QTimer.singleShot(500, self.show_next_when_ready)

    def restart(self) -> None:
        self._close_overlay(mark_seen=False)
        for step in self.steps:
            self.settings.remove(self.SETTINGS_PREFIX + step.key)
        self.settings.sync()
        QTimer.singleShot(0, self.show_next_when_ready)

    def notify_context_changed(self) -> None:
        if self.overlay is None:
            QTimer.singleShot(0, self.show_next_when_ready)

    def show_next_when_ready(self) -> None:
        if self.overlay is not None or not self.window.isVisible():
            return
        step = next(
            (
                item
                for item in self.steps
                if not self._was_seen(item) and item.is_available(self.window)
            ),
            None,
        )
        if step is None:
            return
        host = step.host(self.window)
        if host is None or not host.isVisible():
            return
        overlay = CoachMarkOverlay(
            host,
            step.target(self.window),
            step.title,
            step.body,
            "GUIDE  {0} / {1}".format(self.steps.index(step) + 1, len(self.steps)),
        )
        overlay.dismissed.connect(self._dismiss_current)
        overlay.abandoned.connect(self._abandon_current)
        overlay.destroyed.connect(self._overlay_destroyed)
        self.current_step = step
        self.overlay = overlay

    def _dismiss_current(self) -> None:
        if self.current_step is not None:
            self.settings.setValue(self.SETTINGS_PREFIX + self.current_step.key, True)
            self.settings.sync()
        self._close_overlay(mark_seen=False)
        QTimer.singleShot(140, self.show_next_when_ready)

    def _abandon_current(self) -> None:
        self._close_overlay(mark_seen=False)

    def _close_overlay(self, mark_seen: bool) -> None:
        if mark_seen and self.current_step is not None:
            self.settings.setValue(self.SETTINGS_PREFIX + self.current_step.key, True)
        overlay = self.overlay
        self.overlay = None
        self.current_step = None
        if overlay is not None:
            overlay.detach()

    def _overlay_destroyed(self) -> None:
        self.overlay = None
        self.current_step = None

    def _was_seen(self, step: HintStep) -> bool:
        return self.settings.value(
            self.SETTINGS_PREFIX + step.key, False, type=bool
        )

    @staticmethod
    def _build_steps() -> list[HintStep]:
        def main_host(window):
            return window

        return [
            HintStep(
                "welcome",
                "A private workspace for photo shoots",
                "PhotoChoose supports JPEG, HEIC/HEIF, PNG, and supported camera RAW "
                "files. A matching HEIC + MOV Live Photo stays one item and exports "
                "together, but video playback and ordinary video files are not supported. "
                "Analysis runs locally and your original files are never changed.",
                lambda _window: True,
                main_host,
                lambda _window: None,
            ),
            HintStep(
                "folders",
                "Begin with a shoot folder",
                "Click Add shoot, drag one or more folders onto the window, or use "
                "File → Add shoot folder. PhotoChoose scans and analyzes new folders. "
                "Select a folder here later; press Delete to close it without losing "
                "stored decisions. A phone or camera must expose a normal folder path.",
                lambda _window: True,
                main_host,
                lambda window: window.add_shoot_button,
            ),
            HintStep(
                "workspace",
                "Filter and reanalyze the shoot",
                "All, Best, Keep, Review, and Reject change what is visible. Reanalyze "
                "rescans the folder and applies the selected local analysis profile. "
                "Settings controls the profile and automatic Best count; the Folder "
                "menu also provides Rescan and a calibration report.",
                lambda window: window.current_project is not None
                and window.photo_model.rowCount() > 0,
                main_host,
                lambda window: window.view_tabs,
            ),
            HintStep(
                "review",
                "Select photographs and make decisions",
                "Click a thumbnail, or use Shift/Ctrl to select several. K keeps, R "
                "marks Review, X rejects, U clears a decision, B toggles Best, and O "
                "shows analysis evidence. Double-click a photograph for fullscreen.",
                lambda window: window.current_project is not None
                and window.photo_model.rowCount() > 0,
                main_host,
                lambda window: window.photo_grid,
            ),
            HintStep(
                "inspector",
                "Inspect before you decide",
                "The inspector shows the preview, decision and Best controls, analysis "
                "details, and Evidence. Open enters fullscreen. Compare becomes available "
                "when the selected photograph belongs to a similar-photo group.",
                lambda window: bool(window._selected_photo_ids()),
                main_host,
                lambda window: window.inspector,
            ),
            HintStep(
                "finish",
                "Export without touching the originals",
                "File → Export Keep + Best copies your chosen photographs to a new "
                "destination. File → Close folder removes its cached RAW previews while "
                "preserving metadata and decisions. Return to Help → Show hints whenever "
                "you want to replay this guide.",
                lambda window: bool(window._selected_photo_ids()),
                main_host,
                lambda window: window.menuBar(),
            ),
            HintStep(
                "fullscreen",
                "Navigate and inspect at full size",
                "Use Left/Right or the filmstrip to move through photos. The same K, R, "
                "X, U, B, and O shortcuts work here. Zoom with the slider, a two-finger "
                "pinch, or Ctrl + mouse wheel; drag the enlarged photo to pan. With O "
                "active, the right-side Faces, Eyes, and Objects menus can show all, "
                "hide, or individually override evidence for this photo. Press Esc or "
                "Close to return.",
                lambda window: window.fullscreen_dialog is not None
                and window.fullscreen_dialog.isVisible(),
                lambda window: window.fullscreen_dialog,
                lambda window: window.fullscreen_dialog.image_scroll,
            ),
            HintStep(
                "comparison",
                "Compare similar frames side by side",
                "Reference stays on the left while Previous and Next change the candidate; "
                "Swap sides promotes the candidate. Zoom and pan are synchronized, and "
                "each side has its own Keep, Review, Reject, Clear, and Best controls.",
                lambda window: window.comparison_dialog is not None
                and window.comparison_dialog.isVisible(),
                lambda window: window.comparison_dialog,
                lambda window: window.comparison_dialog.left_pane,
            ),
        ]
