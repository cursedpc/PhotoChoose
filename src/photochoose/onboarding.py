from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget


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
        self.setObjectName("coachMarkOverlay")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMouseTracking(True)

        self.card = QFrame(self)
        self.card.setObjectName("coachMarkCard")
        self.card.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(20, 18, 20, 16)
        card_layout.setSpacing(9)

        progress_label = QLabel(progress)
        progress_label.setObjectName("coachMarkProgress")
        card_layout.addWidget(progress_label)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("coachMarkTitle")
        self.title_label.setWordWrap(True)
        card_layout.addWidget(self.title_label)

        self.body_label = QLabel(body)
        self.body_label.setObjectName("coachMarkBody")
        self.body_label.setWordWrap(True)
        self.body_label.setTextFormat(Qt.PlainText)
        card_layout.addWidget(self.body_label)

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
        )
        self.card.setFixedWidth(400)
        self.host.installEventFilter(self)
        if self.target is not None:
            self.target.installEventFilter(self)
        self._sync_geometry()
        self.show()
        self.raise_()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self.host:
            if event.type() in (QEvent.Resize, QEvent.Move, QEvent.Show):
                QTimer.singleShot(0, self._sync_geometry)
            elif event.type() in (QEvent.Close, QEvent.Hide):
                self.abandoned.emit()
        elif watched is self.target and event.type() in (
            QEvent.Resize,
            QEvent.Move,
            QEvent.Show,
            QEvent.Hide,
        ):
            QTimer.singleShot(0, self._sync_geometry)
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
        top_left = self.mapFromGlobal(self.target.mapToGlobal(QPoint(0, 0)))
        target_rect = QRect(top_left, self.target.size()).adjusted(-6, -6, 6, 6)
        return target_rect.intersected(self.rect().adjusted(8, 8, -8, -8))

    def _position_card(self) -> None:
        margin = 22
        gap = 16
        card_size = self.card.sizeHint()
        card_width = min(400, max(300, self.width() - margin * 2))
        self.card.setFixedWidth(card_width)
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
        self.host.removeEventFilter(self)
        if self.target is not None:
            self.target.removeEventFilter(self)
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
                "A private workspace for JPEG and RAW shoots",
                "PhotoChoose's supported workflow is JPEG files and supported camera "
                "RAW files only. HEIC, PNG, video, and other formats are outside the "
                "supported workflow. Analysis runs locally and your original files are "
                "never changed.",
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
                "pinch, or Ctrl + mouse wheel; drag the enlarged photo to pan. Press Esc "
                "or Close to return.",
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
