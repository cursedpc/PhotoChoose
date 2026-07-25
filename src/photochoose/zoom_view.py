from __future__ import annotations

import math
import time
from typing import Optional

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QMouseEvent, QNativeGestureEvent, QWheelEvent
from PySide6.QtWidgets import QGestureEvent, QPinchGesture, QScrollArea, QWidget


class ZoomPanScrollArea(QScrollArea):
    """Image viewport with shared wheel, trackpad, and drag-pan behavior."""

    zoomFactorRequested = Signal(float)
    zoomInteractionStarted = Signal()
    zoomInteractionFinished = Signal()
    horizontalNavigationRequested = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._zoom_percent = 100
        self._drag_origin: Optional[QPointF] = None
        self._drag_scroll_origin = QPoint()
        self._horizontal_navigation_enabled = False
        self._zoom_source: Optional[str] = None
        self._last_zoom_input = 0.0
        self._pinch_tracking = False
        self._pinch_direction = 0
        self._pinch_effective_scale = 1.0
        self._zoom_interaction_active = False
        self._pending_zoom_log = 0.0
        self._zoom_frame_timer = QTimer(self)
        self._zoom_frame_timer.setSingleShot(True)
        self._zoom_frame_timer.setInterval(16)
        self._zoom_frame_timer.timeout.connect(self._flush_zoom_factor)
        self._zoom_end_timer = QTimer(self)
        self._zoom_end_timer.setSingleShot(True)
        self._zoom_end_timer.setInterval(140)
        self._zoom_end_timer.timeout.connect(self._finish_zoom_interaction)
        self.setAlignment(Qt.AlignCenter)
        self.setWidgetResizable(False)
        self.viewport().grabGesture(Qt.PinchGesture)

    def set_zoom_percent(self, value: int) -> None:
        self._zoom_percent = value
        self._update_pan_cursor()

    def set_horizontal_navigation_enabled(self, enabled: bool) -> None:
        self._horizontal_navigation_enabled = enabled

    def normalized_center(self) -> tuple[float, float]:
        return (
            self._normalized_scroll(self.horizontalScrollBar()),
            self._normalized_scroll(self.verticalScrollBar()),
        )

    def restore_normalized_center(self, center: tuple[float, float]) -> None:
        for scrollbar, position in zip(
            (self.horizontalScrollBar(), self.verticalScrollBar()), center
        ):
            scrollbar.setValue(round(position * scrollbar.maximum()))
        self._update_pan_cursor()

    def center_image(self) -> None:
        for scrollbar in (self.horizontalScrollBar(), self.verticalScrollBar()):
            scrollbar.setValue(scrollbar.maximum() // 2)
        self._update_pan_cursor()

    @staticmethod
    def _normalized_scroll(scrollbar) -> float:
        if scrollbar.maximum() <= 0:
            return 0.5
        return max(0.0, min(1.0, scrollbar.value() / scrollbar.maximum()))

    def _can_pan(self) -> bool:
        return self._zoom_percent > 100 and (
            self.horizontalScrollBar().maximum() > 0
            or self.verticalScrollBar().maximum() > 0
        )

    def _update_pan_cursor(self) -> None:
        if self._drag_origin is not None:
            self.viewport().setCursor(Qt.ClosedHandCursor)
        elif self._can_pan():
            self.viewport().setCursor(Qt.OpenHandCursor)
        else:
            self.viewport().unsetCursor()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.ControlModifier:
            pixel_delta = event.pixelDelta().y()
            angle_delta = event.angleDelta().y()
            if pixel_delta or angle_delta:
                accepted = self._accept_zoom_source("wheel")
                if accepted:
                    logarithmic_delta = (
                        pixel_delta * 0.003
                        if pixel_delta
                        else angle_delta * 0.00075
                    )
                    self._queue_zoom_log(logarithmic_delta)
                event.accept()
                return
        horizontal = event.pixelDelta().x() or event.angleDelta().x()
        vertical = event.pixelDelta().y() or event.angleDelta().y()
        if (
            self._horizontal_navigation_enabled
            and self._zoom_percent <= 100
            and abs(horizontal) >= max(20, abs(vertical))
        ):
            self.horizontalNavigationRequested.emit(1 if horizontal < 0 else -1)
            event.accept()
            return
        super().wheelEvent(event)

    def viewportEvent(self, event: QEvent) -> bool:
        if event.type() == QEvent.NativeGesture and isinstance(
            event, QNativeGestureEvent
        ):
            if event.gestureType() == Qt.BeginNativeGesture:
                return super().viewportEvent(event)
            if event.gestureType() == Qt.ZoomNativeGesture:
                accepted = self._accept_zoom_source("native")
                if accepted:
                    factor = max(0.01, 1.0 + event.value())
                    self._queue_zoom_log(math.log(factor) * 2.0)
                event.accept()
                return True
            if event.gestureType() == Qt.EndNativeGesture:
                if self._zoom_source == "native":
                    self._flush_zoom_factor()
                    self._finish_zoom_interaction()
                event.accept()
                return True

        if event.type() == QEvent.Gesture and isinstance(event, QGestureEvent):
            pinch = event.gesture(Qt.PinchGesture)
            if isinstance(pinch, QPinchGesture):
                state = pinch.state()
                if state == Qt.GestureStarted or not self._pinch_tracking:
                    self._pinch_tracking = True
                    self._pinch_direction = 0
                    self._pinch_effective_scale = 1.0
                raw_scale = pinch.totalScaleFactor()
                if raw_scale <= 0:
                    raw_scale = pinch.scaleFactor()
                accepted = raw_scale > 0 and self._accept_zoom_source("pinch")
                if accepted:
                    if self._pinch_direction == 0 and abs(raw_scale - 1.0) >= 0.003:
                        self._pinch_direction = 1 if raw_scale > 1.0 else -1
                    effective_scale = self._pinch_effective_scale
                    if self._pinch_direction > 0:
                        effective_scale = max(effective_scale, raw_scale)
                    elif self._pinch_direction < 0:
                        effective_scale = min(effective_scale, raw_scale)
                    if abs(effective_scale - self._pinch_effective_scale) >= 1e-6:
                        incremental = effective_scale / self._pinch_effective_scale
                        self._pinch_effective_scale = effective_scale
                        self._queue_zoom_log(math.log(incremental) * 1.5)
                if state in (Qt.GestureFinished, Qt.GestureCanceled):
                    if self._zoom_source == "pinch":
                        self._flush_zoom_factor()
                        self._finish_zoom_interaction()
                    self._pinch_tracking = False
                event.accept(pinch)
                return True

        if event.type() == QEvent.MouseButtonPress and isinstance(event, QMouseEvent):
            if event.button() == Qt.LeftButton and self._can_pan():
                self._drag_origin = event.position()
                self._drag_scroll_origin = QPoint(
                    self.horizontalScrollBar().value(),
                    self.verticalScrollBar().value(),
                )
                self._update_pan_cursor()
                event.accept()
                return True

        if event.type() == QEvent.MouseMove and isinstance(event, QMouseEvent):
            if self._drag_origin is not None and event.buttons() & Qt.LeftButton:
                movement = event.position() - self._drag_origin
                self.horizontalScrollBar().setValue(
                    self._drag_scroll_origin.x() - round(movement.x())
                )
                self.verticalScrollBar().setValue(
                    self._drag_scroll_origin.y() - round(movement.y())
                )
                event.accept()
                return True

        if event.type() == QEvent.MouseButtonRelease and isinstance(event, QMouseEvent):
            if event.button() == Qt.LeftButton and self._drag_origin is not None:
                self._drag_origin = None
                self._update_pan_cursor()
                event.accept()
                return True

        result = super().viewportEvent(event)
        if event.type() in (QEvent.Resize, QEvent.LayoutRequest):
            self._update_pan_cursor()
        return result

    def _accept_zoom_source(self, source: str) -> bool:
        now = time.monotonic()
        if self._zoom_source is None or now - self._last_zoom_input > 0.18:
            self._zoom_source = source
        if self._zoom_source != source:
            return False
        self._last_zoom_input = now
        self._begin_zoom_interaction()
        return True

    def _queue_zoom_log(self, logarithmic_delta: float) -> None:
        self._begin_zoom_interaction()
        self._pending_zoom_log += logarithmic_delta
        if not self._zoom_frame_timer.isActive():
            self._zoom_frame_timer.start()
        self._zoom_end_timer.start()

    def _flush_zoom_factor(self) -> None:
        if abs(self._pending_zoom_log) < 1e-6:
            return
        factor = math.exp(self._pending_zoom_log)
        self._pending_zoom_log = 0.0
        self.zoomFactorRequested.emit(max(0.80, min(1.25, factor)))

    def _begin_zoom_interaction(self) -> None:
        if self._zoom_interaction_active:
            return
        self._zoom_interaction_active = True
        self.zoomInteractionStarted.emit()

    def _finish_zoom_interaction(self) -> None:
        self._flush_zoom_factor()
        self._zoom_source = None
        if not self._zoom_interaction_active:
            return
        self._zoom_interaction_active = False
        self.zoomInteractionFinished.emit()
