from __future__ import annotations

from typing import Dict

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF

from photochoose.models import EyeEvidence, Photo

EYE_COLORS: Dict[str, QColor] = {
    "open": QColor("#35d07f"),
    "closed": QColor("#ff5c62"),
    "uncertain": QColor("#ffbf47"),
}


def render_analysis_overlay(image: QImage, photo: Photo) -> QImage:
    """Return a copy of an image with its stored local-analysis evidence painted."""

    if image.isNull() or not photo.subject_regions:
        return image
    output = image.convertToFormat(QImage.Format_ARGB32_Premultiplied)
    painter = QPainter(output)
    painter.setRenderHint(QPainter.Antialiasing, True)
    paint_analysis_evidence(
        painter,
        QRectF(0.0, 0.0, float(output.width()), float(output.height())),
        photo,
        show_labels=True,
    )
    painter.end()
    return output


def paint_analysis_evidence(
    painter: QPainter,
    image_rect: QRectF,
    photo: Photo,
    show_labels: bool,
) -> None:
    """Paint normalized face and eye evidence into an image's displayed rectangle."""

    if not photo.subject_regions or image_rect.isEmpty():
        return
    painter.save()
    painter.setClipRect(image_rect)
    for index, face in enumerate(photo.subject_regions, start=1):
        x, y, width, height = face.bounds
        face_rect = QRectF(
            image_rect.left() + x * image_rect.width(),
            image_rect.top() + y * image_rect.height(),
            width * image_rect.width(),
            height * image_rect.height(),
        )
        face_scale = min(face_rect.width(), face_rect.height())
        line_width = max(0.85, min(3.0, face_scale / 90.0))
        face_color = QColor("#4da3ff") if face.important else QColor("#9aa1aa")
        pen = QPen(face_color, line_width * (1.15 if face.important else 0.85))
        if not face.important:
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(face_rect, 4.0, 4.0)

        if show_labels and face_scale >= 54.0:
            font = painter.font()
            font.setPixelSize(max(8, min(13, round(face_scale / 15.0))))
            painter.setFont(font)
            importance = "primary" if face.important else "background"
            focus = "—" if face.focus_score is None else "{0:.0f}".format(face.focus_score)
            label_height = painter.fontMetrics().height() + 6
            _paint_label(
                painter,
                face_rect.left(),
                max(image_rect.top(), face_rect.top() - label_height),
                "F{0} · {1} · focus {2}".format(index, importance, focus),
                face_color,
                image_rect,
            )

        for eye in face.eyes:
            _paint_eye(
                painter,
                image_rect,
                eye,
                line_width,
                show_labels and face_scale >= 125.0,
                face_scale,
            )
    painter.restore()


def _paint_eye(
    painter: QPainter,
    image_rect: QRectF,
    eye: EyeEvidence,
    line_width: float,
    show_label: bool,
    face_scale: float,
) -> None:
    if not eye.landmarks:
        return
    points = QPolygonF(
        [
            QPointF(
                image_rect.left() + x * image_rect.width(),
                image_rect.top() + y * image_rect.height(),
            )
            for x, y in eye.landmarks
        ]
    )
    color = EYE_COLORS.get(eye.state, EYE_COLORS["uncertain"])
    eye_bounds = points.boundingRect()
    eye_line_width = max(0.85, min(line_width, eye_bounds.width() / 14.0))
    painter.setPen(QPen(color, eye_line_width))
    painter.setBrush(QColor(color.red(), color.green(), color.blue(), 38))
    painter.drawPolygon(points)
    if show_label:
        font = painter.font()
        font.setPixelSize(max(8, min(11, round(face_scale / 22.0))))
        painter.setFont(font)
        focus = "—" if eye.focus_score is None else "{0:.0f}".format(eye.focus_score)
        _paint_label(
            painter,
            eye_bounds.left(),
            eye_bounds.bottom() + 2.0,
            "{0} · {1}".format(eye.state.title(), focus),
            color,
            image_rect,
        )


def _paint_label(
    painter: QPainter,
    left: float,
    top: float,
    text: str,
    accent: QColor,
    limit_rect: QRectF,
) -> None:
    metrics = painter.fontMetrics()
    width = metrics.horizontalAdvance(text) + 12
    height = metrics.height() + 6
    left = max(limit_rect.left(), min(left, limit_rect.right() - width))
    top = max(limit_rect.top(), min(top, limit_rect.bottom() - height))
    box = QRectF(left, top, width, height)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(10, 12, 16, 205))
    painter.drawRoundedRect(box, 3.0, 3.0)
    painter.setPen(accent)
    painter.drawText(box.adjusted(6.0, 2.0, -4.0, -2.0), Qt.AlignVCenter, text)
