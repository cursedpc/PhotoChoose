from __future__ import annotations

from typing import Dict, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF

from photochoose.models import EyeEvidence, FaceEvidence, ObjectEvidence, Photo

MAX_CROWDED_FACES = 6
MAX_CROWDED_OBJECTS = 4

EYE_COLORS: Dict[str, QColor] = {
    "open": QColor("#35d07f"),
    "closed": QColor("#ff5c62"),
    "uncertain": QColor("#ffbf47"),
    "obscured": QColor("#c084fc"),
}
OBJECT_COLORS: Dict[str, QColor] = {
    "primary": QColor("#b084ff"),
    "supporting": QColor("#55c2c8"),
    "background": QColor("#9aa1aa"),
}


def render_analysis_overlay(image: QImage, photo: Photo) -> QImage:
    """Return a copy of an image with its stored local-analysis evidence painted."""

    if image.isNull() or not (photo.subject_regions or photo.subject_objects):
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
    """Paint normalized subject, face, and eye evidence into the displayed image."""

    if not (photo.subject_regions or photo.subject_objects) or image_rect.isEmpty():
        return
    visible_objects, visible_faces, crowded = _visible_evidence(photo)
    painter.save()
    painter.setClipRect(image_rect)
    for object_index, item in enumerate(visible_objects):
        x, y, width, height = item.bounds
        object_rect = QRectF(
            image_rect.left() + x * image_rect.width(),
            image_rect.top() + y * image_rect.height(),
            width * image_rect.width(),
            height * image_rect.height(),
        )
        object_scale = min(object_rect.width(), object_rect.height())
        line_width = max(0.85, min(2.6, object_scale / 110.0))
        color = OBJECT_COLORS.get(item.importance, OBJECT_COLORS["background"])
        pen = QPen(color, line_width)
        if item.importance == "background":
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(object_rect, 4.0, 4.0)
        label_limit = 2 if crowded else 4
        if show_labels and object_index < label_limit and object_scale >= 48.0:
            font = painter.font()
            font.setPixelSize(max(8, min(12, round(object_scale / 18.0))))
            painter.setFont(font)
            focus = (
                "—"
                if item.focus_score is None
                else "{0:.0f}".format(item.focus_score)
            )
            label_height = painter.fontMetrics().height() + 6
            _paint_label(
                painter,
                object_rect.left(),
                max(image_rect.top(), object_rect.top() - label_height),
                "{0} · focus {1}".format(
                    "Subject" if item.importance == "primary" else "Supporting",
                    focus,
                ),
                color,
                image_rect,
            )
    for index, face in enumerate(visible_faces, start=1):
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

        face_label_limit = 3 if crowded else 6
        if show_labels and index <= face_label_limit and face_scale >= 54.0:
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
                show_labels
                and not crowded
                and len(visible_faces) <= 3
                and face_scale >= 125.0,
                face_scale,
            )
    painter.restore()


def _visible_evidence(
    photo: Photo,
) -> Tuple[Tuple[ObjectEvidence, ...], Tuple[FaceEvidence, ...], bool]:
    """Choose useful overlay evidence without turning crowded scenes into diagrams."""

    objects = tuple(
        item
        for item in photo.subject_objects
        if item.importance != "background"
        and not (photo.subject_regions and item.label == "person")
    )
    faces = photo.subject_regions
    crowded = len(faces) > 4 or len(objects) + len(faces) > 6
    if not crowded:
        return objects, faces, False

    ranked_objects = sorted(
        objects,
        key=lambda item: (
            item.importance == "primary",
            item.confidence * item.bounds[2] * item.bounds[3],
        ),
        reverse=True,
    )
    ranked_faces = sorted(
        faces,
        key=lambda face: (
            face.important,
            face.confidence * face.bounds[2] * face.bounds[3],
        ),
        reverse=True,
    )
    important_faces = tuple(face for face in ranked_faces if face.important)
    if not important_faces:
        important_faces = tuple(ranked_faces)
    return (
        tuple(ranked_objects[:MAX_CROWDED_OBJECTS]),
        important_faces[:MAX_CROWDED_FACES],
        True,
    )


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
