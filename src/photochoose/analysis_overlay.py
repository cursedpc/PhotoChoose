from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
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


class EvidenceDisplayMode(str, Enum):
    """How a whole evidence category is selected for display."""

    COMMON = "common"
    ALL = "all"
    OFF = "off"


class EvidenceItemMode(str, Enum):
    """A per-item override layered on top of its category mode."""

    AUTO = "auto"
    SHOW = "show"
    HIDE = "hide"


@dataclass(frozen=True)
class AnalysisOverlayVisibility:
    """Per-photo category modes and individual evidence overrides."""

    faces: EvidenceDisplayMode = EvidenceDisplayMode.COMMON
    eyes: EvidenceDisplayMode = EvidenceDisplayMode.COMMON
    objects: EvidenceDisplayMode = EvidenceDisplayMode.COMMON
    face_overrides: Tuple[Tuple[int, EvidenceItemMode], ...] = ()
    eye_overrides: Tuple[Tuple[int, int, EvidenceItemMode], ...] = ()
    object_overrides: Tuple[Tuple[int, EvidenceItemMode], ...] = ()


def render_analysis_overlay(
    image: QImage,
    photo: Photo,
    visibility: AnalysisOverlayVisibility = AnalysisOverlayVisibility(),
) -> QImage:
    """Return a copy of an image with its stored local-analysis evidence painted."""

    if image.isNull() or not _has_selected_evidence(photo, visibility):
        return image
    output = image.convertToFormat(QImage.Format_ARGB32_Premultiplied)
    painter = QPainter(output)
    painter.setRenderHint(QPainter.Antialiasing, True)
    paint_analysis_evidence(
        painter,
        QRectF(0.0, 0.0, float(output.width()), float(output.height())),
        photo,
        show_labels=True,
        visibility=visibility,
    )
    painter.end()
    return output


def paint_analysis_evidence(
    painter: QPainter,
    image_rect: QRectF,
    photo: Photo,
    show_labels: bool,
    visibility: AnalysisOverlayVisibility = AnalysisOverlayVisibility(),
) -> None:
    """Paint normalized subject, face, and eye evidence into the displayed image."""

    if not _has_selected_evidence(photo, visibility) or image_rect.isEmpty():
        return
    visible_objects, visible_faces, visible_eyes, crowded = _selected_evidence(
        photo, visibility
    )
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
            focus = (
                "—" if face.focus_score is None else "{0:.0f}".format(face.focus_score)
            )
            label_height = painter.fontMetrics().height() + 6
            _paint_label(
                painter,
                face_rect.left(),
                max(image_rect.top(), face_rect.top() - label_height),
                "F{0} · {1} · focus {2}".format(index, importance, focus),
                face_color,
                image_rect,
            )

    for face, eye in visible_eyes:
        face_scale = min(
            face.bounds[2] * image_rect.width(),
            face.bounds[3] * image_rect.height(),
        )
        line_width = max(0.85, min(3.0, face_scale / 90.0))
        _paint_eye(
            painter,
            image_rect,
            eye,
            line_width,
            show_labels
            and not crowded
            and len(visible_eyes) <= 6
            and face_scale >= 125.0,
            face_scale,
        )
    painter.restore()


def _visible_evidence(
    photo: Photo,
) -> Tuple[Tuple[ObjectEvidence, ...], Tuple[FaceEvidence, ...], bool]:
    """Choose useful overlay evidence without turning crowded scenes into diagrams."""

    faces = tuple(photo.subject_regions)
    objects = tuple(
        item
        for item in photo.subject_objects
        if item.importance != "background"
        and not (faces and item.label == "person")
    )
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


def _selected_evidence(
    photo: Photo,
    visibility: AnalysisOverlayVisibility,
) -> Tuple[
    Tuple[ObjectEvidence, ...],
    Tuple[FaceEvidence, ...],
    Tuple[Tuple[FaceEvidence, EyeEvidence], ...],
    bool,
]:
    """Resolve category modes and per-item overrides for one photograph."""

    common_objects, common_faces, crowded = _visible_evidence(photo)
    objects = _resolve_items(
        tuple(photo.subject_objects),
        common_objects,
        visibility.objects,
        visibility.object_overrides,
    )
    faces = _resolve_items(
        tuple(photo.subject_regions),
        common_faces,
        visibility.faces,
        visibility.face_overrides,
    )

    all_eyes = tuple(
        (face_index, eye_index, face, eye)
        for face_index, face in enumerate(photo.subject_regions)
        for eye_index, eye in enumerate(face.eyes)
    )
    common_face_ids = {id(face) for face in common_faces}
    if visibility.eyes == EvidenceDisplayMode.ALL:
        selected_eye_keys = {
            (face_index, eye_index)
            for face_index, eye_index, _face, _eye in all_eyes
        }
    elif visibility.eyes == EvidenceDisplayMode.COMMON:
        selected_eye_keys = {
            (face_index, eye_index)
            for face_index, eye_index, face, _eye in all_eyes
            if id(face) in common_face_ids
        }
    else:
        selected_eye_keys = set()
    for face_index, eye_index, mode in visibility.eye_overrides:
        key = (face_index, eye_index)
        if mode == EvidenceItemMode.SHOW:
            selected_eye_keys.add(key)
        elif mode == EvidenceItemMode.HIDE:
            selected_eye_keys.discard(key)
    eyes = tuple(
        (face, eye)
        for face_index, eye_index, face, eye in all_eyes
        if (face_index, eye_index) in selected_eye_keys
    )
    return objects, faces, eyes, crowded


def _resolve_items(
    all_items: Tuple[ObjectEvidence, ...] | Tuple[FaceEvidence, ...],
    common_items: Tuple[ObjectEvidence, ...] | Tuple[FaceEvidence, ...],
    mode: EvidenceDisplayMode,
    overrides: Tuple[Tuple[int, EvidenceItemMode], ...],
) -> tuple:
    if mode == EvidenceDisplayMode.ALL:
        selected_indices = set(range(len(all_items)))
    elif mode == EvidenceDisplayMode.COMMON:
        common_ids = {id(item) for item in common_items}
        selected_indices = {
            index for index, item in enumerate(all_items) if id(item) in common_ids
        }
    else:
        selected_indices = set()
    for index, item_mode in overrides:
        if item_mode == EvidenceItemMode.SHOW:
            selected_indices.add(index)
        elif item_mode == EvidenceItemMode.HIDE:
            selected_indices.discard(index)
    return tuple(
        item for index, item in enumerate(all_items) if index in selected_indices
    )


def _has_selected_evidence(
    photo: Photo,
    visibility: AnalysisOverlayVisibility,
) -> bool:
    objects, faces, eyes, _crowded = _selected_evidence(photo, visibility)
    return bool(objects or faces or eyes)


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
