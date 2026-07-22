from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from photochoose.models import Photo


class SelectionInspector(QFrame):
    """Collapsible summary and action panel for the current selection."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("selectionInspector")
        self.setFrameShape(QFrame.StyledPanel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 7, 10, 9)
        layout.setSpacing(7)

        header = QHBoxLayout()
        self.toggle_button = QToolButton()
        self.toggle_button.setText("Photo inspector")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.DownArrow)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(True)
        self.toggle_button.setStyleSheet("font-weight: 600;")
        self.toggle_button.toggled.connect(self.set_expanded)
        header.addWidget(self.toggle_button)
        header.addStretch(1)
        self.visible_count_label = QLabel("0 photos")
        self.visible_count_label.setStyleSheet("color: #808792;")
        header.addWidget(self.visible_count_label)
        layout.addLayout(header)

        self.contents = QWidget()
        contents_layout = QVBoxLayout(self.contents)
        contents_layout.setContentsMargins(2, 0, 2, 0)
        contents_layout.setSpacing(7)

        self.details_label = QLabel("Select a photo to inspect the analysis and classify it.")
        self.details_label.setWordWrap(True)
        self.details_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        contents_layout.addWidget(self.details_label)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        self.open_button = QPushButton("Open fullscreen")
        self.compare_button = QPushButton("Compare group")
        self.keep_button = QPushButton("Keep  K")
        self.review_button = QPushButton("Review  R")
        self.reject_button = QPushButton("Reject  X")
        self.clear_button = QPushButton("Clear  U")
        self.best_button = QPushButton("Toggle Best  B")
        self.analysis_overlay_button = QPushButton("Analysis overlay  O")
        self.analysis_overlay_button.setCheckable(True)
        self.analysis_overlay_button.setToolTip(
            "Show detected face boxes and eye-state evidence on previews"
        )
        for button in (
            self.open_button,
            self.compare_button,
            self.keep_button,
            self.review_button,
            self.reject_button,
            self.clear_button,
            self.best_button,
            self.analysis_overlay_button,
        ):
            actions.addWidget(button)
        actions.addStretch(1)
        contents_layout.addLayout(actions)
        layout.addWidget(self.contents)
        self.set_photos([], 0)

    def set_expanded(self, expanded: bool) -> None:
        self.contents.setVisible(expanded)
        self.toggle_button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)

    def set_photos(self, photos: Sequence[Photo], visible_count: int) -> None:
        self.visible_count_label.setText("{0} photos visible".format(visible_count))
        enabled = bool(photos)
        for button in (
            self.open_button,
            self.compare_button,
            self.keep_button,
            self.review_button,
            self.reject_button,
            self.clear_button,
            self.best_button,
        ):
            button.setEnabled(enabled)
        self.analysis_overlay_button.setEnabled(visible_count > 0)
        self.compare_button.setEnabled(
            len(photos) == 1 and photos[0].similarity_group is not None
        )

        if not photos:
            self.details_label.setMinimumHeight(24)
            self.contents.setMinimumHeight(62)
            self.toggle_button.setText("Photo inspector · none selected")
            self.details_label.setText(
                "Select a photo to inspect its focus, exposure, faces, eyes, grouping, "
                "and automatic reasons."
            )
            return

        self.toggle_button.setText(
            "Photo inspector · {0} selected".format(len(photos))
        )
        if len(photos) > 1:
            self.details_label.setMinimumHeight(40)
            self.contents.setMinimumHeight(80)
            decisions = Counter(photo.decision.value for photo in photos)
            decision_summary = " · ".join(
                "{0} {1}".format(decisions[name], name.title())
                for name in ("keep", "review", "reject", "unreviewed")
                if decisions[name]
            )
            best_count = sum(photo.is_best for photo in photos)
            self.details_label.setText(
                "{0} photos selected · {1} · {2} Best\n"
                "Actions below apply to every selected photo.".format(
                    len(photos), decision_summary, best_count
                )
            )
            return

        photo = photos[0]
        self.details_label.setMinimumHeight(64)
        self.contents.setMinimumHeight(110)
        metrics = [
            "{0}    ·    {1} ({2})    ·    Best: {3} ({4})".format(
                photo.relative_path,
                photo.decision.value.title(),
                photo.decision_source,
                "yes" if photo.is_best else "no",
                photo.best_source,
            ),
            "Files: {0}".format(photo.file_summary),
            "Analysis: {0}    ·    Focus: {1}    ·    Brightness: {2}".format(
                photo.analysis_status,
                self._number(photo.focus_score),
                self._percentage(photo.brightness),
            ),
            "Faces: {0}    ·    Eyes O/C/?: {1}/{2}/{3}    ·    Face focus: {4}    ·    "
            "Model: {5}".format(
                photo.face_count,
                photo.eye_count,
                photo.closed_eye_count,
                photo.uncertain_eye_count,
                self._number(photo.subject_focus_score),
                photo.subject_model,
            ),
            "Face confidence: {0}    ·    Eye-open score: {1}    ·    Eye focus: {2}    ·    "
            "Primary faces: {3}    ·    Group: {4}    ·    Rank: {5}".format(
                self._percentage(photo.face_confidence),
                self._percentage(photo.eye_open_score),
                self._number(photo.eye_focus_score),
                photo.important_face_count or "—",
                photo.similarity_group or "—",
                self._number(photo.best_score),
            ),
            "Objects: {0}    ·    Primary objects: {1}    ·    Subject focus: {2}    ·    "
            "Object model: {3}".format(
                photo.object_count,
                photo.primary_object_count or "—",
                self._number(photo.primary_subject_focus_score),
                photo.object_model,
            ),
        ]
        metadata = self._metadata_summary(photo)
        if metadata:
            metrics.append(metadata)
        if photo.group_reason:
            metrics.append(photo.group_reason)
        if photo.auto_reasons:
            metrics.append("Reasons: " + " · ".join(photo.auto_reasons))
        details_height = max(64, len(metrics) * 18 + 6)
        self.details_label.setMinimumHeight(details_height)
        self.contents.setMinimumHeight(details_height + 46)
        self.details_label.setText("\n".join(metrics))

    @staticmethod
    def _number(value: Optional[float]) -> str:
        return "—" if value is None else "{0:.1f}".format(value)

    @staticmethod
    def _percentage(value: Optional[float]) -> str:
        return "—" if value is None else "{0:.1f}%".format(value * 100.0)

    @staticmethod
    def _metadata_summary(photo: Photo) -> str:
        parts = []
        if photo.captured_at:
            parts.append("Captured: " + photo.captured_at.replace("T", " "))
        camera = " ".join(
            value for value in (photo.camera_make, photo.camera_model) if value
        )
        if camera:
            parts.append("Camera: " + camera)
        if photo.lens_model:
            parts.append("Lens: " + photo.lens_model)
        if photo.focal_length is not None:
            parts.append("{0:g}mm".format(photo.focal_length))
        if photo.aperture is not None:
            parts.append("f/{0:g}".format(photo.aperture))
        if photo.exposure_seconds is not None:
            if 0 < photo.exposure_seconds < 1:
                parts.append("1/{0:d}s".format(round(1.0 / photo.exposure_seconds)))
            else:
                parts.append("{0:g}s".format(photo.exposure_seconds))
        if photo.iso is not None:
            parts.append("ISO {0}".format(photo.iso))
        return "    ·    ".join(parts)
