from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QImageReader, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from photochoose.models import Photo


class SelectionInspector(QFrame):
    """Quiet contextual controls and expandable evidence for the current selection."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._preview_image = QImage()
        self.setObjectName("inspectorSurface")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 14)
        layout.setSpacing(10)
        layout.setAlignment(Qt.AlignTop)

        header = QHBoxLayout()
        panel_title = QLabel("Inspector")
        panel_title.setObjectName("eyebrow")
        panel_title.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        header.addWidget(panel_title)
        header.addStretch(1)
        self.visible_count_label = QLabel("0 photos")
        self.visible_count_label.setObjectName("mutedLabel")
        self.visible_count_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        header.addWidget(self.visible_count_label)
        layout.addLayout(header)

        self.preview_label = QLabel()
        self.preview_label.setObjectName("inspectorPreview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setFixedHeight(168)
        self.preview_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.preview_label.hide()
        layout.addWidget(self.preview_label)

        self.selection_title = QLabel("Nothing selected")
        self.selection_title.setObjectName("selectionTitle")
        self.selection_title.setWordWrap(True)
        self.selection_title.setMaximumHeight(44)
        self.selection_title.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(self.selection_title)
        self.selection_hint = QLabel("Choose a photo to review or classify it.")
        self.selection_hint.setObjectName("inspectorHint")
        self.selection_hint.setWordWrap(True)
        self.selection_hint.setMaximumHeight(44)
        self.selection_hint.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(self.selection_hint)

        layout.addSpacing(4)
        decision_label = QLabel("Decision")
        decision_label.setObjectName("eyebrow")
        decision_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(decision_label)

        decisions = QHBoxLayout()
        decisions.setSpacing(6)
        self.keep_button = QPushButton("Keep")
        self.keep_button.setProperty("kind", "keep")
        self.review_button = QPushButton("Review")
        self.review_button.setProperty("kind", "review")
        self.reject_button = QPushButton("Reject")
        self.reject_button.setProperty("kind", "reject")
        for button, shortcut in (
            (self.keep_button, "K"),
            (self.review_button, "R"),
            (self.reject_button, "X"),
        ):
            button.setCheckable(True)
            button.setToolTip("Keyboard: {0}".format(shortcut))
            decisions.addWidget(button)
        layout.addLayout(decisions)

        self.best_button = QPushButton("Mark Best")
        self.best_button.setProperty("kind", "best")
        self.best_button.setCheckable(True)
        self.best_button.setToolTip("Keyboard: B")
        layout.addWidget(self.best_button)

        self.clear_button = QPushButton("Clear decision")
        self.clear_button.setProperty("kind", "quiet")
        self.clear_button.setToolTip("Return the selection to Unreviewed · Keyboard: U")
        layout.addWidget(self.clear_button)

        open_row = QHBoxLayout()
        open_row.setSpacing(6)
        self.open_button = QPushButton("Open")
        self.compare_button = QPushButton("Compare")
        self.open_button.setToolTip("Open the selected photo fullscreen")
        self.compare_button.setToolTip("Compare this photo with its similar group")
        open_row.addWidget(self.open_button)
        open_row.addWidget(self.compare_button)
        layout.addLayout(open_row)

        view_label = QLabel("Display")
        view_label.setObjectName("eyebrow")
        view_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(view_label)
        self.analysis_overlay_button = QPushButton("Analysis overlay")
        self.analysis_overlay_button.setCheckable(True)
        self.analysis_overlay_button.setToolTip(
            "Show stored subject and eye evidence on previews · Keyboard: O"
        )
        layout.addWidget(self.analysis_overlay_button)

        self.toggle_button = QToolButton()
        self.toggle_button.setText("Analysis details")
        self.toggle_button.setProperty("kind", "quiet")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.RightArrow)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.toggle_button.toggled.connect(self.set_expanded)
        layout.addWidget(self.toggle_button)

        self.contents = QScrollArea()
        self.contents.setWidgetResizable(True)
        self.contents.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.contents.setMinimumHeight(260)
        details_body = QWidget()
        contents_layout = QVBoxLayout(details_body)
        contents_layout.setContentsMargins(2, 2, 8, 2)
        contents_layout.setSpacing(7)

        self.details_label = QLabel("Select a photo to inspect the analysis and classify it.")
        self.details_label.setWordWrap(True)
        self.details_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.details_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        contents_layout.addWidget(self.details_label)
        contents_layout.addStretch(1)
        self.contents.setWidget(details_body)
        layout.addWidget(self.contents, 1)
        self.set_expanded(False)
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
            self._clear_preview()
            self.selection_title.setText("Nothing selected")
            self.selection_hint.setText("Choose a photo to review or classify it.")
            self._set_checked_state(None, False)
            self.details_label.setText(
                "Select a photo to inspect its focus, exposure, faces, eyes, grouping, "
                "and automatic reasons."
            )
            return

        if len(photos) > 1:
            self._clear_preview()
            self.selection_title.setText("{0} photos selected".format(len(photos)))
            decisions = Counter(photo.decision.value for photo in photos)
            decision_summary = " · ".join(
                "{0} {1}".format(decisions[name], name.title())
                for name in ("keep", "review", "reject", "unreviewed")
                if decisions[name]
            )
            best_count = sum(photo.is_best for photo in photos)
            self.selection_hint.setText(
                "{0} · {1} Best · actions apply to all".format(decision_summary, best_count)
            )
            self._set_checked_state(None, False)
            self.details_label.setText(
                "{0} photos selected · {1} · {2} Best\n"
                "Actions below apply to every selected photo.".format(
                    len(photos), decision_summary, best_count
                )
            )
            return

        photo = photos[0]
        self._load_preview(photo.display_path)
        self.selection_title.setText(photo.filename)
        self.selection_hint.setText(
            "{0}{1}".format(
                photo.decision.value.title(),
                " · Best" if photo.is_best else "",
            )
        )
        self._set_checked_state(photo.decision.value, photo.is_best)
        metrics = [
            "{0}    ·    {1} ({2})    ·    Best: {3} ({4})".format(
                photo.relative_path,
                photo.decision.value.title(),
                photo.decision_source,
                "yes" if photo.is_best else "no",
                photo.best_source,
            ),
            "Files: {0}".format(photo.file_summary),
            "Analysis: {0}    ·    Profile: {1} / {2}    ·    Focus: {3}    ·    "
            "Brightness: {4}".format(
                photo.analysis_status,
                (photo.analysis_profile or "—").title(),
                photo.analysis_version or "—",
                self._number(photo.focus_score),
                self._percentage(photo.brightness),
            ),
            "Faces: {0}    ·    Eyes O/C/?/Obs: {1}/{2}/{3}/{4}    ·    "
            "Face focus: {5}    ·    Model: {6}".format(
                photo.face_count,
                photo.eye_count,
                photo.closed_eye_count,
                photo.uncertain_eye_count,
                photo.obscured_eye_count,
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
        if photo.eye_visibility_reasons:
            metrics.append(
                "Eye visibility: " + " · ".join(photo.eye_visibility_reasons)
            )
        if photo.auto_reasons:
            metrics.append("Reasons: " + " · ".join(photo.auto_reasons))
        self.details_label.setText("\n".join(metrics))

    def _load_preview(self, path: str) -> None:
        reader = QImageReader(path)
        reader.setAutoTransform(True)
        source_size = reader.size()
        if source_size.isValid():
            reader.setScaledSize(source_size.scaled(900, 600, Qt.KeepAspectRatio))
        self._preview_image = reader.read()
        if self._preview_image.isNull():
            self.preview_label.setText("Preview unavailable")
        else:
            self.preview_label.setText("")
        self.preview_label.show()
        QTimer.singleShot(0, self._render_preview)

    def _clear_preview(self) -> None:
        self._preview_image = QImage()
        self.preview_label.clear()
        self.preview_label.hide()

    def _render_preview(self) -> None:
        if self._preview_image.isNull() or not self.preview_label.isVisible():
            return
        target = self.preview_label.contentsRect().size()
        target.setWidth(max(1, target.width() - 2))
        target.setHeight(max(1, target.height() - 2))
        image = self._preview_image.scaled(
            target, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.preview_label.setPixmap(QPixmap.fromImage(image))

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._render_preview()

    def _set_checked_state(self, decision: Optional[str], is_best: bool) -> None:
        self.keep_button.setChecked(decision == "keep")
        self.review_button.setChecked(decision == "review")
        self.reject_button.setChecked(decision == "reject")
        self.best_button.setChecked(is_best)
        self.best_button.setText("Best" if is_best else "Mark Best")

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
