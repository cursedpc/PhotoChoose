from __future__ import annotations

from PySide6.QtCore import QRect, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPalette, QPen
from PySide6.QtWidgets import QStyle, QStyledItemDelegate, QStyleOptionViewItem

from photochoose.analysis_overlay import paint_analysis_evidence
from photochoose.models import Decision
from photochoose.thumbnail_model import PhotoListModel

GRID_SIZE = QSize(236, 216)


class PhotoDelegate(QStyledItemDelegate):
    """Draw consistent thumbnail labels instead of relying on platform styles."""

    def sizeHint(self, option: QStyleOptionViewItem, index) -> QSize:
        return GRID_SIZE

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        painter.save()
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        card = option.rect.adjusted(4, 4, -4, -4)
        selected = bool(option.state & QStyle.State_Selected)
        hovered = bool(option.state & QStyle.State_MouseOver)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#292d33") if hovered else QColor("#23262b"))
        painter.drawRect(card)

        image_box = QRect(card.left() + 8, card.top() + 8, card.width() - 16, 164)
        painter.setBrush(QColor("#111316"))
        painter.drawRect(image_box)
        pixmap = index.data(Qt.DecorationRole)
        if pixmap is not None and not pixmap.isNull():
            scaled = pixmap.scaled(image_box.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = image_box.left() + (image_box.width() - scaled.width()) // 2
            y = image_box.top() + (image_box.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
            if self.parent().property("analysisOverlayEnabled"):
                photo = index.data(PhotoListModel.AnalysisEvidenceRole)
                if photo is not None:
                    paint_analysis_evidence(
                        painter,
                        QRectF(float(x), float(y), float(scaled.width()), float(scaled.height())),
                        photo,
                        show_labels=False,
                        highlighted_face_indices=index.data(
                            PhotoListModel.HighlightedFacesRole
                        )
                        or (),
                    )

        decision = index.data(PhotoListModel.DecisionRole)
        if decision != Decision.UNREVIEWED.value:
            self._paint_decision_badge(painter, image_box, decision)
        asset_badge = index.data(PhotoListModel.AssetRole)
        if asset_badge:
            self._paint_asset_badge(painter, image_box, asset_badge)
        group_id = index.data(PhotoListModel.GroupRole)
        if group_id:
            self._paint_group_badge(painter, image_box, group_id)
        if index.data(PhotoListModel.BestRole):
            self._paint_best_badge(painter, image_box)

        painter.setPen(option.palette.color(QPalette.Text))
        label_rect = QRect(card.left() + 10, image_box.bottom() + 8, card.width() - 20, 25)
        metrics = QFontMetrics(option.font)
        filename = metrics.elidedText(
            index.data(Qt.DisplayRole), Qt.ElideMiddle, label_rect.width()
        )
        painter.drawText(label_rect, Qt.AlignLeft | Qt.AlignVCenter, filename)

        if selected:
            painter.setPen(QPen(QColor("#75a7ef"), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(card.adjusted(1, 1, -1, -1))
        painter.restore()

    @staticmethod
    def _paint_decision_badge(painter: QPainter, image_box: QRect, decision: str) -> None:
        colors = {
            Decision.KEEP.value: QColor("#247a4a"),
            Decision.REVIEW.value: QColor("#a86f16"),
            Decision.REJECT.value: QColor("#b43a3a"),
        }
        text = decision.upper()
        width = painter.fontMetrics().horizontalAdvance(text) + 16
        badge = QRect(image_box.left() + 8, image_box.bottom() - 29, width, 21)
        painter.setPen(Qt.NoPen)
        painter.setBrush(colors[decision])
        painter.drawRect(badge)
        painter.setPen(QColor("white"))
        painter.drawText(badge, Qt.AlignCenter, text)

    @staticmethod
    def _paint_best_badge(painter: QPainter, image_box: QRect) -> None:
        badge = QRect(image_box.right() - 54, image_box.top() + 8, 46, 21)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#d49a24"))
        painter.drawRect(badge)
        painter.setPen(QColor("#171717"))
        painter.drawText(badge, Qt.AlignCenter, "BEST")

    @staticmethod
    def _paint_group_badge(painter: QPainter, image_box: QRect, group_id: str) -> None:
        width = painter.fontMetrics().horizontalAdvance(group_id) + 14
        badge = QRect(image_box.left() + 8, image_box.top() + 8, width, 21)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#316f9f") if group_id.startswith("B") else QColor("#59616d"))
        painter.drawRect(badge)
        painter.setPen(QColor("white"))
        painter.drawText(badge, Qt.AlignCenter, group_id)

    @staticmethod
    def _paint_asset_badge(painter: QPainter, image_box: QRect, text: str) -> None:
        width = painter.fontMetrics().horizontalAdvance(text) + 14
        badge = QRect(image_box.right() - width - 8, image_box.bottom() - 30, width, 22)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#43386f"))
        painter.drawRect(badge)
        painter.setPen(QColor("white"))
        painter.drawText(badge, Qt.AlignCenter, text)
