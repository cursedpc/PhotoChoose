from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from photochoose.calibration_report import CalibrationReport


class CalibrationReportDialog(QDialog):
    """Read-only view of private, local calibration evidence."""

    def __init__(
        self,
        report: CalibrationReport,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.report = report
        self.setWindowTitle("PhotoChoose calibration report")
        self.setObjectName("calibrationDialog")
        self.resize(820, 680)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 16)
        layout.setSpacing(9)
        heading = QLabel("Calibration report · {0}".format(report.project_name))
        heading.setObjectName("projectTitle")
        layout.addWidget(heading)

        notice = QLabel(
            "Local evidence only · suggestions are provisional · profile thresholds are "
            "never changed automatically"
        )
        notice.setWordWrap(True)
        notice.setObjectName("mutedLabel")
        layout.addWidget(notice)

        self.report_text = QPlainTextEdit(report.to_text())
        self.report_text.setReadOnly(True)
        self.report_text.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        layout.addWidget(self.report_text, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
