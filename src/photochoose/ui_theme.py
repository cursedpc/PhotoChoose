from __future__ import annotations

from PySide6.QtGui import QColor, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication

APP_STYLESHEET = """
QWidget {
    color: #dedfe2;
}

QMainWindow, QDialog {
    background: #191a1d;
}

QFrame#libraryRail {
    background: #202125;
    border: none;
    border-right: 1px solid #303136;
}

QFrame#workspaceSurface {
    background: #18191c;
    border: none;
}

QFrame#inspectorSurface {
    background: #202125;
    border: none;
    border-left: 1px solid #303136;
}

QFrame#comparisonPane {
    background: #1b1c1f;
    border: 1px solid #303136;
}

QDialog#fullscreenSurface { background: #101114; }

QScrollArea#fullscreenEvidenceControls,
QWidget#fullscreenEvidenceControlsBody {
    background: #151619;
    border: none;
}

QScrollArea#fullscreenEvidenceControls {
    border-left: 1px solid #303136;
}

QLabel#overlayControlsTitle {
    color: #eeeeef;
    font-size: 14px;
    font-weight: 600;
}

QFrame#overlayEvidenceSection {
    background: transparent;
    border: none;
    border-top: 1px solid #303136;
    padding-top: 7px;
}

QLabel#overlayCategoryTitle {
    color: #dedfe2;
    font-weight: 600;
}

QLabel#overlayItemLabel {
    color: #9fa2aa;
    font-size: 11px;
}

QScrollArea#fullscreenEvidenceControls QComboBox {
    padding: 0 18px 0 6px;
}

QComboBox#overlayCategoryMode { min-height: 0; }

QScrollArea#fullscreenEvidenceControls QComboBox::drop-down { width: 18px; }

QLabel#fullscreenCanvas {
    background: #0b0c0e;
    border: 1px solid #25262a;
}

QLabel#inspectorPreview {
    background: #141518;
    border: 1px solid #303136;
}

QWidget#fullscreenFilmstrip {
    background: #151619;
    border-top: 1px solid #303136;
}

QListView#fullscreenCarousel {
    background: transparent;
    border: none;
    outline: none;
}

QLabel#carouselPosition {
    color: #c9cbd0;
    font-size: 12px;
    font-weight: 600;
}

QFrame#workspaceHeader, QFrame#workspaceNavigation {
    background: transparent;
    border: none;
}

QFrame#emptyState {
    background: #16171a;
    border: none;
}

QLabel#brandMark {
    color: #f0f0f2;
    font-size: 17px;
    font-weight: 600;
}

QLabel#eyebrow {
    color: #b3b5ba;
    font-size: 12px;
    font-weight: 600;
}

QLabel#mutedLabel, QLabel#projectPath, QLabel#inspectorHint {
    color: #8d9098;
}

QLabel#projectTitle {
    color: #eeeeef;
    font-size: 19px;
    font-weight: 600;
}

QLabel#panelTitle, QLabel#selectionTitle {
    color: #eeeeef;
    font-size: 14px;
    font-weight: 600;
}

QLabel#decisionBadge {
    color: #d8dade;
    background: #303136;
    border: 1px solid #424349;
    border-radius: 3px;
    padding: 3px 8px;
    font-weight: 600;
}

QLabel#decisionBadge[state="keep"] {
    color: #e4f2e9;
    background: #315543;
    border-color: #47705a;
}

QLabel#decisionBadge[state="review"] {
    color: #f5ead6;
    background: #5b4b30;
    border-color: #77633e;
}

QLabel#decisionBadge[state="reject"] {
    color: #f4e1e1;
    background: #5b383b;
    border-color: #75484c;
}

QLabel#statusKeep { color: #72b58f; font-weight: 600; }
QLabel#statusReview { color: #d2a761; font-weight: 600; }
QLabel#statusReject { color: #d97a7a; font-weight: 600; }

QPushButton, QToolButton, QComboBox, QSpinBox {
    background: #292a2e;
    border: 1px solid #3a3b40;
    border-radius: 3px;
    min-height: 28px;
    padding: 0 10px;
}

QPushButton:hover, QToolButton:hover, QComboBox:hover, QSpinBox:hover {
    background: #323338;
    border-color: #4a4b51;
}

QPushButton:pressed, QToolButton:pressed {
    background: #242529;
}

QPushButton:disabled, QToolButton:disabled, QComboBox:disabled, QSpinBox:disabled {
    color: #62656c;
    background: #222327;
    border-color: #2c2d31;
}

QPushButton[kind="primary"] {
    color: #ffffff;
    background: #3568ae;
    border-color: #4779ba;
    font-weight: 600;
    padding: 0 15px;
}

QPushButton[kind="primary"]:hover { background: #3d72ba; }
QPushButton[kind="primary"]:pressed { background: #305e9d; }
QPushButton[kind="primary"]:disabled {
    color: #76849b;
    background: #263247;
    border-color: #30394a;
}

QPushButton[kind="quiet"], QToolButton[kind="quiet"] {
    background: transparent;
    border-color: transparent;
}

QPushButton[kind="quiet"]:hover, QToolButton[kind="quiet"]:hover {
    background: #292a2e;
    border-color: #35363b;
}

QPushButton[kind="keep"]:checked {
    color: #e4f2e9;
    background: #315543;
    border-color: #47705a;
}

QPushButton[kind="review"]:checked {
    color: #f5ead6;
    background: #5b4b30;
    border-color: #77633e;
}

QPushButton[kind="reject"]:checked {
    color: #f4e1e1;
    background: #5b383b;
    border-color: #75484c;
}

QPushButton[kind="best"]:checked {
    color: #1b1b1d;
    background: #cda552;
    border-color: #d9b76c;
    font-weight: 600;
}

QListWidget, QListView, QScrollArea {
    background: transparent;
    border: none;
    outline: none;
}

QPlainTextEdit {
    background: #151619;
    border: 1px solid #303136;
    padding: 8px;
    selection-background-color: #3b557a;
}

QListWidget::item {
    min-height: 34px;
    padding: 0 9px;
    margin: 1px 0;
}

QListWidget::item:hover { background: #292a2e; }
QListWidget::item:selected {
    color: #f0f2f5;
    background: #37445a;
}

QListWidget#projectList[dropActive="true"] {
    background: #242a33;
    border: 1px solid #5577a8;
}

QListView#photoGrid {
    background: #141518;
    border: none;
    border-top: 1px solid #292a2e;
    padding: 7px;
}

QTabBar::tab {
    color: #92959c;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    min-height: 32px;
    padding: 0 11px;
    margin-right: 2px;
}

QTabBar::tab:hover { color: #c8cacf; }
QTabBar::tab:selected {
    color: #eeeeef;
    border-bottom-color: #5f8ac5;
}

QToolButton#peopleFilterButton {
    color: #92959c;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    border-radius: 0;
    min-height: 36px;
    padding: 0 14px;
    font-size: 14px;
    font-weight: 600;
}

QToolButton#peopleFilterButton:hover { color: #c8cacf; background: transparent; }
QToolButton#peopleFilterButton:checked {
    color: #eeeeef;
    border-bottom-color: #5f8ac5;
}

QMenu#peopleMenu::item {
    min-height: 48px;
    padding: 5px 18px 5px 10px;
    font-size: 14px;
    font-weight: 600;
}

QMenu#peopleMenu::item:selected { background: #3b557a; }

QMenuBar {
    background: #191a1d;
    border-bottom: 1px solid #292a2e;
}

QMenuBar::item { padding: 5px 9px; background: transparent; }
QMenuBar::item:selected { background: #2b2c30; }

QMenu {
    background: #242529;
    border: 1px solid #3a3b40;
    padding: 4px;
}

QMenu::item { padding: 7px 28px 7px 11px; }
QMenu::item:selected { background: #3b557a; }
QMenu::separator { height: 1px; background: #3a3b40; margin: 4px 7px; }

QComboBox::drop-down { border: none; width: 23px; }
QComboBox QAbstractItemView {
    background: #242529;
    border: 1px solid #3a3b40;
    selection-background-color: #3b557a;
    outline: none;
}

QSpinBox::up-button, QSpinBox::down-button { width: 18px; border: none; }

QSlider::groove:horizontal {
    height: 3px;
    background: #333439;
}

QSlider::sub-page:horizontal { background: #567db3; }

QSlider::handle:horizontal {
    width: 12px;
    margin: -5px 0;
    background: #d4d5d8;
    border: 1px solid #6d7077;
    border-radius: 6px;
}

QToolTip {
    color: #eeeeef;
    background: #292a2e;
    border: 1px solid #45464c;
    padding: 5px;
}

QSplitter::handle { background: #303136; width: 1px; }
QSplitter::handle:hover { background: #526b90; }

QScrollBar:vertical {
    background: transparent;
    width: 9px;
    margin: 1px;
}
QScrollBar::handle:vertical {
    background: #44464d;
    border-radius: 3px;
    min-height: 28px;
}
QScrollBar::handle:vertical:hover { background: #585b64; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }

QStatusBar {
    color: #858891;
    background: #191a1d;
    border-top: 1px solid #292a2e;
}

QStatusBar::item { border: none; }
"""


def apply_application_theme(app: QApplication) -> None:
    """Apply PhotoChoose's integrated cross-platform studio theme."""

    app.setStyle("Fusion")
    app.setFont(QFontDatabase.systemFont(QFontDatabase.GeneralFont))
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#191a1d"))
    palette.setColor(QPalette.WindowText, QColor("#dedfe2"))
    palette.setColor(QPalette.Base, QColor("#18191c"))
    palette.setColor(QPalette.AlternateBase, QColor("#202125"))
    palette.setColor(QPalette.ToolTipBase, QColor("#292a2e"))
    palette.setColor(QPalette.ToolTipText, QColor("#eeeeef"))
    palette.setColor(QPalette.Text, QColor("#dedfe2"))
    palette.setColor(QPalette.Button, QColor("#292a2e"))
    palette.setColor(QPalette.ButtonText, QColor("#dedfe2"))
    palette.setColor(QPalette.BrightText, QColor("#ffffff"))
    palette.setColor(QPalette.Highlight, QColor("#4b6f9f"))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.PlaceholderText, QColor("#747780"))
    app.setPalette(palette)
    app.setStyleSheet(APP_STYLESHEET)
