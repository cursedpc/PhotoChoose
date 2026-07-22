from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

APP_STYLESHEET = """
QWidget {
    color: #e7e9ed;
    font-size: 13px;
}

QMainWindow, QDialog {
    background: #141619;
}

QFrame#libraryRail {
    background: rgba(28, 31, 36, 238);
    border: 1px solid #30343b;
    border-radius: 14px;
}

QFrame#workspaceSurface, QFrame#inspectorSurface, QFrame#comparisonPane {
    background: rgba(31, 34, 39, 242);
    border: 1px solid #30343b;
    border-radius: 14px;
}

QDialog#fullscreenSurface { background: #0f1114; }

QLabel#fullscreenCanvas {
    background: #0b0d0f;
    border: 1px solid #20242a;
    border-radius: 10px;
}

QFrame#actionShelf {
    background: rgba(35, 39, 45, 236);
    border: 1px solid #373c44;
    border-radius: 12px;
}

QFrame#workspaceHeader, QFrame#workspaceNavigation {
    background: transparent;
    border: none;
}

QFrame#emptyState {
    background: #191b1f;
    border: 1px solid #292d33;
    border-radius: 11px;
}

QLabel#brandMark {
    color: #f4f5f7;
    font-size: 19px;
    font-weight: 700;
    letter-spacing: 0.4px;
}

QLabel#eyebrow, QLabel#mutedLabel, QLabel#projectPath, QLabel#inspectorHint {
    color: #9299a5;
}

QLabel#eyebrow {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}

QLabel#projectTitle {
    color: #f2f3f5;
    font-size: 22px;
    font-weight: 650;
}

QLabel#panelTitle {
    color: #f0f1f3;
    font-size: 15px;
    font-weight: 650;
}

QLabel#selectionTitle {
    color: #f0f1f3;
    font-size: 16px;
    font-weight: 650;
}

QLabel#statusKeep { color: #73c69a; font-weight: 650; }
QLabel#statusReview { color: #e0b36b; font-weight: 650; }
QLabel#statusReject { color: #e58484; font-weight: 650; }

QPushButton, QToolButton, QComboBox, QSpinBox {
    background: #292d33;
    border: 1px solid #3a3f47;
    border-radius: 8px;
    min-height: 30px;
    padding: 0 11px;
}

QPushButton:hover, QToolButton:hover, QComboBox:hover, QSpinBox:hover {
    background: #323740;
    border-color: #4b525e;
}

QPushButton:pressed, QToolButton:pressed {
    background: #24282e;
}

QPushButton:disabled, QToolButton:disabled, QComboBox:disabled, QSpinBox:disabled {
    color: #666c76;
    background: #22252a;
    border-color: #2d3036;
}

QPushButton[kind="primary"] {
    color: #f8fbff;
    background: #3d71c7;
    border-color: #4b7ed1;
    font-weight: 650;
    padding: 0 16px;
}

QPushButton[kind="primary"]:hover { background: #477dd5; }
QPushButton[kind="primary"]:pressed { background: #3566b7; }
QPushButton[kind="primary"]:disabled {
    color: #7d8ca5;
    background: #26344a;
    border-color: #303c50;
}

QPushButton[kind="quiet"], QToolButton[kind="quiet"] {
    background: transparent;
    border-color: transparent;
}

QPushButton[kind="quiet"]:hover, QToolButton[kind="quiet"]:hover {
    background: #2a2e34;
    border-color: #343941;
}

QPushButton[kind="keep"]:checked {
    color: #dff7e9;
    background: #285840;
    border-color: #37755a;
}

QPushButton[kind="review"]:checked {
    color: #fff0d1;
    background: #66502b;
    border-color: #886b3a;
}

QPushButton[kind="reject"]:checked {
    color: #ffe5e5;
    background: #693b3e;
    border-color: #8c4d52;
}

QPushButton[kind="best"]:checked {
    color: #17191c;
    background: #ddb15b;
    border-color: #e5c376;
    font-weight: 700;
}

QListWidget, QListView, QScrollArea {
    background: transparent;
    border: none;
    outline: none;
}

QPlainTextEdit {
    background: #191b1f;
    border: 1px solid #30343b;
    border-radius: 9px;
    padding: 9px;
    selection-background-color: #3a5279;
}

QListWidget::item {
    border-radius: 8px;
    min-height: 35px;
    padding: 0 10px;
    margin: 2px 0;
}

QListWidget::item:hover { background: #292d33; }
QListWidget::item:selected {
    color: #eef4ff;
    background: #354c70;
}

QListView#photoGrid {
    background: #191b1f;
    border: 1px solid #292d33;
    border-radius: 11px;
    padding: 8px;
}

QTabBar::tab {
    color: #9ba2ad;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    min-height: 34px;
    padding: 0 13px;
    margin-right: 4px;
}

QTabBar::tab:hover { color: #d3d7dd; }
QTabBar::tab:selected {
    color: #f1f3f6;
    border-bottom-color: #6f9fe8;
}

QMenuBar {
    background: #141619;
    border-bottom: 1px solid #262a30;
}

QMenuBar::item { padding: 5px 9px; background: transparent; }
QMenuBar::item:selected { background: #2b2f35; border-radius: 5px; }

QMenu {
    background: #24272c;
    border: 1px solid #3a3e45;
    border-radius: 9px;
    padding: 6px;
}

QMenu::item { padding: 7px 28px 7px 12px; border-radius: 5px; }
QMenu::item:selected { background: #3a5279; }
QMenu::separator { height: 1px; background: #3a3e45; margin: 5px 8px; }

QComboBox::drop-down { border: none; width: 24px; }
QComboBox QAbstractItemView {
    background: #24272c;
    border: 1px solid #3a3e45;
    selection-background-color: #3a5279;
    outline: none;
}

QSpinBox::up-button, QSpinBox::down-button { width: 18px; border: none; }

QSlider::groove:horizontal {
    height: 4px;
    background: #30343a;
    border-radius: 2px;
}

QSlider::sub-page:horizontal {
    background: #608bcf;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    width: 14px;
    margin: -5px 0;
    background: #d7dae0;
    border: 1px solid #737a85;
    border-radius: 7px;
}

QToolTip {
    color: #f2f3f5;
    background: #292d33;
    border: 1px solid #454a53;
    padding: 5px;
}

QSplitter::handle { background: transparent; width: 7px; }
QSplitter::handle:hover { background: #3b414a; }

QScrollBar:vertical {
    background: transparent;
    width: 10px;
    margin: 2px;
}
QScrollBar::handle:vertical {
    background: #444a54;
    border-radius: 4px;
    min-height: 28px;
}
QScrollBar::handle:vertical:hover { background: #5a626f; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }

QStatusBar {
    color: #8f96a1;
    background: #141619;
    border-top: 1px solid #24272c;
}

QStatusBar::item { border: none; }
"""


def apply_application_theme(app: QApplication) -> None:
    """Apply PhotoChoose's calm, consistent cross-platform studio theme."""

    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#141619"))
    palette.setColor(QPalette.WindowText, QColor("#e7e9ed"))
    palette.setColor(QPalette.Base, QColor("#202328"))
    palette.setColor(QPalette.AlternateBase, QColor("#25292f"))
    palette.setColor(QPalette.ToolTipBase, QColor("#292d33"))
    palette.setColor(QPalette.ToolTipText, QColor("#f2f3f5"))
    palette.setColor(QPalette.Text, QColor("#e7e9ed"))
    palette.setColor(QPalette.Button, QColor("#292d33"))
    palette.setColor(QPalette.ButtonText, QColor("#e7e9ed"))
    palette.setColor(QPalette.BrightText, QColor("#ffffff"))
    palette.setColor(QPalette.Highlight, QColor("#4f78b9"))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.PlaceholderText, QColor("#747b86"))
    app.setPalette(palette)
    app.setStyleSheet(APP_STYLESHEET)
