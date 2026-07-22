from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import (
    QItemSelectionModel,
    QObject,
    QPoint,
    QRunnable,
    Qt,
    QThreadPool,
    Signal,
    Slot,
)
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStackedLayout,
    QTabBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from photochoose.analysis_profiles import ANALYSIS_PROFILES, ANALYSIS_VERSION
from photochoose.analyzer import BaselineAnalyzer
from photochoose.calibration_dialog import CalibrationReportDialog
from photochoose.calibration_report import build_calibration_report
from photochoose.comparison_viewer import GroupComparisonDialog
from photochoose.database import PhotoDatabase
from photochoose.exporter import ProjectExporter
from photochoose.fullscreen_viewer import FullscreenPhotoDialog
from photochoose.models import Decision, PhotoView, Project
from photochoose.photo_delegate import GRID_SIZE, PhotoDelegate
from photochoose.selection_inspector import SelectionInspector
from photochoose.thumbnail_model import THUMBNAIL_SIZE, PhotoListModel

VIEWS = [
    PhotoView.ALL,
    PhotoView.BEST,
    PhotoView.KEEP,
    PhotoView.REVIEW,
    PhotoView.REJECT,
]


class _ScanSignals(QObject):
    finished = Signal(int, int)
    failed = Signal(int, str)


class _ScanTask(QRunnable):
    def __init__(self, database: PhotoDatabase, project_id: int) -> None:
        super().__init__()
        self.database = database
        self.project_id = project_id
        self.signals = _ScanSignals()

    def run(self) -> None:
        try:
            count = self.database.scan_project(self.project_id)
        except Exception as exc:  # displayed to the user at the UI boundary
            self.signals.failed.emit(self.project_id, str(exc))
            return
        self.signals.finished.emit(self.project_id, count)


class _AnalysisSignals(QObject):
    progress = Signal(int, int, int)
    finished = Signal(int, object)
    failed = Signal(int, str)


class _AnalysisTask(QRunnable):
    def __init__(
        self, database: PhotoDatabase, project_id: int, analysis_profile: str
    ) -> None:
        super().__init__()
        self.database = database
        self.project_id = project_id
        self.analysis_profile = analysis_profile
        self.signals = _AnalysisSignals()

    def run(self) -> None:
        analyzer = BaselineAnalyzer(self.database, self.analysis_profile)
        try:
            totals = analyzer.analyze_project(
                self.project_id,
                lambda complete, total: self.signals.progress.emit(
                    self.project_id, complete, total
                ),
            )
        except Exception as exc:  # displayed to the user at the UI boundary
            self.signals.failed.emit(self.project_id, str(exc))
            return
        self.signals.finished.emit(self.project_id, totals)


class _ExportSignals(QObject):
    progress = Signal(int, int, int)
    finished = Signal(int, object)
    failed = Signal(int, str)


class _ExportTask(QRunnable):
    def __init__(self, database: PhotoDatabase, project_id: int, destination: Path) -> None:
        super().__init__()
        self.database = database
        self.project_id = project_id
        self.destination = destination
        self.signals = _ExportSignals()

    def run(self) -> None:
        exporter = ProjectExporter(self.database)
        try:
            result = exporter.export_keep_and_best(
                self.project_id,
                self.destination,
                lambda complete, total: self.signals.progress.emit(
                    self.project_id, complete, total
                ),
            )
        except Exception as exc:  # displayed to the user at the UI boundary
            self.signals.failed.emit(self.project_id, str(exc))
            return
        self.signals.finished.emit(self.project_id, result)


class MainWindow(QMainWindow):
    def __init__(self, database: PhotoDatabase) -> None:
        super().__init__()
        self.database = database
        self.current_project: Optional[Project] = None
        self.current_view = PhotoView.ALL
        self.thread_pool = QThreadPool.globalInstance()
        self.busy_projects = set()
        self.analyze_after_scan = set()
        self.photo_model = PhotoListModel(database, self)
        self.fullscreen_dialog: Optional[FullscreenPhotoDialog] = None
        self.comparison_dialog: Optional[GroupComparisonDialog] = None
        self.calibration_dialog: Optional[CalibrationReportDialog] = None

        self.setWindowTitle("PhotoChoose")
        self.setObjectName("mainWindow")
        self.setMinimumSize(1080, 700)
        self.resize(1440, 900)
        self._create_project_actions()
        self._build_ui()
        self._build_menus()
        self._install_shortcuts()
        self._load_projects()

    def _create_project_actions(self) -> None:
        self.add_project_action = QAction("Add shoot folder…", self)
        self.add_project_action.triggered.connect(self._choose_project_folder)
        self.rescan_button = QAction("Rescan folder", self)
        self.rescan_button.triggered.connect(self._rescan_current_project)
        self.analyze_action = QAction("Analyze and separate", self)
        self.analyze_action.triggered.connect(self._analyze_current_project)
        self.calibration_button = QAction("Calibration report…", self)
        self.calibration_button.triggered.connect(self._open_calibration_report)
        self.export_button = QAction("Export Keep + Best…", self)
        self.export_button.triggered.connect(self._choose_export_folder)
        self.close_project_button = QAction("Close project", self)
        self.close_project_button.triggered.connect(self._close_current_project)
        for action in (
            self.rescan_button,
            self.analyze_action,
            self.calibration_button,
            self.export_button,
            self.close_project_button,
        ):
            action.setEnabled(False)

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)
        splitter.addWidget(self._build_project_sidebar())
        splitter.addWidget(self._build_photo_workspace())
        self.inspector = self._build_selection_inspector()
        splitter.addWidget(self.inspector)
        splitter.setSizes([220, 900, 310])
        splitter.setStretchFactor(1, 1)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setCollapsible(2, False)
        root_layout.addWidget(splitter)
        self.setCentralWidget(root)
        self.statusBar().showMessage("Add a shoot folder to begin")

    def _build_project_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setObjectName("libraryRail")
        sidebar.setMinimumWidth(190)
        sidebar.setMaximumWidth(270)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(14, 16, 14, 14)
        layout.setSpacing(10)

        brand = QLabel("PhotoChoose")
        brand.setObjectName("brandMark")
        layout.addWidget(brand)
        layout.addSpacing(12)

        title = QLabel("Folders")
        title.setObjectName("eyebrow")
        layout.addWidget(title)

        self.project_list = QListWidget()
        self.project_list.setObjectName("projectList")
        self.project_list.setSpacing(2)
        self.project_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.project_list.customContextMenuRequested.connect(
            self._show_project_context_menu
        )
        self.project_list.currentItemChanged.connect(self._project_changed)
        layout.addWidget(self.project_list, 1)

        layout.addSpacing(6)
        add_button = QPushButton("Add shoot…")
        add_button.setObjectName("addShootButton")
        add_button.setProperty("kind", "quiet")
        add_button.setToolTip("Add a folder as a new private photo project")
        add_button.clicked.connect(self._choose_project_folder)
        layout.addWidget(add_button)
        return sidebar

    def _build_photo_workspace(self) -> QWidget:
        workspace = QFrame()
        workspace.setObjectName("workspaceSurface")
        layout = QVBoxLayout(workspace)
        layout.setContentsMargins(16, 12, 16, 0)
        layout.setSpacing(10)

        header = QFrame()
        header.setObjectName("workspaceHeader")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(3)
        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        title_block = QVBoxLayout()
        title_block.setSpacing(2)
        self.project_title = QLabel("No project selected")
        self.project_title.setObjectName("projectTitle")
        title_block.addWidget(self.project_title)
        self.project_path = QLabel("Select a project from the library")
        self.project_path.setObjectName("projectPath")
        self.project_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        title_block.addWidget(self.project_path)
        title_row.addLayout(title_block, 1)

        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.setProperty("kind", "primary")
        self.analyze_button.setToolTip(
            "Refresh metadata, then classify this project entirely on this computer"
        )
        self.analyze_button.clicked.connect(self._analyze_current_project)
        self.analyze_button.setEnabled(False)
        title_row.addWidget(self.analyze_button)

        self.project_options_button = QToolButton()
        self.project_options_button.setText("Settings")
        self.project_options_button.setPopupMode(QToolButton.InstantPopup)
        self.project_options_button.setEnabled(False)
        self.project_options_menu = QMenu(self.project_options_button)
        self.project_options_menu.addAction(self._build_project_settings_action())
        self.project_options_button.setMenu(self.project_options_menu)
        title_row.addWidget(self.project_options_button)
        header_layout.addLayout(title_row)

        meta_row = QHBoxLayout()
        meta_row.setContentsMargins(0, 5, 0, 0)
        self.analysis_profile_status = QLabel("Safe profile · ready")
        self.analysis_profile_status.setObjectName("mutedLabel")
        meta_row.addWidget(self.analysis_profile_status)
        meta_row.addStretch(1)
        header_layout.addLayout(meta_row)
        layout.addWidget(header)

        navigation = QFrame()
        navigation.setObjectName("workspaceNavigation")
        navigation_layout = QHBoxLayout(navigation)
        navigation_layout.setContentsMargins(0, 6, 0, 0)
        navigation_layout.setSpacing(10)
        self.view_tabs = QTabBar()
        self.view_tabs.setExpanding(False)
        self.view_tabs.setDocumentMode(True)
        self.view_tabs.setDrawBase(False)
        for view in VIEWS:
            self.view_tabs.addTab(view.value.title())
        self.view_tabs.currentChanged.connect(self._view_changed)
        navigation_layout.addWidget(self.view_tabs)
        navigation_layout.addStretch(1)
        self.count_label = QLabel("0 photos")
        self.count_label.setObjectName("mutedLabel")
        navigation_layout.addWidget(self.count_label)
        layout.addWidget(navigation)

        self.photo_grid = QListView()
        self.photo_grid.setObjectName("photoGrid")
        self.photo_grid.setModel(self.photo_model)
        self.photo_grid.setItemDelegate(PhotoDelegate(self.photo_grid))
        self.photo_grid.setViewMode(QListView.IconMode)
        self.photo_grid.setResizeMode(QListView.Adjust)
        self.photo_grid.setMovement(QListView.Static)
        self.photo_grid.setSelectionMode(QListView.ExtendedSelection)
        self.photo_grid.setUniformItemSizes(True)
        self.photo_grid.setWordWrap(True)
        self.photo_grid.setIconSize(THUMBNAIL_SIZE)
        self.photo_grid.setGridSize(GRID_SIZE)
        self.photo_grid.setSpacing(7)
        self.photo_grid.setContextMenuPolicy(Qt.CustomContextMenu)
        self.photo_grid.customContextMenuRequested.connect(self._show_photo_context_menu)
        self.photo_grid.selectionModel().selectionChanged.connect(self._selection_changed)
        self.photo_grid.doubleClicked.connect(self._open_fullscreen)

        canvas = QWidget()
        self.canvas_stack = QStackedLayout(canvas)
        self.canvas_stack.setContentsMargins(0, 0, 0, 0)
        empty_state = QFrame()
        empty_state.setObjectName("emptyState")
        empty_layout = QVBoxLayout(empty_state)
        empty_layout.setContentsMargins(30, 30, 30, 30)
        empty_layout.addStretch(1)
        self.empty_title = QLabel("Your workspace is ready")
        self.empty_title.setObjectName("projectTitle")
        self.empty_title.setAlignment(Qt.AlignCenter)
        empty_layout.addWidget(self.empty_title)
        self.empty_hint = QLabel(
            "Add a shoot or choose one from the library. Originals always stay untouched."
        )
        self.empty_hint.setObjectName("mutedLabel")
        self.empty_hint.setAlignment(Qt.AlignCenter)
        self.empty_hint.setWordWrap(True)
        empty_layout.addWidget(self.empty_hint)
        empty_layout.addStretch(1)
        self.canvas_stack.addWidget(empty_state)
        self.canvas_stack.addWidget(self.photo_grid)
        self.canvas_stack.setCurrentIndex(0)
        layout.addWidget(canvas, 1)
        return workspace

    def _build_project_settings_action(self) -> QWidgetAction:
        settings = QWidget()
        settings.setMinimumWidth(270)
        layout = QVBoxLayout(settings)
        layout.setContentsMargins(12, 9, 12, 10)
        layout.setSpacing(7)

        label = QLabel("Project settings")
        label.setObjectName("eyebrow")
        layout.addWidget(label)

        profile_row = QHBoxLayout()
        profile_row.addWidget(QLabel("Analysis profile"))
        self.analysis_profile_combo = QComboBox()
        for profile in ANALYSIS_PROFILES:
            self.analysis_profile_combo.addItem(profile.label, profile.key)
            self.analysis_profile_combo.setItemData(
                self.analysis_profile_combo.count() - 1,
                profile.description,
                Qt.ToolTipRole,
            )
        self.analysis_profile_combo.setEnabled(False)
        self.analysis_profile_combo.currentIndexChanged.connect(
            self._analysis_profile_changed
        )
        profile_row.addWidget(self.analysis_profile_combo, 1)
        layout.addLayout(profile_row)

        best_row = QHBoxLayout()
        best_row.addWidget(QLabel("Best per group"))
        best_row.addStretch(1)
        self.best_count_spin = QSpinBox()
        self.best_count_spin.setRange(1, 5)
        self.best_count_spin.setValue(1)
        self.best_count_spin.setToolTip(
            "Choose up to this many visually distinct automatic Best frames per group"
        )
        self.best_count_spin.setEnabled(False)
        self.best_count_spin.valueChanged.connect(self._best_count_changed)
        best_row.addWidget(self.best_count_spin)
        layout.addLayout(best_row)

        action = QWidgetAction(self)
        action.setDefaultWidget(settings)
        return action

    def _build_selection_inspector(self) -> SelectionInspector:
        inspector = SelectionInspector()
        inspector.setMinimumWidth(275)
        inspector.setMaximumWidth(380)
        inspector.keep_button.clicked.connect(
            lambda: self._set_decision(Decision.KEEP)
        )
        inspector.review_button.clicked.connect(
            lambda: self._set_decision(Decision.REVIEW)
        )
        inspector.reject_button.clicked.connect(
            lambda: self._set_decision(Decision.REJECT)
        )
        inspector.clear_button.clicked.connect(
            lambda: self._set_decision(Decision.UNREVIEWED)
        )
        inspector.best_button.clicked.connect(self._toggle_best)
        inspector.open_button.clicked.connect(self._open_selected_fullscreen)
        inspector.compare_button.clicked.connect(self._open_selected_comparison)
        inspector.analysis_overlay_button.toggled.connect(self._set_analysis_overlay)
        self.keep_button = inspector.keep_button
        self.review_button = inspector.review_button
        self.reject_button = inspector.reject_button
        self.clear_button = inspector.clear_button
        self.best_button = inspector.best_button
        self._set_action_buttons_enabled(False)
        return inspector

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self.add_project_action)
        file_menu.addSeparator()
        file_menu.addAction(self.export_button)
        file_menu.addAction(self.close_project_button)

        project_menu = self.menuBar().addMenu("Project")
        project_menu.addAction(self.analyze_action)
        project_menu.addAction(self.rescan_button)
        project_menu.addSeparator()
        project_menu.addAction(self.calibration_button)

        photo_menu = self.menuBar().addMenu("Photo")
        for text, handler in (
            ("Keep    K", lambda: self._set_decision(Decision.KEEP)),
            ("Review    R", lambda: self._set_decision(Decision.REVIEW)),
            ("Reject    X", lambda: self._set_decision(Decision.REJECT)),
            ("Clear decision    U", lambda: self._set_decision(Decision.UNREVIEWED)),
        ):
            photo_menu.addAction(text).triggered.connect(handler)
        photo_menu.addSeparator()
        photo_menu.addAction("Toggle Best    B").triggered.connect(self._toggle_best)
        photo_menu.addAction("Open fullscreen").triggered.connect(
            self._open_selected_fullscreen
        )
        photo_menu.addAction("Compare group").triggered.connect(
            self._open_selected_comparison
        )

        view_menu = self.menuBar().addMenu("View")
        for index, view in enumerate(VIEWS):
            view_menu.addAction(view.value.title()).triggered.connect(
                lambda _checked=False, selected=index: self.view_tabs.setCurrentIndex(selected)
            )
        view_menu.addSeparator()
        view_menu.addAction("Analysis overlay    O").triggered.connect(
            self._toggle_analysis_overlay
        )

    def _install_shortcuts(self) -> None:
        shortcuts = [
            ("K", lambda: self._set_decision(Decision.KEEP)),
            ("R", lambda: self._set_decision(Decision.REVIEW)),
            ("X", lambda: self._set_decision(Decision.REJECT)),
            ("U", lambda: self._set_decision(Decision.UNREVIEWED)),
            ("B", self._toggle_best),
            ("O", self._toggle_analysis_overlay),
            (QKeySequence.SelectAll, self.photo_grid.selectAll),
        ]
        for key, handler in shortcuts:
            action = QAction(self)
            action.setShortcut(key)
            action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
            action.triggered.connect(handler)
            self.addAction(action)

    @Slot(bool)
    def _set_analysis_overlay(self, enabled: bool) -> None:
        self.photo_grid.setProperty("analysisOverlayEnabled", enabled)
        self.photo_grid.viewport().update()

    @Slot()
    def _toggle_analysis_overlay(self) -> None:
        button = self.inspector.analysis_overlay_button
        button.setChecked(not button.isChecked())

    def _load_projects(self, select_project_id: Optional[int] = None) -> None:
        self.project_list.blockSignals(True)
        self.project_list.clear()
        selected_item = None
        for project in self.database.list_projects():
            item = QListWidgetItem(project.name)
            item.setData(Qt.UserRole, project.id)
            item.setToolTip(project.root_path)
            self.project_list.addItem(item)
            if project.id == select_project_id:
                selected_item = item
        self.project_list.blockSignals(False)

        if selected_item is not None:
            self.project_list.setCurrentItem(selected_item)
        elif self.project_list.count():
            self.project_list.setCurrentRow(0)
        else:
            self._clear_current_project()

    @Slot()
    def _choose_project_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose a shoot folder")
        if not folder:
            return
        try:
            project = self.database.add_project(Path(folder))
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot add folder", str(exc))
            return
        self._load_projects(project.id)
        self._start_scan(project, analyze_after=True)

    @Slot(QListWidgetItem, QListWidgetItem)
    def _project_changed(self, current: Optional[QListWidgetItem], _previous) -> None:
        if current is None:
            self._clear_current_project()
            return
        project = self.database.get_project(current.data(Qt.UserRole))
        if project is None:
            return
        self.current_project = project
        self.project_title.setText(project.name)
        self.project_path.setText(project.root_path)
        self.setWindowTitle("{0} · PhotoChoose".format(project.name))
        self._update_analysis_profile_controls()
        self._update_best_count_control()
        self._update_project_action_buttons()
        self._refresh_photos()

    @Slot(int)
    def _analysis_profile_changed(self, index: int) -> None:
        if self.current_project is None or index < 0:
            return
        profile = self.analysis_profile_combo.itemData(index)
        if not profile or profile == self.current_project.analysis_profile:
            return
        self.database.set_project_analysis_profile(self.current_project.id, profile)
        updated = self.database.get_project(self.current_project.id)
        if updated is None:
            return
        self.current_project = updated
        self._update_analysis_profile_controls()
        self._refresh_photos()
        self.statusBar().showMessage(
            "{0} profile selected · click Analyze and separate to apply it".format(
                self.analysis_profile_combo.currentText()
            ),
            10000,
        )

    def _update_analysis_profile_controls(self) -> None:
        if self.current_project is None:
            return
        index = self.analysis_profile_combo.findData(
            self.current_project.analysis_profile
        )
        self.analysis_profile_combo.blockSignals(True)
        if index >= 0:
            self.analysis_profile_combo.setCurrentIndex(index)
        self.analysis_profile_combo.blockSignals(False)
        stale_count = self.database.count_stale_analysis(self.current_project.id)
        profile_label = self.analysis_profile_combo.currentText() or "Safe"
        if stale_count:
            self.analysis_profile_status.setText(
                "{0} profile · {1} photo{2} need reanalysis".format(
                    profile_label,
                    stale_count,
                    "" if stale_count == 1 else "s",
                )
            )
            self.analysis_profile_status.setObjectName("statusReview")
        else:
            self.analysis_profile_status.setText(
                "{0} profile · {1} · local processing".format(
                    profile_label, ANALYSIS_VERSION
                )
            )
            self.analysis_profile_status.setObjectName("mutedLabel")
        self.analysis_profile_status.style().unpolish(self.analysis_profile_status)
        self.analysis_profile_status.style().polish(self.analysis_profile_status)

    @Slot(int)
    def _best_count_changed(self, count: int) -> None:
        if self.current_project is None:
            return
        if count == self.current_project.best_count_per_group:
            return
        automatic_count = self.database.set_project_best_count(
            self.current_project.id, count
        )
        updated = self.database.get_project(self.current_project.id)
        if updated is None:
            return
        self.current_project = updated
        self._refresh_photos()
        self.statusBar().showMessage(
            "Up to {0} diverse Best per group · {1} automatic Best selected from "
            "stored rankings".format(count, automatic_count),
            10000,
        )

    def _update_best_count_control(self) -> None:
        if self.current_project is None:
            return
        self.best_count_spin.blockSignals(True)
        self.best_count_spin.setValue(self.current_project.best_count_per_group)
        self.best_count_spin.blockSignals(False)

    @Slot(QPoint)
    def _show_project_context_menu(self, position: QPoint) -> None:
        item = self.project_list.itemAt(position)
        if item is None:
            return
        self.project_list.setCurrentItem(item)
        menu = self._project_context_menu()
        menu.exec(self.project_list.viewport().mapToGlobal(position))

    def _project_context_menu(self) -> QMenu:
        menu = QMenu(self)
        rescan = menu.addAction("Rescan folder")
        rescan.triggered.connect(self._rescan_current_project)
        analyze = menu.addAction("Analyze and separate")
        analyze.triggered.connect(self._analyze_current_project)
        calibration = menu.addAction("Calibration report…")
        calibration.triggered.connect(self._open_calibration_report)
        export = menu.addAction("Export Keep + Best…")
        export.triggered.connect(self._choose_export_folder)
        menu.addSeparator()
        close = menu.addAction("Close folder")
        close.triggered.connect(self._close_current_project)
        enabled = (
            self.current_project is not None
            and self.current_project.id not in self.busy_projects
        )
        for action in (rescan, analyze, calibration, export, close):
            action.setEnabled(enabled)
        return menu

    @Slot(int)
    def _view_changed(self, index: int) -> None:
        if 0 <= index < len(VIEWS):
            self.current_view = VIEWS[index]
            self._refresh_photos()

    @Slot()
    def _rescan_current_project(self) -> None:
        if self.current_project is not None:
            self._start_scan(self.current_project)

    def _start_scan(self, project: Project, analyze_after: bool = False) -> None:
        if analyze_after:
            self.analyze_after_scan.add(project.id)
        self.busy_projects.add(project.id)
        self._update_project_action_buttons()
        self.statusBar().showMessage("Scanning {0}…".format(project.name))
        task = _ScanTask(self.database, project.id)
        task.signals.finished.connect(self._scan_finished)
        task.signals.failed.connect(self._scan_failed)
        self.thread_pool.start(task)

    @Slot(int, int)
    def _scan_finished(self, project_id: int, count: int) -> None:
        self.busy_projects.discard(project_id)
        if self.current_project and self.current_project.id == project_id:
            self._update_project_action_buttons()
            self._refresh_photos()
        if project_id in self.analyze_after_scan:
            self.analyze_after_scan.discard(project_id)
            project = self.database.get_project(project_id)
            if project is not None:
                self._start_analysis(project)
                return
        self.statusBar().showMessage("Scan complete: {0} supported photos".format(count), 5000)

    @Slot(int, str)
    def _scan_failed(self, project_id: int, message: str) -> None:
        self.analyze_after_scan.discard(project_id)
        self.busy_projects.discard(project_id)
        if self.current_project and self.current_project.id == project_id:
            self._update_project_action_buttons()
        QMessageBox.warning(self, "Scan failed", message)
        self.statusBar().showMessage("Scan failed", 5000)

    @Slot()
    def _analyze_current_project(self) -> None:
        if self.current_project is not None:
            self._start_scan(self.current_project, analyze_after=True)

    def _start_analysis(self, project: Project) -> None:
        if not self.database.list_photos(project.id, PhotoView.ALL):
            QMessageBox.information(
                self,
                "Nothing to analyze",
                "This project has no supported indexed photos. Rescan it first.",
            )
            return
        self.busy_projects.add(project.id)
        self._update_project_action_buttons()
        self.statusBar().showMessage("Starting local analysis of {0}…".format(project.name))
        task = _AnalysisTask(self.database, project.id, project.analysis_profile)
        task.signals.progress.connect(self._analysis_progress)
        task.signals.finished.connect(self._analysis_finished)
        task.signals.failed.connect(self._analysis_failed)
        self.thread_pool.start(task)

    @Slot(int, int, int)
    def _analysis_progress(self, project_id: int, complete: int, total: int) -> None:
        if self.current_project and self.current_project.id == project_id:
            self.statusBar().showMessage("Analyzing locally… {0} / {1}".format(complete, total))

    @Slot(int, object)
    def _analysis_finished(self, project_id: int, totals) -> None:
        self.busy_projects.discard(project_id)
        if self.current_project and self.current_project.id == project_id:
            self._update_project_action_buttons()
            self._refresh_photos()
            self._update_analysis_profile_controls()
        self.statusBar().showMessage(
            "Analysis complete · Keep {0} · Review {1} · Reject {2}".format(
                totals.get(Decision.KEEP.value, 0),
                totals.get(Decision.REVIEW.value, 0),
                totals.get(Decision.REJECT.value, 0),
            ),
            10000,
        )

    @Slot(int, str)
    def _analysis_failed(self, project_id: int, message: str) -> None:
        self.busy_projects.discard(project_id)
        if self.current_project and self.current_project.id == project_id:
            self._update_project_action_buttons()
        QMessageBox.warning(self, "Analysis failed", message)
        self.statusBar().showMessage("Analysis failed", 5000)

    @Slot()
    def _open_calibration_report(self) -> None:
        if self.current_project is None:
            return
        if self.calibration_dialog is not None:
            self.calibration_dialog.raise_()
            self.calibration_dialog.activateWindow()
            return
        report = build_calibration_report(
            self.database, self.current_project.id
        )
        dialog = CalibrationReportDialog(report, self)
        dialog.finished.connect(self._calibration_report_closed)
        self.calibration_dialog = dialog
        dialog.show()

    @Slot(int)
    def _calibration_report_closed(self, _result: int) -> None:
        dialog = self.sender()
        if dialog is self.calibration_dialog:
            self.calibration_dialog = None

    @Slot()
    def _close_current_project(self) -> None:
        if self.current_project is None:
            return
        project = self.current_project
        if project.id in self.busy_projects:
            return
        removed_previews, removed_bytes = self.database.close_project(project.id)
        self.current_project = None
        self._load_projects()
        cache_message = ""
        if removed_previews:
            cache_message = " · cleared {0} RAW preview{1} ({2})".format(
                removed_previews,
                "" if removed_previews == 1 else "s",
                self._file_size(removed_bytes),
            )
        self.statusBar().showMessage(
            "Closed {0}{1} · metadata and decisions were preserved".format(
                project.name, cache_message
            ),
            7000,
        )

    @staticmethod
    def _file_size(size_bytes: int) -> str:
        size = float(size_bytes)
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024.0 or unit == "GB":
                if unit == "B":
                    return "{0:.0f} {1}".format(size, unit)
                return "{0:.1f} {1}".format(size, unit)
            size /= 1024.0
        return "{0} B".format(size_bytes)

    @Slot(object)
    def _open_fullscreen(self, index) -> None:
        if not index.isValid() or not 0 <= index.row() < len(self.photo_model.photos):
            return
        if self.fullscreen_dialog is not None:
            self.fullscreen_dialog.raise_()
            self.fullscreen_dialog.activateWindow()
            return
        dialog = FullscreenPhotoDialog(
            self.database,
            list(self.photo_model.photos),
            index.row(),
            None,
        )
        dialog.finished.connect(self._fullscreen_closed)
        self.fullscreen_dialog = dialog
        dialog.showFullScreen()

    @Slot(int)
    def _fullscreen_closed(self, _result: int) -> None:
        dialog = self.sender()
        if dialog is self.fullscreen_dialog:
            self.fullscreen_dialog = None
        if self.current_project is not None:
            self._refresh_photos()

    @Slot()
    def _open_selected_fullscreen(self) -> None:
        indexes = self.photo_grid.selectionModel().selectedIndexes()
        if indexes:
            self._open_fullscreen(indexes[0])

    @Slot(QPoint)
    def _show_photo_context_menu(self, position: QPoint) -> None:
        index = self.photo_grid.indexAt(position)
        if not index.isValid():
            return
        selection_model = self.photo_grid.selectionModel()
        if not selection_model.isSelected(index):
            selection_model.select(index, QItemSelectionModel.ClearAndSelect)
            self.photo_grid.setCurrentIndex(index)
        menu = self._photo_context_menu(index)
        menu.exec(self.photo_grid.viewport().mapToGlobal(position))

    def _photo_context_menu(self, index=None) -> QMenu:
        menu = QMenu(self)
        open_action = menu.addAction("Open fullscreen")
        if index is None:
            indexes = self.photo_grid.selectionModel().selectedIndexes()
            index = indexes[0] if indexes else None
        open_action.setEnabled(index is not None and index.isValid())
        open_action.triggered.connect(
            lambda _checked=False, selected=index: self._open_fullscreen(selected)
        )
        compare_action = menu.addAction("Compare similar group")
        selected_photo = None
        if index is not None and index.isValid():
            selected_photo = self.photo_model.photos[index.row()]
        compare_action.setEnabled(
            selected_photo is not None and selected_photo.similarity_group is not None
        )
        compare_action.triggered.connect(
            lambda _checked=False, photo=selected_photo: self._open_comparison(photo)
        )
        menu.addSeparator()
        keep = menu.addAction("Keep")
        keep.triggered.connect(lambda: self._set_decision(Decision.KEEP))
        review = menu.addAction("Review")
        review.triggered.connect(lambda: self._set_decision(Decision.REVIEW))
        reject = menu.addAction("Reject")
        reject.triggered.connect(lambda: self._set_decision(Decision.REJECT))
        clear = menu.addAction("Clear decision")
        clear.triggered.connect(lambda: self._set_decision(Decision.UNREVIEWED))
        menu.addSeparator()
        best = menu.addAction("Toggle Best")
        best.triggered.connect(self._toggle_best)
        enabled = bool(self._selected_photo_ids())
        for action in (keep, review, reject, clear, best):
            action.setEnabled(enabled)
        return menu

    @Slot()
    def _open_selected_comparison(self) -> None:
        indexes = self.photo_grid.selectionModel().selectedIndexes()
        if not indexes or self.current_project is None:
            return
        selected = self.photo_model.photos[indexes[0].row()]
        self._open_comparison(selected)

    def _open_comparison(self, selected) -> None:
        if selected is None or not selected.similarity_group:
            return
        if self.comparison_dialog is not None:
            self.comparison_dialog.raise_()
            self.comparison_dialog.activateWindow()
            return
        all_photos = self.database.list_photos(self.current_project.id, PhotoView.ALL)
        group = [
            photo
            for photo in all_photos
            if photo.similarity_group == selected.similarity_group
        ]
        group.sort(
            key=lambda photo: (
                photo.captured_at is None,
                photo.captured_at or "",
                photo.relative_path.casefold(),
            )
        )
        if len(group) < 2:
            return
        dialog = GroupComparisonDialog(self.database, group, selected.id, None)
        dialog.finished.connect(self._comparison_closed)
        self.comparison_dialog = dialog
        dialog.showMaximized()

    @Slot(int)
    def _comparison_closed(self, _result: int) -> None:
        dialog = self.sender()
        if dialog is self.comparison_dialog:
            self.comparison_dialog = None
        if self.current_project is not None:
            self._refresh_photos()

    @Slot()
    def _choose_export_folder(self) -> None:
        if self.current_project is None:
            return
        project = self.current_project
        exporter = ProjectExporter(self.database)
        if exporter.export_count(project.id) == 0:
            QMessageBox.information(
                self,
                "Nothing to export",
                "This project has no Keep or Best photographs yet.",
            )
            return
        folder = QFileDialog.getExistingDirectory(self, "Choose export destination")
        if not folder:
            return
        self.busy_projects.add(project.id)
        self._update_project_action_buttons()
        self.statusBar().showMessage("Preparing export…")
        task = _ExportTask(self.database, project.id, Path(folder))
        task.signals.progress.connect(self._export_progress)
        task.signals.finished.connect(self._export_finished)
        task.signals.failed.connect(self._export_failed)
        self.thread_pool.start(task)

    @Slot(int, int, int)
    def _export_progress(self, project_id: int, complete: int, total: int) -> None:
        if self.current_project and self.current_project.id == project_id:
            self.statusBar().showMessage("Copying export… {0} / {1}".format(complete, total))

    @Slot(int, object)
    def _export_finished(self, project_id: int, result) -> None:
        self.busy_projects.discard(project_id)
        if self.current_project and self.current_project.id == project_id:
            self._update_project_action_buttons()
        message = "Copied: {0}\nSkipped existing: {1}\nDestination: {2}".format(
            result.copied, result.skipped, result.destination
        )
        if result.failed:
            message += "\nFailed: {0}".format(len(result.failed))
            message += "\n\n" + "\n".join(result.failed[:5])
        QMessageBox.information(self, "Export complete", message)
        self.statusBar().showMessage(
            "Export complete · {0} copied · {1} skipped".format(result.copied, result.skipped),
            10000,
        )

    @Slot(int, str)
    def _export_failed(self, project_id: int, message: str) -> None:
        self.busy_projects.discard(project_id)
        if self.current_project and self.current_project.id == project_id:
            self._update_project_action_buttons()
        QMessageBox.warning(self, "Export failed", message)
        self.statusBar().showMessage("Export failed", 5000)

    def _update_project_action_buttons(self) -> None:
        enabled = (
            self.current_project is not None and self.current_project.id not in self.busy_projects
        )
        self.rescan_button.setEnabled(enabled)
        self.analyze_button.setEnabled(enabled)
        self.analyze_action.setEnabled(enabled)
        self.calibration_button.setEnabled(enabled)
        self.export_button.setEnabled(enabled)
        self.close_project_button.setEnabled(enabled)
        self.project_options_button.setEnabled(enabled)
        self.analysis_profile_combo.setEnabled(enabled)
        self.best_count_spin.setEnabled(enabled)

    def _clear_current_project(self) -> None:
        self.current_project = None
        self.project_title.setText("No project selected")
        self.project_path.setText("Select a project from the library")
        self.setWindowTitle("PhotoChoose")
        self.analysis_profile_combo.setEnabled(False)
        self.best_count_spin.blockSignals(True)
        self.best_count_spin.setValue(1)
        self.best_count_spin.blockSignals(False)
        self.best_count_spin.setEnabled(False)
        self.analysis_profile_status.setText("Safe profile · ready")
        self.analysis_profile_status.setObjectName("mutedLabel")
        self.photo_model.clear()
        self.empty_title.setText("Your workspace is ready")
        self.empty_hint.setText(
            "Add a shoot or choose one from the library. Originals always stay untouched."
        )
        self.canvas_stack.setCurrentIndex(0)
        self.inspector.set_photos([], 0)
        for index, view in enumerate(VIEWS):
            self.view_tabs.setTabText(index, view.value.title())
        self.rescan_button.setEnabled(False)
        self.analyze_button.setEnabled(False)
        self.analyze_action.setEnabled(False)
        self.calibration_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.close_project_button.setEnabled(False)
        self.project_options_button.setEnabled(False)

    def _selected_photo_ids(self):
        indexes = self.photo_grid.selectionModel().selectedIndexes()
        return self.photo_model.photo_ids(indexes)

    def _set_decision(self, decision: Decision) -> None:
        photo_ids = self._selected_photo_ids()
        if not photo_ids:
            return
        self.database.set_decision(photo_ids, decision)
        self._refresh_photos()

    def _toggle_best(self) -> None:
        photo_ids = self._selected_photo_ids()
        if not photo_ids:
            return
        self.database.toggle_best(photo_ids)
        self._refresh_photos()

    def _refresh_photos(self) -> None:
        if self.current_project is None:
            return
        self.photo_model.refresh(self.current_project.id, self.current_view)
        if self.photo_model.rowCount():
            self.canvas_stack.setCurrentIndex(1)
        else:
            self.empty_title.setText("Nothing in this view")
            self.empty_hint.setText(
                "Try another filter, or use Project → Rescan if files were added."
            )
            self.canvas_stack.setCurrentIndex(0)
        self.photo_grid.clearSelection()
        self._update_counts()
        self._selection_changed()

    def _update_counts(self) -> None:
        if self.current_project is None:
            return
        counts = self.database.counts(self.current_project.id)
        for index, view in enumerate(VIEWS):
            label = "{0}  {1}".format(view.value.title(), counts[view.value])
            self.view_tabs.setTabText(index, label)
        self.count_label.setText("{0} photos visible".format(self.photo_model.rowCount()))

    @Slot()
    def _selection_changed(self, *_args) -> None:
        indexes = self.photo_grid.selectionModel().selectedIndexes()
        selected_photos = self.photo_model.selected_photos(indexes)
        selected_count = len(selected_photos)
        self._set_action_buttons_enabled(selected_count > 0)
        self.inspector.open_button.setEnabled(selected_count > 0)
        self.inspector.set_photos(selected_photos, self.photo_model.rowCount())
        if selected_count:
            self.statusBar().showMessage("{0} selected".format(selected_count))
        elif self.current_project is not None:
            self.statusBar().showMessage(
                "{0} visible · source photos unchanged".format(self.photo_model.rowCount())
            )

    def _set_action_buttons_enabled(self, enabled: bool) -> None:
        for button in (
            self.keep_button,
            self.review_button,
            self.reject_button,
            self.clear_button,
            self.best_button,
        ):
            button.setEnabled(enabled)
