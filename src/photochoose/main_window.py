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
    QFileDialog,
    QFrame,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

from photochoose.analyzer import BaselineAnalyzer
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
    def __init__(self, database: PhotoDatabase, project_id: int) -> None:
        super().__init__()
        self.database = database
        self.project_id = project_id
        self.signals = _AnalysisSignals()

    def run(self) -> None:
        analyzer = BaselineAnalyzer(self.database)
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

        self.setWindowTitle("PhotoChoose")
        self.resize(1280, 820)
        self._build_ui()
        self._install_shortcuts()
        self._load_projects()

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_project_sidebar())
        splitter.addWidget(self._build_photo_workspace())
        splitter.setSizes([260, 1020])
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("Add a shoot folder to begin")

    def _build_project_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setMinimumWidth(220)
        layout = QVBoxLayout(sidebar)

        title = QLabel("Projects")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(title)

        self.project_list = QListWidget()
        self.project_list.setAlternatingRowColors(True)
        self.project_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.project_list.customContextMenuRequested.connect(
            self._show_project_context_menu
        )
        self.project_list.currentItemChanged.connect(self._project_changed)
        layout.addWidget(self.project_list, 1)

        add_button = QPushButton("Add shoot folder…")
        add_button.clicked.connect(self._choose_project_folder)
        layout.addWidget(add_button)

        self.rescan_button = QPushButton("Rescan selected folder")
        self.rescan_button.clicked.connect(self._rescan_current_project)
        self.rescan_button.setEnabled(False)
        layout.addWidget(self.rescan_button)

        self.analyze_button = QPushButton("Analyze and separate")
        self.analyze_button.setToolTip(
            "Refresh metadata, then classify this folder entirely on this computer"
        )
        self.analyze_button.clicked.connect(self._analyze_current_project)
        self.analyze_button.setEnabled(False)
        layout.addWidget(self.analyze_button)

        self.export_button = QPushButton("Export Keep + Best…")
        self.export_button.setToolTip(
            "Copy kept and best photographs into Keep and Best export folders"
        )
        self.export_button.clicked.connect(self._choose_export_folder)
        self.export_button.setEnabled(False)
        layout.addWidget(self.export_button)

        self.close_project_button = QPushButton("Close selected folder")
        self.close_project_button.setToolTip(
            "Hide this project while preserving its analysis and decisions"
        )
        self.close_project_button.clicked.connect(self._close_current_project)
        self.close_project_button.setEnabled(False)
        layout.addWidget(self.close_project_button)
        return sidebar

    def _build_photo_workspace(self) -> QWidget:
        workspace = QWidget()
        layout = QVBoxLayout(workspace)

        self.project_title = QLabel("No project selected")
        self.project_title.setStyleSheet("font-size: 20px; font-weight: 600;")
        layout.addWidget(self.project_title)

        self.project_path = QLabel("")
        self.project_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.project_path.setStyleSheet("color: #808792;")
        layout.addWidget(self.project_path)

        self.view_tabs = QTabBar()
        self.view_tabs.setExpanding(False)
        self.view_tabs.setDocumentMode(True)
        for view in VIEWS:
            self.view_tabs.addTab(view.value.title())
        self.view_tabs.currentChanged.connect(self._view_changed)
        layout.addWidget(self.view_tabs)

        self.inspector = SelectionInspector()
        self.inspector.keep_button.clicked.connect(
            lambda: self._set_decision(Decision.KEEP)
        )
        self.inspector.review_button.clicked.connect(
            lambda: self._set_decision(Decision.REVIEW)
        )
        self.inspector.reject_button.clicked.connect(
            lambda: self._set_decision(Decision.REJECT)
        )
        self.inspector.clear_button.clicked.connect(
            lambda: self._set_decision(Decision.UNREVIEWED)
        )
        self.inspector.best_button.clicked.connect(self._toggle_best)
        self.inspector.open_button.clicked.connect(self._open_selected_fullscreen)
        self.inspector.compare_button.clicked.connect(self._open_selected_comparison)
        self.inspector.analysis_overlay_button.toggled.connect(
            self._set_analysis_overlay
        )
        self.keep_button = self.inspector.keep_button
        self.review_button = self.inspector.review_button
        self.reject_button = self.inspector.reject_button
        self.clear_button = self.inspector.clear_button
        self.best_button = self.inspector.best_button
        self.count_label = self.inspector.visible_count_label
        layout.addWidget(self.inspector)

        self.photo_grid = QListView()
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
        self.photo_grid.setSpacing(8)
        self.photo_grid.setContextMenuPolicy(Qt.CustomContextMenu)
        self.photo_grid.customContextMenuRequested.connect(self._show_photo_context_menu)
        self.photo_grid.selectionModel().selectionChanged.connect(self._selection_changed)
        self.photo_grid.doubleClicked.connect(self._open_fullscreen)
        layout.addWidget(self.photo_grid, 1)
        self._set_action_buttons_enabled(False)
        return workspace

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
        self._update_project_action_buttons()
        self._refresh_photos()

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
        export = menu.addAction("Export Keep + Best…")
        export.triggered.connect(self._choose_export_folder)
        menu.addSeparator()
        close = menu.addAction("Close folder")
        close.triggered.connect(self._close_current_project)
        enabled = (
            self.current_project is not None
            and self.current_project.id not in self.busy_projects
        )
        for action in (rescan, analyze, export, close):
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
        task = _AnalysisTask(self.database, project.id)
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
        self.export_button.setEnabled(enabled)
        self.close_project_button.setEnabled(enabled)

    def _clear_current_project(self) -> None:
        self.current_project = None
        self.project_title.setText("No project selected")
        self.project_path.setText("")
        self.photo_model.clear()
        self.inspector.set_photos([], 0)
        for index, view in enumerate(VIEWS):
            self.view_tabs.setTabText(index, view.value.title())
        self.rescan_button.setEnabled(False)
        self.analyze_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.close_project_button.setEnabled(False)

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
