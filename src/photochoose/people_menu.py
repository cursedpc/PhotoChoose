from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Sequence

from PySide6.QtCore import QObject, QPoint, QRunnable, QSize, Qt, QThreadPool, Signal, Slot
from PySide6.QtGui import QAction, QIcon, QImage, QPixmap
from PySide6.QtWidgets import QMenu, QProxyStyle, QStyle

from photochoose.database import PhotoDatabase
from photochoose.image_loader import read_qimage
from photochoose.models import PersonIdentity

FACE_ICON_SIZE = QSize(46, 46)
MAX_FACE_ICON_CACHE = 200


class _PeopleMenuStyle(QProxyStyle):
    def pixelMetric(self, metric, option=None, widget=None) -> int:
        if metric == QStyle.PM_SmallIconSize:
            return FACE_ICON_SIZE.width()
        return super().pixelMetric(metric, option, widget)


class _FaceIconSignals(QObject):
    loaded = Signal(int, object)


class _FaceIconTask(QRunnable):
    def __init__(
        self,
        person_id: int,
        path: str,
        bounds: Sequence[float],
    ) -> None:
        super().__init__()
        self.person_id = person_id
        self.path = path
        self.bounds = tuple(bounds)
        self.signals = _FaceIconSignals()

    def run(self) -> None:
        self.signals.loaded.emit(
            self.person_id,
            _person_face_image(self.path, self.bounds),
        )


class PeopleMenu(QMenu):
    """People found in the folder currently open in the workspace."""

    personSelected = Signal(object, object, str)
    renameRequested = Signal(int, str)
    mergeRequested = Signal(int)
    clearRequested = Signal()

    def __init__(self, database: PhotoDatabase, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("peopleMenu")
        self.database = database
        self.project_id: Optional[int] = None
        self.project_name = ""
        self.active_filter = False
        self._clear_filter_action: Optional[QAction] = None
        self._management_menu: Optional[QMenu] = None
        self._thread_pool = QThreadPool.globalInstance()
        self._icon_cache: OrderedDict[int, QIcon] = OrderedDict()
        self._icon_requests = set()
        self._icon_tasks: dict[int, _FaceIconTask] = {}
        self._person_actions: dict[int, QAction] = {}
        self._people_by_action: dict[QAction, PersonIdentity] = {}
        self._menu_style = _PeopleMenuStyle()
        self._menu_style.setParent(self)
        self.setStyle(self._menu_style)
        self.setMinimumWidth(290)
        self.aboutToShow.connect(self.rebuild)

    def set_project(self, project_id: Optional[int], project_name: str = "") -> None:
        self.project_id = project_id
        self.project_name = project_name
        if project_id is not None:
            # Begin loading the face crops as soon as a folder becomes active so
            # the previews are normally ready before the menu is opened.
            for person in self.database.list_people(project_id):
                self._request_icon(person)
        if self.isVisible():
            self.rebuild()

    def set_active_filter(self, active: bool) -> None:
        self.active_filter = active
        if self.isVisible():
            self.rebuild()

    def rebuild(self) -> None:
        self.clear()
        self._person_actions.clear()
        self._people_by_action.clear()
        self._clear_filter_action = None
        if self.project_id is None:
            empty = self.addAction("Open a folder to browse people")
            empty.setEnabled(False)
            return
        if self.active_filter:
            self._clear_filter_action = self.addAction("Clear person filter")
            self._clear_filter_action.triggered.connect(self.clearRequested.emit)
            self.addSeparator()
        people = self.database.list_people(self.project_id)
        unknown_count = self.database.identity_unknown_count(self.project_id)
        if not people and not unknown_count:
            empty = self.addAction("No recognized people in this folder")
            empty.setEnabled(False)
            return
        if unknown_count:
            unknown = self.addAction("Unknown  ·  {0}".format(unknown_count))
            unknown.setToolTip("Faces that were not assigned confidently")
            unknown.triggered.connect(
                lambda _checked=False: self._select(None, "Unknown")
            )
        if unknown_count and people:
            self.addSeparator()
        for person in people:
            icon = self._icon_cache.get(person.id)
            if icon is not None:
                self._icon_cache.move_to_end(person.id)
            action = self.addAction(
                icon or QIcon(),
                "{0}  ·  {1}".format(person.display_name, person.photo_count),
            )
            # macOS hides QAction menu icons by default. Person crops are content,
            # not decorative command icons, so explicitly keep them visible.
            action.setIconVisibleInMenu(True)
            action.setToolTip(
                "Show photographs of {0} in the open folder".format(
                    person.display_name
                )
            )
            action.triggered.connect(
                lambda _checked=False, selected=person: self._select(
                    selected.id, selected.display_name
                )
            )
            self._person_actions[person.id] = action
            self._people_by_action[action] = person
            self._request_icon(person)

    def _request_icon(self, person: PersonIdentity) -> None:
        if (
            person.id in self._icon_cache
            or person.id in self._icon_requests
            or not person.cover_photo_path
            or not person.cover_bounds
        ):
            return
        self._icon_requests.add(person.id)
        task = _FaceIconTask(
            person.id,
            person.cover_photo_path,
            person.cover_bounds,
        )
        task.signals.loaded.connect(self._icon_loaded)
        # Keep the runnable and its signal object alive until Qt delivers the
        # queued result. Without this, fast menu rebuilds could lose previews.
        self._icon_tasks[person.id] = task
        self._thread_pool.start(task)

    @Slot(int, object)
    def _icon_loaded(self, person_id: int, image: QImage) -> None:
        self._icon_requests.discard(person_id)
        self._icon_tasks.pop(person_id, None)
        if image.isNull():
            return
        icon = QIcon(QPixmap.fromImage(image))
        self._icon_cache[person_id] = icon
        self._icon_cache.move_to_end(person_id)
        while len(self._icon_cache) > MAX_FACE_ICON_CACHE:
            self._icon_cache.popitem(last=False)
        action = self._person_actions.get(person_id)
        if action is not None:
            action.setIcon(icon)
            action.setIconVisibleInMenu(True)

    def _select(
        self,
        person_id: Optional[int],
        label: str,
    ) -> None:
        self.close()
        self.personSelected.emit(person_id, self.project_id, label)

    def _show_person_actions(self, person: PersonIdentity, point: QPoint) -> None:
        if self._management_menu is not None:
            self._management_menu.close()
        menu = QMenu(self)
        rename = menu.addAction("Rename…")
        rename.triggered.connect(
            lambda _checked=False: self._rename(person.id, person.display_name)
        )
        merge = menu.addAction("Merge into…")
        merge.triggered.connect(
            lambda _checked=False: self._merge(person.id)
        )
        self._management_menu = menu
        menu.popup(point)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.RightButton:
            action = self.actionAt(event.position().toPoint())
            person = self._people_by_action.get(action)
            if person is not None:
                self._show_person_actions(person, event.globalPosition().toPoint())
                event.accept()
                return
        super().mouseReleaseEvent(event)

    def _rename(self, person_id: int, current_name: str) -> None:
        if self._management_menu is not None:
            self._management_menu.close()
        self.close()
        self.renameRequested.emit(person_id, current_name)

    def _merge(self, person_id: int) -> None:
        if self._management_menu is not None:
            self._management_menu.close()
        self.close()
        self.mergeRequested.emit(person_id)


def _person_face_image(path: str, bounds: Sequence[float]) -> QImage:
    image = read_qimage(path, QSize(320, 320))
    if image.isNull():
        return QImage()
    x, y, width, height = bounds
    left = max(0, min(image.width() - 1, round(x * image.width())))
    top = max(0, min(image.height() - 1, round(y * image.height())))
    crop_width = max(1, min(image.width() - left, round(width * image.width())))
    crop_height = max(1, min(image.height() - top, round(height * image.height())))
    crop = image.copy(left, top, crop_width, crop_height)
    return crop.scaled(
        FACE_ICON_SIZE,
        Qt.KeepAspectRatioByExpanding,
        Qt.SmoothTransformation,
    )
