from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Sequence

from PySide6.QtCore import (
    QObject,
    QPoint,
    QRunnable,
    QSize,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import QCursor, QEnterEvent, QIcon, QImage, QPixmap
from PySide6.QtWidgets import QMenu, QPushButton, QWidgetAction

from photochoose.database import PhotoDatabase
from photochoose.image_loader import read_qimage
from photochoose.models import PersonIdentity

FACE_ICON_SIZE = QSize(46, 46)
MAX_FACE_ICON_CACHE = 200


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


class _PersonRow(QPushButton):
    hovered = Signal(int, object)

    def __init__(
        self,
        person: PersonIdentity,
        icon: Optional[QIcon] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.person = person
        self.setText("{0}  ·  {1}    ›".format(person.display_name, person.photo_count))
        if icon is not None:
            self.setIcon(icon)
        self.setIconSize(FACE_ICON_SIZE)
        self.setMinimumSize(280, 58)
        self.setCursor(Qt.PointingHandCursor)
        self.setProperty("kind", "menuRow")
        self.setToolTip("Show all photographs of {0}".format(person.display_name))

    def enterEvent(self, event: QEnterEvent) -> None:
        self.hovered.emit(self.person.id, self.mapToGlobal(QPoint(self.width(), 0)))
        super().enterEvent(event)


class PeopleMenu(QMenu):
    """Compact people browser with hover navigation into source folders."""

    personSelected = Signal(object, object, str)
    renameRequested = Signal(int, str)
    mergeRequested = Signal(int)

    def __init__(self, database: PhotoDatabase, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("peopleMenu")
        self.database = database
        self._folder_menu: Optional[QMenu] = None
        self._thread_pool = QThreadPool.globalInstance()
        self._icon_cache: OrderedDict[int, QIcon] = OrderedDict()
        self._icon_requests = set()
        self._rows = {}
        self._people_by_id = {}
        self._folder_cache = {}
        self._active_folder_person_id: Optional[int] = None
        self._hover_timer = QTimer(self)
        self._hover_timer.setInterval(35)
        self._hover_timer.timeout.connect(self._follow_person_hover)
        self.aboutToShow.connect(self.rebuild)
        self.aboutToShow.connect(self._hover_timer.start)
        self.aboutToHide.connect(self._menu_hidden)

    def rebuild(self) -> None:
        self.clear()
        self._rows.clear()
        self._folder_cache.clear()
        people = self.database.list_people()
        self._people_by_id = {person.id: person for person in people}
        unknown_count = self.database.identity_unknown_count()
        if not people and not unknown_count:
            empty = self.addAction("No recognized people yet")
            empty.setEnabled(False)
            return
        if unknown_count:
            unknown = self.addAction("Unknown  ·  {0}".format(unknown_count))
            unknown.setToolTip("Faces that were not assigned confidently")
            unknown.triggered.connect(
                lambda _checked=False: self._select(None, None, "Unknown")
            )
        if unknown_count and people:
            self.addSeparator()
        for person in people:
            action = QWidgetAction(self)
            icon = self._icon_cache.get(person.id)
            if icon is not None:
                self._icon_cache.move_to_end(person.id)
            row = _PersonRow(person, icon, self)
            self._rows[person.id] = row
            row.clicked.connect(
                lambda _checked=False, selected=person: self._select(
                    selected.id, None, selected.display_name
                )
            )
            row.hovered.connect(
                lambda person_id, point, selected=person: self._show_folders(
                    selected, point
                )
            )
            action.setDefaultWidget(row)
            self.addAction(action)
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
        self._thread_pool.start(task)

    @Slot(int, object)
    def _icon_loaded(self, person_id: int, image: QImage) -> None:
        self._icon_requests.discard(person_id)
        if image.isNull():
            return
        icon = QIcon(QPixmap.fromImage(image))
        self._icon_cache[person_id] = icon
        self._icon_cache.move_to_end(person_id)
        while len(self._icon_cache) > MAX_FACE_ICON_CACHE:
            self._icon_cache.popitem(last=False)
        row = self._rows.get(person_id)
        if row is not None:
            row.setIcon(icon)

    def _select(
        self,
        person_id: Optional[int],
        project_id: Optional[int],
        label: str,
    ) -> None:
        if self._folder_menu is not None:
            self._folder_menu.close()
        self.close()
        self.personSelected.emit(person_id, project_id, label)

    def _show_folders(self, person: PersonIdentity, point: QPoint) -> None:
        if (
            self._active_folder_person_id == person.id
            and self._folder_menu is not None
            and self._folder_menu.isVisible()
        ):
            return
        if self._folder_menu is not None:
            self._folder_menu.close()
        menu = QMenu(self)
        all_action = menu.addAction("All folders  ·  {0}".format(person.photo_count))
        all_action.triggered.connect(
            lambda _checked=False: self._select(
                person.id, None, person.display_name
            )
        )
        rename = menu.addAction("Rename…")
        rename.triggered.connect(
            lambda _checked=False: self._rename(person.id, person.display_name)
        )
        merge = menu.addAction("Merge into…")
        merge.triggered.connect(
            lambda _checked=False: self._merge(person.id)
        )
        folders = self._folder_cache.get(person.id)
        if folders is None:
            folders = self.database.list_person_folders(person.id)
            self._folder_cache[person.id] = folders
        if folders:
            menu.addSeparator()
        for folder in folders:
            action = menu.addAction(
                "{0}  ·  {1}".format(folder.name, folder.photo_count)
            )
            action.triggered.connect(
                lambda _checked=False, selected=folder: self._select(
                    person.id, selected.project_id, person.display_name
                )
            )
        self._folder_menu = menu
        self._active_folder_person_id = person.id
        menu.popup(point)

    def _follow_person_hover(self) -> None:
        """Keep the folder flyout synchronized even while a QMenu owns the mouse."""

        if not self.isVisible():
            return
        cursor = QCursor.pos()
        for person_id, row in self._rows.items():
            if row.isVisible() and row.rect().contains(row.mapFromGlobal(cursor)):
                if person_id != self._active_folder_person_id:
                    person = self._people_by_id.get(person_id)
                    if person is not None:
                        self._show_folders(
                            person,
                            row.mapToGlobal(QPoint(row.width(), 0)),
                        )
                return
        if self.rect().contains(self.mapFromGlobal(cursor)):
            if self._folder_menu is not None:
                self._folder_menu.close()
                self._folder_menu = None
            self._active_folder_person_id = None

    def _menu_hidden(self) -> None:
        self._hover_timer.stop()
        if self._folder_menu is not None:
            self._folder_menu.close()
            self._folder_menu = None
        self._active_folder_person_id = None

    def _rename(self, person_id: int, current_name: str) -> None:
        if self._folder_menu is not None:
            self._folder_menu.close()
        self.close()
        self.renameRequested.emit(person_id, current_name)

    def _merge(self, person_id: int) -> None:
        if self._folder_menu is not None:
            self._folder_menu.close()
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
