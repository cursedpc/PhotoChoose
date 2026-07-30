from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPoint, QSize, Qt, Signal
from PySide6.QtGui import QEnterEvent, QIcon, QPixmap
from PySide6.QtWidgets import QMenu, QPushButton, QWidgetAction

from photochoose.database import PhotoDatabase
from photochoose.image_loader import read_qimage
from photochoose.models import PersonIdentity

FACE_ICON_SIZE = QSize(36, 36)


class _PersonRow(QPushButton):
    hovered = Signal(int, object)

    def __init__(self, person: PersonIdentity, parent=None) -> None:
        super().__init__(parent)
        self.person = person
        self.setText("{0}  ·  {1}".format(person.display_name, person.photo_count))
        self.setIcon(_person_icon(person))
        self.setIconSize(FACE_ICON_SIZE)
        self.setMinimumHeight(44)
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
        self.database = database
        self._folder_menu: Optional[QMenu] = None
        self.aboutToShow.connect(self.rebuild)

    def rebuild(self) -> None:
        self.clear()
        people = self.database.list_people()
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
            row = _PersonRow(person, self)
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
        folders = self.database.list_person_folders(person.id)
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
        menu.popup(point)

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


def _person_icon(person: PersonIdentity) -> QIcon:
    if not person.cover_photo_path or not person.cover_bounds:
        return QIcon()
    image = read_qimage(person.cover_photo_path, QSize(240, 240))
    if image.isNull():
        return QIcon()
    x, y, width, height = person.cover_bounds
    left = max(0, min(image.width() - 1, round(x * image.width())))
    top = max(0, min(image.height() - 1, round(y * image.height())))
    crop_width = max(1, min(image.width() - left, round(width * image.width())))
    crop_height = max(1, min(image.height() - top, round(height * image.height())))
    crop = image.copy(left, top, crop_width, crop_height)
    pixmap = QPixmap.fromImage(crop).scaled(
        FACE_ICON_SIZE,
        Qt.KeepAspectRatioByExpanding,
        Qt.SmoothTransformation,
    )
    return QIcon(pixmap)
