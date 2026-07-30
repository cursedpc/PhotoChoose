from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from photochoose.database import PhotoDatabase
from photochoose.identity_recognition import IdentityMatcher
from photochoose.models import IdentityFaceEvidence, PhotoView


class IdentityRecognitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.database = PhotoDatabase(self.root / "state.sqlite3")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _photo(self, folder: str, filename: str):
        shoot = self.root / folder
        shoot.mkdir(exist_ok=True)
        (shoot / filename).write_bytes(b"identity-test-image")
        project = self.database.add_project(shoot)
        self.database.scan_project(project.id)
        photo = next(
            photo
            for photo in self.database.list_photos(project.id, PhotoView.ALL)
            if photo.filename == filename
        )
        return project, photo

    @staticmethod
    def _face(index: int, embedding, quality: float = 0.9, x: float = 0.1):
        vector = np.asarray(embedding, dtype=np.float32)
        vector /= np.linalg.norm(vector)
        return IdentityFaceEvidence(
            face_index=index,
            bounds=(x, 0.1, 0.3, 0.4),
            landmarks=((x + 0.1, 0.2),) * 5,
            embedding=tuple(float(value) for value in vector),
            quality=quality,
            model_name="test-identity-model",
        )

    def test_same_identity_is_grouped_across_folders(self) -> None:
        first_project, first = self._photo("First", "one.jpg")
        second_project, second = self._photo("Second", "two.jpg")
        third_project, third = self._photo("Third", "three.jpg")
        self.database.replace_identity_faces(first.id, (self._face(0, [1.0, 0.0]),))
        self.database.replace_identity_faces(
            second.id, (self._face(0, [0.99, 0.05]),)
        )
        self.database.replace_identity_faces(
            third.id, (self._face(0, [0.98, 0.08]),)
        )

        stats = IdentityMatcher(self.database).assign_unknown_faces()

        people = self.database.list_people()
        self.assertEqual(1, len(people))
        self.assertEqual(3, people[0].photo_count)
        self.assertEqual(1, stats.created_people)
        self.assertEqual(
            {"First", "Second", "Third"},
            {folder.name for folder in self.database.list_person_folders(people[0].id)},
        )
        self.assertEqual(
            [first.id],
            [
                photo.id
                for photo in self.database.list_photos_for_person(
                    people[0].id, first_project.id
                )
            ],
        )
        self.assertEqual(second_project.id, second.project_id)
        self.assertEqual(third_project.id, third.project_id)

    def test_distinct_faces_and_same_photo_faces_are_not_forced_together(self) -> None:
        _project, photo = self._photo("Group", "group.jpg")
        self.database.replace_identity_faces(
            photo.id,
            (
                self._face(0, [1.0, 0.0], x=0.05),
                self._face(1, [0.999, 0.01], x=0.60),
            ),
        )

        IdentityMatcher(self.database).assign_unknown_faces()

        self.assertEqual([], self.database.list_people())
        self.assertEqual(2, self.database.identity_unknown_count())

    def test_kick_out_is_remembered_and_stays_unknown(self) -> None:
        _project, first = self._photo("Portraits", "one.jpg")
        _project, second = self._photo("Portraits", "two.jpg")
        _project, third = self._photo("Portraits", "three.jpg")
        self.database.replace_identity_faces(first.id, (self._face(0, [1.0, 0.0]),))
        self.database.replace_identity_faces(second.id, (self._face(0, [0.99, 0.04]),))
        self.database.replace_identity_faces(third.id, (self._face(0, [0.98, 0.07]),))
        IdentityMatcher(self.database).assign_unknown_faces()
        person = self.database.list_people()[0]

        self.assertEqual(1, self.database.reject_photo_person(first.id, person.id))
        IdentityMatcher(self.database).assign_unknown_faces()

        self.assertNotIn(
            first.id,
            [photo.id for photo in self.database.list_photos_for_person(person.id)],
        )
        self.assertEqual(1, self.database.identity_unknown_count())
        face = self.database.list_identity_faces()[0]
        self.assertEqual("manual", face["assignment_source"])
        self.assertIn(person.id, self.database.rejected_person_ids(int(face["id"])))

        shifted = self._face(0, [1.0, 0.0], x=0.12)
        self.database.replace_identity_faces(first.id, (shifted,))
        IdentityMatcher(self.database).assign_unknown_faces()
        preserved = self.database.list_identity_faces()[0]
        self.assertEqual(face["id"], preserved["id"])
        self.assertIsNone(preserved["person_id"])
        self.assertEqual("manual", preserved["assignment_source"])

    def test_people_can_be_named_and_duplicate_groups_merged(self) -> None:
        _first_project, first = self._photo("First", "one.jpg")
        _second_project, second = self._photo("Second", "two.jpg")
        self.database.replace_identity_faces(first.id, (self._face(0, [1.0, 0.0]),))
        self.database.replace_identity_faces(second.id, (self._face(0, [0.0, 1.0]),))
        faces = self.database.list_identity_faces()
        first_person_id = self.database.create_person()
        second_person_id = self.database.create_person()
        self.database.assign_identity_face(
            int(faces[0]["id"]), first_person_id, 1.0, source="manual"
        )
        self.database.assign_identity_face(
            int(faces[1]["id"]), second_person_id, 1.0, source="manual"
        )

        self.database.rename_person(first_person_id, "Anna")
        self.database.merge_persons(first_person_id, second_person_id)

        people = self.database.list_people()
        self.assertEqual(1, len(people))
        self.assertEqual("Anna", people[0].display_name)
        self.assertEqual(2, people[0].photo_count)

    def test_low_quality_single_face_remains_unknown(self) -> None:
        _project, photo = self._photo("Distant", "small.jpg")
        self.database.replace_identity_faces(
            photo.id, (self._face(0, [1.0, 0.0], quality=0.2),)
        )

        stats = IdentityMatcher(self.database).assign_unknown_faces()

        self.assertEqual([], self.database.list_people())
        self.assertEqual(1, stats.unknown)
        self.assertEqual(1, self.database.identity_unknown_count())

    def test_obsolete_automatic_singletons_are_regrouped_to_unknown(self) -> None:
        _first_project, first = self._photo("First", "one.jpg")
        _second_project, second = self._photo("Second", "two.jpg")
        self.database.replace_identity_faces(first.id, (self._face(0, [1.0, 0.0]),))
        self.database.replace_identity_faces(second.id, (self._face(0, [0.0, 1.0]),))
        faces = self.database.list_identity_faces()
        for face in faces:
            person_id = self.database.create_person()
            self.database.assign_identity_face(
                int(face["id"]), person_id, 1.0, source="automatic"
            )
        self.assertEqual(2, len(self.database.list_people()))

        IdentityMatcher(self.database).assign_unknown_faces()

        self.assertEqual([], self.database.list_people())
        self.assertEqual(2, self.database.identity_unknown_count())

    def test_people_filter_opens_global_and_folder_scoped_photos(self) -> None:
        from PIL import Image
        from PySide6.QtCore import QPoint
        from PySide6.QtTest import QTest

        from photochoose.app import create_application

        first_project, first = self._photo("First", "one.jpg")
        second_project, second = self._photo("Second", "two.jpg")
        _third_project, third = self._photo("Third", "three.jpg")
        self.database.replace_identity_faces(first.id, (self._face(0, [1.0, 0.0]),))
        self.database.replace_identity_faces(
            second.id, (self._face(0, [0.99, 0.05]),)
        )
        self.database.replace_identity_faces(
            third.id, (self._face(0, [0.98, 0.08]),)
        )
        for photo in (first, second, third):
            Image.new("RGB", (160, 120), (110, 90, 70)).save(
                photo.absolute_path, format="JPEG"
            )
        IdentityMatcher(self.database).assign_unknown_faces()
        person = self.database.list_people()[0]
        _fourth_project, fourth = self._photo("Fourth", "other.jpg")
        self.database.replace_identity_faces(
            fourth.id, (self._face(0, [0.0, 1.0]),)
        )
        fourth_face = next(
            face
            for face in self.database.list_identity_faces()
            if int(face["photo_id"]) == fourth.id
        )
        other_person_id = self.database.create_person()
        self.database.assign_identity_face(
            int(fourth_face["id"]), other_person_id, 1.0, source="manual"
        )

        app = create_application(
            ["photochoose-identity-ui-test"], self.database.path
        )
        window = app.main_window
        window.people_menu.rebuild()
        self.assertEqual(2, len(window.people_menu._rows))
        QTest.qWait(300)
        app.processEvents()
        self.assertIn(person.id, window.people_menu._icon_cache)
        menu = window.people_menu
        menu.popup(QPoint(20, 20))
        app.processEvents()
        first_row = menu._rows[person.id]
        second_row = menu._rows[other_person_id]
        with patch(
            "photochoose.people_menu.QCursor.pos",
            return_value=first_row.mapToGlobal(first_row.rect().center()),
        ):
            menu._active_folder_person_id = None
            menu._follow_person_hover()
        self.assertEqual(person.id, menu._active_folder_person_id)
        with patch(
            "photochoose.people_menu.QCursor.pos",
            return_value=second_row.mapToGlobal(second_row.rect().center()),
        ):
            menu._follow_person_hover()
        self.assertEqual(other_person_id, menu._active_folder_person_id)
        menu.close()
        window._person_filter_selected(person.id, None, person.display_name)
        app.processEvents()
        self.assertTrue(window.people_button.isChecked())
        self.assertEqual(3, window.photo_model.rowCount())
        self.assertEqual(
            (0,),
            window.photo_model.index(0, 0).data(
                window.photo_model.HighlightedFacesRole
            ),
        )

        window._person_filter_selected(
            person.id, first_project.id, person.display_name
        )
        app.processEvents()
        self.assertEqual(1, window.photo_model.rowCount())
        self.assertEqual(first.id, window.photo_model.photos[0].id)
        self.assertEqual(second_project.id, second.project_id)
        window.close()


if __name__ == "__main__":
    unittest.main()
