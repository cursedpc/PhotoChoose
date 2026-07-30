from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

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
        self.database.replace_identity_faces(first.id, (self._face(0, [1.0, 0.0]),))
        self.database.replace_identity_faces(
            second.id, (self._face(0, [0.99, 0.05]),)
        )

        stats = IdentityMatcher(self.database).assign_unknown_faces()

        people = self.database.list_people()
        self.assertEqual(1, len(people))
        self.assertEqual(2, people[0].photo_count)
        self.assertEqual(1, stats.created_people)
        self.assertEqual(
            {"First", "Second"},
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

        self.assertEqual(2, len(self.database.list_people()))

    def test_kick_out_is_remembered_and_stays_unknown(self) -> None:
        _project, first = self._photo("Portraits", "one.jpg")
        self.database.replace_identity_faces(first.id, (self._face(0, [1.0, 0.0]),))
        IdentityMatcher(self.database).assign_unknown_faces()
        person = self.database.list_people()[0]

        self.assertEqual(1, self.database.reject_photo_person(first.id, person.id))
        IdentityMatcher(self.database).assign_unknown_faces()

        self.assertEqual([], self.database.list_photos_for_person(person.id))
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
        IdentityMatcher(self.database).assign_unknown_faces()
        first_person, second_person = self.database.list_people()

        self.database.rename_person(first_person.id, "Anna")
        self.database.merge_persons(first_person.id, second_person.id)

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

    def test_people_filter_opens_global_and_folder_scoped_photos(self) -> None:
        from photochoose.app import create_application

        first_project, first = self._photo("First", "one.jpg")
        second_project, second = self._photo("Second", "two.jpg")
        self.database.replace_identity_faces(first.id, (self._face(0, [1.0, 0.0]),))
        self.database.replace_identity_faces(
            second.id, (self._face(0, [0.99, 0.05]),)
        )
        IdentityMatcher(self.database).assign_unknown_faces()
        person = self.database.list_people()[0]

        app = create_application(
            ["photochoose-identity-ui-test"], self.database.path
        )
        window = app.main_window
        window._person_filter_selected(person.id, None, person.display_name)
        app.processEvents()
        self.assertTrue(window.people_button.isChecked())
        self.assertEqual(2, window.photo_model.rowCount())

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
