from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from photochoose.database import PhotoDatabase
from photochoose.identity_recognition import (
    OPENVINO_REID_MODEL_NAME,
    IdentityMatcher,
)
from photochoose.models import Decision, IdentityFaceEvidence, PhotoView


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
    def _face(
        index: int,
        embedding,
        quality: float = 0.9,
        x: float = 0.1,
        model_name: str = "test-identity-model",
    ):
        vector = np.asarray(embedding, dtype=np.float32)
        vector /= np.linalg.norm(vector)
        return IdentityFaceEvidence(
            face_index=index,
            bounds=(x, 0.1, 0.3, 0.4),
            landmarks=((x + 0.1, 0.2),) * 5,
            embedding=tuple(float(value) for value in vector),
            quality=quality,
            model_name=model_name,
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

    def test_lookalike_pair_does_not_join_a_consistent_person(self) -> None:
        photos = [
            self._photo("Portraits", filename)[1]
            for filename in (
                "one.jpg",
                "two.jpg",
                "three.jpg",
                "lookalike-one.jpg",
                "lookalike-two.jpg",
            )
        ]
        embeddings = (
            [1.0, 0.0, 0.0],
            [0.99, 0.05, 0.0],
            [0.98, 0.08, 0.0],
            [0.72, 0.0, 0.694],
            [0.71, 0.01, 0.704],
        )
        for photo, embedding in zip(photos, embeddings):
            self.database.replace_identity_faces(
                photo.id, (self._face(0, embedding),)
            )

        IdentityMatcher(self.database).assign_unknown_faces()

        people = self.database.list_people()
        self.assertEqual(1, len(people))
        self.assertEqual(3, people[0].photo_count)
        self.assertEqual(2, self.database.identity_unknown_count())

    def test_recurring_person_absorbs_a_moderate_singleton(self) -> None:
        photos = [
            self._photo("Recurring", "portrait-{0}.jpg".format(index))[1]
            for index in range(13)
        ]
        for index, photo in enumerate(photos[:12]):
            self.database.replace_identity_faces(
                photo.id,
                (self._face(0, [1.0, 0.002 * index, 0.0]),),
            )
        self.database.replace_identity_faces(
            photos[-1].id,
            (self._face(0, [0.6, 0.8, 0.0]),),
        )

        IdentityMatcher(self.database).assign_unknown_faces()

        people = self.database.list_people()
        self.assertEqual(1, len(people))
        self.assertEqual(13, people[0].photo_count)
        self.assertEqual(0, self.database.identity_unknown_count())

    def test_openvino_album_profile_bridges_large_pose_change(self) -> None:
        photos = [
            self._photo("OpenVINO", "portrait-{0}.jpg".format(index))[1]
            for index in range(13)
        ]
        for photo in photos[:12]:
            self.database.replace_identity_faces(
                photo.id,
                (
                    self._face(
                        0,
                        [1.0, 0.0, 0.0],
                        model_name=OPENVINO_REID_MODEL_NAME,
                    ),
                ),
            )
        self.database.replace_identity_faces(
            photos[-1].id,
            (
                self._face(
                    0,
                    [0.42, 0.9075, 0.0],
                    model_name=OPENVINO_REID_MODEL_NAME,
                ),
            ),
        )

        IdentityMatcher(self.database).assign_unknown_faces()

        people = self.database.list_people()
        self.assertEqual(1, len(people))
        self.assertEqual(13, people[0].photo_count)

    def test_mixed_embedding_models_are_grouped_independently(self) -> None:
        for index in range(3):
            _project, photo = self._photo("Legacy", "legacy-{0}.jpg".format(index))
            self.database.replace_identity_faces(
                photo.id,
                (self._face(0, [1.0, 0.01 * index]),),
            )
        for index in range(3):
            _project, photo = self._photo("Current", "current-{0}.jpg".format(index))
            self.database.replace_identity_faces(
                photo.id,
                (
                    self._face(
                        0,
                        [0.0, 1.0, 0.01 * index],
                        model_name=OPENVINO_REID_MODEL_NAME,
                    ),
                ),
            )

        IdentityMatcher(self.database).assign_unknown_faces()

        self.assertEqual(
            [3, 3],
            sorted(person.photo_count for person in self.database.list_people()),
        )

    def test_seed_clusters_merge_by_aggregate_identity(self) -> None:
        matcher = IdentityMatcher(self.database)

        def cluster(start_id: int, start_photo: int, vector):
            normalized = np.asarray(vector, dtype=np.float32)
            normalized /= np.linalg.norm(normalized)
            return [
                {
                    "id": start_id + offset,
                    "photo_id": start_photo + offset,
                    "project_id": 1,
                    "embedding": normalized,
                    "quality": 0.9,
                }
                for offset in range(3)
            ]

        merged = matcher._merge_seed_clusters(
            (
                cluster(1, 1, [1.0, 0.0]),
                cluster(10, 10, [0.8, 0.6]),
                cluster(20, 20, [0.0, 1.0]),
            )
        )

        self.assertEqual([3, 6], sorted(len(group) for group in merged))

    def test_automatic_fragment_merges_into_named_identity(self) -> None:
        photos = [
            self._photo("Consolidate", "portrait-{0}.jpg".format(index))[1]
            for index in range(6)
        ]
        for photo in photos[:3]:
            self.database.replace_identity_faces(
                photo.id,
                (
                    self._face(
                        0,
                        [1.0, 0.0, 0.0],
                        model_name=OPENVINO_REID_MODEL_NAME,
                    ),
                ),
            )
        for photo in photos[3:]:
            self.database.replace_identity_faces(
                photo.id,
                (
                    self._face(
                        0,
                        [0.56, 0.8285, 0.0],
                        model_name=OPENVINO_REID_MODEL_NAME,
                    ),
                ),
            )
        faces = self.database.list_identity_faces()
        named_id = self.database.create_person()
        fragment_id = self.database.create_person()
        for index, face in enumerate(faces):
            self.database.assign_identity_face(
                int(face["id"]),
                named_id if index < 3 else fragment_id,
                1.0,
                source="manual" if index == 0 else "automatic",
            )
        self.database.rename_person(named_id, "Anna")
        matcher = IdentityMatcher(self.database)
        current = self.database.list_identity_faces()

        merged = matcher._consolidate_automatic_people(
            matcher._prototype_map(current), matcher._member_map(current)
        )

        self.assertEqual(1, merged)
        people = self.database.list_people()
        self.assertEqual([(named_id, "Anna", 6)], [
            (person.id, person.display_name, person.photo_count) for person in people
        ])

    def test_dominant_person_uses_multiple_appearance_examples(self) -> None:
        photos = [
            self._photo("Dominant", "portrait-{0}.jpg".format(index))[1]
            for index in range(13)
        ]
        for index, photo in enumerate(photos):
            embedding = [1.0, 0.0, 0.0] if index < 9 else [0.0, 1.0, 0.0]
            quality = 0.9 if index < 9 else 0.6
            self.database.replace_identity_faces(
                photo.id,
                (
                    self._face(
                        0,
                        embedding,
                        quality=quality,
                        model_name=OPENVINO_REID_MODEL_NAME,
                    ),
                ),
            )
        faces = self.database.list_identity_faces()
        person_id = self.database.create_person()
        for face in faces[:-1]:
            self.database.assign_identity_face(
                int(face["id"]), person_id, 1.0, source="automatic"
            )
        self.database.rename_person(person_id, "Anna")

        IdentityMatcher(self.database).assign_unknown_faces()

        person = self.database.list_people()[0]
        self.assertEqual("Anna", person.display_name)
        self.assertEqual(13, person.photo_count)
        self.assertEqual(0, self.database.identity_unknown_count())

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

    def test_model_upgrade_resets_unnamed_groups_but_preserves_named_people(self) -> None:
        _project, unnamed_photo = self._photo("Migration", "unnamed.jpg")
        _project, named_photo = self._photo("Migration", "named.jpg")
        for photo in (unnamed_photo, named_photo):
            self.database.replace_identity_faces(
                photo.id, (self._face(0, [1.0, 0.0]),)
            )
        faces = self.database.list_identity_faces()
        unnamed_person = self.database.create_person()
        named_person = self.database.create_person()
        self.database.assign_identity_face(
            int(faces[0]["id"]), unnamed_person, 1.0, source="automatic"
        )
        self.database.assign_identity_face(
            int(faces[1]["id"]), named_person, 1.0, source="automatic"
        )
        self.database.rename_person(named_person, "Anna")

        upgraded = self._face(
            0,
            [1.0, 0.0, 0.0],
            model_name=OPENVINO_REID_MODEL_NAME,
        )
        self.database.replace_identity_faces(unnamed_photo.id, (upgraded,))
        self.database.replace_identity_faces(named_photo.id, (upgraded,))

        updated = {
            int(face["photo_id"]): face
            for face in self.database.list_identity_faces()
        }
        self.assertIsNone(updated[unnamed_photo.id]["person_id"])
        self.assertEqual("unknown", updated[unnamed_photo.id]["assignment_source"])
        self.assertEqual(named_person, updated[named_photo.id]["person_id"])
        self.assertEqual(["Anna"], [person.display_name for person in self.database.list_people()])

    def test_people_browser_is_scoped_to_the_open_folder(self) -> None:
        from PIL import Image
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest

        from photochoose.app import create_application

        first_project, first = self._photo("First", "one.jpg")
        second_project, second = self._photo("Second", "two.jpg")
        _third_project, third = self._photo("Third", "three.jpg")
        self.database.replace_identity_faces(first.id, (self._face(0, [1.0, 0.0]),))
        self.database.set_decision((first.id,), Decision.KEEP)
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
        fourth_project, fourth = self._photo("Fourth", "other.jpg")
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
        fifth_project, fifth = self._photo("Fifth", "named-other.jpg")
        self.database.replace_identity_faces(
            fifth.id, (self._face(0, [0.0, 0.0, 1.0]),)
        )
        fifth_face = next(
            face
            for face in self.database.list_identity_faces()
            if int(face["photo_id"]) == fifth.id
        )
        named_other_id = self.database.create_person()
        self.database.assign_identity_face(
            int(fifth_face["id"]), named_other_id, 1.0, source="manual"
        )
        self.database.rename_person(named_other_id, "Saved person")

        app = create_application(
            ["photochoose-identity-ui-test"], self.database.path
        )
        window = app.main_window
        first_row_index = next(
            row
            for row in range(window.project_list.count())
            if window.project_list.item(row).data(Qt.UserRole) == first_project.id
        )
        window.project_list.setCurrentRow(first_row_index)
        app.processEvents()
        window.people_menu.rebuild()
        self.assertEqual(first_project.id, window.people_menu.project_id)
        self.assertEqual([person.id], list(window.people_menu._person_actions))
        self.assertEqual(
            1,
            window.people_menu._people_by_action[
                window.people_menu._person_actions[person.id]
            ].photo_count,
        )
        self.assertNotIn(other_person_id, window.people_menu._person_actions)
        self.assertEqual(
            [named_other_id],
            [candidate.id for candidate in window._merge_candidates(person.id)],
        )
        for _attempt in range(40):
            if person.id in window.people_menu._icon_cache:
                break
            app.processEvents()
            QTest.qWait(25)
        self.assertIn(person.id, window.people_menu._icon_cache)
        self.assertFalse(
            window.people_menu._person_actions[person.id].icon().isNull()
        )
        self.assertTrue(
            window.people_menu._person_actions[person.id].isIconVisibleInMenu()
        )
        window.people_menu._person_actions[person.id].trigger()
        app.processEvents()
        self.assertTrue(window.people_button.isChecked())
        self.assertEqual(first_project.id, window.identity_project_id)
        self.assertEqual(1, window.photo_model.rowCount())
        self.assertEqual(first.id, window.photo_model.photos[0].id)
        self.assertEqual(
            (0,),
            window.photo_model.index(0, 0).data(
                window.photo_model.HighlightedFacesRole
            ),
        )
        window.view_tabs.setCurrentIndex(2)
        app.processEvents()
        self.assertTrue(window.identity_filter_active)
        self.assertEqual(PhotoView.KEEP, window.current_view)
        self.assertEqual(1, window.photo_model.rowCount())
        window.view_tabs.setCurrentIndex(3)
        app.processEvents()
        self.assertTrue(window.identity_filter_active)
        self.assertEqual(PhotoView.REVIEW, window.current_view)
        self.assertEqual(0, window.photo_model.rowCount())
        window.people_menu.rebuild()
        self.assertIsNotNone(window.people_menu._clear_filter_action)
        self.assertEqual(
            "Clear person filter", window.people_menu._clear_filter_action.text()
        )
        window.people_menu._clear_filter_action.trigger()
        app.processEvents()
        self.assertFalse(window.identity_filter_active)
        self.assertEqual(PhotoView.REVIEW, window.current_view)
        window.view_tabs.setCurrentIndex(0)
        app.processEvents()

        fourth_row_index = next(
            row
            for row in range(window.project_list.count())
            if window.project_list.item(row).data(Qt.UserRole) == fourth_project.id
        )
        window.project_list.setCurrentRow(fourth_row_index)
        app.processEvents()
        window.people_menu.rebuild()
        self.assertFalse(window.identity_filter_active)
        self.assertEqual(fourth_project.id, window.people_menu.project_id)
        self.assertEqual(
            [other_person_id], list(window.people_menu._person_actions)
        )
        self.assertNotIn(person.id, window.people_menu._person_actions)
        self.assertNotIn(named_other_id, window.people_menu._person_actions)

        self.assertEqual(second_project.id, second.project_id)
        self.assertEqual("Fifth", fifth_project.name)
        window.close()


if __name__ == "__main__":
    unittest.main()
