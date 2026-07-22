from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from photochoose.database import PhotoDatabase
from photochoose.metadata import read_photo_metadata
from photochoose.models import PhotoView


def jpeg_with_test_exif() -> bytes:
    date = b"2026:07:22 14:15:16\x00"
    make = b"Canon\x00"
    model = b"EOS R6\x00"
    ifd0_offset = 8
    ifd0_size = 2 + 3 * 12 + 4
    make_offset = ifd0_offset + ifd0_size
    model_offset = make_offset + len(make)
    exif_offset = model_offset + len(model)
    exif_size = 2 + 6 * 12 + 4
    date_offset = exif_offset + exif_size
    exposure_offset = date_offset + len(date)
    aperture_offset = exposure_offset + 8
    focal_offset = aperture_offset + 8

    tiff = bytearray(b"II" + struct.pack("<HI", 42, ifd0_offset))
    tiff += struct.pack("<H", 3)
    tiff += struct.pack("<HHII", 0x010F, 2, len(make), make_offset)
    tiff += struct.pack("<HHII", 0x0110, 2, len(model), model_offset)
    tiff += struct.pack("<HHII", 0x8769, 4, 1, exif_offset)
    tiff += struct.pack("<I", 0)
    tiff += make + model
    tiff += struct.pack("<H", 6)
    tiff += struct.pack("<HHII", 0x9003, 2, len(date), date_offset)
    tiff += struct.pack("<HHI", 0x9291, 2, 4) + b"125\x00"
    tiff += struct.pack("<HHI", 0x8827, 3, 1) + struct.pack("<H", 800) + b"\x00\x00"
    tiff += struct.pack("<HHII", 0x829A, 5, 1, exposure_offset)
    tiff += struct.pack("<HHII", 0x829D, 5, 1, aperture_offset)
    tiff += struct.pack("<HHII", 0x920A, 5, 1, focal_offset)
    tiff += struct.pack("<I", 0)
    tiff += date
    tiff += struct.pack("<II", 1, 250)
    tiff += struct.pack("<II", 28, 10)
    tiff += struct.pack("<II", 85, 1)
    payload = b"Exif\x00\x00" + bytes(tiff)
    return b"\xff\xd8\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload + b"\xff\xd9"


class MetadataTests(unittest.TestCase):
    def test_reads_capture_and_camera_settings_from_exif(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frame.jpg"
            path.write_bytes(jpeg_with_test_exif())

            metadata = read_photo_metadata(path)

            self.assertEqual("2026-07-22T14:15:16.125000", metadata.captured_at)
            self.assertEqual("Canon", metadata.camera_make)
            self.assertEqual("EOS R6", metadata.camera_model)
            self.assertEqual(800, metadata.iso)
            self.assertAlmostEqual(1 / 250, metadata.exposure_seconds)
            self.assertAlmostEqual(2.8, metadata.aperture)
            self.assertAlmostEqual(85.0, metadata.focal_length)

    def test_scan_persists_exif_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shoot = root / "Shoot"
            shoot.mkdir()
            (shoot / "frame.jpg").write_bytes(jpeg_with_test_exif())
            database = PhotoDatabase(root / "state.sqlite3")
            project = database.add_project(shoot)

            database.scan_project(project.id)
            photo = database.list_photos(project.id, PhotoView.ALL)[0]

            self.assertEqual("2026-07-22T14:15:16.125000", photo.captured_at)
            self.assertEqual("EOS R6", photo.camera_model)
            self.assertEqual(800, photo.iso)


if __name__ == "__main__":
    unittest.main()
