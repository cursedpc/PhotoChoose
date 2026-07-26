from __future__ import annotations

import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path


def _load_normalizer():
    script = Path(__file__).parents[1] / "packaging/macos/normalize_openvino_arm64.py"
    spec = importlib.util.spec_from_file_location("normalize_openvino_arm64", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


normalizer = _load_normalizer()


def _arm64_macho(*, padding: bytes = bytes(16)) -> bytearray:
    header_size = 32
    segment_size = 72
    signature_command_size = 16
    commands_size = segment_size + signature_command_size
    linkedit_offset = header_size + commands_size
    signature_size = 32
    signature_offset = linkedit_offset
    signature_end = signature_offset + signature_size
    file_size = signature_end + len(padding)

    contents = bytearray(file_size)
    struct.pack_into(
        "<IiiIIIII",
        contents,
        0,
        normalizer.MH_MAGIC_64,
        normalizer.CPU_TYPE_ARM64,
        0,
        6,
        2,
        commands_size,
        0,
        0,
    )
    struct.pack_into(
        "<II16sQQQQiiII",
        contents,
        header_size,
        normalizer.LC_SEGMENT_64,
        segment_size,
        b"__LINKEDIT".ljust(16, b"\0"),
        0,
        0,
        linkedit_offset,
        file_size - linkedit_offset,
        0,
        0,
        0,
        0,
    )
    struct.pack_into(
        "<IIII",
        contents,
        header_size + segment_size,
        normalizer.LC_CODE_SIGNATURE,
        signature_command_size,
        signature_offset,
        signature_size,
    )
    contents[signature_offset:signature_end] = bytes(range(signature_size))
    contents[signature_end:] = padding
    return contents


class OpenVINOArm64NormalizerTests(unittest.TestCase):
    def test_known_zero_padding_is_removed_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            library = Path(temporary) / "libopenvino.dylib"
            contents = _arm64_macho()
            library.write_bytes(contents)

            self.assertTrue(normalizer.normalize_macho_signature_padding(library))
            self.assertEqual(len(contents) - 16, library.stat().st_size)

            normalized = library.read_bytes()
            linkedit_size = struct.unpack_from("<Q", normalized, 32 + 48)[0]
            self.assertEqual(32, linkedit_size)
            self.assertFalse(normalizer.normalize_macho_signature_padding(library))

    def test_unexpected_padding_is_rejected_without_modifying_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            library = Path(temporary) / "libopenvino.dylib"
            contents = _arm64_macho(padding=b"unexpected bytes")
            library.write_bytes(contents)

            with self.assertRaisesRegex(RuntimeError, "refusing to remove unexpected"):
                normalizer.normalize_macho_signature_padding(library)
            self.assertEqual(contents, library.read_bytes())


if __name__ == "__main__":
    unittest.main()
