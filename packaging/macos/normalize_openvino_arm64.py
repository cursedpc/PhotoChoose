"""Repair known OpenVINO ARM64 Mach-O padding before PyInstaller runs.

Some OpenVINO macOS ARM64 dylibs contain 16 zero bytes after the declared code
signature while ``__LINKEDIT`` still claims those bytes. Apple's binary tools
reject that inconsistent layout when PyInstaller adjusts and re-signs the
libraries. This script narrows ``__LINKEDIT`` to the signature boundary and
removes only that exact, known padding pattern.
"""

from __future__ import annotations

import argparse
import platform
import struct
import sys
from pathlib import Path

MH_MAGIC_64 = 0xFEEDFACF
CPU_TYPE_ARM64 = 0x0100000C
LC_SEGMENT_64 = 0x19
LC_CODE_SIGNATURE = 0x1D
MACH_HEADER_64_SIZE = 32
SEGMENT_COMMAND_64_SIZE = 72
LINKEDIT_FILESIZE_OFFSET = 48
KNOWN_PADDING_SIZE = 16


def normalize_macho_signature_padding(path: Path) -> bool:
    """Normalize the known ARM64 OpenVINO signature padding in *path*.

    Returns ``True`` only when the file was changed. Non-Mach-O and non-ARM64
    files are ignored. Unexpected layouts are rejected instead of being
    modified.
    """

    contents = bytearray(path.read_bytes())
    if len(contents) < MACH_HEADER_64_SIZE:
        return False

    magic, cpu_type = struct.unpack_from("<II", contents)
    if magic != MH_MAGIC_64 or cpu_type != CPU_TYPE_ARM64:
        return False

    command_count, command_bytes = struct.unpack_from("<II", contents, 16)
    commands_end = MACH_HEADER_64_SIZE + command_bytes
    if commands_end > len(contents):
        raise RuntimeError(f"{path}: Mach-O load commands extend beyond the file")

    linkedit: tuple[int, int, int] | None = None
    signature: tuple[int, int] | None = None
    command_offset = MACH_HEADER_64_SIZE

    for _ in range(command_count):
        if command_offset + 8 > commands_end:
            raise RuntimeError(f"{path}: truncated Mach-O load command")
        command, command_size = struct.unpack_from("<II", contents, command_offset)
        if command_size < 8 or command_offset + command_size > commands_end:
            raise RuntimeError(f"{path}: invalid Mach-O load command size")

        if command == LC_SEGMENT_64 and command_size >= SEGMENT_COMMAND_64_SIZE:
            segment_name = bytes(contents[command_offset + 8 : command_offset + 24])
            if segment_name.rstrip(b"\0") == b"__LINKEDIT":
                file_offset, file_size = struct.unpack_from(
                    "<QQ", contents, command_offset + 40
                )
                linkedit = (command_offset, file_offset, file_size)
        elif command == LC_CODE_SIGNATURE and command_size >= 16:
            signature = struct.unpack_from("<II", contents, command_offset + 8)

        command_offset += command_size

    if linkedit is None or signature is None:
        return False

    linkedit_command, linkedit_offset, linkedit_size = linkedit
    signature_offset, signature_size = signature
    linkedit_end = linkedit_offset + linkedit_size
    signature_end = signature_offset + signature_size
    file_size = len(contents)

    if signature_end == file_size:
        return False
    if linkedit_end != file_size or signature_end > file_size:
        raise RuntimeError(f"{path}: unexpected __LINKEDIT or code-signature boundary")

    padding = contents[signature_end:]
    if len(padding) != KNOWN_PADDING_SIZE or any(padding):
        raise RuntimeError(
            f"{path}: refusing to remove unexpected {len(padding)}-byte signature padding"
        )
    if signature_end < linkedit_offset:
        raise RuntimeError(f"{path}: code signature precedes __LINKEDIT")

    normalized_linkedit_size = signature_end - linkedit_offset
    struct.pack_into(
        "<Q",
        contents,
        linkedit_command + LINKEDIT_FILESIZE_OFFSET,
        normalized_linkedit_size,
    )
    path.write_bytes(contents[:signature_end])
    return True


def normalize_openvino_tree(root: Path) -> list[Path]:
    """Normalize affected ARM64 Mach-O files below an OpenVINO package root."""

    libraries = root / "libs"
    if not libraries.is_dir():
        raise RuntimeError(f"OpenVINO library directory was not found: {libraries}")

    normalized: list[Path] = []
    for path in sorted(candidate for candidate in libraries.rglob("*") if candidate.is_file()):
        if normalize_macho_signature_padding(path):
            normalized.append(path)
    return normalized


def _installed_openvino_root() -> Path:
    import openvino

    return Path(openvino.__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        help="OpenVINO package root; defaults to the installed package",
    )
    args = parser.parse_args()

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        if args.root is None:
            parser.error("automatic normalization is only supported on macOS ARM64")

    root = args.root.resolve() if args.root else _installed_openvino_root()
    normalized = normalize_openvino_tree(root)
    print(f"Normalized {len(normalized)} OpenVINO ARM64 Mach-O files under {root}")
    for path in normalized:
        print(f"  {path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
