from __future__ import annotations

import struct
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class PhotoMetadata:
    captured_at: Optional[str] = None
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    lens_model: Optional[str] = None
    iso: Optional[int] = None
    exposure_seconds: Optional[float] = None
    aperture: Optional[float] = None
    focal_length: Optional[float] = None


def read_photo_metadata(path: Path) -> PhotoMetadata:
    """Read a small, safe subset of TIFF/EXIF metadata without decoding pixels."""

    try:
        with path.open("rb") as stream:
            signature = stream.read(4)
            stream.seek(0)
            if signature[:2] == b"\xff\xd8":
                exif = _jpeg_exif(stream)
            elif signature[:2] in (b"II", b"MM"):
                exif = stream.read(1024 * 1024)
            else:
                exif = None
    except OSError:
        return PhotoMetadata()
    if not exif:
        return PhotoMetadata()
    try:
        return _parse_tiff_exif(exif)
    except (IndexError, OverflowError, struct.error, ValueError):
        return PhotoMetadata()


def _jpeg_exif(stream) -> Optional[bytes]:
    if stream.read(2) != b"\xff\xd8":
        return None
    while True:
        prefix = stream.read(1)
        if not prefix:
            return None
        if prefix != b"\xff":
            continue
        marker = stream.read(1)
        while marker == b"\xff":
            marker = stream.read(1)
        if not marker or marker in (b"\xd9", b"\xda"):
            return None
        if marker in (b"\x01",) or 0xD0 <= marker[0] <= 0xD7:
            continue
        length_bytes = stream.read(2)
        if len(length_bytes) != 2:
            return None
        length = struct.unpack(">H", length_bytes)[0]
        if length < 2:
            return None
        payload = stream.read(length - 2)
        if marker == b"\xe1" and payload.startswith(b"Exif\x00\x00"):
            return payload[6:]


def _parse_tiff_exif(data: bytes) -> PhotoMetadata:
    endian_marker = data[:2]
    if endian_marker == b"II":
        endian = "<"
    elif endian_marker == b"MM":
        endian = ">"
    else:
        raise ValueError("Invalid TIFF byte order")
    if _unpack(endian, "H", data, 2) != 42:
        raise ValueError("Invalid TIFF marker")

    ifd0_offset = _unpack(endian, "I", data, 4)
    ifd0 = _read_ifd(data, endian, ifd0_offset)
    exif_offset = _integer(ifd0.get(0x8769))
    exif = _read_ifd(data, endian, exif_offset) if exif_offset else {}

    date_text = _text(exif.get(0x9003)) or _text(ifd0.get(0x0132))
    subseconds = _text(exif.get(0x9291))
    captured_at = _capture_time(date_text, subseconds)
    return PhotoMetadata(
        captured_at=captured_at,
        camera_make=_text(ifd0.get(0x010F)),
        camera_model=_text(ifd0.get(0x0110)),
        lens_model=_text(exif.get(0xA434)),
        iso=_integer(exif.get(0x8827)),
        exposure_seconds=_number(exif.get(0x829A)),
        aperture=_number(exif.get(0x829D)),
        focal_length=_number(exif.get(0x920A)),
    )


def _read_ifd(data: bytes, endian: str, offset: int) -> Dict[int, object]:
    if offset < 0 or offset + 2 > len(data):
        raise ValueError("Invalid IFD offset")
    count = _unpack(endian, "H", data, offset)
    if count > 512:
        raise ValueError("Unreasonable IFD entry count")
    values: Dict[int, object] = {}
    type_sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 9: 4, 10: 8}
    for index in range(count):
        entry_offset = offset + 2 + index * 12
        if entry_offset + 12 > len(data):
            break
        tag, value_type = struct.unpack_from(endian + "HH", data, entry_offset)
        item_count = _unpack(endian, "I", data, entry_offset + 4)
        item_size = type_sizes.get(value_type)
        if item_size is None or item_count > 65536:
            continue
        byte_count = item_size * item_count
        if byte_count <= 4:
            raw = data[entry_offset + 8 : entry_offset + 8 + byte_count]
        else:
            value_offset = _unpack(endian, "I", data, entry_offset + 8)
            if value_offset < 0 or value_offset + byte_count > len(data):
                continue
            raw = data[value_offset : value_offset + byte_count]
        values[tag] = _decode_value(raw, endian, value_type, item_count)
    return values


def _decode_value(raw: bytes, endian: str, value_type: int, count: int):
    if value_type == 2:
        return raw.rstrip(b"\x00").decode("utf-8", errors="replace").strip()
    formats = {1: "B", 3: "H", 4: "I", 9: "i"}
    if value_type in formats:
        values = struct.unpack(endian + formats[value_type] * count, raw)
        return values[0] if count == 1 else values
    if value_type in (5, 10):
        number_format = "I" if value_type == 5 else "i"
        values = []
        for index in range(count):
            numerator, denominator = struct.unpack_from(
                endian + number_format * 2, raw, index * 8
            )
            values.append(float(numerator) / denominator if denominator else 0.0)
        return values[0] if count == 1 else tuple(values)
    return raw


def _unpack(endian: str, value_format: str, data: bytes, offset: int) -> int:
    return struct.unpack_from(endian + value_format, data, offset)[0]


def _text(value: object) -> Optional[str]:
    if isinstance(value, str) and value:
        return value
    return None


def _integer(value: object) -> Optional[int]:
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _number(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _capture_time(date_text: Optional[str], subseconds: Optional[str]) -> Optional[str]:
    if not date_text:
        return None
    try:
        captured = datetime.strptime(date_text.strip(), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None
    digits = "".join(character for character in (subseconds or "") if character.isdigit())
    if digits:
        captured = captured.replace(microsecond=int((digits + "000000")[:6]))
    return captured.isoformat(timespec="microseconds")
