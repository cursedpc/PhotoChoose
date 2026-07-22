from __future__ import annotations

import hashlib
import mmap
import os
import struct
from pathlib import Path
from typing import Optional, Tuple

MIN_PREVIEW_EDGE = 320
MAX_EMBEDDED_JPEG_BYTES = 64 * 1024 * 1024
MAX_JPEG_CANDIDATES = 128


def extract_embedded_jpeg(raw_path: Path, cache_directory: Path) -> Optional[Path]:
    """Extract the largest plausible embedded JPEG without altering the RAW file."""

    try:
        stat = raw_path.stat()
    except OSError:
        return None
    fingerprint = "{0}\0{1}\0{2}".format(raw_path, stat.st_size, stat.st_mtime_ns)
    cache_name = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:24] + ".jpg"
    cache_path = cache_directory / cache_name
    if cache_path.is_file():
        return cache_path

    best: Optional[bytes] = None
    best_area = 0
    try:
        with raw_path.open("rb") as stream:
            with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
                offset = 0
                candidates = 0
                while candidates < MAX_JPEG_CANDIDATES:
                    start = data.find(b"\xff\xd8\xff", offset)
                    if start < 0:
                        break
                    end = data.find(b"\xff\xd9", start + 3)
                    if end < 0:
                        break
                    end += 2
                    offset = start + 3
                    candidates += 1
                    length = end - start
                    if length > MAX_EMBEDDED_JPEG_BYTES:
                        continue
                    candidate = data[start:end]
                    dimensions = _jpeg_dimensions(candidate)
                    if dimensions is None or min(dimensions) < MIN_PREVIEW_EDGE:
                        continue
                    area = dimensions[0] * dimensions[1]
                    if area > best_area:
                        best = candidate
                        best_area = area
    except (OSError, ValueError):
        return None

    if best is None:
        return None
    try:
        cache_directory.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_bytes(best)
        os.replace(str(temporary), str(cache_path))
    except OSError:
        return None
    return cache_path


def _jpeg_dimensions(data: bytes) -> Optional[Tuple[int, int]]:
    if not data.startswith(b"\xff\xd8"):
        return None
    offset = 2
    sof_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    while offset + 4 <= len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            return None
        marker = data[offset]
        offset += 1
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        if marker == 0xDA or offset + 2 > len(data):
            return None
        length = struct.unpack_from(">H", data, offset)[0]
        if length < 2 or offset + length > len(data):
            return None
        if marker in sof_markers and length >= 7:
            height, width = struct.unpack_from(">HH", data, offset + 3)
            return width, height
        offset += length
    return None
