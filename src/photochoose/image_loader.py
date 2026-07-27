from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pi_heif
from PIL import Image, ImageCms, ImageOps
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QImage, QImageIOHandler, QImageReader

HEIF_EXTENSIONS = {".heic", ".heif"}
PILLOW_ANALYSIS_EXTENSIONS = HEIF_EXTENSIONS | {".png"}
PILLOW_UI_EXTENSIONS = PILLOW_ANALYSIS_EXTENSIONS


class ImageDecodeError(RuntimeError):
    """A supported image could not be decoded safely."""


def is_heif_path(path: str | Path) -> bool:
    return Path(path).suffix.casefold() in HEIF_EXTENSIONS


def read_qimage(path: str | Path, target_size: Optional[QSize] = None) -> QImage:
    """Decode an oriented image, using the portable HEIF backend when needed."""

    if Path(path).suffix.casefold() in PILLOW_UI_EXTENSIONS:
        try:
            image = _flatten_alpha(_open_pillow_image(Path(path)))
            if target_size is not None and target_size.isValid():
                image.thumbnail(
                    (max(1, target_size.width()), max(1, target_size.height())),
                    Image.Resampling.LANCZOS,
                )
            return _pillow_to_qimage(image)
        except (IndexError, OSError, RuntimeError, TypeError, ValueError):
            return QImage()

    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    if target_size is not None and target_size.isValid():
        source_size = reader.size()
        if source_size.isValid():
            reader.setScaledSize(
                _qt_decode_size(
                    source_size,
                    target_size,
                    _transformation_rotates(reader.transformation()),
                )
            )
    return reader.read()


def decode_analysis_bgr(path: str | Path, encoded: Optional[bytes] = None) -> np.ndarray:
    """Return an oriented 8-bit BGR image for the analysis pipeline."""

    source = Path(path)
    payload = encoded if encoded is not None else source.read_bytes()
    if source.suffix.casefold() not in PILLOW_ANALYSIS_EXTENSIONS:
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ImageDecodeError("Image data cannot be decoded")
        return image

    try:
        image = _open_pillow_image(source, payload)
        image = _flatten_alpha(image)
        rgb = np.asarray(image, dtype=np.uint8)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ImageDecodeError("Image data cannot be decoded") from exc
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ImageDecodeError("Image data has an unsupported pixel layout")
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def read_heif_metadata_blocks(path: Path) -> Tuple[Optional[bytes], Tuple[bytes, ...]]:
    """Return primary-image EXIF and XMP-like metadata without decoding pixels."""

    try:
        container = pi_heif.open_heif(path)
        primary = container[container.primary_index]
    except (IndexError, OSError, RuntimeError, TypeError, ValueError):
        return None, ()
    info = primary.info
    exif = info.get("exif")
    xmp_blocks = []
    direct_xmp = info.get("xmp")
    if isinstance(direct_xmp, bytes):
        xmp_blocks.append(direct_xmp)
    for item in info.get("metadata") or ():
        if not isinstance(item, dict):
            continue
        content_type = str(item.get("content_type") or "").casefold()
        metadata_type = str(item.get("type") or "").casefold()
        data = item.get("data")
        if isinstance(data, bytes) and (
            "xml" in content_type or "rdf" in content_type or metadata_type == "xmp"
        ):
            xmp_blocks.append(data)
    return exif if isinstance(exif, bytes) else None, tuple(xmp_blocks)


def heif_primary_image_count(path: Path) -> int:
    """Return the number of top-level HEIF images, excluding auxiliary images."""

    try:
        return len(pi_heif.open_heif(path))
    except (IndexError, OSError, RuntimeError, TypeError, ValueError):
        return 0


def _open_pillow_image(path: Path, encoded: Optional[bytes] = None) -> Image.Image:
    if path.suffix.casefold() in HEIF_EXTENSIONS:
        source = BytesIO(encoded) if encoded is not None else path
        container = pi_heif.open_heif(source)
        if not len(container):
            raise ImageDecodeError("HEIF container has no primary image")
        image = container[container.primary_index].to_pillow()
    else:
        source = BytesIO(encoded) if encoded is not None else path
        image = Image.open(source)
        image.load()
    image = ImageOps.exif_transpose(image)
    return _convert_to_srgb(image)


def _convert_to_srgb(image: Image.Image) -> Image.Image:
    profile = image.info.get("icc_profile")
    if not isinstance(profile, bytes) or not profile:
        return image
    try:
        alpha = image.getchannel("A") if "A" in image.getbands() else None
        converted = ImageCms.profileToProfile(
            image.convert("RGB"),
            ImageCms.ImageCmsProfile(BytesIO(profile)),
            ImageCms.createProfile("sRGB"),
            outputMode="RGB",
        )
        if alpha is not None:
            converted.putalpha(alpha)
        return converted
    except (ImageCms.PyCMSError, OSError, ValueError):
        return image


def _flatten_alpha(image: Image.Image) -> Image.Image:
    if "A" not in image.getbands():
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (127, 127, 127, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def _pillow_to_qimage(image: Image.Image) -> QImage:
    rgba = image.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    return QImage(
        data,
        rgba.width,
        rgba.height,
        rgba.width * 4,
        QImage.Format_RGBA8888,
    ).copy()


def _qt_decode_size(source_size: QSize, target_size: QSize, rotated: bool) -> QSize:
    oriented_source = (
        QSize(source_size.height(), source_size.width()) if rotated else source_size
    )
    oriented_result = oriented_source.scaled(target_size, Qt.KeepAspectRatio)
    if rotated:
        return QSize(oriented_result.height(), oriented_result.width())
    return oriented_result


def _transformation_rotates(
    transformation: QImageIOHandler.Transformation,
) -> bool:
    return transformation in {
        QImageIOHandler.TransformationRotate90,
        QImageIOHandler.TransformationMirrorAndRotate90,
        QImageIOHandler.TransformationFlipAndRotate90,
        QImageIOHandler.TransformationRotate270,
    }
