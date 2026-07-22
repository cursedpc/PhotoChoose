from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from photochoose.metadata import PhotoMetadata, read_photo_metadata
from photochoose.raw_preview import extract_embedded_jpeg

PREVIEW_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}
JPEG_EXTENSIONS = {".jpg", ".jpeg"}
RAW_EXTENSIONS = {
    ".3fr",
    ".arw",
    ".cr2",
    ".cr3",
    ".dng",
    ".erf",
    ".fff",
    ".iiq",
    ".kdc",
    ".mef",
    ".mos",
    ".mrw",
    ".nef",
    ".nrw",
    ".orf",
    ".pef",
    ".raf",
    ".raw",
    ".rw2",
    ".rwl",
    ".sr2",
    ".srf",
    ".srw",
    ".x3f",
}
SIDECAR_EXTENSIONS = {".xmp"}
SUPPORTED_ASSET_EXTENSIONS = PREVIEW_EXTENSIONS | RAW_EXTENSIONS | SIDECAR_EXTENSIONS


@dataclass(frozen=True)
class LogicalAsset:
    asset_key: str
    relative_path: str
    extension: str
    size_bytes: int
    modified_ns: int
    preview_relative_path: Optional[str]
    cached_preview_path: Optional[str]
    asset_paths: Tuple[str, ...]
    raw_count: int
    sidecar_count: int
    metadata: PhotoMetadata


def discover_logical_assets(
    root: Path,
    cache_directory: Path,
    existing_paths: Sequence[str] = (),
) -> Mapping[str, LogicalAsset]:
    grouped: Dict[Tuple[str, str], Dict[str, List[Path]]] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        extension = path.suffix.casefold()
        if extension not in SUPPORTED_ASSET_EXTENSIONS:
            continue
        parent_key = path.parent.relative_to(root).as_posix().casefold()
        stem = path.stem
        if extension in SIDECAR_EXTENSIONS:
            nested_extension = Path(stem).suffix.casefold()
            if nested_extension in PREVIEW_EXTENSIONS | RAW_EXTENSIONS:
                stem = Path(stem).stem
        key = (parent_key, stem.casefold())
        group = grouped.setdefault(key, {"preview": [], "raw": [], "sidecar": []})
        if extension in PREVIEW_EXTENSIONS:
            group["preview"].append(path)
        elif extension in RAW_EXTENSIONS:
            group["raw"].append(path)
        else:
            group["sidecar"].append(path)

    discovered: Dict[str, LogicalAsset] = {}
    existing = set(existing_paths)
    for key, group in grouped.items():
        previews = sorted(group["preview"], key=lambda path: path.name.casefold())
        raws = sorted(group["raw"], key=lambda path: path.name.casefold())
        sidecars = sorted(group["sidecar"], key=lambda path: path.name.casefold())
        jpeg_previews = [path for path in previews if path.suffix.casefold() in JPEG_EXTENSIONS]

        if raws:
            companion = jpeg_previews[0] if jpeg_previews else None
            image_members = raws + ([companion] if companion is not None else [])
            representative = _existing_or_preferred(root, image_members, existing, companion)
            members = image_members + sidecars
            asset = _logical_asset(
                root,
                _asset_key(*key),
                representative,
                companion,
                members,
                raws,
                sidecars,
            )
            if companion is None:
                cached = extract_embedded_jpeg(representative, cache_directory)
                asset = replace(
                    asset,
                    cached_preview_path=str(cached) if cached is not None else None,
                )
            discovered[asset.relative_path] = asset
            previews = [path for path in previews if path != companion]

        for preview in previews:
            members = [preview] + (sidecars if not raws else [])
            logical_key = _asset_key(*key)
            if raws or len(previews) > 1:
                logical_key += "|preview|" + preview.name.casefold()
            asset = _logical_asset(
                root,
                logical_key,
                preview,
                preview,
                members,
                [],
                sidecars if not raws else [],
            )
            discovered[asset.relative_path] = asset
    return discovered


def _logical_asset(
    root: Path,
    asset_key: str,
    representative: Path,
    preview: Optional[Path],
    members: Sequence[Path],
    raws: Sequence[Path],
    sidecars: Sequence[Path],
) -> LogicalAsset:
    image_members = [path for path in members if path.suffix.casefold() not in SIDECAR_EXTENSIONS]
    stats = [path.stat() for path in image_members]
    metadata_paths = list(raws) + ([preview] if preview is not None else [])
    metadata = _merge_metadata(read_photo_metadata(path) for path in metadata_paths)
    return LogicalAsset(
        asset_key=asset_key,
        relative_path=representative.relative_to(root).as_posix(),
        extension=representative.suffix.casefold(),
        size_bytes=sum(stat.st_size for stat in stats),
        modified_ns=max(stat.st_mtime_ns for stat in stats),
        preview_relative_path=(preview.relative_to(root).as_posix() if preview else None),
        cached_preview_path=None,
        asset_paths=tuple(path.relative_to(root).as_posix() for path in members),
        raw_count=len(raws),
        sidecar_count=len(sidecars),
        metadata=metadata,
    )


def _existing_or_preferred(
    root: Path,
    candidates: Sequence[Path],
    existing: set,
    preferred: Optional[Path],
) -> Path:
    for candidate in candidates:
        if candidate.relative_to(root).as_posix() in existing:
            return candidate
    return preferred or candidates[0]


def _asset_key(parent: str, stem: str) -> str:
    return "{0}/{1}".format(parent, stem) if parent != "." else stem


def _merge_metadata(values: Iterable[PhotoMetadata]) -> PhotoMetadata:
    merged = {field.name: None for field in fields(PhotoMetadata)}
    for metadata in values:
        for name in merged:
            if merged[name] is None:
                merged[name] = getattr(metadata, name)
    return PhotoMetadata(**merged)
