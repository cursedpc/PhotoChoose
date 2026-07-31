# PhotoChoose

PhotoChoose is a private, local-first desktop application for reviewing large photo shoots.
It helps you find strong frames, compare similar photographs, organize people, and export your
selection without modifying the originals.

[Download the latest release](https://github.com/cursedpc/easyphoto/releases/latest)

## What PhotoChoose does

- Analyzes photographs locally and suggests **Keep**, **Review**, or **Reject**.
- Keeps **Best** as a manual mark that only you control.
- Groups bursts and similar photographs for side-by-side comparison.
- Detects faces, eyes, and common objects, with an optional evidence overlay.
- Recognizes recurring people across folders and lets you name, merge, correct, and filter them.
- Combines People with **All**, **Best**, **Keep**, **Review**, and **Reject** filters.
- Watches open folders and processes files added or changed later.
- Exports selected photographs or separate Keep and Best collections.
- Preserves RAW companions, Live Photo video, XMP, and AAE sidecars.
- Stores decisions between sessions and never alters source photographs.

All analysis runs on the computer's CPU. Photographs, face identities, and review history are not
uploaded.

## Install

### Windows

Download the Windows x64 setup file from
[Releases](https://github.com/cursedpc/easyphoto/releases/latest) and run it. A newer setup file
upgrades the existing installation while preserving the local catalog and settings.

### macOS

Download the DMG matching your Mac:

- **Apple Silicon** for M-series Macs.
- **Intel** for Intel Macs.

Drag **PhotoChoose.app** into **Applications**. The current builds are unsigned, so run these
commands once after each newly downloaded or replaced copy:

~~~bash
xattr -dr com.apple.quarantine /Applications/PhotoChoose.app
open /Applications/PhotoChoose.app
~~~

The Apple Silicon build is native ARM64 and does not require Rosetta.

## Quick start

1. Choose **Folder → Add folder**, or drag one or more photo folders into PhotoChoose.
2. Let the initial scan and local analysis finish.
3. Browse **All**, **Best**, **Keep**, **Review**, and **Reject**.
4. Use **People** to view one person, then combine it with any decision filter.
5. Open a photograph for fullscreen review, or compare frames from the same group.
6. Apply your decisions and export from the **File** menu or photograph context menu.

Choose **Reanalyze** after changing the Safe, Balanced, or Strict analysis profile. Manual
decisions remain protected when analysis runs again.

## Review tools

### Fullscreen and comparison

Fullscreen review provides a centered filmstrip, quick decisions, zoom, and pan. Use a trackpad
pinch, **Ctrl/Cmd + mouse wheel**, or the zoom slider. Drag an enlarged photograph to inspect
another area. Comparison mode provides the same zoom controls for similar frames.

Press **O** to show the analysis overlay. Its **Details** menu can show or hide individual faces,
eyes, and objects for the current photograph.

### People

PhotoChoose automatically groups likely instances of the same person. You can:

- Give a person a name.
- Merge duplicate identities.
- Remove an incorrect match with **Not [person]**.
- Browse the person's photographs in the open folder.
- Combine the person with Keep, Review, Reject, Best, or All.
- Use **Clear person filter** to return to the normal folder view.

Named identities remain available for matching and manual merging across folders. Uncertain faces
stay under **Unknown** instead of being forced into an identity.

### Export

- **Export selected…** copies exactly the selected photographs.
- **Export Keep…** copies only Keep photographs.
- **Export Best…** copies only Best photographs.

Files go directly into the destination you choose. PhotoChoose preserves required relative paths
and companion assets, skips existing destination files, and refuses to export inside the source
folder.

## Supported photographs

PhotoChoose displays and analyzes:

- JPEG
- HEIC and HEIF
- PNG
- WebP
- BMP
- TIFF

Common Canon, Nikon, Sony, Fujifilm, Olympus/OM, Panasonic, Pentax, Hasselblad, Phase One,
Samsung, Sigma, Leica, and DNG RAW extensions are recognized. RAW-only photographs use a
sufficiently large embedded JPEG preview when one is available; PhotoChoose does not develop RAW
sensor data.

Matching RAW, JPEG/HEIC preview, XMP/AAE sidecar, and Live Photo MOV files appear as one item and
are exported together. HEIC Live Photo video is preserved but is not played or analyzed.

## Keyboard shortcuts

The letter shortcuts follow physical key positions, so they continue working with non-English
keyboard layouts.

| Key | Action |
| --- | --- |
| **K** | Keep |
| **R** | Review |
| **X** | Reject |
| **U** | Clear decision |
| **B** | Toggle Best |
| **O** | Toggle analysis overlay |
| **Cmd/Ctrl+A** | Select every visible photograph |
| **Left / Right** | Previous / next photograph |
| **+ / - / 0** | Zoom in / zoom out / fit |
| **Esc** | Close fullscreen or comparison |

## Local data

The catalog contains file references, decisions, analysis results, and local face-identity
embeddings. It is stored in the operating system's per-user application-data directory, never
inside the photo folder:

- Windows: **%APPDATA%\PhotoChoose\PhotoChoose**
- macOS: **~/Library/Application Support/PhotoChoose/PhotoChoose**

Closing a folder removes it from the sidebar but preserves its catalog information. Original
photographs are never deleted or rewritten.

## Run from source

PhotoChoose requires Python 3.11.

~~~bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m photochoose
~~~

On Windows, create the environment with **py -3.11 -m venv .venv**, activate it with
**.venv\Scripts\activate**, and then run the same install and launch commands.

Run the test suite with:

~~~bash
QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests
~~~

Platform packaging instructions are available in
[packaging/windows/README.md](packaging/windows/README.md) and
[packaging/macos/README.md](packaging/macos/README.md).
