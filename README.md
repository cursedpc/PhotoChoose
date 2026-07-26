# PhotoChoose

PhotoChoose is a private, local-first desktop application for professional photo culling. It provides project-folder navigation, fast thumbnail browsing, automatic baseline separation, and persistent manual `Keep`, `Review`, `Reject`, and `Best` decisions. It never modifies source photographs.

## Current behavior

- Import one or more shoot folders as projects.
- Automatically analyze a newly imported folder in a background worker.
- Re-run analysis with **Analyze and separate**.
- Detect unreadable images, exact duplicates, severe global blur, and extreme exposure.
- Detect frontal faces and five facial landmarks with the bundled YuNet ONNX model.
- Measure eyelid geometry with a bundled 478-point MediaPipe face-mesh model running
  locally through LiteRT, producing conservative open, closed, or uncertain states.
- Measure sharpness separately for detected faces and eyes, preserving the weakest
  primary-subject result for conservative burst ranking.
- Treat faces smaller than one quarter of the largest detected face as background evidence
  for ranking purposes, while still showing and counting them.
- Detect people and common animals, vehicles, furnishings, food, electronics, and other
  COCO objects with a bundled NanoDet model, then classify their importance as primary,
  supporting, or background.
- Find an unlabeled prominent region with a local spectral-saliency fallback when no known
  object class is detected, and measure focus inside every retained subject region.
- Read EXIF capture time, camera/lens identity, ISO, shutter, aperture, and focal length.
- Group bursts using capture timing, camera identity, and perceptual similarity, with a
  conservative visual fallback when EXIF is unavailable.
- Mark the strongest usable photograph in each similar group as Best.
- Promote a questionable grouped frame to Reject only when another frame from the same
  moment ranks at least 25 points higher; unique photos and manual decisions stay protected.
- Populate conservative `Keep`, `Review`, and `Reject` recommendations with visible reasons.
- Navigate projects from the quiet left library rail; project operations stay in the
  **Project** menu instead of competing with the photographic workspace.
- Right-click a project (or use a macOS two-finger secondary click) to rescan,
  analyze, export, or close it.
- Close a project to remove it from the sidebar while preserving metadata, analysis, and
  decisions. Generated RAW-only previews are deleted automatically and recreated if the
  project is opened again; original RAW/JPEG/XMP files are never touched.
- Browse `All`, `Best`, `Keep`, `Review`, and `Reject` inside the selected project.
- Classify the active selection in the contextual right inspector. Detailed focus, exposure,
  face/eye/object, similarity, ranking, and decision evidence stays collapsed until requested.
- Toggle the **Analysis overlay** (`O`) in the grid, fullscreen viewer, or comparison view
  to inspect primary/supporting object boxes and primary/background face boxes,
  open/closed/uncertain eye geometry, and local sharpness measurements.
- Automatically declutter crowded overlays: face evidence replaces redundant person boxes,
  background object guesses stay hidden, the strongest regions are capped, and text labels
  are reduced before useful eye-state colors and primary focus boxes.
- Right-click photographs to open fullscreen or apply Keep, Review, Reject, Clear,
  and Best decisions to the selection.
- Open **Compare group** for any blue `B####` burst or gray `G####` similarity badge.
  The two-up view provides synchronized zoom and pan, candidate cycling, score/reason
  evidence, and independent decisions for each frame.
- Cull in fullscreen using the **More** menu, context menu, or keyboard shortcuts and navigate with
  Left/Right, A/D, Space/Backspace, the centered decision-colored filmstrip, or a horizontal
  trackpad gesture; press `Esc` to close it. The filmstrip loads thumbnails lazily, keeps the
  current photo centered, and shows its exact position in the visible sequence.
- Mark selected photographs using the inspector, context menus, or keyboard shortcuts.
- Export kept and best photographs to a user-selected destination.
- Resume with the same projects and decisions after restarting.
- Rescan a project without losing decisions for files that still exist.
- Read JPEG, PNG, WebP, BMP, and TIFF previews.
- Recognize common Canon, Nikon, Sony, Fujifilm, Olympus/OM, Panasonic, Pentax,
  Hasselblad, Phase One, Samsung, Sigma, Leica, and DNG RAW extensions.
- Treat matching RAW, JPEG, and XMP files as one culling item and export the complete
  file set together.
- Extract sufficiently large embedded JPEG previews from RAW-only files into the local
  application cache. RAW files without a usable preview remain safely in Review.

`Best` is an independent flag. A photograph can be both `Keep` and `Best`, while `Keep`, `Review`, and `Reject` are mutually exclusive decisions.

Manual decisions are protected: after you explicitly classify a photograph, subsequent analysis runs update its metrics and reasons but do not overwrite your choice.

Each project has a **Safe**, **Balanced**, or **Strict** analysis profile in **Settings**.
Safe is the conservative default. Changing the profile marks existing results as stale but does not
silently reclassify anything; click **Analyze and separate** when you are ready to apply the
new profile. Every result stores its profile and analysis version for later inspection.
Manual decision and Best changes are recorded only in the local SQLite database as private
calibration evidence; photographs and calibration history are never uploaded.

Open **Calibration report…** for the selected project to summarize automatic-versus-manual
decision disagreements, Best corrections, affected analysis signals, and profile versions.
Suggestions require repeated matching corrections and remain informational: PhotoChoose does
not modify thresholds automatically. Small local tests validate the report machinery, while
meaningful threshold tuning and performance validation require representative large shoots on
the target machine.

**Settings** also sets **Best per group** from 1–5. PhotoChoose reuses stored group rankings,
so changing this value does not rerun image analysis. Automatic selections favor the highest
scores while skipping near-identical perceptual hashes; the limit is therefore a maximum, not
a promise to fill a group with redundant frames. Explicit manual Best choices are always
preserved and count toward the selected maximum.

The analysis intentionally sends uncertain soft images, confidently detected closed eyes,
uncertain or obscured/unresolved eye states, and very soft primary detected objects to `Review`;
it does not reject a unique photograph solely from those model signals. Within a burst, open
eyes and sharper
primary subjects improve ranking. The current face models work best on sufficiently large,
near-frontal faces, and the general detector recognizes only its COCO-80 classes. Heavily
obscured eyes, very small faces, expressions, uncatalogued subjects, and the initial profile
thresholds still need photographer-corpus calibration.

Eye visibility is handled separately from eyelid geometry. If an eye region is too small,
dark/covered, glare-clipped, or lacks sufficient local contrast/detail, PhotoChoose stores an
**Obscured** state and its reason instead of trusting an open/closed estimate. The purple overlay
and inspector expose this evidence. These conservative visibility heuristics still require
large-shoot validation on the target machine, especially for glasses and profile faces.

Object class names are confidence-qualified model guesses, not authoritative metadata. A
wrong class name can still surround the correct subject and provide a useful regional-focus
measurement. PhotoChoose therefore keeps class names out of the normal overlay and exposes
them as `Likely …` details in the photograph tooltip.

## Interface direction

PhotoChoose uses a restrained dark studio interface built from three stable zones: library,
photographic canvas, and contextual inspector. These zones form one continuous graphite workspace
separated by thin structural dividers rather than floating cards. Native system typography,
restrained spacing, and one blue primary action replace decorative labels and a wall of equally
weighted buttons. It borrows the working rhythm of professional creative software without copying
another product's panels or iconography. Qt's Fusion base and the shared theme keep the hierarchy
consistent on macOS and Windows; platform scaling still requires review on the target machine.
Shoot folders can be dragged from Finder or Windows Explorer onto the application. Multiple
folders are accepted together and processed sequentially so model analysis does not compete with
itself for CPU and memory; ordinary files and non-local links are ignored.

## Local subject models

All three neural model files are installed with PhotoChoose and run through OpenVINO on the CPU.
Analysis does not contact a server or download model files at runtime. If OpenVINO model loading
or inference fails, PhotoChoose automatically returns to OpenCV DNN for YuNet and NanoDet and
LiteRT for face mesh. Its earlier OpenCV Haar face/eye checks and spectral-saliency fallback remain
available when the neural models cannot provide evidence. Analysis details identify the runtime
that actually produced each result.

- Face detection: OpenCV Zoo YuNet, MIT licensed, executed through OpenVINO CPU.
- Eye state: MediaPipe Face Mesh V2 landmarks, Apache-2.0 licensed, executed locally
  through OpenVINO CPU.
- General objects: OpenCV Zoo NanoDet, Apache-2.0 licensed, executed locally through
  OpenVINO CPU with a model-free saliency fallback.
- Exact upstream locations, licenses, filenames, and SHA-256 hashes are recorded in
  `src/photochoose/model_assets/MODELS.md`.

The current inference path is entirely local and CPU-only. OpenVINO CPU is implemented and tested
for output parity on the macOS development machine; throughput, memory, and thermal validation on
large shoots remains for the target Windows 11 machine. Intel GPU and OpenVINO `AUTO` execution are
outside the current implementation scope.

## Export behavior

**Export Keep + Best…** creates `Keep` and `Best` folders inside the destination you choose and preserves the shoot's relative subfolder structure. A photograph marked both Keep and Best is copied into both collections. Existing destination files are skipped rather than overwritten, and the source shoot folder is never modified. PhotoChoose refuses to export inside the source shoot because a later rescan would otherwise import the copies again.

Closing a project is also non-destructive. It clears only regenerable RAW preview files to
release disk space. Adding the same source folder again reopens its existing database record
with prior classifications intact and recreates any required RAW previews.

## RAW and sidecar behavior

A set such as `IMG_1234.CR3`, `IMG_1234.JPG`, and `IMG_1234.XMP` appears once in the
grid. The JPEG is used for local analysis and viewing while Keep, Review, Reject, and Best
apply to the complete set. Purple thumbnail badges show `RAW`, `RAW+JPG`, and `XMP`
membership. Export preserves original filenames and relative folders; extracted cache
previews are never exported as source assets.

RAW-only support depends on the camera file containing a discoverable embedded JPEG of
useful size. PhotoChoose does not develop RAW sensor data, alter originals, or classify an
unpreviewable RAW as a technical failure.

## Development setup

PhotoChoose requires Python 3.11. Keeping one Python minor version across macOS development,
Windows delivery, and CI avoids native-wheel differences and prepares the inference layer for the
selected OpenVINO CPU runtime.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m photochoose
```

On Windows, create the environment with `py -3.11 -m venv .venv`, then activate it with
`.venv\\Scripts\\activate` before installing and running the application. Confirm the selected
interpreter with `python -VV`; it must report Python 3.11 before dependencies are installed.

## Windows and macOS release builds

Windows releases use a PyInstaller one-folder application wrapped in an Inno Setup installer under
Program Files. The installer requests elevation, and uninstalling it preserves the
PhotoChoose catalog under `%APPDATA%\\PhotoChoose\\PhotoChoose`. The same application folder is
also emitted as a portable ZIP for target-machine diagnostics.

The build must run on Windows; PyInstaller does not cross-build Windows executables from macOS.
Version tags and manual workflow runs execute the release builders in
`.github/workflows/windows-installer.yml`. A matching tag such as `v0.1.4` publishes the tested
installer, portable ZIP, native Intel and Apple Silicon macOS DMGs, and SHA-256 checksums directly
to GitHub Releases. The macOS apps are built on matching native GitHub runners and are intentionally
unsigned. Local build instructions are in `packaging/windows/README.md` and
`packaging/macos/README.md`. To upgrade on Windows, download and run the newer setup file directly:
its stable application identity reuses the existing installation and preserves the catalog and
settings, so PhotoChoose does not need to be uninstalled first.

### Install on macOS

Download the DMG matching the Mac—`Apple-Silicon` for M-series Macs or `Intel` for Intel Macs—open
it, and drag `PhotoChoose.app` into `Applications`. The installation and first-launch procedure is
the same on Intel and Apple Silicon; only the downloaded DMG differs. The Apple Silicon build is
native ARM64 and does not require Rosetta.

Because both builds are unsigned, remove macOS quarantine from the newly installed copy and open it
from Terminal:

```bash
xattr -dr com.apple.quarantine /Applications/PhotoChoose.app
open /Applications/PhotoChoose.app
```

This is required only once for each newly downloaded or replaced copy of the app, not on every
launch. As an alternative, try opening the app and then select **Open Anyway** under **System
Settings → Privacy & Security**. Updating currently means replacing `PhotoChoose.app` with the
version from the new DMG; the catalog and settings in Application Support are preserved.

## Keyboard workflow

The letter shortcuts use physical key positions on Windows and macOS, so they keep working when a
non-English keyboard layout is active.

- `K`: Keep
- `R`: Review
- `X`: Reject
- `U`: Clear the decision
- `B`: Toggle Best
- `O`: Toggle face/eye/object analysis evidence
- `Cmd/Ctrl+A`: Select every visible photograph
- `Left` / `Right` (fullscreen): previous / next photo
- Horizontal swipe or two-finger scroll (fullscreen): previous / next photo
- `Left` / `Right` (comparison): previous / next candidate
- `+` / `-` / `0` (comparison): zoom in / zoom out / fit

The local database and cached application state are stored in the operating system's standard per-user application-data directory, not in the source shoot folder.

### macOS Qt plugin repair

PhotoChoose checks Qt's local plugin directory before opening the first window. If macOS has applied `UF_HIDDEN` to plugins inside a dot-prefixed virtual environment, PhotoChoose clears that flag so the `cocoa` platform and JPEG reader remain discoverable. This repair is limited to the installed PySide6 plugin directory and does not touch photographs.

## Tests

```bash
QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -v
```

See `ROADMAP.md` for the planned analysis pipeline, RAW support, best-shot ranking, and Intel/OpenVINO validation.
