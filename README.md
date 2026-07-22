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
- Read EXIF capture time, camera/lens identity, ISO, shutter, aperture, and focal length.
- Group bursts using capture timing, camera identity, and perceptual similarity, with a
  conservative visual fallback when EXIF is unavailable.
- Mark the strongest usable photograph in each similar group as Best.
- Promote a questionable grouped frame to Reject only when another frame from the same
  moment ranks at least 25 points higher; unique photos and manual decisions stay protected.
- Populate conservative `Keep`, `Review`, and `Reject` recommendations with visible reasons.
- Navigate projects from the left sidebar.
- Right-click a project (or use a macOS two-finger secondary click) to rescan,
  analyze, export, or close it.
- Close a project to remove it from the sidebar while preserving metadata, analysis, and
  decisions. Generated RAW-only previews are deleted automatically and recreated if the
  project is opened again; original RAW/JPEG/XMP files are never touched.
- Browse `All`, `Best`, `Keep`, `Review`, and `Reject` inside the selected project.
- Inspect focus, exposure, face/eye, similarity, ranking, and decision reasons in a
  collapsible panel above the thumbnail grid.
- Toggle the **Analysis overlay** (`O`) in the grid, fullscreen viewer, or comparison view
  to inspect primary/background face boxes, open/closed/uncertain eye geometry, and local
  sharpness measurements.
- Right-click photographs to open fullscreen or apply Keep, Review, Reject, Clear,
  and Best decisions to the selection.
- Open **Compare group** for any blue `B####` burst or gray `G####` similarity badge.
  The two-up view provides synchronized zoom and pan, candidate cycling, score/reason
  evidence, and independent decisions for each frame.
- Cull in fullscreen using its toolbar or context menu and navigate with Left/Right,
  A/D, Space/Backspace, or a horizontal trackpad gesture; press `Esc` to close it.
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

The analysis intentionally sends uncertain soft images, confidently detected closed eyes,
and uncertain eye states to `Review`; it does not reject a unique portrait merely because a
model suspects a blink. Within a burst, closed eyes reduce the ranking score so an otherwise
similar open-eye frame is preferred. The current models work best on sufficiently large,
near-frontal faces. Profiles, heavily obscured eyes, very small faces, expressions, and
general non-human subject detection still need later models and photographer-corpus
calibration.

## Local subject models

Both neural model files are installed with PhotoChoose. YuNet runs through OpenCV DNN;
the face-mesh model runs through LiteRT CPU on the current macOS development machine.
Analysis does not contact a server or download model files at runtime. If model loading or
inference fails, PhotoChoose falls back to its earlier OpenCV Haar face/eye checks and marks
the backend in the analysis details.

- Face detection and landmarks: OpenCV Zoo YuNet, MIT licensed.
- Eye state: MediaPipe Face Mesh V2 landmarks, Apache-2.0 licensed, executed locally
  through LiteRT.
- Exact upstream locations, licenses, filenames, and SHA-256 hashes are recorded in
  `src/photochoose/model_assets/MODELS.md`.

YuNet remains ONNX/OpenVINO compatible, while OpenVINO can import the face-mesh TFLite
model. The currently selected runtime is LiteRT CPU because it is stable and entirely local
on both development and target platforms. Actual Intel CPU, integrated-GPU, and OpenVINO
`AUTO` performance will be measured on the target Windows 11 machine; the present macOS
work does not claim those performance numbers yet.

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

Python 3.11 is recommended. Python 3.9 remains supported for the initial macOS development environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m photochoose
```

On Windows, activate the environment with `.venv\\Scripts\\activate` before installing and running the application.

## Keyboard workflow

- `K`: Keep
- `R`: Review
- `X`: Reject
- `U`: Clear the decision
- `B`: Toggle Best
- `O`: Toggle face/eye analysis evidence
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
