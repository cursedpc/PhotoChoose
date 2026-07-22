# PhotoChoose Roadmap

## Product goal

PhotoChoose is a private, local-first desktop application for professional photo culling. Its primary job is to separate clearly bad or redundant photographs from usable photographs so that a photographer does not need to manually inspect every frame. Its secondary job is to rank the strongest photographs within bursts and visually similar groups.

The initial development environment is macOS. The target production environment is Windows 11 on an Intel CPU with Intel integrated graphics. The application must remain usable in CPU-only mode.

## Product principles

- **Conservative rejection:** false rejection is more costly than leaving a questionable image for review.
- **Non-destructive by default:** analysis never deletes or modifies an original photograph.
- **Explainable decisions:** every classification includes human-readable reasons and component scores.
- **Compare related photographs:** closed eyes or weaker focus should cause rejection primarily when a better photograph of the same moment exists.
- **Local processing:** photographs, previews, embeddings, and analysis results stay on the computer.
- **Responsive desktop workflow:** scanning runs in background workers and can be paused or resumed.
- **Hardware fallback:** Intel GPU acceleration is optional; unsupported models automatically run on the CPU.

## Intended workflow

1. Drag a shoot folder into the application or select it with a folder picker.
2. Navigate imported project/shoot folders from the folder-only sidebar.
3. Inside the selected folder, switch between **All**, **Best**, **Keep**, **Review**, and **Reject**.
4. Discover supported photographs and associate RAW+JPEG pairs and sidecar files.
5. Extract embedded previews where possible instead of decoding full-resolution RAW files.
6. Run inexpensive technical checks and group duplicates, bursts, and similar scenes.
7. Analyze faces, eyes, subjects, focus, exposure, and image quality.
8. Classify photographs into **Keep**, **Review**, and **Reject**. These states are mutually exclusive.
9. Mark the highest-ranked photographs as **Best**, an independent flag normally applied to kept photographs.
10. Let the photographer compare, override, and lock decisions.
11. On explicit confirmation, export XMP ratings or copy/move complete file groups.

## Initial scope

### Required

- Native-style graphical interface for macOS development and Windows 11 delivery.
- Folder selection and drag-and-drop import.
- JPEG support and common camera RAW formats.
- RAW+JPEG and sidecar-aware file grouping.
- Resumable analysis with visible progress and cancellation.
- Duplicate and near-duplicate detection.
- Burst and visually similar photograph grouping.
- Global and subject-aware blur/focus analysis.
- Face detection, eye-state estimation, and face/eye sharpness analysis.
- Basic exposure and clipping checks.
- Conservative Keep/Review/Reject classification.
- Best-in-group ranking.
- Side-by-side comparison of related photographs.
- Reasons and confidence for every recommendation.
- Local database for cached results and user overrides.
- Non-destructive XMP or folder export after confirmation.

### Deferred

- RAW development or color editing.
- Cloud synchronization, accounts, or collaborative galleries.
- Training large models from scratch.
- General-purpose photo-library management.
- Automatic permanent deletion.
- Mobile application.

## Proposed technical direction

- **Language:** Python, using a supported 3.x version shared by macOS and Windows.
- **Desktop UI:** PySide6/Qt.
- **Inference format:** ONNX where practical.
- **Target acceleration:** OpenVINO with `AUTO` selection for Intel GPU/CPU and an explicit CPU mode.
- **Image processing:** OpenCV plus Pillow; LibRaw/rawpy or ExifTool for RAW previews and metadata.
- **Storage:** SQLite for scan state, scores, groups, decisions, and overrides.
- **Background work:** separate worker processes for decoding and inference so the UI remains responsive.
- **Packaging:** evaluate PyInstaller and Nuitka after the inference pipeline is stable.
- **Testing:** pytest for logic and regression tests, plus a fixed, photographer-labeled image corpus.

The inference layer must be isolated behind a small backend interface. macOS development can use CPU inference while Windows uses the same model files through OpenVINO. No feature should require the Intel GPU to function.

## Analysis pipeline

### Tier 1: inexpensive checks

- Readability and file-integrity checks.
- Embedded-preview extraction.
- Perceptual hashes for exact and near duplicates.
- Capture-time and metadata-based burst candidates.
- Histogram, clipping, and simple sharpness measurements.

Tier 1 reduces the number of expensive model evaluations and should run on all photographs.

### Tier 2: subject-aware analysis

- Detect people, faces, and primary non-human subjects.
- Estimate whether each visible eye is open, closed, obscured, or uncertain.
- Measure sharpness on faces, eyes, and the primary subject rather than relying only on global sharpness.
- Estimate technical image quality at preview resolution.
- Produce embeddings for visually similar grouping where Tier 1 is insufficient.

### Tier 3: contextual decisions

- Compare candidates only within duplicate, burst, or similar-scene groups.
- Prefer open eyes, sharper important subjects, usable exposure, and stronger expressions.
- Protect unique photographs from aggressive automatic rejection.
- Assign one of three outcomes:
  - **Keep:** usable or the strongest representative of a group.
  - **Review:** uncertain, unique, intentionally blurred, partially occluded, or near a threshold.
  - **Reject:** high-confidence technical failure or redundant weaker alternative.
- Rank the strongest candidates independently of the rejection decision.

## Milestones

### M0 — Repository and decision log

- Establish repository structure, dependency management, coding conventions, and test commands.
- Record supported input formats and target Intel hardware when available.
- Assemble a small representative development corpus without committing private photographs.
- Define a dependency/model license inventory even though the application is private.

**Exit:** a reproducible macOS development setup and documented architecture decisions.

### M1 — Desktop shell and scan database

- Build the Qt application shell.
- Implement folder picker, drag-and-drop, navigation, and settings.
- Add SQLite schema and migrations.
- Implement discovery, metadata indexing, progress, cancellation, and resume.
- Display fast cached thumbnails in a virtualized grid.

**Exit:** thousands of files can be indexed and browsed without analysis or UI stalls.

### M2 — Safe file and preview handling

- Support JPEG first, followed by prioritized RAW formats.
- Extract embedded RAW previews and normalize orientation and color metadata.
- Associate RAW+JPEG pairs, XMP files, and other sidecars.
- Add a file-operation planner with collision checks, dry-run output, and undo information.

**Exit:** scans and planned exports preserve complete photograph groups and never alter originals unexpectedly.

### M3 — Baseline technical rejection

- Add corrupt-file, extreme exposure, global blur, and duplicate checks.
- Calibrate thresholds against labeled photographs.
- Add Keep/Review/Reject views and per-image explanations.
- Ensure low-confidence failures go to Review.

**Exit:** the application safely removes a useful portion of obvious failures without relying on neural models.

### M4 — Faces, eyes, and subject focus

- **Implemented baseline:** bundled YuNet ONNX face detection and five landmarks,
  with a local Haar fallback if model inference fails.
- **Implemented baseline:** bundled MediaPipe Face Mesh V2 landmarks executed locally
  through LiteRT CPU, with conservative eyelid-geometry open, closed, and uncertain results.
- **Implemented:** per-face and per-eye sharpness, face confidence, eye-open score,
  normalized region evidence persisted in SQLite, and visible explanations in the UI.
- **Implemented:** primary-face weighting for group photographs; small background faces
  remain visible but do not spoil ranking of the main subjects.
- **Implemented:** toggleable evidence overlays in the grid, fullscreen viewer, and
  side-by-side comparison, including face importance and eye state/focus.
- Add a compact local detector for people and primary non-human subjects such as animals,
  vehicles, products, documents, flowers, and prominent architectural elements.
- Rank detected objects as primary, supporting, or background evidence using size,
  position, confidence, and overlap with detected faces.
- Measure sharpness inside each important object region, persist the evidence, and display
  it through the existing analysis overlay.
- Use general-subject sharpness in separation and Best ranking conservatively: a unique
  photograph must not be rejected solely because object importance is uncertain.
- Add explicit obscured-eye separation beyond the current uncertain state.

**Exit:** portrait and event photographs receive reliable, explainable subject-aware quality signals on macOS CPU.

### M5 — Bursts, similar scenes, and best shots

- Group photographs using timestamps, perceptual similarity, and embeddings.
- Build a comparison view with synchronized zoom and quick keyboard decisions.
- Rank candidates within each group.
- Add configurable best-shot count and diversity protection.
- Make relative evidence influence rejection: a blink is stronger evidence when an open-eye alternative exists.

**Exit:** the photographer can review groups rather than individual files and see useful best-shot suggestions.

### M6 — Photographer calibration

- Create simple rejection-strictness controls with safe defaults.
- Record manual overrides and disagreement reasons locally.
- Add a calibration report showing false rejects, missed rejects, and best-shot agreement.
- Optionally learn lightweight scoring weights from user decisions; do not retrain large models.

**Exit:** thresholds and ranking weights can be tuned to the photographer's real work without making the system opaque.

### M7 — Intel Windows backend

- Install and validate OpenVINO on the exact Windows 11 target.
- Confirm device discovery for the Intel integrated GPU.
- Benchmark CPU, GPU, and `AUTO` modes with the same corpus.
- Convert or replace unsupported models.
- Tune input sizes, batch sizes, precision, caching, and worker counts.
- Verify that CPU fallback produces equivalent decisions within documented tolerance.

**Exit:** stable end-to-end inference on the target machine with measured photos-per-minute, memory usage, and thermals.

### M8 — Windows packaging and field validation

- Produce a signed or locally trusted Windows build.
- Bundle models and runtimes without network downloads during normal use.
- Test installation, upgrades, uninstallation, long paths, Unicode paths, external drives, and interrupted scans.
- Run full shoots through the program and compare time saved against manual culling.
- Add backup/restore for the analysis database and settings.

**Exit:** a repeatable Windows 11 build that can safely process real professional shoots.

## Testing strategy

### During macOS development

- Run models through the common CPU path.
- Use fixed golden images for preprocessing and score-regression tests.
- Test the GUI with synthetic libraries containing tens of thousands of records.
- Keep model inference separate from UI and file operations.
- Test every destructive-looking operation as a dry run first.

### Automated Windows checks

- Run installation, import, unit, database, and small CPU inference tests on a Windows CI runner.
- Build a Windows artifact regularly to catch packaging failures early.
- Do not treat virtualized CI as evidence of Intel integrated-GPU performance.

### Exact target validation

- Use the real Windows 11 Intel machine for device compatibility, throughput, memory, driver, and long-running stability tests.
- Maintain a representative private benchmark shoot and record hardware and driver versions with every result.
- Compare classifications between macOS CPU and Windows OpenVINO to catch backend drift.

## Quality and safety targets

Initial numerical targets are provisional and must be calibrated with real labeled shoots.

- Favor extremely high precision for automatic Reject; uncertain images belong in Review.
- Never reject a unique image solely because its aesthetic score is low.
- Never delete originals automatically.
- Preserve RAW, JPEG, and sidecar relationships during export.
- Resume an interrupted scan without losing manual decisions.
- Explain every Reject using inspectable signals rather than a single unexplained score.
- Measure best-shot quality as agreement with photographer choices within groups, not across unrelated scenes.

## Known risks

- **Intentional blur resembles missed focus.** Mitigate with conservative thresholds, subject-aware measurements, group comparison, and Review classification.
- **Closed-eye detection fails with glasses, profiles, or small faces.** Use an uncertain state and reject only with strong evidence and preferably a better alternative.
- **Aesthetics are subjective.** Keep aesthetic scoring secondary and allow learned personal weights.
- **RAW decoding is expensive.** Prefer embedded previews and cache normalized thumbnails.
- **Integrated GPU support varies by generation and driver.** Keep CPU fallback and validate every selected model on the exact target.
- **Large shoots can exhaust memory.** Stream work, bound queues and caches, and store intermediate results in SQLite.
- **File operations can separate related assets.** Treat each photograph and its RAW/JPEG/sidecars as one transactional unit.

## First implementation slice

The first useful build should deliberately stay small:

1. Qt window with folder selection and thumbnail grid.
2. JPEG discovery and persistent SQLite scan state.
3. Exact duplicates, perceptual duplicates, simple exposure checks, and baseline blur scoring.
4. Keep/Review/Reject views with visible reasons and manual overrides.
5. Non-destructive export manifest—no physical move/delete yet.

This slice validates the workflow and safety model before adding face analysis, RAW support, or GPU acceleration.

## Next implementation slice

The next active slice extends subject-aware analysis beyond portraits:

1. Select a compact, locally bundled, license-compatible object detector that runs on the
   macOS CPU path and can later use OpenVINO on Intel Windows.
2. Detect and classify people, animals, vehicles, products, documents, flowers, and other
   common prominent objects without contacting a server.
3. Identify primary versus supporting/background subjects and measure regional sharpness.
4. Store object labels, confidence, importance, bounds, and focus scores in SQLite.
5. Show this evidence in the existing thumbnail, fullscreen, and comparison overlays.
6. Add the signals to Review and within-burst Best ranking, while keeping unique-photo
   rejection conservative.

**Exit:** non-portrait photographs receive inspectable primary-subject focus evidence on
the current CPU backend without weakening PhotoChoose's false-reject protections.

## Current implementation status

The working prototype now exceeds the first slice: it scans and analyzes in background
workers, persists protected manual overrides, performs baseline technical checks, uses
local face-mesh eye state plus face/eye-focus signals, groups perceptually similar frames, assigns
automatic Best choices inside groups, and copies Keep/Best selections non-destructively.
Project and photo context menus, an expandable analysis inspector, and a navigable
fullscreen culling surface are also implemented. Scanning now stores core EXIF capture and
camera settings; capture time and visual similarity drive burst groups, and a two-up view
provides synchronized zoom/pan and per-frame culling. A conservative relative rule rejects
an automatically questionable frame only when a much stronger alternative exists in its
group. RAW/JPEG/XMP sets are now one logical photograph, RAW-only files use cached embedded
JPEG previews when available, and complete file sets export together. Closing a project
releases its generated RAW previews without discarding catalog metadata or decisions.
Broader RAW preview validation, stronger subject analysis, calibration, OpenVINO, and
Windows packaging remain on the mainline roadmap. Subject evidence is now stored per face
and eye and can be overlaid in all culling views; burst ranking uses the weakest primary
face/eye sharpness rather than allowing a tiny background face to dominate.
