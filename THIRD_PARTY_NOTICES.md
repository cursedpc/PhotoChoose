# Third-party notices

PhotoChoose is distributed with open-source libraries and inference models developed by other
projects. Those components are not relicensed under the PhotoChoose license. They remain governed
by the licenses identified below.

Binary releases include the license files supplied by installed Python distributions. The exact
set can vary by operating system and processor architecture.

## Runtime components

| Component | Versions used by PhotoChoose | License | Upstream |
| --- | --- | --- | --- |
| Python | 3.11 | Python Software Foundation License 2.0 | https://www.python.org/ |
| PySide6, Qt for Python, and Shiboken | 6.9.x | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only; PhotoChoose uses the GPL-3.0 option | https://doc.qt.io/qtforpython-6/ |
| NumPy | 1.26.x | BSD-3-Clause, with bundled-component notices in its distribution | https://numpy.org/ |
| OpenCV Python Headless | 4.11.x | Apache-2.0, with bundled-component notices in its distribution | https://github.com/opencv/opencv-python |
| Pillow | 12.x | MIT-CMU | https://python-pillow.org/ |
| pi-heif | 1.4.0 | BSD-3-Clause; its wheels contain separately licensed codec libraries | https://github.com/bigcat88/pillow_heif |
| LiteRT | 1.0.1 or 2.1.6, depending on platform | Apache-2.0 | https://github.com/google-ai-edge/LiteRT |
| OpenVINO | 2025.4.1 or 2026.1.0, depending on platform | Apache-2.0, with bundled-component notices in its distribution | https://github.com/openvinotoolkit/openvino |
| OpenVINO Telemetry | Installed with OpenVINO | Apache-2.0 | https://github.com/openvinotoolkit/openvino |
| packaging | Installed with OpenVINO | Apache-2.0 OR BSD-2-Clause | https://github.com/pypa/packaging |

The pi-heif wheel's own **LICENSES_bundled.txt** identifies the precise native codecs included for
each platform and their licenses. NumPy, OpenCV, OpenVINO, and Pillow likewise ship their applicable
copyright and bundled-component notices in their distribution metadata; PhotoChoose's packaging
process preserves those files.

## Bundled inference models

| Model | License | Local notice | Upstream |
| --- | --- | --- | --- |
| YuNet face detector | MIT | **YUNET_LICENSE** | https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet |
| SFace identity model | Apache-2.0 | **SFACE_LICENSE** | https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface |
| face-reidentification-retail-0095 | Apache-2.0 | **OPENVINO_MODEL_ZOO_LICENSE** | https://github.com/openvinotoolkit/open_model_zoo |
| MediaPipe Face Mesh V2 | Apache-2.0 | **MEDIAPIPE_LICENSE** | https://github.com/google-ai-edge/mediapipe |
| NanoDet object detector | Apache-2.0 | **NANODET_LICENSE** | https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_nanodet |

The model notice files and model hashes are stored under
**src/photochoose/model_assets/** and are included in packaged applications. See
**src/photochoose/model_assets/MODELS.md** for exact filenames and provenance.

## Build and packaging tools

PyInstaller is used to create the application bundles. Its build system is GPL-2.0-or-later with a
Bootloader Exception that permits distributing the generated executable. The complete terms are in
**packaging/licenses/PYINSTALLER_COPYING.txt**.

Inno Setup is used to create the Windows installer. It is a build-time tool and is not part of the
PhotoChoose source or installed application.

## Project-owned assets

The PhotoChoose icon and the small synthetic HEIC test fixture were created for this project and
are covered by the PhotoChoose license. They are not third-party photographs.

## No endorsement

The names of third-party projects and copyright holders are used only to identify their software
and licenses. They do not imply endorsement of PhotoChoose.
