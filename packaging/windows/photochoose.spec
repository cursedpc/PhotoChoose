# ruff: noqa: F821

from importlib.metadata import distribution
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, copy_metadata

project_root = Path(SPECPATH).parents[1]
source_root = project_root / "src"
package_root = source_root / "photochoose"

datas = [
    (str(package_root / "branding" / "photochoose-icon.png"), "photochoose/branding"),
    (str(project_root / "LICENSE"), "."),
    (str(project_root / "THIRD_PARTY_NOTICES.md"), "."),
    (
        str(project_root / "packaging" / "licenses" / "PYTHON_LICENSE.txt"),
        "licenses/Python",
    ),
    (
        str(project_root / "packaging" / "licenses" / "PYINSTALLER_COPYING.txt"),
        "licenses/PyInstaller",
    ),
]
datas.extend(
    (str(path), "photochoose/model_assets")
    for path in sorted((package_root / "model_assets").iterdir())
    if path.is_file()
)

binaries = []
hiddenimports = []
for package in ("openvino", "ai_edge_litert", "pi_heif"):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas.extend(package_datas)
    binaries.extend(package_binaries)
    hiddenimports.extend(package_hiddenimports)

license_distributions = (
    "PySide6",
    "PySide6_Essentials",
    "PySide6_Addons",
    "shiboken6",
    "numpy",
    "opencv-python-headless",
    "Pillow",
    "pi-heif",
    "ai-edge-litert",
    "openvino",
    "openvino-telemetry",
    "packaging",
)
for distribution_name in license_distributions:
    datas.extend(copy_metadata(distribution_name))
    package_distribution = distribution(distribution_name)
    notice_directory = "licenses/python-packages/{0}".format(
        package_distribution.metadata["Name"] or distribution_name
    )
    for relative_path in package_distribution.files or ():
        if not Path(str(relative_path)).name.lower().startswith(
            ("license", "copying", "notice")
        ):
            continue
        license_path = Path(package_distribution.locate_file(relative_path))
        if license_path.is_file():
            datas.append((str(license_path), notice_directory))

analysis = Analysis(
    [str(package_root / "__main__.py")],
    pathex=[str(source_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(analysis.pure)

executable = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="PhotoChoose",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(Path(SPECPATH) / "photochoose.ico"),
    version=str(Path(SPECPATH) / "version_info.txt"),
    contents_directory="_internal",
)

bundle = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="PhotoChoose",
)
