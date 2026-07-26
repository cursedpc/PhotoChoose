# ruff: noqa: F821

import platform
import tomllib
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, copy_metadata

project_root = Path(SPECPATH).parents[1]
source_root = project_root / "src"
package_root = source_root / "photochoose"
icon_path = package_root / "branding" / "photochoose-icon.png"
project_version = tomllib.loads(
    (project_root / "pyproject.toml").read_text(encoding="utf-8")
)["project"]["version"]
target_arch = platform.machine()
if target_arch not in ("arm64", "x86_64"):
    raise RuntimeError("Unsupported macOS build architecture: {0}".format(target_arch))

# OpenVINO's ARM64 wheel contains an optional TBB topology helper with
# malformed Mach-O link-edit metadata and a Homebrew-only libhwloc path.
# PyInstaller cannot rewrite that file. The ARM CPU plugin does not link to it;
# oneTBB falls back to its normal scheduler when the helper is unavailable.
excluded_arm64_openvino_binaries = {"libtbbbind_2_5.3.dylib"}

datas = [
    (str(icon_path), "photochoose/branding"),
]
datas.extend(
    (str(path), "photochoose/model_assets")
    for path in sorted((package_root / "model_assets").iterdir())
    if path.is_file()
)

binaries = []
hiddenimports = []
for package in ("openvino", "ai_edge_litert"):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    if package == "openvino" and target_arch == "arm64":
        # collect_all() reports shared libraries in both tables. Filtering only
        # binaries lets PyInstaller reclassify the data entry during Analysis.
        package_datas = [
            data
            for data in package_datas
            if Path(data[0]).name not in excluded_arm64_openvino_binaries
        ]
        package_binaries = [
            binary
            for binary in package_binaries
            if Path(binary[0]).name not in excluded_arm64_openvino_binaries
        ]
    datas.extend(package_datas)
    binaries.extend(package_binaries)
    hiddenimports.extend(package_hiddenimports)

datas.extend(copy_metadata("openvino"))
datas.extend(copy_metadata("ai-edge-litert"))

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
    target_arch=target_arch,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(icon_path),
    contents_directory="Frameworks",
)

collection = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="PhotoChoose",
)

application = BUNDLE(
    collection,
    name="PhotoChoose.app",
    icon=str(icon_path),
    bundle_identifier="io.github.cursedpc.photochoose",
    version=project_version,
    info_plist={
        "CFBundleDisplayName": "PhotoChoose",
        "CFBundleName": "PhotoChoose",
        "CFBundleVersion": project_version,
        "LSMultipleInstancesProhibited": True,
        "NSPrincipalClass": "NSApplication",
        "NSHighResolutionCapable": True,
    },
)
