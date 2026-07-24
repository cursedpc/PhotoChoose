from pathlib import Path

from PyInstaller.utils.hooks import collect_all, copy_metadata


project_root = Path(SPECPATH).parents[1]
source_root = project_root / "src"
package_root = source_root / "photochoose"

datas = [
    (str(package_root / "branding" / "photochoose-icon.png"), "photochoose/branding"),
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
