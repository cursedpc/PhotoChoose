# macOS packaging

PhotoChoose's Intel and Apple Silicon macOS releases are native PyInstaller application bundles
distributed as separate unsigned DMGs. Build each bundle on the matching macOS architecture with
Python 3.11; PyInstaller does not cross-build macOS applications from Windows or Linux. The spec
selects `x86_64` or `arm64` from the build machine automatically.

Intel builds retain OpenVINO 2025.4.1 and LiteRT 1.0.1, the latest LiteRT line published for Intel
macOS. Apple Silicon builds use OpenVINO 2026.1 and LiteRT 2.1.6. A normalization step removes a
known 16-byte padding defect from four OpenVINO ARM64 dylibs before PyInstaller re-signs them. The
native tests, recursive Mach-O architecture check, and frozen smoke test validate the complete ARM
runtime before publication. Windows also uses LiteRT 2.1.6 because no 1.0.1 Windows wheel exists.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,packaging]'
# Apple Silicon only:
python packaging/macos/normalize_openvino_arm64.py
python -m PyInstaller --noconfirm --clean packaging/macos/photochoose.spec
xattr -cr dist/PhotoChoose.app
codesign --force --deep --sign - dist/PhotoChoose.app
codesign --verify --deep --strict --verbose=2 dist/PhotoChoose.app
QT_QPA_PLATFORM=offscreen dist/PhotoChoose.app/Contents/MacOS/PhotoChoose --smoke-test
```

Create the release disk image from the repository root. Use `Intel` on an Intel Mac or
`Apple-Silicon` on an Apple Silicon Mac in the output filename:

```bash
mkdir -p release/dmg-root
ditto --noextattr --noacl dist/PhotoChoose.app release/dmg-root/PhotoChoose.app
ln -s /Applications release/dmg-root/Applications
hdiutil create -volname PhotoChoose -srcfolder release/dmg-root -ov -format UDZO \
  release/PhotoChoose-0.1.4-macOS-Apple-Silicon.dmg
```

The ad-hoc signature keeps the bundle internally consistent but does not identify or establish trust
in the developer. The GitHub Actions release workflow performs these steps on native Intel and
Apple Silicon runners and publishes both DMGs beside the Windows installer. Because the app does
not use an Apple Developer ID, install it in `/Applications`, then remove the quarantine attribute
and launch it once:

```bash
xattr -dr com.apple.quarantine /Applications/PhotoChoose.app
open /Applications/PhotoChoose.app
```

Repeat this only after downloading or replacing the app with a new unsigned build, not on every
launch. The catalog remains in the standard
per-user Application Support directory and is not stored inside the app bundle.
