# Intel macOS packaging

PhotoChoose's Intel macOS release is a PyInstaller application bundle distributed in an unsigned
DMG. Build it on Intel macOS with Python 3.11; PyInstaller does not cross-build macOS applications
from Windows or Linux.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,packaging]'
python -m PyInstaller --noconfirm --clean packaging/macos/photochoose.spec
xattr -cr dist/PhotoChoose.app
codesign --force --deep --sign - dist/PhotoChoose.app
codesign --verify --deep --strict --verbose=2 dist/PhotoChoose.app
QT_QPA_PLATFORM=offscreen dist/PhotoChoose.app/Contents/MacOS/PhotoChoose --smoke-test
```

Create the release disk image from the repository root:

```bash
mkdir -p release/dmg-root
ditto --noextattr --noacl dist/PhotoChoose.app release/dmg-root/PhotoChoose.app
ln -s /Applications release/dmg-root/Applications
hdiutil create -volname PhotoChoose -srcfolder release/dmg-root -ov -format UDZO \
  release/PhotoChoose-0.1.4-macOS-Intel.dmg
```

The ad-hoc signature keeps the bundle internally consistent but does not identify or establish trust
in the developer. The GitHub Actions release workflow performs these steps on an Intel runner and
publishes the DMG beside the Windows installer. Because the app does not use an Apple Developer ID,
macOS Gatekeeper
may require the user to Control-click PhotoChoose, choose **Open**, then confirm **Open** on first
launch. The application itself belongs in `/Applications`; its catalog remains in the standard
per-user Application Support directory and is not stored inside the app bundle.
