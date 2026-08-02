# Windows packaging

The build embeds PhotoChoose's GPL license, third-party notices, and the license files supplied by
packaged Python distributions. The Inno Setup wizard displays the PhotoChoose license before
installation, and the portable ZIP keeps the main notices beside the executable.

PhotoChoose is built as an x64 PyInstaller one-folder application and wrapped in an Inno Setup
installer. Build on 64-bit Windows with Python 3.11; PyInstaller bundles are specific to the
operating system that creates them.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,packaging]"
python packaging\windows\make_icon.py
python -m PyInstaller --noconfirm --clean packaging\windows\photochoose.spec
dist\PhotoChoose\PhotoChoose.exe --smoke-test
```

Install Inno Setup, then compile the installer from the repository root:

```powershell
& "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe" `
  "/DAppVersion=0.1.7" packaging\windows\photochoose.iss
```

The installer is written to `release\PhotoChoose-Setup-0.1.7-x64.exe`. It installs the application
under `C:\Program Files\PhotoChoose` by default and requests Windows elevation. The destination can
still be changed in the setup wizard. The catalog database and settings remain under
`%APPDATA%\PhotoChoose\PhotoChoose` and are intentionally not removed by uninstall.

Version 0.1.3 migrates the earlier per-user application copy from
`%LOCALAPPDATA%\Programs\PhotoChoose` after the Program Files installation succeeds. The catalog and
settings live outside either application directory and remain untouched. After migration, the
stable Inno Setup `AppId` and `UsePreviousAppDir=yes` reuse the Program Files or user-selected
installation directory. CI installs 0.1.2 first, upgrades it to the new setup, verifies the old
application copy was removed, and smoke-tests a second in-place installation.

The automated workflow additionally installs the generated setup into a temporary directory,
starts the installed executable in smoke-test mode, uninstalls it, creates a portable ZIP, and
publishes SHA-256 checksums. To create a downloadable GitHub Release, update the application
version in `pyproject.toml`, commit it to `main`, then push the matching `v<version>` tag. The
workflow publishes the installer only after every Windows build, test, install, smoke-test, and
in-place-upgrade/uninstall step succeeds. Public releases remain unsigned until a trusted signing
identity is configured.
