# Windows packaging

PhotoChoose is built as an x64 PyInstaller one-folder application and wrapped in a per-user Inno
Setup installer. Build on 64-bit Windows with Python 3.11; PyInstaller bundles are specific to the
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
  "/DAppVersion=0.1.2" packaging\windows\photochoose.iss
```

The installer is written to `release\PhotoChoose-Setup-0.1.2-x64.exe`. It installs under the
current user's Local AppData and does not require elevation. The catalog database remains under
`%APPDATA%\PhotoChoose\PhotoChoose` and is intentionally not removed by uninstall.

Running a newer PhotoChoose setup upgrades the existing installation in place. The stable Inno
Setup `AppId` and `UsePreviousAppDir=yes` reuse the current install directory and uninstall record;
there is no need to uninstall the earlier release. The catalog and settings live outside the
application directory and remain untouched. CI installs the setup twice and smoke-tests the
second installation to protect this upgrade path.

The automated workflow additionally installs the generated setup into a temporary directory,
starts the installed executable in smoke-test mode, uninstalls it, creates a portable ZIP, and
publishes SHA-256 checksums. To create a downloadable GitHub Release, update the application
version in `pyproject.toml`, commit it to `main`, then push the matching `v<version>` tag. The
workflow publishes the installer only after every Windows build, test, install, smoke-test, and
in-place-upgrade/uninstall step succeeds. Public releases remain unsigned until a trusted signing
identity is configured.
