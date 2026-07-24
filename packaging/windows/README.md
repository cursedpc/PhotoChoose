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
  "/DAppVersion=0.1.0" packaging\windows\photochoose.iss
```

The installer is written to `release\PhotoChoose-Setup-0.1.0-x64.exe`. It installs under the
current user's Local AppData and does not require elevation. The catalog database remains under
`%APPDATA%\PhotoChoose\PhotoChoose` and is intentionally not removed by uninstall.

The automated workflow additionally installs the generated setup into a temporary directory,
starts the installed executable in smoke-test mode, uninstalls it, creates a portable ZIP, and
publishes SHA-256 checksums. Public releases remain unsigned until a trusted signing identity is
configured.
