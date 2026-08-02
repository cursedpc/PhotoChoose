#ifndef AppVersion
  #define AppVersion "0.1.7"
#endif

[Setup]
AppId={{BD9F71EE-5A25-44FF-AA4A-12D92966E921}
AppName=PhotoChoose
AppVersion={#AppVersion}
AppVerName=PhotoChoose {#AppVersion}
AppPublisher=PhotoChoose
AppPublisherURL=https://github.com/cursedpc/PhotoChoose
AppSupportURL=https://github.com/cursedpc/PhotoChoose/issues
AppUpdatesURL=https://github.com/cursedpc/PhotoChoose/releases/latest
LicenseFile=..\..\LICENSE
DefaultDirName={autopf}\PhotoChoose
DefaultGroupName=PhotoChoose
DisableProgramGroupPage=yes
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\..\release
OutputBaseFilename=PhotoChoose-Setup-{#AppVersion}-x64
SetupIconFile=photochoose.ico
UninstallDisplayIcon={app}\PhotoChoose.exe
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
CloseApplications=force
RestartApplications=no
UsePreviousAppDir=yes
VersionInfoVersion={#AppVersion}.0

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "..\..\dist\PhotoChoose\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\PhotoChoose"; Filename: "{app}\PhotoChoose.exe"
Name: "{autodesktop}\PhotoChoose"; Filename: "{app}\PhotoChoose.exe"; Tasks: desktopicon

[Run]
Filename: "{code:LegacyUninstaller}"; Parameters: "/VERYSILENT /SUPPRESSMSGBOXES /NORESTART"; StatusMsg: "Removing the previous per-user application files…"; Flags: runhidden waituntilterminated skipifdoesntexist; Check: HasLegacyPerUserInstall
Filename: "{app}\PhotoChoose.exe"; Description: "Launch PhotoChoose"; Flags: nowait postinstall skipifsilent

[Code]
const
  LegacyUninstallKey = 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{BD9F71EE-5A25-44FF-AA4A-12D92966E921}_is1';

var
  LegacyUninstallerPath: String;
  LegacyExecutablePath: String;

function InitializeSetup: Boolean;
var
  LegacyInstallDir: String;
  UninstallCommand: String;
begin
  LegacyUninstallerPath := '';
  LegacyExecutablePath := '';
  if RegQueryStringValue(HKCU, LegacyUninstallKey, 'InstallLocation', LegacyInstallDir) then
    LegacyExecutablePath := AddBackslash(LegacyInstallDir) + 'PhotoChoose.exe';
  if RegQueryStringValue(HKCU, LegacyUninstallKey, 'UninstallString', UninstallCommand) then
    LegacyUninstallerPath := RemoveQuotes(Trim(UninstallCommand));
  if not FileExists(LegacyUninstallerPath) then
    LegacyUninstallerPath := '';
  Result := True;
end;

function LegacyUninstaller(Param: String): String;
begin
  Result := LegacyUninstallerPath;
end;

function HasLegacyPerUserInstall: Boolean;
begin
  Result := LegacyUninstallerPath <> '';
end;

procedure RegisterExtraCloseApplicationsResources;
begin
  if FileExists(LegacyExecutablePath) then
    RegisterExtraCloseApplicationsResource(False, LegacyExecutablePath);
end;
