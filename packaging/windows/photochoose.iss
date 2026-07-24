#ifndef AppVersion
  #define AppVersion "0.1.1"
#endif

[Setup]
AppId={{BD9F71EE-5A25-44FF-AA4A-12D92966E921}
AppName=PhotoChoose
AppVersion={#AppVersion}
AppVerName=PhotoChoose {#AppVersion}
AppPublisher=PhotoChoose
DefaultDirName={localappdata}\Programs\PhotoChoose
DefaultGroupName=PhotoChoose
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
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
Filename: "{app}\PhotoChoose.exe"; Description: "Launch PhotoChoose"; Flags: nowait postinstall skipifsilent
