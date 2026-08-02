[Setup]
AppId={{BD9F71EE-5A25-44FF-AA4A-12D92966E921}
AppName=PhotoChoose
AppVersion=0.1.2
AppVerName=PhotoChoose legacy-layout migration fixture
DefaultDirName={localappdata}\Programs\PhotoChoose
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\..\release
OutputBaseFilename=PhotoChoose-Legacy-Layout-Test
Compression=lzma2
SolidCompression=yes
UninstallDisplayIcon={app}\PhotoChoose.exe

[Files]
Source: "..\..\LICENSE"; DestDir: "{app}"; DestName: "PhotoChoose.exe"
