#define MyAppName "DenZin Auto-Dubber V2.2"
#define MyAppVersion "2.2.5"
#define MyAppPublisher "By Den"
#define MyAppExeName "AutoDubber.exe"

[Setup]
AppId={{9E3A1B84-6D2C-4F18-8B74-2950AEC1D422}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\DenZin_AutoDubber
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..
OutputBaseFilename=Auto_Translate_Movie_Setup
SetupIconFile=favicon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/fast
SolidCompression=yes
DiskSpanning=no
WizardStyle=modern
DisableDirPage=no
DisableWelcomePage=no
CloseApplications=force

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: checkedonce

[Files]
; Compiled PyInstaller Standalone Bundle (Includes Python 3.10, PyTorch CUDA, Faster-Whisper, Demucs, VoxCPM, Edge-TTS)
Source: "dist\AutoDubber\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; Live Hot-Patchable Python & Web UI Files Next to AutoDubber.exe
Source: "server.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "dubber.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "voice_converter.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "dialog_picker.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "run_voxcpm_studio.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "vertex_batch_processor.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "system_diagnostic.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "index.html"; DestDir: "{app}"; Flags: ignoreversion
Source: "app.js"; DestDir: "{app}"; Flags: ignoreversion
Source: "style.css"; DestDir: "{app}"; Flags: ignoreversion
Source: "favicon.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "start_tool.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "start_voxcpm.bat"; DestDir: "{app}"; Flags: ignoreversion

; Embedded FFmpeg & FFprobe Media Engine
Source: "ffmpeg.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "ffprobe.exe"; DestDir: "{app}"; Flags: ignoreversion

; Voice Presets, Samples, Fonts & OpenVoiceV2 Tone-Color Weights
Source: "custom_voices\*"; DestDir: "{app}\custom_voices"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "voice_samples\*"; DestDir: "{app}\voice_samples"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "fonts\*"; DestDir: "{app}\fonts"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "models\OpenVoiceV2\converter\*"; DestDir: "{app}\models\OpenVoiceV2\converter"; Flags: ignoreversion recursesubdirs createallsubdirs

[Dirs]
Name: "{app}\subtitles"
Name: "{app}\models"
Name: "{app}\plugins"
Name: "{app}\patches"

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; IconFilename: "{app}\favicon.ico"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; IconFilename: "{app}\favicon.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent

