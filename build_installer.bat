@echo off
title Compile AutoDubber All-In-One Setup EXE
cd /d "%~dp0"
echo ========================================================
echo   COMPILING AUTODUBBER ALL-IN-ONE SETUP EXE
echo ========================================================
echo.

echo [1/2] Building standalone Python runtime with PyInstaller...
"C:\Users\Den\AppData\Local\Programs\Python\Python310\Scripts\pyinstaller.exe" --noconfirm --clean AutoDubber.spec
if %ERRORLEVEL% neq 0 (
    echo [ERROR] PyInstaller build failed!
    pause
    exit /b 1
)

echo.
echo [2/2] Packing Single-File Setup EXE with Inno Setup (ISCC)...
"C:\Users\Den\AppData\Local\Programs\Inno Setup 6\ISCC.exe" "installer_setup.iss"

echo.
echo ========================================================
echo [DONE] Compiled installer saved to your Desktop:
echo C:\Users\Den\Desktop\Auto_Translate_Movie_Setup.exe
echo ========================================================
pause

