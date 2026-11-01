@echo off
title Publish Update to GitHub (dylyden/AutoDubber)
cd /d "%~dp0"
echo ========================================================
echo   PUBLISH UPDATE TO GITHUB (dylyden/AutoDubber)
echo ========================================================
echo.

REM Ensure git repo is initialized and author identity is set
if not exist ".git" (
    echo Initializing Git repository...
    git init
)
git branch -M main
git config user.name "dylyden"
git config user.email "dylyden@users.noreply.github.com"
git remote set-url origin https://github.com/dylyden/AutoDubber.git 2>nul || git remote add origin https://github.com/dylyden/AutoDubber.git

set /p NEW_VER="Enter new version number (e.g. 2.2.6) [leave blank to keep current]: "
set /p CHANGELOG="Enter update changelog message [default: Bug fixes and extractor updates]: "
if "%CHANGELOG%"=="" set "CHANGELOG=Bug fixes and extractor updates"

set "PY_EXE=python"
if exist "%~dp0python310\python.exe" (
    set "PY_EXE=%~dp0python310\python.exe"
) else if exist "C:\Users\Den\AppData\Local\Programs\Python\Python310\python.exe" (
    set "PY_EXE=C:\Users\Den\AppData\Local\Programs\Python\Python310\python.exe"
)

"%PY_EXE%" publish_patch.py "%NEW_VER%" "%CHANGELOG%"

echo.
echo Staging core update files...
git add .
git commit -m "Release update %NEW_VER%: %CHANGELOG%"
echo.
echo Pushing to https://github.com/dylyden/AutoDubber.git ...
git push -u origin main --force

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================================
    echo [SUCCESS] Update pushed! All users clicking "Check Update" will now receive this version.
    echo ========================================================
) else (
    echo.
    echo ========================================================
    echo [WARNING] Git push required sign-in or failed. Please sign in to GitHub if prompted and run again.
    echo ========================================================
)
pause
