@echo off
title DenZin Auto-Dubber Launcher
echo ===================================================
echo             DENZIN AUTO-DUBBER LAUNCHER            
echo ===================================================
echo.
echo [1/2] Starting Python FastAPI Backend Server...
echo [2/2] Opening browser to http://127.0.0.1:8500 in 3 seconds...
echo.

:: Add local ffmpeg to PATH if it exists
set "PATH=%~dp0ffmpeg-master-latest-win64-gpl\bin;%PATH%"

:: Launch browser in a background timer thread
start /b cmd /c "ping 127.0.0.1 -n 4 > nul && start http://127.0.0.1:8500"

:: Run the FastAPI server in the foreground
python server.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Server crashed or failed to start.
    echo Please make sure you have installed all requirements: pip install -r requirements.txt
    echo.
    pause
)
