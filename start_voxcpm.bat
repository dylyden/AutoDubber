@echo off
title VoxCPM Private Studio Launcher
echo ===================================================
echo             VOXCPM PRIVATE STUDIO LAUNCHER            
echo ===================================================
echo.
echo Starting Local Web Server on Port 8080...
echo.
python run_voxcpm_studio.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Server crashed or failed to start.
    echo.
    pause
)
