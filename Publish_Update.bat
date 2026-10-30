@echo off
title DenZin AutoDubber - GitHub Patch Publisher
cd /d "%~dp0"
echo =======================================================
echo   DENZIN AUTODUBBER - GITHUB PATCH PUBLISHER
echo =======================================================
echo.
python publish_patch.py
echo.
pause
