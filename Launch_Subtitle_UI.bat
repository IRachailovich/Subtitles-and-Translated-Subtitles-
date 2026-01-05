@echo off
REM Subtitle Generator Desktop Launcher
REM This script activates the conda environment and launches the UI

cd /d "%~dp0"

REM Activate conda environment
call conda activate subtitles

REM Launch the subtitle UI
python subtitle_ui.py

pause
