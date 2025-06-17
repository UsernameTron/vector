@echo off
title Vector RAG Database Launcher

echo 🚀 Starting Vector RAG Database Launcher...
echo Current directory: %cd%

REM Check if we're in the right directory
if not exist "desktop_launcher.py" (
    echo ❌ Error: desktop_launcher.py not found
    echo Please run this script from the Vector RAG Database directory
    pause
    exit /b 1
)

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH
    echo Please install Python 3.7+ and try again
    pause
    exit /b 1
)

REM Launch the desktop application
echo 🎯 Launching Vector RAG Database...
python desktop_launcher.py

echo Launcher closed.
pause
