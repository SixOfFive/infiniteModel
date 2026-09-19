@echo off
REM InfiniteModel controller launcher (Windows). Points the app's writable home
REM at %LOCALAPPDATA%\InfiniteModel and runs the venv console script. The
REM console script is itself a supervisor: it relaunches server.py on a
REM self-update (exit 42), so no relaunch loop is needed here.
setlocal
title InfiniteModel Controller
set "INFINITEMODEL_HOME=%LOCALAPPDATA%\InfiniteModel"
set "APP=%~dp0"
if not exist "%APP%venv\Scripts\infinitemodel-controller.exe" (
  echo [!] venv not built yet. Run:  "%APP%bootstrap.ps1"  ^(or re-run the installer's setup^).
  echo.
  pause
  exit /b 1
)
"%APP%venv\Scripts\infinitemodel-controller.exe" %*
echo.
echo [controller exited code %errorlevel%] - press any key to close.
pause >nul
