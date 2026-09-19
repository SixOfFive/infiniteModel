@echo off
REM InfiniteModel worker launcher (Windows). Sets the writable home and runs the
REM venv console script (a self-update supervisor; relaunches client.py on exit
REM 42). The worker auto-discovers the controller by UDP broadcast; pass
REM --controller <ip> for cross-subnet setups, or --device cpu to force CPU.
setlocal
title InfiniteModel Worker
set "INFINITEMODEL_HOME=%LOCALAPPDATA%\InfiniteModel"
set "APP=%~dp0"
if not exist "%APP%venv\Scripts\infinitemodel-worker.exe" (
  echo [!] venv not built yet. Run:  "%APP%bootstrap.ps1"  ^(or re-run the installer's setup^).
  echo.
  pause
  exit /b 1
)
"%APP%venv\Scripts\infinitemodel-worker.exe" %*
echo.
echo [worker exited code %errorlevel%] - press any key to close.
pause >nul
