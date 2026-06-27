@echo off
REM ===========================================================================
REM  Start the InfiniteModel WORKER using the offline venv built by install.bat.
REM  Self-update aware: client.py exits 42 when it pulls new code -> relaunch.
REM
REM    start-client.bat                       CPU+GPU (auto-falls-back to CPU)
REM    start-client.bat --device cpu          force CPU (silences GPU notice)
REM    start-client.bat --controller 10.0.0.5 point at a different controller
REM    start-client.bat --name mybox          override reported hostname
REM ===========================================================================
title InfiniteModel Worker -^> BEAST
set "HERE=%~dp0"
if not exist "%HERE%.venv-win\Scripts\python.exe" (
  echo [!] Worker env not built yet.
  echo     Run  install.bat  in the parent folder first.
  echo.
  pause >nul
  exit /b 1
)
cd /d "%HERE%.."
:loop
"%HERE%.venv-win\Scripts\python.exe" client.py %*
if %errorlevel%==42 (
  echo [update] new code pulled - relaunching ...
  goto loop
)
echo.
echo [worker exited] - press any key to close.
pause >nul
