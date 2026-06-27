@echo off
REM ===========================================================================
REM  InfiniteModel CONTROLLER  ->  run on BEAST
REM  Ollama keeps :11434; InfiniteModel's Ollama-compatible API is on :21434.
REM  Dashboard:  http://192.168.15.103:21434/   (control :50100, data :50101)
REM
REM  Pure supervisor loop. server.py SELF-UPDATES from GitLab (writes the new
REM  code, then exits with code 42); this loop relaunches it on the new file.
REM
REM  There is deliberately NO `git pull` in here: a .bat that git-resets itself
REM  mid-run corrupts under cmd (it reads the batch by byte offset). server.py
REM  keeps itself current instead (only when idle - no model loaded).
REM
REM  One-time setup / to force a manual code update, run in this folder:
REM      git fetch
REM      git reset --hard origin/main
REM
REM  Extra args pass through, e.g.:  server.bat --http-port 22000
REM ===========================================================================
title InfiniteModel Controller (BEAST)
cd /d "%~dp0"
:loop
python server.py --host 0.0.0.0 --http-port 21434 --control-port 50100 --data-port 50101 %*
set EC=%errorlevel%
if %EC%==42 (
  echo [update] new code pulled - relaunching ...
  goto loop
)
if not %EC%==0 (
  REM ANY non-zero exit = a crash (e.g. the Windows asyncio ProactorEventLoop shutdown
  REM InvalidStateError) — auto-relaunch instead of leaving the controller dead. The 3s
  REM delay both avoids a tight crash-loop and gives a window to Ctrl+C out to stop.
  echo.
  echo [supervisor] controller exited code %EC% - relaunching in 3s ^(Ctrl+C to stop^) ...
  timeout /t 3 /nobreak >nul
  goto loop
)
echo.
echo [controller exited cleanly] - press any key to close.
pause >nul
