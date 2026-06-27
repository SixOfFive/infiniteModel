@echo off
REM ===========================================================================
REM  InfiniteModel WORKER (Windows)  ->  connects to the BEAST controller
REM
REM  Pure supervisor loop. client.py SELF-UPDATES from GitLab (writes the new
REM  code, then exits with code 42); this loop relaunches it on the new file.
REM
REM  There is deliberately NO `git pull` in here: a .bat that git-resets itself
REM  mid-run corrupts under cmd (cmd reads a batch by byte offset, so rewriting
REM  this file while it runs makes it resume on a garbled line). client.py keeps
REM  itself current instead.
REM
REM  One-time setup / to force a manual code update, run in this folder:
REM      git fetch
REM      git reset --hard origin/main
REM
REM  Extra client.py flags pass through, e.g.:  client.bat --device cpu+gpu
REM ===========================================================================
title InfiniteModel Worker -^> BEAST
cd /d "%~dp0"
:loop
python client.py %*
if %errorlevel%==42 (
  echo [update] new code pulled - relaunching ...
  goto loop
)
echo.
echo [worker exited] - press any key to close.
pause >nul
