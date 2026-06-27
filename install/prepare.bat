@echo off
REM ===========================================================================
REM  PREPARE the portable worker bundle (run ONCE, on an internet-connected box).
REM  Downloads standalone Python (win+linux) + the worker wheel closure into
REM  install\python\ and install\wheels\. Nothing is installed into the OS.
REM
REM    prepare.bat            both platforms
REM    prepare.bat --win      Windows wheels only (still grabs both Pythons)
REM    prepare.bat --force    re-download everything
REM ===========================================================================
setlocal
cd /d "%~dp0"
set "PY="
where python >nul 2>&1 && set "PY=python"
if not defined PY ( where py >nul 2>&1 && set "PY=py -3" )
if not defined PY ( where python3 >nul 2>&1 && set "PY=python3" )
if not defined PY (
  echo [ERROR] Python 3 with pip is required to PREPARE the bundle.
  echo         Install it on this build box, then re-run prepare.bat.
  exit /b 1
)
echo Using %PY%
%PY% _fetch.py %*
