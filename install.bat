@echo off
REM ===========================================================================
REM  InfiniteModel - portable WORKER installer (Windows)
REM
REM  Builds a self-contained worker environment under .\install\ from the
REM  pre-downloaded bundle. NOTHING is installed into Windows. Re-runnable.
REM  Designed to run straight off a USB stick / copied folder, offline.
REM
REM  Prereq: the bundle was populated once with  install\prepare.bat  (online).
REM ===========================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"
set "ROOT=%CD%"
set "INS=%ROOT%\install"
set "VENV=%INS%\.venv-win"
set "WHEELS=%INS%\wheels\win"
set "REQ=%INS%\requirements-client.txt"

echo ===========================================================================
echo  InfiniteModel portable worker installer (Windows)
echo  Repo: %ROOT%
echo ===========================================================================

REM --- 1) locate a Python: bundled standalone -> py -3.13 -> python -----------
set "BOOT_EXE="
set "BOOT_LAUNCH="
set "RT=%INS%\runtime\win\python"
if exist "%RT%\python.exe" set "BOOT_EXE=%RT%\python.exe"
if not defined BOOT_EXE if exist "%INS%\python\cpython-3.13-windows.tar.gz" (
  echo [1/5] extracting bundled Python ...
  if not exist "%INS%\runtime\win" mkdir "%INS%\runtime\win"
  tar -xf "%INS%\python\cpython-3.13-windows.tar.gz" -C "%INS%\runtime\win"
  if exist "%RT%\python.exe" set "BOOT_EXE=%RT%\python.exe"
)
if not defined BOOT_EXE ( where py >nul 2>&1 && set "BOOT_LAUNCH=py -3.13" )
if not defined BOOT_EXE if not defined BOOT_LAUNCH ( where python >nul 2>&1 && set "BOOT_EXE=python" )
if not defined BOOT_EXE if not defined BOOT_LAUNCH (
  echo [ERROR] No Python found and no bundled Python in install\python\.
  echo         Run install\prepare.bat on an internet-connected box first,
  echo         or install Python 3.13 on this machine.
  goto :fail
)
if defined BOOT_EXE  ( echo [1/5] Python: "%BOOT_EXE%" )
if defined BOOT_LAUNCH ( echo [1/5] Python: %BOOT_LAUNCH% )

REM --- 2) create the venv (offline) ------------------------------------------
if not exist "%VENV%\Scripts\python.exe" (
  echo [2/5] creating venv at install\.venv-win ...
  if defined BOOT_EXE ( "%BOOT_EXE%" -m venv "%VENV%" ) else ( %BOOT_LAUNCH% -m venv "%VENV%" )
) else (
  echo [2/5] venv already present - reusing
)
set "VPY=%VENV%\Scripts\python.exe"
if not exist "%VPY%" ( echo [ERROR] venv creation failed & goto :fail )

REM --- 3) install worker deps: offline from bundle, else online fallback ------
echo [3/5] installing worker deps (offline from install\wheels\win) ...
"%VPY%" -m pip install --no-index --find-links "%WHEELS%" -r "%REQ%"
if errorlevel 1 (
  echo.
  echo [warn] offline install incomplete - falling back to ONLINE install ...
  "%VPY%" -m pip install --upgrade pip
  "%VPY%" -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
  "%VPY%" -m pip install transformers==5.12.1 safetensors==0.8.0 huggingface_hub==1.19.0 numpy==2.4.6 psutil==7.2.2
  if errorlevel 1 ( echo [ERROR] dependency install failed ^(offline AND online^) & goto :fail )
)

REM --- 4) verify -------------------------------------------------------------
echo [4/5] verifying imports ...
"%VPY%" -c "import torch,transformers,safetensors,huggingface_hub,numpy,psutil;print('  torch',torch.__version__,'| transformers',transformers.__version__,'| numpy',numpy.__version__)"
if errorlevel 1 ( echo [ERROR] dependency verification failed & goto :fail )

REM --- 5) done ---------------------------------------------------------------
echo [5/5] ready.
echo.
echo ===========================================================================
echo  READY - the worker environment is built (nothing installed into Windows).
echo.
echo  START THE WORKER:
echo      install\start-client.bat
echo.
echo  Useful variants:
echo      install\start-client.bat --device cpu          ^(force CPU^)
echo      install\start-client.bat --controller ^<ip^>     ^(other controller^)
echo      install\start-client.bat --name ^<label^>        ^(override hostname^)
echo.
echo  Default controller: 192.168.15.103:50100  (edit start-client.bat to change)
echo ===========================================================================
echo.
pause >nul
exit /b 0

:fail
echo.
echo Installation did not complete. See messages above.
pause >nul
exit /b 1
