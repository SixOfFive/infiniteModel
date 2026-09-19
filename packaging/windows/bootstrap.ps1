<#
  InfiniteModel - Windows bootstrap (venv builder), the Windows twin of
  packaging/bootstrap.sh. Builds a Python venv next to the app, installs the
  torch build for this machine, then the packaged wheel + chosen backend extras.
  Invoked by the installer's post-install step, or run by hand to change
  hardware/backends. torch is NOT bundled (hardware/version-specific).

  Examples:
    powershell -ExecutionPolicy Bypass -File bootstrap.ps1 -Flavor cu128 -Extras "worker,vision,stt"
    powershell -ExecutionPolicy Bypass -File bootstrap.ps1 -Flavor cpu   -Extras "controller,worker"
    powershell -ExecutionPolicy Bypass -File bootstrap.ps1 -Flavor cpu   -Extras "controller" -NoTorch
#>
[CmdletBinding()]
param(
  [ValidateSet('cpu','cu128','cu126','cu124')]
  [string]$Flavor = 'cpu',
  [string]$Extras = '',
  [switch]$WithAcestep,
  [switch]$NoTorch,
  # Default: this script's own folder (the app dir the installer laid down).
  [string]$InstallDir = $PSScriptRoot,
  [string]$AppHome = (Join-Path $env:LOCALAPPDATA 'InfiniteModel')
)
$ErrorActionPreference = 'Stop'

function Find-Python {
  # Prefer the py launcher (py -3), then python.exe on PATH. Require >= 3.10.
  foreach ($cand in @(@('py','-3'), @('python'), @('python3'))) {
    $exe = $cand[0]; $pre = $cand[1..($cand.Count-1)]
    if (Get-Command $exe -ErrorAction SilentlyContinue) {
      $v = & $exe @pre -c "import sys;print('%d.%d'%sys.version_info[:2])" 2>$null
      if ($LASTEXITCODE -eq 0 -and $v) {
        $mm = $v.Split('.'); if ([int]$mm[0] -gt 3 -or ([int]$mm[0] -eq 3 -and [int]$mm[1] -ge 10)) {
          return ,($exe) + $pre
        }
      }
    }
  }
  throw "No Python 3.10+ found. Install it (winget install Python.Python.3.12) and re-run."
}

$wheel = Get-ChildItem -Path (Join-Path $InstallDir 'wheel') -Filter 'infinitemodel-*.whl' -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $wheel) { throw "No infinitemodel wheel under $InstallDir\wheel" }

$venv = Join-Path $InstallDir 'venv'
$vpy  = Join-Path $venv 'Scripts\python.exe'

Write-Host "== infinitemodel bootstrap =="
Write-Host "   flavor=$Flavor  extras=$Extras  venv=$venv  home=$AppHome"

# --- venv -------------------------------------------------------------------
if (-not (Test-Path $vpy)) {
  $py = Find-Python
  Write-Host "[1/4] creating venv with $($py -join ' ') ..."
  & $py[0] $py[1..($py.Count-1)] -m venv $venv
}
& $vpy -m pip install --upgrade pip | Out-Null

# --- torch (flavor-specific, first) -----------------------------------------
if (-not $NoTorch) {
  $idx = "https://download.pytorch.org/whl/$Flavor"
  Write-Host "[2/4] installing torch ($Flavor) from $idx ..."
  & $vpy -m pip install torch --index-url $idx
  if ($LASTEXITCODE -ne 0) { throw "torch install failed" }
} else {
  Write-Host "[2/4] torch: skipped (-NoTorch)"
}

# --- wheel + extras ---------------------------------------------------------
Write-Host "[3/4] installing infinitemodel$(if($Extras){"[$Extras]"}) ..."
if ($Extras) { & $vpy -m pip install "$($wheel.FullName)[$Extras]" }
else         { & $vpy -m pip install $wheel.FullName }
if ($LASTEXITCODE -ne 0) { throw "wheel install failed" }

# --- special backends -------------------------------------------------------
if (",$Extras," -like '*,tts,*') {
  Write-Host "== Kokoro TTS: kokoro+misaki (--no-deps) =="
  & $vpy -m pip install --no-deps kokoro misaki
  Write-Host "   NOTE: Windows TTS also needs the espeak-ng runtime. Install espeak-ng"
  Write-Host "   (github.com/espeak-ng/espeak-ng/releases) and set PHONEMIZER_ESPEAK_LIBRARY"
  Write-Host "   to its libespeak-ng.dll if phonemizer can't find it."
}
if ($WithAcestep) {
  Write-Host "== ACE-Step (t2a): NOT auto-installed — needs a --no-deps source install under"
  Write-Host "   a constraints file + torchaudio + a bf16-capable Ampere+ NVIDIA GPU. Follow"
  Write-Host "   docs/T2A.md, installing into $venv."
}

# --- app home ---------------------------------------------------------------
Write-Host "[4/4] ensuring app home $AppHome ..."
New-Item -ItemType Directory -Force -Path $AppHome | Out-Null

Write-Host ""
Write-Host "== ready =="
Write-Host "   Controller dashboard: http://localhost:21434"
Write-Host "   State + models:       $AppHome"
Write-Host "   Launch from the Start Menu, or the .cmd launchers in $InstallDir."
