; ===========================================================================
;  InfiniteModel - Windows installer (NSIS / MUI2).
;  Per-user install (no admin). Installs the wheel + launchers + bootstrap into
;  %LOCALAPPDATA%\Programs\InfiniteModel, lets the user pick role + backend
;  components and a torch flavour, then runs bootstrap.ps1 to build the venv.
;  torch is not bundled (hardware/version-specific).
;
;  Build (from the repo root):
;    makensis -DAPPVERSION=<ver> -DWHEEL=dist/infinitemodel-<ver>-py3-none-any.whl \
;             -DSRCDIR=packaging/windows -DLICENSEFILE=LICENSE \
;             packaging/windows/win-installer.nsi
;  (packaging/windows/build_installer.sh fills these in and builds the wheel first.)
; ===========================================================================
Unicode true
!include "MUI2.nsh"
!include "nsDialogs.nsh"
!include "LogicLib.nsh"
!include "Sections.nsh"

!ifndef APPVERSION
  !define APPVERSION "0.0.0"
!endif
!ifndef WHEEL
  !error "pass -DWHEEL=<path to the built wheel>"
!endif
!ifndef SRCDIR
  !define SRCDIR "packaging/windows"
!endif
!ifndef LICENSEFILE
  !define LICENSEFILE "LICENSE"
!endif
!ifndef OUTFILE
  !define OUTFILE "dist/infinitemodel-${APPVERSION}-setup.exe"
!endif

Name "InfiniteModel ${APPVERSION}"
OutFile "${OUTFILE}"
RequestExecutionLevel user
InstallDir "$LOCALAPPDATA\Programs\InfiniteModel"
ShowInstDetails show
ShowUninstDetails show
BrandingText "InfiniteModel ${APPVERSION}"

Var Flavor        ; cu128 | cu126 | cpu
Var Extras        ; comma-joined pip extras from selected components
Var WithAce       ; "1" if ACE-Step selected
Var Dlg
Var RbCuda128
Var RbCuda126
Var RbCpu

; --- pages ------------------------------------------------------------------
!define MUI_ABORTWARNING
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "${LICENSEFILE}"
!insertmacro MUI_PAGE_COMPONENTS
Page custom TorchPageCreate TorchPageLeave
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

Function .onInit
  StrCpy $Flavor "cu128"
  StrCpy $Extras ""
  StrCpy $WithAce "0"
FunctionEnd

; --- custom torch-flavour page ---------------------------------------------
Function TorchPageCreate
  !insertmacro MUI_HEADER_TEXT "Compute backend" "Choose the PyTorch build for this machine."
  nsDialogs::Create 1018
  Pop $Dlg
  ${If} $Dlg == error
    Abort
  ${EndIf}
  ${NSD_CreateLabel} 0 0 100% 26u "Pick the backend for this box. NVIDIA options need a recent driver. ACE-Step music (if selected) needs an NVIDIA Ampere+ GPU and is skipped on CPU."
  ${NSD_CreateRadioButton} 8u 32u 95% 12u "NVIDIA GPU - CUDA 12.8 (recommended)"
  Pop $RbCuda128
  ${NSD_CreateRadioButton} 8u 46u 95% 12u "NVIDIA GPU - CUDA 12.6"
  Pop $RbCuda126
  ${NSD_CreateRadioButton} 8u 60u 95% 12u "CPU only (no GPU)"
  Pop $RbCpu
  ${NSD_Check} $RbCuda128
  nsDialogs::Show
FunctionEnd

Function TorchPageLeave
  ${NSD_GetState} $RbCuda128 $0
  ${If} $0 == ${BST_CHECKED}
    StrCpy $Flavor "cu128"
  ${Else}
    ${NSD_GetState} $RbCuda126 $0
    ${If} $0 == ${BST_CHECKED}
      StrCpy $Flavor "cu126"
    ${Else}
      StrCpy $Flavor "cpu"
    ${EndIf}
  ${EndIf}
FunctionEnd

; --- sections ---------------------------------------------------------------
Section "-Core (required)" SEC_CORE
  SectionIn RO
  SetOutPath "$INSTDIR"
  File "/oname=infinitemodel-controller.cmd" "${SRCDIR}/infinitemodel-controller.cmd"
  File "/oname=infinitemodel-worker.cmd" "${SRCDIR}/infinitemodel-worker.cmd"
  File "/oname=bootstrap.ps1" "${SRCDIR}/bootstrap.ps1"
  File "/oname=README.md" "${SRCDIR}/README.md"
  File "${LICENSEFILE}"
  SetOutPath "$INSTDIR\wheel"
  File "${WHEEL}"
  WriteUninstaller "$INSTDIR\uninstall.exe"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\InfiniteModel" "DisplayName" "InfiniteModel ${APPVERSION}"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\InfiniteModel" "DisplayVersion" "${APPVERSION}"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\InfiniteModel" "Publisher" "sixoffive"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\InfiniteModel" "UninstallString" "$\"$INSTDIR\uninstall.exe$\""
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\InfiniteModel" "InstallLocation" "$INSTDIR"
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\InfiniteModel" "NoModify" 1
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\InfiniteModel" "NoRepair" 1
SectionEnd

Section "Controller (server + dashboard)" SEC_CTRL
  StrCpy $Extras "$Extras,controller"
SectionEnd

Section "Worker (model execution)" SEC_WORK
  StrCpy $Extras "$Extras,worker"
SectionEnd

SectionGroup "Optional model backends"
  Section /o "Vision (image input)" SEC_VIS
    StrCpy $Extras "$Extras,vision"
  SectionEnd
  Section /o "Speech-to-text (Whisper)" SEC_STT
    StrCpy $Extras "$Extras,stt"
  SectionEnd
  Section /o "Text-to-speech (Kokoro)" SEC_TTS
    StrCpy $Extras "$Extras,tts"
  SectionEnd
  Section /o "Text-to-image (diffusers)" SEC_T2I
    StrCpy $Extras "$Extras,t2i"
  SectionEnd
  Section /o "Music (MusicGen)" SEC_MUS
    StrCpy $Extras "$Extras,music"
  SectionEnd
  Section /o "Kimi-Linear architecture" SEC_KIMI
    StrCpy $Extras "$Extras,kimi"
  SectionEnd
  Section /o "ACE-Step music - t2a (NVIDIA Ampere+ only)" SEC_ACE
    StrCpy $WithAce "1"
  SectionEnd
SectionGroupEnd

; runs last: shortcuts + venv bootstrap
Section "-Finalize"
  ; trim a leading comma from $Extras
  StrCpy $0 $Extras 1
  ${If} $0 == ","
    StrCpy $Extras $Extras "" 1
  ${EndIf}

  ${If} $WithAce == "1"
  ${AndIf} $Flavor == "cpu"
    MessageBox MB_OK "ACE-Step music needs an NVIDIA Ampere+ GPU; with the CPU build selected it will be skipped. See docs/T2A.md."
  ${EndIf}

  CreateDirectory "$SMPROGRAMS\InfiniteModel"
  ${If} ${SectionIsSelected} ${SEC_CTRL}
    CreateShortCut "$SMPROGRAMS\InfiniteModel\InfiniteModel Controller.lnk" "$INSTDIR\infinitemodel-controller.cmd" "" "$INSTDIR\infinitemodel-controller.cmd" 0
  ${EndIf}
  ${If} ${SectionIsSelected} ${SEC_WORK}
    CreateShortCut "$SMPROGRAMS\InfiniteModel\InfiniteModel Worker.lnk" "$INSTDIR\infinitemodel-worker.cmd" "" "$INSTDIR\infinitemodel-worker.cmd" 0
  ${EndIf}
  CreateShortCut "$SMPROGRAMS\InfiniteModel\Uninstall InfiniteModel.lnk" "$INSTDIR\uninstall.exe"

  StrCpy $1 ""
  ${If} $WithAce == "1"
    StrCpy $1 "-WithAcestep"
  ${EndIf}
  DetailPrint "Building venv (torch=$Flavor, extras=$Extras). This downloads packages and can take several minutes..."
  nsExec::ExecToLog 'powershell -NoProfile -ExecutionPolicy Bypass -File "$INSTDIR\bootstrap.ps1" -Flavor $Flavor -Extras "$Extras" $1'
  Pop $2
  ${If} $2 != "0"
    MessageBox MB_OK "The setup step exited with code $2. You can re-run it any time:$\n  powershell -ExecutionPolicy Bypass -File $\"$INSTDIR\bootstrap.ps1$\" -Flavor $Flavor -Extras $\"$Extras$\""
  ${EndIf}
SectionEnd

; --- component descriptions -------------------------------------------------
!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_CTRL} "The controller: HTTP API + dashboard on port 21434."
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_WORK} "A worker: runs model execution and joins a controller."
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_VIS}  "Image inputs for vision models (pillow)."
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_STT}  "Whisper speech-to-text (soundfile)."
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_TTS}  "Kokoro text-to-speech (also needs the espeak-ng runtime)."
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_T2I}  "Text-to-image (diffusers + accelerate)."
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_MUS}  "MusicGen text-to-music."
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_KIMI} "Kimi-Linear architecture (fla-core + triton)."
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_ACE}  "ACE-Step music. Needs an NVIDIA Ampere+ GPU; guided install (see docs/T2A.md)."
!insertmacro MUI_FUNCTION_DESCRIPTION_END

; --- uninstaller ------------------------------------------------------------
Section "Uninstall"
  RMDir /r "$INSTDIR\venv"
  RMDir /r "$INSTDIR\wheel"
  Delete "$INSTDIR\infinitemodel-controller.cmd"
  Delete "$INSTDIR\infinitemodel-worker.cmd"
  Delete "$INSTDIR\bootstrap.ps1"
  Delete "$INSTDIR\README.md"
  Delete "$INSTDIR\LICENSE"
  Delete "$INSTDIR\uninstall.exe"
  RMDir "$INSTDIR"
  RMDir /r "$SMPROGRAMS\InfiniteModel"
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\InfiniteModel"
  ; State + models under %LOCALAPPDATA%\InfiniteModel are intentionally left in place.
  MessageBox MB_OK "InfiniteModel removed. Your models and state remain in $LOCALAPPDATA\InfiniteModel (delete by hand if you no longer need them)."
SectionEnd
