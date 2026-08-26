!macro SUBGEN_STOP_RUNNING_PROCESSES
  DetailPrint "Stopping running SubGen processes..."
  nsExec::ExecToLog '"$SYSDIR\taskkill.exe" /F /T /IM "subgen-desktop.exe"'
  Pop $0
  nsExec::ExecToLog '"$SYSDIR\taskkill.exe" /F /T /IM "SubGenBackend.exe"'
  Pop $0
  Sleep 1500
!macroend

!macro SUBGEN_REMOVE_VOLATILE_UPLOADS
  RMDir /r "$LOCALAPPDATA\com.subgen.studio\uploads\jobs"
!macroend

!macro NSIS_HOOK_PREINSTALL
  !insertmacro SUBGEN_STOP_RUNNING_PROCESSES
  !insertmacro SUBGEN_REMOVE_VOLATILE_UPLOADS
!macroend

!macro NSIS_HOOK_PREUNINSTALL
  !insertmacro SUBGEN_STOP_RUNNING_PROCESSES
  !insertmacro SUBGEN_REMOVE_VOLATILE_UPLOADS
!macroend
