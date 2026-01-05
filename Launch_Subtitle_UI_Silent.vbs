Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' Get the directory where this script is located
scriptPath = fso.GetParentFolderName(WScript.ScriptFullName)

' Run the batch file without showing a window
WshShell.Run """" & scriptPath & "\Launch_Subtitle_UI.bat" & """", 0, False
