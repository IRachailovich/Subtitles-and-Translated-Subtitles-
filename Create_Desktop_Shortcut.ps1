# PowerShell script to create a desktop shortcut

$WshShell = New-Object -ComObject WScript.Shell
$DesktopPath = [System.Environment]::GetFolderPath('Desktop')
$ShortcutPath = Join-Path $DesktopPath "Subtitle Generator.lnk"

$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = Join-Path $PSScriptRoot "Launch_Subtitle_UI_Silent.vbs"
$Shortcut.WorkingDirectory = $PSScriptRoot
$Shortcut.Description = "Subtitle Generator & Translator"
$Shortcut.IconLocation = "shell32.dll,177"  # Video camera icon
$Shortcut.Save()

Write-Host "Desktop shortcut created successfully!" -ForegroundColor Green
Write-Host "Location: $ShortcutPath" -ForegroundColor Cyan
Write-Host "`nYou can now double-click 'Subtitle Generator' on your desktop to launch the app." -ForegroundColor Yellow
pause
