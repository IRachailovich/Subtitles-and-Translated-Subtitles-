# Desktop Application Installation Guide

## Quick Setup (3 Steps)

### Step 1: Create Desktop Shortcut
Right-click `Create_Desktop_Shortcut.ps1` and select **"Run with PowerShell"**

A shortcut called "Subtitle Generator" will appear on your desktop.

---

### Step 2: Launch the App
Double-click **"Subtitle Generator"** on your desktop - the UI will open automatically!

---

### Step 3: Use the App
No terminal needed! Just:
1. Select your video
2. Configure subtitle style
3. Click "Generate Subtitles"

---

## Alternative Launch Methods

### Option 1: Silent Launch (No Terminal Window)
Double-click: `Launch_Subtitle_UI_Silent.vbs`

### Option 2: With Terminal (for debugging)
Double-click: `Launch_Subtitle_UI.bat`

### Option 3: Direct Python (from terminal)
```powershell
conda activate subtitles
python subtitle_ui.py
```

---

## Troubleshooting

### "Cannot run scripts" error when creating shortcut:
1. Open PowerShell as Administrator
2. Run: `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`
3. Type `Y` and press Enter
4. Try creating the shortcut again

### Desktop shortcut doesn't work:
- Make sure you ran `Create_Desktop_Shortcut.ps1` from the Subtitles_generator folder
- Verify the conda environment "subtitles" exists
- Try using `Launch_Subtitle_UI.bat` instead to see error messages

### UI doesn't open:
- Open PowerShell in this folder
- Run: `conda activate subtitles`
- Run: `python subtitle_ui.py`
- Check for error messages

---

## Customization

### Change Desktop Icon:
1. Right-click the desktop shortcut → Properties
2. Click "Change Icon"
3. Browse for an .ico file or select from Windows icons

### Pin to Taskbar:
1. Create desktop shortcut first
2. Right-click the shortcut
3. Select "Pin to taskbar"

### Pin to Start Menu:
1. Create desktop shortcut first
2. Drag it to Start Menu
3. Or right-click → "Pin to Start"

---

## What Each File Does

- **Launch_Subtitle_UI.bat** - Activates conda and starts UI (shows terminal)
- **Launch_Subtitle_UI_Silent.vbs** - Starts UI silently (no terminal window)
- **Create_Desktop_Shortcut.ps1** - Creates desktop shortcut automatically
- **subtitle_ui.py** - The main GUI application

---

## Uninstall

To remove:
1. Delete the desktop shortcut
2. (Optional) Delete the Subtitles_generator folder
3. (Optional) Run: `conda remove -n subtitles --all`
