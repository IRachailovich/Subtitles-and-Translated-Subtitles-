# Subtitle Generator UI Instructions

## Quick Start

1. **Launch the UI:**
   ```powershell
   python subtitle_ui.py
   ```

2. **Configure Your Subtitles:**
   - Select a video file using the "Browse" button
   - Choose your target language from the dropdown
   - Customize the subtitle appearance:
     - **Font Settings:** Choose font family and size
     - **Colors:** Click color buttons to pick:
       - Text Color (default: white)
       - Outline Color (default: pink #E361F7)
       - Background Color (default: black)
     - **Style:** Adjust outline width, shadow, and border style

3. **Preview:** The "ABC" preview shows how your subtitles will look

4. **Generate:** Click "Generate Subtitles" to process your video

## Features

### Border Styles
- **Standard Outline (1):** Text with outline, no background box
- **Opaque Box (3):** Text in a solid background box
- **Outline + Box (4):** Text with outline AND background box

### Color Conversion
The UI automatically converts web colors (#RRGGBB) to ASS format (&HBBGGRR&)

### Save/Load Config
- Save your favorite configurations as JSON files
- Load them for future videos

## Example Configurations

### Professional Look (White text, black outline, semi-transparent box)
- Text: #FFFFFF (white)
- Outline: #000000 (black)
- Background: #000000 (black)
- Outline Width: 2
- Shadow: 1
- Border Style: Opaque Box (3)

### Colorful Style (White text, pink outline)
- Text: #FFFFFF (white)
- Outline: #E361F7 (pink)
- Background: #000000 (black)
- Outline Width: 2
- Shadow: 1
- Border Style: Outline + Box (4)

### Arabic/Persian (Amiri font)
- Font: Amiri
- Text: #FFFFFF (white)
- Outline: #000000 (black)
- Font Size: 36
- Border Style: Standard Outline (1)

## Command Line Alternative

You can still use the command line with a saved config:
```powershell
python video_subtitles_translator.py "video.mp4" --target-language en --config my_config.json
```

## Troubleshooting

- **UI doesn't launch:** Ensure tkinter is installed (comes with Python by default)
- **Preview doesn't match output:** The canvas preview is simplified; actual output will use proper ASS rendering
- **Colors look different:** Remember ASS uses BGR format, but the UI handles conversion automatically
