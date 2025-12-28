# Video Subtitle Generator

This project provides a Python script to automatically generate and burn subtitles into a video file. It performs the following steps:
1.  Extracts the audio from the video file.
2.  Transcribes the audio to text using the `faster-whisper` library for high performance.
3.  Generates a `.srt` (SubRip Subtitle) file from the transcription.
4.  Burns the subtitles directly into the video frames to create a new, subtitled video file.

## Environment Setup

This project uses `conda` for environment management to ensure all dependencies, including complex binary libraries, are handled correctly. The following instructions provide the most robust and stable setup method.

**1. Create a clean conda environment:**

First, create a new conda environment. We recommend using Python 3.10.

```bash
conda create -n subtitles python=3.10 -y
```

**2. Activate the new environment:**

You must activate the environment before installing packages.

```bash
conda activate subtitles
```

**3. Install all dependencies:**

This is the most critical step. The following command uses the `conda-forge` channel to install all necessary packages, including `ffmpeg` for video processing, `PyTorch` for the AI models, and `intel-openmp` to prevent common library conflicts.

```bash
conda install -c conda-forge ffmpeg pytorch transformers sentencepiece intel-openmp tqdm -y
```

After the main dependencies are installed via `conda`, install `faster-whisper` using `pip`:

```bash
pip install faster-whisper
```

Your environment is now correctly configured to run both the transcription and translation scripts without library conflicts.

## Usage

The project contains two main scripts:

### 1. Basic Transcription (`video_subtitles.py`)

This script generates subtitles in the original language detected in the video.

**Command:**
```bash
python video_subtitles.py <path_to_your_video.mp4>
```

**Example:**
```bash
python video_subtitles.py "MyMovie.mp4"
```

### 2. Transcription and Translation (`video_subtitles_translator.py`)

This enhanced script can either generate subtitles in the original language or translate them to a new language.

**Where to place your video:**

You can place your video file in the same directory as the script or provide a full path to its location.

**Usage Instructions:**

*   **Windows:** Use `python` or `py` in your shell.
*   **macOS/Linux:** Use `python3`.

**A. Generate subtitles in the original detected language:**

This works just like the basic script, creating an SRT file and a subtitled video in the language detected in the audio.

**Command:**
```bash
python video_subtitles_translator.py "MyPresentation.mp4"
```
*(Output: `MyPresentation.en.srt` and `MyPresentation_subtitled_en.mp4` if English was detected)*

**B. Transcribe and then TRANSLATE to another language:**

Provide a target language code to translate the subtitles before burning them into the video.

**Command:**
```bash
python video_subtitles_translator.py <path_to_your_video.mp4> --target-language <language_code>
```
**Example (translating to Arabic):**
```bash
python video_subtitles_translator.py "MyPresentation.mp4" -t ar
```
*(Output: `MyPresentation.ar.srt` and `MyPresentation_subtitled_ar.mp4`)*

**C. Burn an existing SRT file:**

If you already have an SRT file, you can burn it directly into the video without running transcription or translation.

**Command:**
```bash
python video_subtitles_translator.py <path_to_your_video.mp4> <path_to_your_subtitle_file.srt>
```
**Example:**
```bash
python video_subtitles_translator.py "MyPresentation.mp4" "MyPresentation.fr.srt"
```
*(Output: `MyPresentation_subtitled.mp4`)*

**Supported Target Language Codes:**

You can use the following language codes with the `--target-language` (or `-t`) flag:

*   `ar`: Arabic
*   `de`: German
*   `es`: Spanish
*   `fr`: French
*   `he`: Hebrew
*   `it`: Italian
*   `ja`: Japanese
*   `pt`: Portuguese
*   `ru`: Russian
*   `zh`: Chinese
*   `en`: English

