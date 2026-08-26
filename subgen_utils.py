import os
import subprocess
import re
import time
from tqdm import tqdm


def hidden_subprocess_kwargs():
    """Prevent bundled command-line media tools from flashing windows on Windows."""
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}
    return {}


def is_windows_interrupted_process_returncode(returncode):
    """Return True for Windows STATUS_CONTROL_C_EXIT (0xC000013A)."""
    try:
        return (int(returncode) & 0xFFFFFFFF) == 0xC000013A
    except (TypeError, ValueError):
        return False

# -----------------------------
# Utility: seconds → SRT time
# -----------------------------
def sec_to_srt(t):
    """Converts seconds to SRT time format (HH:MM:SS,ms)."""
    total_ms = max(0, int(float(t) * 1000 + 0.5))
    h, rem = divmod(total_ms, 3600 * 1000)
    m, rem = divmod(rem, 60 * 1000)
    s, ms = divmod(rem, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


# ---------------------------------
# Utility: Progress bar for ffmpeg
# ---------------------------------
def get_video_duration(video_path):
    """Gets the total duration of a video file in seconds using ffprobe."""
    command = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            **hidden_subprocess_kwargs(),
        )
    except FileNotFoundError as exc:
        raise RuntimeError("FFprobe is unavailable. Install or repair the bundled FFmpeg runtime.") from exc
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "").strip()
        detail_suffix = f" Details: {details}" if details else ""
        raise RuntimeError(
            f"FFprobe could not inspect the video (exit code {exc.returncode}).{detail_suffix}"
        ) from exc
    try:
        return float(result.stdout)
    except (ValueError, IndexError):
        # Fallback if ffprobe fails
        print("Warning: Could not determine video duration.")
        return None

def run_ffmpeg_with_progress(command, video_path, desc, cwd=None, interrupt_retries=1):
    """
    Runs an ffmpeg command with a tqdm progress bar.
    Captures and raises an error with detailed output if the command fails.
    """
    total_duration = get_video_duration(video_path)
    attempts = max(1, int(interrupt_retries) + 1)

    for attempt in range(attempts):
        try:
            if total_duration is None:
                subprocess.run(command, check=True, cwd=cwd, **hidden_subprocess_kwargs())
                return

            time_regex = re.compile(r"time=(\d{2}):(\d{2}):(\d{2})\.(\d{2})")
            output_lines = []
            with tqdm(total=round(total_duration), desc=desc, unit='s', dynamic_ncols=True) as pbar:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    encoding='utf-8',
                    errors='replace',
                    cwd=cwd,
                    **hidden_subprocess_kwargs(),
                )

                last_seconds = 0
                for line in process.stdout:
                    output_lines.append(line)
                    match = time_regex.search(line)
                    if match:
                        hours, minutes, seconds, _ = map(int, match.groups())
                        current_seconds = hours * 3600 + minutes * 60 + seconds
                        update_amount = current_seconds - last_seconds
                        if update_amount > 0:
                            pbar.update(update_amount)
                            last_seconds = current_seconds

            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    process.returncode,
                    command,
                    output="".join(output_lines),
                )
            return
        except subprocess.CalledProcessError as exc:
            retryable = is_windows_interrupted_process_returncode(exc.returncode)
            if not retryable or attempt + 1 >= attempts:
                raise
            print(
                f"Warning: {desc} was interrupted by Windows (0xC000013A). "
                f"Retrying ({attempt + 2}/{attempts})..."
            )
            time.sleep(0.5)
