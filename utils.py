import subprocess
import re
from tqdm import tqdm

# -----------------------------
# Utility: seconds → SRT time
# -----------------------------
def sec_to_srt(t):
    """Converts seconds to SRT time format (HH:MM:SS,ms)."""
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
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
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    try:
        return float(result.stdout)
    except (ValueError, IndexError):
        # Fallback if ffprobe fails
        print("Warning: Could not determine video duration.")
        return None

def run_ffmpeg_with_progress(command, video_path, desc, cwd=None):
    """
    Runs an ffmpeg command with a tqdm progress bar.
    Captures and raises an error with detailed output if the command fails.
    """
    total_duration = get_video_duration(video_path)
    if total_duration is None:
        # If we can't get the duration, run without a progress bar
        subprocess.run(command, check=True, cwd=cwd)
        return

    # Regex to find the time in ffmpeg's output
    time_regex = re.compile(r"time=(\d{2}):(\d{2}):(\d{2})\.(\d{2})")
    
    output_lines = []

    with tqdm(total=round(total_duration), desc=desc, unit='s', dynamic_ncols=True) as pbar:
        # Use Popen to stream output
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            encoding='utf-8',
            cwd=cwd
        )
        
        last_seconds = 0
        for line in process.stdout:
            output_lines.append(line)
            match = time_regex.search(line)
            if match:
                hours, minutes, seconds, _ = map(int, match.groups())
                current_seconds = hours * 3600 + minutes * 60 + seconds
                
                # Update progress bar
                update_amount = current_seconds - last_seconds
                if update_amount > 0:
                    pbar.update(update_amount)
                    last_seconds = current_seconds
    
    process.wait()
    if process.returncode != 0:
        error_output = "".join(output_lines)
        raise subprocess.CalledProcessError(
            process.returncode, 
            command, 
            output=error_output
        )
