import subprocess
import sys
from faster_whisper import WhisperModel
from tqdm import tqdm
import os
from pathlib import Path
from utils import sec_to_srt, get_video_duration, run_ffmpeg_with_progress


# --- CONFIGURATION ---
CONFIG = {
    "model_size": "base",
    "audio_sample_rate": 16000,
}
# ---------------------


# -----------------------------
# Step 1: Extract audio
# -----------------------------
def extract_audio(video_path, audio_path):
    command = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ac", "1",
        "-ar", str(CONFIG["audio_sample_rate"]),
        audio_path
    ]
    run_ffmpeg_with_progress(command, video_path, "Extracting audio")


# -----------------------------
# Step 2: Transcribe audio (with faster-whisper)
# -----------------------------
def transcribe_audio(audio_path, model_size="medium"):
    # faster-whisper is a reimplementation of Whisper using CTranslate2 for faster inference.
    # Using 'int8' quantization for good speed on CPU.
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    
    segments_iterator, info = model.transcribe(audio_path, beam_size=5)
    
    # The rest of the script expects a list of dictionaries, so we convert the output.
    segments_list = []
    for segment in segments_iterator:
        segments_list.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text
        })
    
    print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
    return segments_list


# -----------------------------
# Step 3: Write SRT file
# -----------------------------
def write_srt(segments, srt_path):
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start = sec_to_srt(seg["start"])
            end = sec_to_srt(seg["end"])
            text = seg["text"].strip()

            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(text + "\n\n")


# -----------------------------
# Step 4: Burn subtitles
# -----------------------------
def burn_subtitles(video_path, srt_path, output_path):
    # Reverting to the original strategy of changing the CWD to avoid path escaping issues.
    video_path_obj = Path(video_path).resolve()
    srt_path_obj = Path(srt_path).resolve()
    output_path_obj = Path(output_path).resolve()

    # The directory where the ffmpeg command will be run.
    cwd_dir = video_path_obj.parent
    
    # Create paths relative to the new CWD.
    video_filename_relative = video_path_obj.name
    # Use as_posix() to ensure forward slashes, which is safer.
    srt_filename_relative = srt_path_obj.relative_to(cwd_dir).as_posix()
    output_filename_relative = output_path_obj.name

    # Using the 'subtitles' filter with a simple relative path.
    vf_arg = f"subtitles=filename='{srt_filename_relative}'"
    
    command = [
        "ffmpeg", "-y",
        "-i", video_filename_relative,
        "-vf", vf_arg,
        "-c:a", "copy",
        output_filename_relative,
    ]
    
    # Run the command from the video's directory.
    run_ffmpeg_with_progress(command, video_path, "Burning subtitles", cwd=cwd_dir)





# -----------------------------
# Main pipeline
# -----------------------------
def main(video_path, srt_path_arg=None):
    video_path_obj = Path(video_path).resolve()

    if not video_path_obj.exists():
        print("Video file not found.")
        sys.exit(1)

    base = video_path_obj.stem
    output_path = f"{base}_subtitled.mp4"

    if srt_path_arg:
        # A specific SRT file was provided, so only run the burn-in step.
        print("SRT file provided. Skipping transcription and burning subtitles directly...")
        srt_path = Path(srt_path_arg)
        if not srt_path.exists():
            print(f"Error: Provided SRT file not found at {srt_path_arg}")
            sys.exit(1)
        
        burn_subtitles(str(video_path_obj), str(srt_path), output_path)

    else:
        # No SRT file provided, run the full pipeline.
        audio_path = f"{base}_audio.wav"
        srt_path = f"{base}.srt"

        print("Extracting audio...")
        extract_audio(str(video_path_obj), audio_path)

        print("Transcribing audio...")
        segments = transcribe_audio(audio_path, model_size=CONFIG["model_size"])

        print("Writing subtitles...")
        write_srt(segments, srt_path)

        print("Burning subtitles into video...")
        burn_subtitles(str(video_path_obj), srt_path, output_path)

        # Optional cleanup
        os.remove(audio_path)

    print("\nDone.")
    print(f"Output video: {output_path}")


if __name__ == "__main__":
    try:
        if len(sys.argv) == 2:
            # Run the full pipeline
            main(sys.argv[1])
        elif len(sys.argv) == 3:
            # Run only the burn-in step
            main(sys.argv[1], srt_path_arg=sys.argv[2])
        else:
            print("Usage (full pipeline): python video_subtitles.py <video_file>")
            print("Usage (burn-in only):  python video_subtitles.py <video_file> <srt_file>")
            sys.exit(1)
    except subprocess.CalledProcessError as e:
        print("\n--- FFMPEG COMMAND FAILED ---")
        # The e.cmd is a list of arguments, join them for readability
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Return Code: {e.returncode}")
        print("\n--- FFMPEG OUTPUT ---")
        print(e.output)
        sys.exit(1)

