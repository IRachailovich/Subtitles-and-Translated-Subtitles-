import subprocess
import sys
import argparse
from faster_whisper import WhisperModel
from tqdm import tqdm
import os
import json
from pathlib import Path
from utils import sec_to_srt, get_video_duration, run_ffmpeg_with_progress

# Import translation libraries
try:
    from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
except ImportError:
    print("Transformers library not found. Please install it with 'pip install transformers torch sentencepiece'")
    sys.exit(1)


# --- CONFIGURATION ---
CONFIG = {
    "model_size": "small",  # Using 'small' for a good balance of speed and accuracy
    "audio_sample_rate": 16000,
}
# ---------------------


# Helper function for color conversion
def hex_to_ass_color(hex_color):
    """Convert web hex color (#RRGGBB) to ASS subtitle format (&HBBGGRR&)"""
    hex_color = hex_color.lstrip('#')
    r = hex_color[0:2]
    g = hex_color[2:4]
    b = hex_color[4:6]
    return f"&H{b.upper()}{g.upper()}{r.upper()}&"


# ---------------------------------
# Supported Languages for Translation
# ---------------------------------
def get_supported_languages():
    """Returns a non-exhaustive dictionary of supported target languages for translation."""
    # Based on Helsinki-NLP Opus-MT models. Format: "target_code": "LanguageName"
    return {
        "ar": "Arabic", "de": "German", "es": "Spanish", "fr": "French",
        "he": "Hebrew", "it": "Italian", "ja": "Japanese", "pt": "Portuguese",
        "ru": "Russian", "zh": "Chinese", "en": "English", "fa": "Persian"
    }


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
# Step 2: Transcribe audio
# -----------------------------
def transcribe_audio(audio_path, model_size="medium"):
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments_iterator, info = model.transcribe(audio_path, beam_size=5)
    
    segments_list = [
        {"start": s.start, "end": s.end, "text": s.text}
        for s in segments_iterator
    ]
    
    print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
    return segments_list, info


# -----------------------------
# Step 3: Translate segments
# -----------------------------
def translate_segments(segments, src_lang, tgt_lang):
    """
    Translates subtitle segments using the best available
    open-source model for each language.
    """

    # -----------------------------
    # Model selection
    # -----------------------------
    if tgt_lang == "ar":
        # Best open-source English → Arabic MT
        model_name = "Helsinki-NLP/opus-mt-tc-big-en-ar"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

    elif tgt_lang == "fa":
        # Best open-source English → Persian MT
        model_name = "SeyedAli/English-to-Persian-Translation-mT5-V1"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    elif src_lang == "he" and tgt_lang == "en":
        # Hebrew → English (note: direction is reversed in model name)
        model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

    else:
        # Default fallback (existing behavior)
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

    print(f"Loading translation model: {model_name}")

    # -----------------------------
    # Batch translation
    # -----------------------------
    original_texts = [seg["text"] for seg in segments]
    translated_texts = []
    batch_size = 8

    print(f"Translating {len(original_texts)} segments to '{tgt_lang}'...")

    for i in tqdm(range(0, len(original_texts), batch_size), desc="Translating"):
        batch = original_texts[i:i + batch_size]

        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        with torch.no_grad():
            outputs = model.generate(**encoded, max_length=512)

        translated_texts.extend(
            tokenizer.batch_decode(outputs, skip_special_tokens=True)
        )

    # -----------------------------
    # Rebuild segments
    # -----------------------------
    translated_segments = []
    for i, seg in enumerate(segments):
        translated_segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": translated_texts[i]
        })

    return translated_segments



# -----------------------------
# Step 4: Write SRT file
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
# Step 5: Burn subtitles
# -----------------------------
def burn_subtitles(video_path, srt_path, output_path, lang_code=None, style_config=None):
    video_path_obj = Path(video_path).resolve()
    srt_path_obj = Path(srt_path).resolve()
    output_path_obj = Path(output_path).resolve()
    cwd_dir = video_path_obj.parent
    
    video_filename_relative = video_path_obj.name
    srt_filename_relative = srt_path_obj.relative_to(cwd_dir).as_posix()
    output_filename_relative = output_path_obj.name

    # Base subtitle filter
    
    # Use custom style config if provided, otherwise use defaults
    if style_config:
        font_name = style_config.get("font_name", "Arial")
        font_size = style_config.get("font_size", 20)
        primary_color = hex_to_ass_color(style_config.get("primary_color", "#FFFFFF"))
        text_outline_color = hex_to_ass_color(style_config.get("outline_color", "#000000"))  # Text outline from outline_color field
        background_color = hex_to_ass_color(style_config.get("back_color", "#000000"))  # Background from back_color field
        outline_width = style_config.get("outline_width", 1)
        shadow = style_config.get("shadow", 1)
        border_style = style_config.get("border_style", 3)
        
        # Apply font directory if using special fonts
        font_path = "Amiri-Regular.ttf"
        fonts_dir = Path(font_path).parent.as_posix()
        
        if font_name == "Amiri":
            vf_arg = (
                f"subtitles=filename='{srt_filename_relative}':"
                f"charenc=UTF-8:"
                f"fontsdir='{fonts_dir}':"
                f"force_style='FontName={font_name},"
                f"FontSize={font_size},"
                f"PrimaryColour={primary_color},"
                f"OutlineColour={text_outline_color},"
                f"BackColour={background_color},"
                f"Shadow={shadow},"
                f"Outline={outline_width},"
                f"BorderStyle={border_style}'"
            )
        else:
            vf_arg = (
                f"subtitles=filename='{srt_filename_relative}':"
                f"charenc=UTF-8:"
                f"force_style='FontName={font_name},"
                f"FontSize={font_size},"
                f"PrimaryColour={primary_color},"
                f"OutlineColour={text_outline_color},"
                f"BackColour={background_color},"
                f"Shadow={shadow},"
                f"Outline={outline_width},"
                f"BorderStyle={border_style}'"
            )
    else:
        # Default styles based on language
        font_path = "Amiri-Regular.ttf"
        fonts_dir = Path(font_path).parent.as_posix()

        if lang_code in ["ar", "fa"]:
            vf_arg = (
                f"subtitles=filename='{srt_filename_relative}':"
                f"charenc=UTF-8:"
                f"fontsdir='{fonts_dir}':"
                "force_style='FontName=Amiri,"
                "FontSize=36,"
                "PrimaryColour=&HFFFFFF&,"
                "OutlineColour=&H000000&,"
                "Shadow=1,"
                "Outline=1,"
                "BorderStyle=1'"
            )
        else:
            vf_arg = (
                f"subtitles=filename='{srt_filename_relative}':"
                f"charenc=UTF-8:"
                "force_style='FontSize=20,"
                "PrimaryColour=&HFFFFFF&,"
                "OutlineColour=&H000000&,"
                "Shadow=1,"
                "Outline=1,"
                "BorderStyle=3'"
            )

    # ----------------------------------

    command = [
        "ffmpeg", "-y",
        "-i", video_filename_relative,
        "-vf", vf_arg,
        "-c:a", "copy",
        output_filename_relative,
    ]
    
    run_ffmpeg_with_progress(command, str(video_path_obj), "Burning subtitles", cwd=cwd_dir)


# -----------------------------
# Main pipeline
# -----------------------------
def main(video_path, srt_path_arg=None, target_language=None, style_config=None):
    video_path_obj = Path(video_path).resolve()

    if not video_path_obj.exists():
        print(f"Error: Video file not found at {video_path}")
        sys.exit(1)

    base = video_path_obj.stem
    
    if srt_path_arg:
        # A specific SRT file was provided, so only run the burn-in step.
        print("SRT file provided. Skipping transcription and burning subtitles directly...")
        srt_path = Path(srt_path_arg)
        if not srt_path.exists():
            print(f"Error: Provided SRT file not found at {srt_path_arg}")
            sys.exit(1)
        
        output_path = f"{base}_subtitled.mp4"
        burn_subtitles(str(video_path_obj), str(srt_path), output_path, style_config=style_config)

    else:
        # No SRT file provided, run the full pipeline.
        audio_path = f"{base}_audio.wav"
        
        print("Step 1: Extracting audio...")
        extract_audio(str(video_path_obj), audio_path)

        print("\\nStep 2: Transcribing audio...")
        segments, info = transcribe_audio(audio_path, model_size=CONFIG["model_size"])

        # Decide the final segments (translated or original)
        final_segments = segments
        srt_lang_code = info.language

        if target_language and target_language != info.language:
            print(f"\\nStep 3: Translating from '{info.language}' to '{target_language}'...")
            final_segments = translate_segments(segments, src_lang=info.language, tgt_lang=target_language)
            srt_lang_code = target_language
        else:
            print("\\nStep 3: Skipping translation.")

        # Define SRT path based on language
        srt_path = f"{base}.{srt_lang_code}.srt"

        print(f"\\nStep 4: Writing subtitles to '{srt_path}'...")
        write_srt(final_segments, srt_path)

        print("\\nStep 5: Burning subtitles into video...")
        output_path = f"{base}_subtitled_{srt_lang_code}.mp4"
        burn_subtitles(str(video_path_obj), srt_path, output_path, srt_lang_code, style_config)

        # Optional cleanup
        os.remove(audio_path)

    print("\\n--- Done ---")
    print(f"Output video: {output_path}")


if __name__ == "__main__":
    supported_langs = get_supported_languages()
    lang_help = ", ".join([f"'{k}' ({v})" for k, v in supported_langs.items()])

    parser = argparse.ArgumentParser(description="Generate and translate subtitles for a video file.")
    parser.add_argument("video_path", type=str, help="Path to the video file.")
    parser.add_argument("srt_path", type=str, nargs='?', default=None, help="(Optional) Path to an existing SRT file to burn directly.")
    parser.add_argument(
        "--target-language", "-t", type=str, default=None,
        help=f"Optional: Language to translate the subtitles into. Supported codes: {list(supported_langs.keys())}"
    )
    parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="Optional: Path to JSON configuration file for subtitle styling"
    )
    
    args = parser.parse_args()
    
    # Load style config if provided
    style_config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                style_config = json.load(f)
            print(f"Loaded style configuration from {args.config}")
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
            print("Using default styles...")

    if args.target_language and args.target_language not in supported_langs:
        print(f"Error: Unsupported target language '{args.target_language}'.")
        print("Please use one of the following language codes:")
        print(lang_help)
        sys.exit(1)
        
    try:
        main(args.video_path, args.srt_path, args.target_language, style_config)
    except subprocess.CalledProcessError as e:
        print("\n--- FFMPEG COMMAND FAILED ---")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Return Code: {e.returncode}")
        print("\n--- FFMPEG OUTPUT ---")
        print(e.output)
        sys.exit(1)
    except Exception as e:
        print(f"\n--- AN ERROR OCCURRED ---")
        print(f"Error: {e}")
        sys.exit(1)

