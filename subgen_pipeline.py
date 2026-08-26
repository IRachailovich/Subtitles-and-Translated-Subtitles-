import os
from pathlib import Path

from subgen_paths import ENV_PATH, MODEL_CACHE_DIR, SOURCE_DIR

# Configure Hugging Face caches before importing libraries that may use them.
_hf_cache_root = MODEL_CACHE_DIR / "huggingface"
_hf_hub_cache = _hf_cache_root / "hub"
_hf_hub_cache.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(_hf_cache_root)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(_hf_hub_cache)
os.environ["HF_HUB_CACHE"] = str(_hf_hub_cache)

import warnings
# Suppress annoying third-party warnings from transformers/huggingface
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")


def load_dotenv(env_path=None):
    """Load simple KEY=VALUE lines from a .env file without requiring python-dotenv."""
    path = Path(env_path) if env_path else ENV_PATH
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv()

import subprocess
import sys
import argparse
import shutil
import textwrap
import urllib.error
import urllib.request
import uuid
import base64
import re
import hashlib
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher
from tqdm import tqdm
from subgen_db import (
    init_db,
    calculate_video_hash,
    get_cached_transcription,
    save_transcription,
    get_cached_translation,
    save_translation,
    get_burned_style,
    save_burned_style,
    save_review_manifest,
    normalize_source_location,
    register_media_source,
)
import json
from subgen_utils import (
    get_video_duration,
    hidden_subprocess_kwargs,
    run_ffmpeg_with_progress,
    sec_to_srt,
)
from subgen_providers import (
    call_google_structured_transcription,
    call_google_transcription,
    call_provider_translation,
    get_provider,
)
from subgen_transcription import (
    CURRENT_PRODUCTION_PROMPT_VERSION,
    GEMINI_TRANSCRIPTION_THINKING_CONFIG_VERSION,
    PROFESSIONAL_PROMPT_VERSION,
)
from subgen_acoustic import (
    discover_terminal_candidate_region,
    discover_text_guided_candidate_region,
    infer_repetition_evidence,
    trim_evidence_from_automatic_report,
)
from subgen_review import (
    add_issue as add_review_issue,
    apply_selected_cue_retranslation,
    assert_burn_allowed,
    complete_burn,
    contiguous_token_repetition_runs,
    estimate_recovery_boundaries,
    independent_speech_coverage,
    load_review,
    make_issue as make_review_issue,
    middle_emission_text_trim_report,
    new_review,
    repetition_anomaly_report,
    timestamp_proposal_report,
    offset_suffix_segments,
    merge_validated_gemini_regions,
    plan_adaptive_recovery_windows,
    save_review,
    selected_source_cues,
    set_ready_for_review,
    sha256_file as sha256_full_file,
    terminal_repetition_trim_report,
)
from subgen_planner import (
    ROUTE_DIRECT_TIMESTAMPED_TRANSCRIPT,
    build_pipeline_plan,
)
from subgen_transcription_adapters import (
    TranscriptionArtifact,
    call_cohere_transcription,
    call_google_timestamped_transcription,
    call_openai_audio_transcription,
    call_openai_compatible_audio_transcription,
    call_xai_transcription,
    group_words_for_subtitles,
    normalize_language_label,
)
from subgen_longform import (
    LONGFORM_PIPELINE_VERSION,
    interval_overlap_seconds,
    run_longform_transcription,
    select_coverage_recovery_gaps,
    validate_timed_segments as validate_longform_timed_segments,
)


# --- CONFIGURATION ---
# Alignment compatibility is determined by the implementation, not by a user's
# persisted settings. Bump this whenever cached timing artifacts become unsafe.
CURRENT_TIMING_ALIGNMENT_VERSION = "2026-08-06-signal-aware-word-anchors-v3"

CONFIG = {
    "model_size": "small",  # Using 'small' for a good balance of speed and accuracy
    "audio_sample_rate": 16000,
    "device": "auto",
    "beam_size": 5,
    "transcription_backend": "local",
    "transcription_provider": "local",
    "transcription_model": "whisper-1",
    "source_language": None,
    "transcription_prompt": "",
    "api_audio_bitrate": "64k",
    "longform_enabled": True,
    "longform_pipeline_version": LONGFORM_PIPELINE_VERSION,
    "longform_chunk_target_seconds": 300,
    "longform_chunk_min_seconds": 180,
    "longform_chunk_max_seconds": 480,
    "longform_chunk_overlap_seconds": 12,
    "longform_boundary_search_seconds": 60,
    "longform_min_boundary_silence_seconds": 1.0,
    "longform_retry_context_seconds": 15,
    "longform_invalid_chunk_split_retry_enabled": True,
    "longform_invalid_chunk_split_retry_min_seconds": 60,
    "longform_invalid_chunk_split_retry_overlap_seconds": 6,
    "longform_coverage_recovery_enabled": True,
    "longform_coverage_recovery_context_seconds": 6,
    "longform_coverage_recovery_max_attempts_per_chunk": 6,
    "longform_coverage_recovery_max_total_attempts": 12,
    "longform_coverage_recovery_min_novel_probability": 0.60,
    "longform_coverage_recovery_existing_match_tolerance_seconds": 0.75,
    "longform_coverage_recovery_allow_unscored_google_segments": True,
    "longform_coverage_recovery_allow_unscored_timing_evidence": True,
    "longform_coverage_recovery_min_unscored_speech_overlap_ratio": 0.80,
    "longform_coverage_recovery_max_window_seconds": 180,
    "local_min_language_probability": 0.35,
    "local_low_word_probability_threshold": 0.35,
    "local_max_low_word_probability_ratio": 0.35,
    "local_multilingual_auto": True,
    "local_condition_on_previous_text": False,
    "local_temperature": 0.0,
    "local_hallucination_silence_threshold": 2.0,
    "local_language_detection_segments": 3,
    "local_language_window_seconds": 15,
    "local_language_max_windows_per_chunk": 24,
    "local_isolated_language_window_override_probability": 0.90,
    "local_language_run_min_windows_for_retranscription": 2,
    "local_language_run_single_window_min_probability": 0.90,
    "local_max_identical_token_run": 7,
    "local_max_character_run": 11,
    "local_character_run_normalization_length": 3,
    # Two occurrences are the mathematical definition of a repetition. These
    # values discover candidates only; text counts never prove hallucination.
    "local_phrase_loop_min_repetitions": 2,
    "local_phrase_loop_min_words": 2,
    "local_phrase_loop_max_unit_words": 0,
    "longform_speech_map_engine": "auto",
    "longform_vad_analysis_window_seconds": 600,
    "longform_vad_analysis_overlap_seconds": 2,
    "longform_vad_threshold": 0.45,
    "longform_vad_min_speech_ms": 150,
    "longform_vad_min_silence_ms": 350,
    "longform_vad_speech_pad_ms": 350,
    "longform_confirmed_silence_seconds": 1.0,
    "longform_max_uncovered_gap_seconds": 1.5,
    "longform_max_uncovered_ratio": 0.03,
    "api_transcription_chunking": "auto",
    "api_transcription_chunk_seconds": 180,
    "api_transcription_chunk_overlap_seconds": 8,
    "api_timing_anchor_chunk_seconds": 30,
    "api_timing_anchor_chunk_overlap_seconds": 4,
    "api_timing_anchor_empty_chunk_retry_context_seconds": 6,
    "api_timing_anchor_silence_noise_db": -45,
    "api_timing_anchor_min_speech_seconds": 0.25,
    "api_timing_anchor_min_speech_ratio": 0.01,
    "timing_alignment_version": CURRENT_TIMING_ALIGNMENT_VERSION,
    "api_transcription_completeness_retries": 1,
    "api_transcription_reuse_text_on_force": True,
    "api_transcription_suspect_output_tokens": 2048,
    "api_transcription_reuse_suspect_transcripts": False,
    "google_transcription_temperature": 0.0,
    "google_transcription_retry_temperature": 0.8,
    "google_transcription_output_tokens_per_second": 32.0,
    "google_transcription_min_output_tokens": 512,
    "google_transcription_output_token_ceiling": 65536,
    # The professional variants remain recovery/diagnostic modes until the
    # representative human-referenced benchmark is complete and non-regressing.
    "google_transcription_prompt_version": CURRENT_PRODUCTION_PROMPT_VERSION,
    "review_before_burn": True,
    "transcript_plausibility_retries": 1,
    "transcript_plausibility_max_words_per_second": 8.0,
    "transcript_plausibility_max_chars_per_second": 80.0,
    "transcript_plausibility_min_repetitive_suffix_repetitions": 2,
    "transcript_plausibility_min_repetitive_suffix_words": 2,
    "transcript_plausibility_max_repetitive_unit_words": 0,
    "automatic_repetition_deletion_enabled": True,
    "automatic_repetition_verification_enabled": True,
    "api_transcript_timing_mode": "precise",
    "timing_anchor_local_fallback_enabled": False,
    "timing_anchor_local_fallback_model_size": "small",
    "timing_anchor_canonical_segment_fallback_enabled": True,
    "translation_batch_size": 8,
    "translation_min_semantic_batch_size": 4,
    "translation_backend": "transformers",
    "translation_provider": "local",
    "llm_model": "gpt-4o",
    "translation_context_window": 2,
    "qa_enabled": True,
    "qa_provider": "openai",
    "qa_model": "gpt-4o-mini",
    "qa_policy": "stop",
    "source_qa_max_prompt_chars": 48000,
    "source_qa_max_segments_per_batch": 200,
    "source_qa_context_segments": 2,
    "source_qa_max_alignment_metadata_chars": 16000,
    "translation_qa_enabled": True,
    "translation_qa_policy": "stop",
    "translation_qa_max_repairs": 1,
    "source_timing_verifier_policy": "stop",
    "source_timing_verifier_block_seconds": 30,
    "source_timing_verifier_padding_seconds": 8,
    "source_timing_verifier_min_token_coverage": 0.75,
    "source_timing_verifier_min_checked_segment_ratio": 0.75,
    "source_timing_verifier_max_start_drift_seconds": 0.85,
    "source_timing_verifier_max_early_end_seconds": 0.30,
    "source_timing_verifier_max_late_end_seconds": 2.50,
    "source_timing_verifier_max_bad_segment_ratio": 0.05,
    "source_timing_verifier_max_bad_segments": 2,
    "source_speech_coverage_max_uncovered_seconds": 8.0,
    "source_speech_coverage_max_uncovered_gap_seconds": 8.0,
    "source_speech_coverage_max_uncovered_ratio": 0.08,
    "source_speech_coverage_padding_seconds": 0.5,
    "subtitle_mode": "auto",
    "tiktok_style": False,
    "visual_style_enabled": True,
    "visual_style_provider": "openai",
    "visual_style_model": "gpt-4o-mini",
    "visual_style_sample_count": 5,
    "visual_style_frame_width": 960,
    "source_dialect": "auto",
    "target_dialect": "natural",
    "translator_notes": "",
    "translation_glossary": [],
    "api_pricing": {
        "openai": {
            "whisper-1": {
                "per_minute": 0.006,
            },
            "gpt-4o": {
                "input_per_1m": 2.50,
                "cached_input_per_1m": 1.25,
                "output_per_1m": 10.00,
            },
            "gpt-4o-mini": {
                "input_per_1m": 0.150,
                "cached_input_per_1m": 0.075,
                "output_per_1m": 0.600,
            },
        },
    },
    "max_chars_per_line": 42,
    "max_lines": 2,
    "min_subtitle_duration": 0.8,
    "custom_providers": {},
}

# ---------------------


# Helper function for color conversion
def hex_to_ass_color(hex_color, opacity_percent=None):
    """Convert web hex color (#RRGGBB) to ASS subtitle format (&HAABBGGRR&)."""
    hex_color = hex_color.lstrip('#')
    r = hex_color[0:2]
    g = hex_color[2:4]
    b = hex_color[4:6]
    if opacity_percent is None:
        return f"&H{b.upper()}{g.upper()}{r.upper()}&"
    opacity = max(0.0, min(100.0, float(opacity_percent)))
    alpha = round(255 * (1.0 - (opacity / 100.0)))
    return f"&H{alpha:02X}{b.upper()}{g.upper()}{r.upper()}&"


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


def normalize_language_shortcuts(argv, supported_langs):
    """Convert legacy language flags like -en or --en into --target-language en."""
    normalized_args = []

    for arg in argv:
        if arg.startswith("--") and arg[2:] in supported_langs:
            normalized_args.extend(["--target-language", arg[2:]])
        elif arg.startswith("-") and arg[1:] in supported_langs:
            normalized_args.extend(["--target-language", arg[1:]])
        else:
            normalized_args.append(arg)

    return normalized_args


def config_default(key, fallback=None):
    return CONFIG.get(key, fallback)


def configured_local_timing_anchor_fallback_model(pipeline_config):
    """Return the explicitly enabled timing-only fallback model, if any."""
    if not bool(
        pipeline_config.get(
            "timing_anchor_local_fallback_enabled",
            config_default(
                "timing_anchor_local_fallback_enabled",
                False,
            ),
        )
    ):
        return None
    return str(
        pipeline_config.get(
            "timing_anchor_local_fallback_model_size",
            config_default(
                "timing_anchor_local_fallback_model_size",
                "small",
            ),
        )
        or "small"
    ).strip()


def resolve_local_timing_anchor_fallback_after_failure(
    pipeline_config,
    error,
):
    model = configured_local_timing_anchor_fallback_model(
        pipeline_config
    )
    if model:
        return model
    raise RuntimeError(
        "OpenAI Whisper-1 timing anchors failed. "
        "SubGen did not start an unrequested local model; "
        "timing validation failed closed."
    ) from error


CACHE_ACTION_POLICIES = {
    "reuse_all": {
        "reuse_gemini_text": True,
        "reuse_source_timing": True,
        "reuse_translation": True,
        "force_burn": False,
    },
    "reburn": {
        "reuse_gemini_text": True,
        "reuse_source_timing": True,
        "reuse_translation": True,
        "force_burn": True,
    },
    "retime": {
        "reuse_gemini_text": True,
        "reuse_source_timing": False,
        "reuse_translation": False,
        "force_burn": True,
    },
    "regenerate_all": {
        "reuse_gemini_text": False,
        "reuse_source_timing": False,
        "reuse_translation": False,
        "force_burn": True,
    },
}


def resolve_cache_action(cache_action=None, force=False):
    """Return the explicit cache policy, preserving legacy --force behavior."""
    action = str(cache_action or "").strip().lower()
    if not action:
        action = "retime" if force else "reuse_all"
    if action not in CACHE_ACTION_POLICIES:
        valid = ", ".join(CACHE_ACTION_POLICIES)
        raise ValueError(f"Unknown cache action '{cache_action}'. Expected one of: {valid}.")
    return action, dict(CACHE_ACTION_POLICIES[action])


def merge_config(config_path=None):
    """Load optional JSON config and merge it over defaults."""
    merged = CONFIG.copy()
    style_config = None

    if not config_path:
        return merged, style_config

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            loaded_config = json.load(f)
        print(f"Loaded configuration from {config_path}")
    except Exception as e:
        print(f"Warning: Could not load config file: {e}")
        print("Using default settings...")
        return merged, style_config

    for key in merged:
        if key in loaded_config:
            merged[key] = loaded_config[key]

    # Backward compatible: the old config file was a flat subtitle style object.
    style_config = loaded_config.get("subtitle", loaded_config)
    return merged, style_config


def check_required_tools(require_video_tools=True):
    """Fail early with a clear message when external binaries are missing."""
    required = ["ffmpeg", "ffprobe"] if require_video_tools else []
    missing = [tool for tool in required if shutil.which(tool) is None]

    if missing:
        print(f"Error: Missing required tool(s): {', '.join(missing)}")
        print("Install them in the conda environment before running the pipeline.")
        sys.exit(1)


def import_torch():
    try:
        import torch
        return torch
    except ImportError as e:
        raise RuntimeError("PyTorch is required for local transcription/translation. Install torch in the environment.") from e


def import_faster_whisper():
    try:
        from faster_whisper import WhisperModel
        return WhisperModel
    except ImportError as e:
        raise RuntimeError("faster-whisper is required for local transcription timing. Install faster-whisper in the environment.") from e


def resolve_faster_whisper_model_reference(
    model_reference,
    *,
    hub_roots=None,
):
    requested = str(model_reference or "").strip()
    if not requested:
        raise RuntimeError("A Faster-Whisper model name or path is required.")

    required_files = (
        "config.json",
        "model.bin",
        "tokenizer.json",
    )

    def is_complete_snapshot(path):
        path = Path(path)
        if not path.is_dir():
            return False
        for name in required_files:
            candidate = path / name
            if not candidate.is_file():
                return False
            try:
                if candidate.stat().st_size <= 0:
                    return False
            except OSError:
                return False
        if not any(
            (path / name).is_file()
            and (path / name).stat().st_size > 0
            for name in ("vocabulary.json", "vocabulary.txt")
        ):
            return False
        try:
            return (path / "model.bin").stat().st_size >= 1024 * 1024
        except OSError:
            return False

    requested_path = Path(requested).expanduser()
    explicitly_relative = requested.startswith(
        (".\\", "./", "..\\", "../", "~")
    )
    if requested_path.is_absolute() or explicitly_relative:
        if not is_complete_snapshot(requested_path):
            raise RuntimeError(
                "The configured Faster-Whisper model path is incomplete: "
                f"{requested_path}"
            )
        return str(requested_path.resolve())

    model_slug = requested.split("/")[-1]
    if not model_slug.startswith("faster-whisper-"):
        model_slug = f"faster-whisper-{model_slug}"
    repository_cache_name = f"models--Systran--{model_slug}"
    roots = [
        Path(root)
        for root in (
            hub_roots
            if hub_roots is not None
            else (
                _hf_hub_cache,
                SOURCE_DIR / ".hf_cache" / "hub",
            )
        )
    ]
    for root in roots:
        snapshots = root / repository_cache_name / "snapshots"
        if not snapshots.is_dir():
            continue
        candidates = sorted(
            (
                path
                for path in snapshots.iterdir()
                if path.is_dir()
            ),
            key=lambda path: path.name,
            reverse=True,
        )
        for candidate in candidates:
            if is_complete_snapshot(candidate):
                return str(candidate.resolve())
    return requested


def resolve_whisperx_alignment_model_reference(
    language_code,
    model_reference,
    *,
    hub_roots=None,
):
    """Resolve a complete local Hugging Face alignment-model snapshot.

    WhisperX otherwise delegates cache discovery to Transformers.  SubGen has
    historically used more than one cache root, so relying on the active
    interpreter's HF_HOME can miss a complete project-local snapshot and try a
    network download.  Alignment must instead be reproducible and fail closed.
    """
    requested = str(model_reference or "").strip()
    if not requested:
        raise RuntimeError(
            f"WhisperX has no Hugging Face alignment model for language "
            f"{language_code or 'unknown'}."
        )

    required_files = (
        "config.json",
        "preprocessor_config.json",
    )

    def is_complete_snapshot(path):
        path = Path(path)
        if not path.is_dir():
            return False
        if any(
            not (path / name).is_file()
            or (path / name).stat().st_size <= 0
            for name in required_files
        ):
            return False
        if not any(
            (path / name).is_file()
            and (path / name).stat().st_size > 0
            for name in ("tokenizer.json", "vocab.json", "vocab.txt")
        ):
            return False
        if not any(
            path.glob("*.safetensors")
        ) and not any(path.glob("pytorch_model*.bin")):
            return False
        return True

    requested_path = Path(requested).expanduser()
    explicitly_relative = requested.startswith(
        (".\\", "./", "..\\", "../", "~")
    )
    if requested_path.is_absolute() or explicitly_relative:
        if not is_complete_snapshot(requested_path):
            raise RuntimeError(
                "The configured WhisperX alignment-model path is incomplete: "
                f"{requested_path}"
            )
        return str(requested_path.resolve())

    repository_cache_name = "models--" + requested.replace("/", "--")
    roots = [
        Path(root)
        for root in (
            hub_roots
            if hub_roots is not None
            else (
                _hf_hub_cache,
                SOURCE_DIR / ".hf_cache" / "hub",
            )
        )
    ]
    for root in roots:
        snapshots = root / repository_cache_name / "snapshots"
        if not snapshots.is_dir():
            continue
        candidates = sorted(
            (
                path
                for path in snapshots.iterdir()
                if path.is_dir()
            ),
            key=lambda path: path.name,
            reverse=True,
        )
        for candidate in candidates:
            if is_complete_snapshot(candidate):
                return str(candidate.resolve())
    raise RuntimeError(
        "No complete local WhisperX alignment-model snapshot is available for "
        f"{language_code or 'unknown'} ({requested})."
    )


def load_whisperx_alignment_model(whisperx, language_code, device):
    """Load a WhisperX aligner without an implicit network fallback."""
    import importlib

    language_code = language_code or "en"
    alignment_module = importlib.import_module("whisperx.alignment")
    torch_models = getattr(
        alignment_module,
        "DEFAULT_ALIGN_MODELS_TORCH",
        {},
    )
    hf_models = getattr(
        alignment_module,
        "DEFAULT_ALIGN_MODELS_HF",
        {},
    )
    if language_code in hf_models:
        model_reference = resolve_whisperx_alignment_model_reference(
            language_code,
            hf_models[language_code],
        )
        model, metadata = whisperx.load_align_model(
            language_code=language_code,
            device=device,
            model_name=model_reference,
            model_cache_only=True,
        )
        return model, metadata, {
            "source": "local_huggingface_snapshot",
            "reference": model_reference,
        }
    if language_code in torch_models:
        model_name = torch_models[language_code]
        checkpoint_name = {
            "WAV2VEC2_ASR_BASE_960H": "wav2vec2_fairseq_base_ls960_asr_ls960.pth",
            "VOXPOPULI_ASR_BASE_10K_FR": "wav2vec2_voxpopuli_base_10k_asr_fr.pt",
            "VOXPOPULI_ASR_BASE_10K_DE": "wav2vec2_voxpopuli_base_10k_asr_de.pt",
            "VOXPOPULI_ASR_BASE_10K_ES": "wav2vec2_voxpopuli_base_10k_asr_es.pt",
            "VOXPOPULI_ASR_BASE_10K_IT": "wav2vec2_voxpopuli_base_10k_asr_it.pt",
        }.get(model_name)
        torch_hub_dir = Path(os.environ.get("TORCH_HOME") or Path.home() / ".cache" / "torch")
        checkpoint_path = (
            torch_hub_dir / "hub" / "checkpoints" / checkpoint_name
            if checkpoint_name
            else None
        )
        if not checkpoint_path or not checkpoint_path.is_file():
            raise RuntimeError(
                "No complete local WhisperX torchaudio alignment checkpoint is "
                f"available for {language_code} ({model_name})."
            )
        model, metadata = whisperx.load_align_model(
            language_code=language_code,
            device=device,
            model_name=model_name,
            model_dir=str(checkpoint_path.parent),
        )
        return model, metadata, {
            "source": "local_torchaudio_checkpoint",
            "reference": str(checkpoint_path),
        }
    raise RuntimeError(
        f"WhisperX has no default alignment model for language {language_code}."
    )


def import_whisperx():
    try:
        import whisperx
        return whisperx
    except ImportError as e:
        raise RuntimeError(
            "WhisperX is required for forced alignment. Install it in the subtitles environment, "
            "or use --api-transcript-timing-mode best/api_fuzzy for fallback timing."
        ) from e


def import_transformers_translation():
    try:
        from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
        return MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
    except ImportError as e:
        raise RuntimeError(
            "Transformers translation requires transformers, torch, and sentencepiece in the environment."
        ) from e


def require_openai_api_key(env_key="OPENAI_API_KEY"):
    env_key = env_key or "OPENAI_API_KEY"
    api_key = os.environ.get(env_key)
    if not api_key:
        raise RuntimeError(
            f"{env_key} is not set. Add it to .env or set it during SubGen setup before using OpenAI API backends."
        )
    return api_key


def resolve_device(requested_device):
    """Resolve cpu/cuda/auto into a concrete device for Whisper and translation."""
    requested_device = (requested_device or "auto").lower()
    if requested_device not in {"auto", "cpu", "cuda"}:
        print("Error: --device must be one of: auto, cpu, cuda")
        sys.exit(1)

    try:
        torch = import_torch()
    except RuntimeError:
        if requested_device == "cuda":
            raise
        return "cpu"

    cuda_available = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())

    if requested_device == "cuda" and not cuda_available:
        print("Error: CUDA was requested, but PyTorch cannot see a CUDA device.")
        sys.exit(1)

    if requested_device == "auto":
        return "cuda" if cuda_available else "cpu"

    return requested_device


def build_multipart_form_data(fields, files):
    boundary = f"----CodexBoundary{uuid.uuid4().hex}"
    body = bytearray()

    for name, value in fields.items():
        if value is None or value == "":
            continue
        values = value if isinstance(value, (list, tuple)) else [value]
        for field_value in values:
            body.extend(f"--{boundary}\r\n".encode("utf-8"))
            body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
            body.extend(str(field_value).encode("utf-8"))
            body.extend(b"\r\n")

    for name, file_path in files.items():
        path = Path(file_path)
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            (
                f'Content-Disposition: form-data; name="{name}"; '
                f'filename="{path.name}"\r\n'
            ).encode("utf-8")
        )
        body.extend(b"Content-Type: application/octet-stream\r\n\r\n")
        body.extend(path.read_bytes())
        body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    return bytes(body), boundary


def post_openai_multipart(endpoint, fields, files, timeout=240, api_env_key="OPENAI_API_KEY"):
    api_key = require_openai_api_key(api_env_key)
    body, boundary = build_multipart_form_data(fields, files)
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI request failed: HTTP {e.code}: {error_body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"OpenAI request failed: {e}") from e


def split_text_into_segment_count(text, segment_count):
    text = normalize_subtitle_text(text)
    if not text or segment_count <= 0:
        return []

    words = text.split()
    if segment_count == 1:
        return [text]

    chunks = []
    cursor = 0
    for index in range(segment_count):
        remaining_words = len(words) - cursor
        remaining_segments = segment_count - index
        take = max(1, round(remaining_words / remaining_segments))
        chunk = " ".join(words[cursor:cursor + take])
        chunks.append(chunk)
        cursor += take

    if cursor < len(words):
        chunks[-1] = normalize_subtitle_text(f"{chunks[-1]} {' '.join(words[cursor:])}")

    return chunks


def compact_text_for_alignment(text):
    normalized = unicodedata.normalize("NFKD", normalize_subtitle_text(text).casefold())
    arabic_fold = {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ؤ": "و",
        "ئ": "ي",
        "ة": "ه",
        "ـ": "",
    }
    chars = []
    for char in normalized:
        if unicodedata.category(char).startswith("M"):
            continue
        char = arabic_fold.get(char, char)
        if char and char.isalnum():
            chars.append(char)
    return "".join(chars)


def similarity_score(left, right):
    left = compact_text_for_alignment(left)
    right = compact_text_for_alignment(right)
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def align_transcript_to_timing_segments_proportional(transcript_text, timing_segments):
    chunks = split_text_into_segment_count(transcript_text, len(timing_segments))
    return [
        {
            "start": timing_segment["start"],
            "end": timing_segment["end"],
            "text": chunk,
        }
        for timing_segment, chunk in zip(timing_segments, chunks)
        if normalize_subtitle_text(chunk)
    ]


def build_rough_transcript_segments(transcript_text, duration_seconds=None, max_segment_seconds=30.0):
    text = normalize_subtitle_text(transcript_text)
    if not text:
        return []
    duration = float(duration_seconds or 0.0)
    if duration <= 0:
        duration = max(1.0, len(text.split()) * 0.35)
    segment_count = max(1, int((duration + max_segment_seconds - 0.001) // max_segment_seconds))
    chunks = split_text_into_segment_count(text, segment_count)
    segments = []
    for index, chunk in enumerate(chunks):
        start = duration * index / segment_count
        end = duration * (index + 1) / segment_count
        segments.append({
            "start": start,
            "end": max(start + 0.1, end),
            "text": chunk,
        })
    return segments


def infer_language_from_text(text, default="en"):
    if not text:
        return default
    counts = {
        "he": len(re.findall(r"[\u0590-\u05FF]", text)),
        "ar": len(re.findall(r"[\u0600-\u06FF]", text)),
        "ru": len(re.findall(r"[\u0400-\u04FF]", text)),
        "zh": len(re.findall(r"[\u4E00-\u9FFF]", text)),
        "ja": len(re.findall(r"[\u3040-\u30FF]", text)),
    }
    language, count = max(counts.items(), key=lambda item: item[1])
    return language if count >= 8 else default


def align_transcript_to_timing_segments(transcript_text, timing_segments, search_window=18):
    """
    Align a full high-quality API transcript to local timed segments.

    The local text is used only as a noisy monotonic anchor. The output text comes
    from the API transcript, while timestamps come from timing_segments.
    """
    api_words = normalize_subtitle_text(transcript_text).split()
    if not api_words or not timing_segments:
        return []

    # Pre-compact all API words
    compact_api_words = ["".join(c.lower() for c in w if c.isalnum()) for w in api_words]

    aligned = []
    cursor = 0

    for segment_index, timing_segment in enumerate(timing_segments):
        local_text = normalize_subtitle_text(timing_segment.get("text"))
        # Pre-compact local_text
        compact_local = "".join(c.lower() for c in local_text if c.isalnum())
        
        local_word_count = max(1, len(local_text.split()))
        remaining_segments = max(1, len(timing_segments) - segment_index)
        remaining_words = max(0, len(api_words) - cursor)

        if segment_index == len(timing_segments) - 1:
            best_start = cursor
            best_end = len(api_words)
        else:
            expected_len = max(1, round(remaining_words / remaining_segments))
            min_len = max(1, min(local_word_count, expected_len) - 3)
            max_len = max(local_word_count, expected_len) + 5
            start_min = max(0, cursor - 3)
            start_max = min(len(api_words), cursor + search_window)
            best_score = -1.0
            best_raw_score = -1.0
            best_start = cursor
            best_end = min(len(api_words), cursor + expected_len)

            for start in range(start_min, start_max + 1):
                for length in range(min_len, max_len + 1):
                    end = min(len(api_words), start + length)
                    if end <= start:
                        continue
                    
                    # Construct the compacted candidate directly from pre-compacted words
                    compact_candidate = "".join(compact_api_words[start:end])
                    if not compact_local or not compact_candidate:
                        raw_score = 0.0
                    else:
                        # Quick length-based pruning: if the length ratio is extremely off, skip SequenceMatcher
                        len_local = len(compact_local)
                        len_cand = len(compact_candidate)
                        if len_cand == 0 or len_local == 0:
                            raw_score = 0.0
                        elif len_local / len_cand < 0.35 or len_cand / len_local < 0.35:
                            raw_score = 0.0
                        else:
                            raw_score = SequenceMatcher(None, compact_local, compact_candidate).ratio()

                    score = raw_score
                    # Prefer monotonic candidates near the current cursor when scores tie.
                    score -= abs(start - cursor) * 0.015
                    if score > best_score:
                        best_score = score
                        best_raw_score = raw_score
                        best_start = start
                        best_end = end

            if best_raw_score < 0.45:
                best_start = cursor
                best_end = min(len(api_words), cursor + expected_len)

        output_start = max(best_start, cursor)
        if best_end <= output_start:
            best_end = min(len(api_words), cursor + max(1, local_word_count))
            output_start = cursor

        text = normalize_subtitle_text(" ".join(api_words[output_start:best_end]))
        if text:
            aligned.append({
                "start": timing_segment["start"],
                "end": timing_segment["end"],
                "text": text,
            })
        cursor = max(best_end, cursor)

    return aligned


def collect_aligned_words(aligned_result):
    words = []
    for item in aligned_result.get("word_segments", []) or []:
        word = normalize_subtitle_text(item.get("word"))
        start = item.get("start")
        end = item.get("end")
        if word and start is not None and end is not None and float(end) > float(start):
            words.append({
                "word": word,
                "start": float(start),
                "end": float(end),
                "score": item.get("score"),
            })

    if words:
        return words

    for segment in aligned_result.get("segments", []) or []:
        for item in segment.get("words", []) or []:
            word = normalize_subtitle_text(item.get("word"))
            start = item.get("start")
            end = item.get("end")
            if word and start is not None and end is not None and float(end) > float(start):
                words.append({
                    "word": word,
                    "start": float(start),
                    "end": float(end),
                    "score": item.get("score"),
                })
    return words


def word_alignment_coverage(transcript_text, aligned_words):
    transcript_words = normalize_subtitle_text(transcript_text).split()
    if not transcript_words:
        return 0.0
    return min(1.0, len(aligned_words) / len(transcript_words))


def build_source_timing_verification_rough_segments(
    source_segments,
    duration_seconds=None,
    block_seconds=30,
    padding_seconds=8,
    max_chars=900,
):
    blocks = []
    current = []
    current_start = None
    current_end = None
    block_seconds = max(5.0, float(block_seconds or 30))
    padding_seconds = max(0.0, float(padding_seconds or 0))

    def flush():
        nonlocal current, current_start, current_end
        if not current:
            return
        start = max(0.0, float(current_start) - padding_seconds)
        end = float(current_end) + padding_seconds
        if duration_seconds is not None:
            end = min(float(duration_seconds), end)
        text = normalize_subtitle_text(" ".join(item["text"] for item in current))
        if text and end > start:
            blocks.append({"start": start, "end": end, "text": text})
        current = []
        current_start = None
        current_end = None

    for segment in source_segments or []:
        text = normalize_subtitle_text(segment.get("text", ""))
        if not text:
            continue
        start = float(segment["start"])
        end = float(segment["end"])
        if not current:
            current = [{"text": text}]
            current_start = start
            current_end = end
            continue

        candidate_text = normalize_subtitle_text(" ".join(item["text"] for item in current + [{"text": text}]))
        candidate_duration = end - float(current_start)
        if candidate_duration > block_seconds or len(candidate_text) > max_chars:
            flush()
            current = [{"text": text}]
            current_start = start
            current_end = end
        else:
            current.append({"text": text})
            current_end = end

    flush()
    return blocks


def whisperx_align_words_to_audio(audio_path, rough_segments, language_code, device="cpu"):
    whisperx = import_whisperx()
    language_code = language_code or "en"
    align_model, metadata, model_info = load_whisperx_alignment_model(
        whisperx,
        language_code,
        device,
    )
    forced_input = [
        {
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "text": normalize_subtitle_text(segment["text"]),
        }
        for segment in rough_segments or []
        if normalize_subtitle_text(segment.get("text"))
    ]
    if not forced_input:
        raise RuntimeError("WhisperX verification has no rough transcript segments to align.")

    try:
        aligned_result = whisperx.align(
            forced_input,
            align_model,
            metadata,
            audio_path,
            device,
            return_char_alignments=False,
        )
    finally:
        try:
            import gc
            del align_model
            gc.collect()
            if device != "cpu":
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception:
            pass

    aligned_words = collect_aligned_words(aligned_result)
    return aligned_words, {
        "rough_segments": len(forced_input),
        "aligned_words": len(aligned_words),
        "language": language_code,
        "alignment_model": model_info,
    }


def transcript_tokens(text):
    return [
        {
            "text": match.group(0),
            "compact": compact_text_for_alignment(match.group(0)),
        }
        for match in re.finditer(r"\S+", normalize_subtitle_text(text))
        if compact_text_for_alignment(match.group(0))
    ]


def align_words_to_canonical_tokens(transcript_text, aligned_words):
    canonical = transcript_tokens(transcript_text)
    timed_words = [
        {
            "word": normalize_subtitle_text(item.get("word", "")),
            "compact": compact_text_for_alignment(item.get("word", "")),
            "start": item.get("start"),
            "end": item.get("end"),
            "score": item.get("score"),
            "supplemental": bool(
                item.get("_supplemental_timing_evidence")
            ),
        }
        for item in aligned_words
        if item.get("start") is not None
        and item.get("end") is not None
        and compact_text_for_alignment(item.get("word", ""))
    ]
    canonical_keys = [item["compact"] for item in canonical]

    matched = 0
    native_matched = 0
    supplemental_matched = 0
    ignored_timed_indexes = set(range(len(timed_words)))

    def apply_equal_matches(timed_indexes, timing_source):
        nonlocal matched, native_matched, supplemental_matched
        if not timed_indexes:
            return
        matcher = SequenceMatcher(
            None,
            canonical_keys,
            [timed_words[index]["compact"] for index in timed_indexes],
            autojunk=False,
        )
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "equal":
                continue
            for canonical_index, subset_index in zip(
                range(i1, i2),
                range(j1, j2),
            ):
                if "start" in canonical[canonical_index]:
                    continue
                timed_index = timed_indexes[subset_index]
                timed = timed_words[timed_index]
                if timed.get("supplemental"):
                    previous_end = next(
                        (
                            float(canonical[index]["end"])
                            for index in range(
                                canonical_index - 1,
                                -1,
                                -1,
                            )
                            if canonical[index].get("end") is not None
                        ),
                        None,
                    )
                    next_start = next(
                        (
                            float(canonical[index]["start"])
                            for index in range(
                                canonical_index + 1,
                                len(canonical),
                            )
                            if canonical[index].get("start") is not None
                        ),
                        None,
                    )
                    if (
                        previous_end is not None
                        and float(timed["start"])
                        < previous_end - 0.25
                    ):
                        continue
                    if (
                        next_start is not None
                        and float(timed["end"])
                        > next_start + 0.25
                    ):
                        continue
                canonical[canonical_index]["start"] = float(timed["start"])
                canonical[canonical_index]["end"] = float(timed["end"])
                canonical[canonical_index]["score"] = timed.get("score")
                canonical[canonical_index]["timing_source"] = timing_source
                canonical[canonical_index]["timed_word_index"] = (
                    timed_index
                )
                ignored_timed_indexes.discard(timed_index)
                matched += 1
                if timed.get("supplemental"):
                    supplemental_matched += 1
                else:
                    native_matched += 1

    apply_equal_matches(
        [
            index
            for index, item in enumerate(timed_words)
            if not item.get("supplemental")
        ],
        "native_timing_anchor",
    )
    apply_equal_matches(
        [
            index
            for index, item in enumerate(timed_words)
            if item.get("supplemental")
        ],
        "supplemental_timing_anchor",
    )

    rescued = 0
    for canonical_index, token in enumerate(canonical):
        if "start" in token or not ignored_timed_indexes:
            continue

        previous_timed_index = -1
        previous_time = None
        for left_index in range(canonical_index - 1, -1, -1):
            if "timed_word_index" in canonical[left_index]:
                previous_timed_index = canonical[left_index]["timed_word_index"]
                previous_time = float(canonical[left_index]["end"])
                break

        next_timed_index = len(timed_words)
        next_time = None
        for right_index in range(canonical_index + 1, len(canonical)):
            if "timed_word_index" in canonical[right_index]:
                next_timed_index = canonical[right_index]["timed_word_index"]
                next_time = float(canonical[right_index]["start"])
                break

        candidates = [
            index
            for index in sorted(
                ignored_timed_indexes,
                key=lambda value: (
                    bool(timed_words[value].get("supplemental")),
                    value,
                ),
            )
            if timed_words[index]["compact"] == token["compact"]
            and (previous_time is None or float(timed_words[index]["start"]) >= previous_time - 0.25)
            and (next_time is None or float(timed_words[index]["end"]) <= next_time + 0.25)
        ]
        if not candidates:
            continue

        timed_index = candidates[0]
        timed = timed_words[timed_index]
        token["start"] = float(timed["start"])
        token["end"] = float(timed["end"])
        token["score"] = timed.get("score")
        token["timing_source"] = (
            "supplemental_timing_anchor_rescued_exact"
            if timed.get("supplemental")
            else "native_timing_anchor_rescued_exact"
        )
        token["timed_word_index"] = timed_index
        ignored_timed_indexes.discard(timed_index)
        matched += 1
        rescued += 1
        if timed.get("supplemental"):
            supplemental_matched += 1
        else:
            native_matched += 1

    return canonical, {
        "canonical_tokens": len(canonical),
        "timed_words": len(timed_words),
        "matched_tokens": matched,
        "native_matched_tokens": native_matched,
        "supplemental_matched_tokens": supplemental_matched,
        "direct_token_coverage": round(matched / len(canonical), 4) if canonical else 0.0,
        "rescued_exact_tokens": rescued,
        "ignored_timed_words": [
            {
                "index": index,
                "word": timed_words[index]["word"],
                "start": timed_words[index]["start"],
                "end": timed_words[index]["end"],
                "supplemental": bool(
                    timed_words[index].get("supplemental")
                ),
            }
            for index in sorted(ignored_timed_indexes)
        ],
    }


def constrain_interpolated_tokens_to_audio(
    tokens,
    interpolated_spans,
    possible_audio_intervals,
    *,
    minimum_excluded_gap_seconds=0.65,
):
    """Redistribute interpolated text over signal without crossing long silence."""
    possible_audio = [
        (float(start), float(end))
        for start, end in possible_audio_intervals or []
        if float(end) > float(start)
    ]
    adjustments = []
    if not possible_audio:
        return tokens, adjustments

    for span in interpolated_spans or []:
        first = int(span["start_index"])
        last = int(span["end_index"])
        indexes = list(range(first, last + 1))
        if not indexes:
            continue
        lower = (
            float(tokens[first - 1]["end"])
            if first > 0 and tokens[first - 1].get("end") is not None
            else float(tokens[first]["start"])
        )
        upper = (
            float(tokens[last + 1]["start"])
            if last + 1 < len(tokens)
            and tokens[last + 1].get("start") is not None
            else float(tokens[last]["end"])
        )
        if upper <= lower:
            continue
        allowed = [
            (max(lower, start), min(upper, end))
            for start, end in possible_audio
            if min(upper, end) > max(lower, start)
        ]
        allowed = [
            (start, end)
            for start, end in allowed
            if end - start >= 0.001
        ]
        allowed_seconds = sum(end - start for start, end in allowed)
        if (
            not allowed
            or (upper - lower) - allowed_seconds
            < float(minimum_excluded_gap_seconds)
        ):
            continue

        assignments = [[] for _ in allowed]
        for offset, token_index in enumerate(indexes):
            target = allowed_seconds * (offset + 0.5) / len(indexes)
            cursor = 0.0
            selected = len(allowed) - 1
            for interval_index, (start, end) in enumerate(allowed):
                interval_seconds = end - start
                if target <= cursor + interval_seconds:
                    selected = interval_index
                    break
                cursor += interval_seconds
            assignments[selected].append(token_index)

        original = [
            {
                "index": index,
                "start": round(float(tokens[index]["start"]), 3),
                "end": round(float(tokens[index]["end"]), 3),
            }
            for index in indexes
        ]
        for (start, end), assigned in zip(allowed, assignments):
            if not assigned:
                continue
            step = (end - start) / len(assigned)
            for offset, token_index in enumerate(assigned):
                token_start = start + step * offset
                token_end = min(
                    end,
                    token_start + max(0.001, step * 0.85),
                )
                tokens[token_index]["start"] = token_start
                tokens[token_index]["end"] = token_end
                tokens[token_index]["timing_source"] = (
                    "interpolated_with_independent_audio_signal"
                )
        adjustments.append({
            "start_index": first,
            "end_index": last,
            "excluded_gap_seconds": round(
                (upper - lower) - allowed_seconds,
                3,
            ),
            "allowed_intervals": [
                [round(start, 3), round(end, 3)]
                for start, end in allowed
            ],
            "original": original,
        })
    return tokens, adjustments


def interpolate_missing_token_timings(
    tokens,
    default_step=0.28,
    ignored_timed_words=None,
    possible_audio_intervals=None,
):
    if not tokens:
        return tokens, []

    interpolated = []
    timed_indexes = [index for index, token in enumerate(tokens) if "start" in token and "end" in token]
    if not timed_indexes:
        return tokens, [{"start_index": 0, "end_index": len(tokens) - 1, "reason": "no_timed_tokens"}]

    for left, right in zip([-1] + timed_indexes, timed_indexes + [len(tokens)]):
        gap_indexes = [index for index in range(left + 1, right) if index not in timed_indexes]
        if not gap_indexes:
            continue

        if left >= 0 and right < len(tokens):
            start_bound = float(tokens[left]["end"])
            end_bound = float(tokens[right]["start"])
            gap_duration = max(0.0, end_bound - start_bound)
            if gap_duration <= 0.001:
                adjacent_containers = []
                left_start = float(tokens[left]["start"])
                left_duration = max(0.0, start_bound - left_start)
                if left_duration > 0.0:
                    adjacent_containers.append({
                        "side": "left",
                        "start": left_start,
                        "end": start_bound,
                        "duration": left_duration,
                    })
                right_end = float(tokens[right]["end"])
                right_duration = max(0.0, right_end - end_bound)
                if right_duration > 0.0:
                    adjacent_containers.append({
                        "side": "right",
                        "start": end_bound,
                        "end": right_end,
                        "duration": right_duration,
                    })
                adjacent_containers.sort(
                    key=lambda item: -float(item["duration"])
                )
                required_duration = max(
                    1.0,
                    float(default_step) * (len(gap_indexes) + 1) * 0.5,
                )
                reclaimed = next(
                    (
                        item
                        for item in adjacent_containers
                        if float(item["duration"]) >= required_duration
                        and (
                            not possible_audio_intervals
                            or interval_overlap_seconds(
                                float(item["start"]),
                                float(item["end"]),
                                possible_audio_intervals,
                            )
                            / max(0.001, float(item["duration"]))
                            >= 0.80
                        )
                    ),
                    None,
                )
                if reclaimed is not None:
                    slots = len(gap_indexes) + 1
                    step = float(reclaimed["duration"]) / slots
                    if reclaimed["side"] == "left":
                        tokens[left]["end"] = min(
                            float(reclaimed["end"]),
                            float(reclaimed["start"]) + step * 0.85,
                        )
                        tokens[left]["timing_source"] = (
                            f"{tokens[left].get('timing_source', 'timed')}"
                            "_split_for_unmatched_tokens"
                        )
                        for offset, index in enumerate(gap_indexes, 1):
                            start = float(reclaimed["start"]) + step * offset
                            end = min(
                                float(reclaimed["end"]),
                                start + max(0.001, step * 0.85),
                            )
                            tokens[index]["start"] = start
                            tokens[index]["end"] = max(
                                start + 0.001,
                                end,
                            )
                    else:
                        for offset, index in enumerate(gap_indexes):
                            start = float(reclaimed["start"]) + step * offset
                            end = min(
                                float(reclaimed["end"]),
                                start + max(0.001, step * 0.85),
                            )
                            tokens[index]["start"] = start
                            tokens[index]["end"] = max(
                                start + 0.001,
                                end,
                            )
                        tokens[right]["start"] = (
                            float(reclaimed["start"])
                            + step * len(gap_indexes)
                        )
                        tokens[right]["timing_source"] = (
                            f"{tokens[right].get('timing_source', 'timed')}"
                            "_split_for_unmatched_tokens"
                        )
                    for index in gap_indexes:
                        tokens[index]["timing_source"] = (
                            "interpolated_within_overlong_adjacent_anchor"
                        )
                    interpolated.append({
                        "start_index": gap_indexes[0],
                        "end_index": gap_indexes[-1],
                        "token_count": len(gap_indexes),
                        "timing_source": (
                            "interpolated_within_overlong_adjacent_anchor"
                        ),
                        "container_side": reclaimed["side"],
                        "container_start": round(
                            float(reclaimed["start"]),
                            3,
                        ),
                        "container_end": round(
                            float(reclaimed["end"]),
                            3,
                        ),
                    })
                    continue
            ignored_containers = [
                item
                for item in ignored_timed_words or []
                if item.get("start") is not None
                and item.get("end") is not None
                and float(item["end"]) > float(item["start"])
                and float(item["start"]) >= start_bound - 0.25
                and float(item["end"]) <= end_bound + 0.25
            ]
            if ignored_containers:
                container_start = max(start_bound, min(float(item["start"]) for item in ignored_containers))
                container_end = min(end_bound, max(float(item["end"]) for item in ignored_containers))
                container_duration = max(0.0, container_end - container_start)
                if container_duration > 0.0:
                    step = container_duration / len(gap_indexes)
                    for offset, index in enumerate(gap_indexes):
                        start = container_start + step * offset
                        end = min(container_end, start + max(0.001, step * 0.85))
                        tokens[index]["start"] = max(container_start, min(start, container_end))
                        tokens[index]["end"] = max(tokens[index]["start"] + 0.001, min(end, container_end))
                        tokens[index]["timing_source"] = "interpolated_over_unmatched_timing_anchors"
                    interpolated.append({
                        "start_index": gap_indexes[0],
                        "end_index": gap_indexes[-1],
                        "token_count": len(gap_indexes),
                        "timing_source": tokens[gap_indexes[0]].get("timing_source"),
                        "container_start": round(container_start, 3),
                        "container_end": round(container_end, 3),
                    })
                    continue
            if gap_duration <= 0.0:
                step = 0.001
                block_start = start_bound
            else:
                desired_duration = default_step * len(gap_indexes)
                if gap_duration > desired_duration * 2:
                    block_start = start_bound + (gap_duration - desired_duration) / 2
                    step = default_step
                else:
                    block_start = start_bound
                    step = gap_duration / len(gap_indexes)
            for offset, index in enumerate(gap_indexes):
                start = block_start + step * offset
                if gap_duration > 0.0:
                    end = min(end_bound, start + max(0.001, step * 0.85))
                else:
                    end = start + step
                tokens[index]["start"] = max(start_bound, min(start, end_bound))
                tokens[index]["end"] = max(tokens[index]["start"] + 0.001, min(end, end_bound))
                tokens[index]["timing_source"] = "interpolated_between_anchors"
        elif left >= 0:
            start_bound = float(tokens[left]["end"])
            ignored_containers = [
                item
                for item in ignored_timed_words or []
                if item.get("start") is not None
                and item.get("end") is not None
                and float(item["end"]) > start_bound
                and float(item["end"]) > float(item["start"])
            ]
            if ignored_containers:
                container_start = max(
                    start_bound,
                    min(float(item["start"]) for item in ignored_containers),
                )
                container_end = max(
                    float(item["end"]) for item in ignored_containers
                )
                step = (
                    (container_end - container_start) / len(gap_indexes)
                )
                for offset, index in enumerate(gap_indexes):
                    start = container_start + step * offset
                    end = min(
                        container_end,
                        start + max(0.001, step * 0.85),
                    )
                    tokens[index]["start"] = start
                    tokens[index]["end"] = max(start + 0.001, end)
                    tokens[index]["timing_source"] = (
                        "interpolated_over_trailing_unmatched_timing_anchors"
                    )
            else:
                cursor = start_bound
                for index in gap_indexes:
                    tokens[index]["start"] = cursor
                    tokens[index]["end"] = cursor + default_step
                    tokens[index]["timing_source"] = (
                        "interpolated_after_last_anchor"
                    )
                    cursor = tokens[index]["end"]
        else:
            end_bound = float(tokens[right]["start"])
            ignored_containers = [
                item
                for item in ignored_timed_words or []
                if item.get("start") is not None
                and item.get("end") is not None
                and float(item["start"]) < end_bound
                and float(item["end"]) > float(item["start"])
            ]
            if ignored_containers:
                container_start = max(
                    0.0,
                    min(float(item["start"]) for item in ignored_containers),
                )
                container_end = min(
                    end_bound,
                    max(float(item["end"]) for item in ignored_containers),
                )
                step = (
                    (container_end - container_start) / len(gap_indexes)
                )
                for offset, index in enumerate(gap_indexes):
                    start = container_start + step * offset
                    end = min(
                        container_end,
                        start + max(0.001, step * 0.85),
                    )
                    tokens[index]["start"] = start
                    tokens[index]["end"] = max(start + 0.001, end)
                    tokens[index]["timing_source"] = (
                        "interpolated_over_leading_unmatched_timing_anchors"
                    )
            else:
                cursor = max(
                    0.0,
                    end_bound - default_step * len(gap_indexes),
                )
                for index in gap_indexes:
                    tokens[index]["start"] = cursor
                    tokens[index]["end"] = cursor + default_step
                    tokens[index]["timing_source"] = (
                        "interpolated_before_first_anchor"
                    )
                    cursor = tokens[index]["end"]

        interpolated.append({
            "start_index": gap_indexes[0],
            "end_index": gap_indexes[-1],
            "token_count": len(gap_indexes),
            "timing_source": tokens[gap_indexes[0]].get("timing_source"),
        })

    tokens, signal_adjustments = constrain_interpolated_tokens_to_audio(
        tokens,
        interpolated,
        possible_audio_intervals,
    )
    return tokens, interpolated, signal_adjustments


def enforce_monotonic_token_timings(tokens, min_duration=0.001):
    cursor = None
    adjustments = []
    for index, token in enumerate(tokens or []):
        if "start" not in token or "end" not in token:
            continue
        start = float(token["start"])
        end = float(token["end"])
        duration = max(min_duration, end - start)
        if cursor is not None and start < cursor:
            original_start = start
            original_end = end
            start = cursor
            end = start + duration
            token["start"] = start
            token["end"] = end
            token["timing_source"] = f"{token.get('timing_source', 'timed')}_monotonic_adjusted"
            adjustments.append({
                "index": index,
                "original_start": round(original_start, 3),
                "original_end": round(original_end, 3),
                "start": round(start, 3),
                "end": round(end, 3),
            })
        cursor = max(cursor if cursor is not None else end, end)
    return tokens, adjustments


def group_canonical_tokens_for_subtitles(
    tokens,
    max_chars=84,
    max_duration=5.0,
    max_gap=0.65,
):
    segments = []
    current = []

    def flush():
        nonlocal current
        if not current:
            return
        start = float(current[0]["start"])
        end = float(current[-1]["end"])
        text = normalize_subtitle_text(" ".join(item["text"] for item in current))
        if text and end > start:
            segment = {"start": start, "end": end, "text": text}
            speakers = {
                item.get("speaker")
                for item in current
                if item.get("speaker") is not None
            }
            languages = {
                item.get("language")
                for item in current
                if item.get("language")
            }
            if len(speakers) == 1:
                segment["speaker"] = next(iter(speakers))
            if len(languages) == 1:
                segment["language"] = next(iter(languages))
            elif len(languages) > 1:
                segment["language"] = "mixed"
            segments.append(segment)
        current = []

    for token in tokens:
        if "start" not in token or "end" not in token:
            continue
        candidate_text = normalize_subtitle_text(" ".join([item["text"] for item in current] + [token["text"]]))
        duration = 0.0 if not current else float(token["end"]) - float(current[0]["start"])
        gap = 0.0 if not current else float(token["start"]) - float(current[-1]["end"])
        speaker_changed = bool(
            current
            and token.get("speaker") is not None
            and current[-1].get("speaker") is not None
            and token.get("speaker") != current[-1].get("speaker")
        )
        language_changed = bool(
            current
            and token.get("language")
            and current[-1].get("language")
            and token.get("language") != current[-1].get("language")
        )
        should_flush = bool(current) and (
            len(candidate_text) > max_chars
            or duration > max_duration
            or gap > max_gap
            or speaker_changed
            or language_changed
        )
        if should_flush:
            flush()
        current.append(token)
        if token["text"][-1:] in {".", "?", "!", "؟", "。"} and len(" ".join(item["text"] for item in current)) >= 24:
            flush()

    flush()
    return segments


def apply_canonical_metadata_to_tokens(tokens, metadata_segments):
    """Transfer provider speaker/language metadata by canonical token order."""
    source_tokens = []
    for segment in metadata_segments or []:
        speaker = segment.get("speaker")
        if speaker is not None:
            speaker = str(speaker).strip()
            if speaker.casefold() in {
                "",
                "none",
                "null",
                "unknown",
                "n/a",
                "na",
                "not specified",
                "unspecified",
            }:
                speaker = None
        language = segment.get("language")
        for token in transcript_tokens(segment.get("text", "")):
            source_tokens.append({
                "compact": token["compact"],
                "speaker": speaker,
                "language": language,
            })
    if not source_tokens or not tokens:
        return {
            "source_metadata_tokens": len(source_tokens),
            "matched_metadata_tokens": 0,
        }
    canonical_keys = [item.get("compact") for item in tokens]
    source_keys = [item["compact"] for item in source_tokens]
    matcher = SequenceMatcher(None, canonical_keys, source_keys, autojunk=False)
    matched = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            continue
        for canonical_index, source_index in zip(range(i1, i2), range(j1, j2)):
            source = source_tokens[source_index]
            if source.get("speaker") is not None:
                tokens[canonical_index]["speaker"] = source["speaker"]
            if source.get("language"):
                tokens[canonical_index]["language"] = source["language"]
            matched += 1
    return {
        "source_metadata_tokens": len(source_tokens),
        "matched_metadata_tokens": matched,
        "metadata_token_coverage": round(matched / len(tokens), 4),
    }


def reconcile_transcript_with_aligned_words(
    transcript_text,
    aligned_words,
    min_direct_coverage=0.80,
    canonical_metadata_segments=None,
    possible_audio_intervals=None,
):
    tokens, report = align_words_to_canonical_tokens(transcript_text, aligned_words)
    if report["direct_token_coverage"] < min_direct_coverage:
        raise RuntimeError(
            "Forced alignment canonical token coverage is too low "
            f"({report['direct_token_coverage']:.2%})."
        )
    tokens, interpolated_spans, signal_adjustments = interpolate_missing_token_timings(
        tokens,
        ignored_timed_words=report.get("ignored_timed_words"),
        possible_audio_intervals=possible_audio_intervals,
    )
    tokens, monotonic_adjustments = enforce_monotonic_token_timings(tokens)
    metadata_report = apply_canonical_metadata_to_tokens(
        tokens,
        canonical_metadata_segments,
    )
    segments = group_canonical_tokens_for_subtitles(tokens)
    reconstructed = normalize_subtitle_text(" ".join(token["text"] for token in tokens))
    report.update({
        "engine": "canonical_token_reconciliation",
        "interpolated_spans": interpolated_spans,
        "signal_constrained_interpolation": signal_adjustments,
        "monotonic_adjustments": monotonic_adjustments,
        "canonical_metadata": metadata_report,
        "reconstructed_similarity": round(similarity_score(transcript_text, reconstructed), 4),
    })
    return segments, report


def group_aligned_words_for_subtitles(
    aligned_words,
    max_chars=84,
    max_duration=5.0,
    max_gap=0.65,
):
    segments = []
    current_words = []
    current_start = None
    current_end = None

    def flush():
        nonlocal current_words, current_start, current_end
        text = normalize_subtitle_text(" ".join(current_words))
        if text and current_start is not None and current_end is not None and current_end > current_start:
            segments.append({
                "start": float(current_start),
                "end": float(current_end),
                "text": text,
            })
        current_words = []
        current_start = None
        current_end = None

    for item in aligned_words:
        word = normalize_subtitle_text(item.get("word"))
        start = item.get("start")
        end = item.get("end")
        if not word or start is None or end is None:
            continue

        start = float(start)
        end = float(end)
        gap = 0.0 if current_end is None else start - current_end
        candidate = normalize_subtitle_text(" ".join(current_words + [word]))
        duration = 0.0 if current_start is None else end - current_start
        should_flush = bool(current_words) and (
            len(candidate) > max_chars
            or duration > max_duration
            or gap > max_gap
        )
        if should_flush:
            flush()

        if current_start is None:
            current_start = start
        current_words.append(word)
        current_end = end

        if word[-1:] in {".", "?", "!", "؟", "。"} and len(" ".join(current_words)) >= 24:
            flush()

    flush()
    return segments


def segments_with_token_spans(segments):
    spans = []
    cursor = 0
    for segment in segments or []:
        words = normalize_subtitle_text(segment.get("text", "")).split()
        if not words:
            continue
        start_token = cursor
        end_token = cursor + len(words)
        spans.append({
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "text": normalize_subtitle_text(segment.get("text", "")),
            "token_start": start_token,
            "token_end": end_token,
        })
        cursor = end_token
    return spans


def timing_drift_report(candidate_segments, reference_segments, drift_threshold=2.25):
    candidate_spans = segments_with_token_spans(candidate_segments)
    reference_spans = segments_with_token_spans(reference_segments)
    if not candidate_spans or not reference_spans:
        return {
            "checked_segments": 0,
            "max_start_drift": 0.0,
            "bad_segment_count": 0,
            "accept": True,
        }

    drifts = []
    examples = []
    for candidate in candidate_spans:
        best_reference = None
        best_overlap = 0
        for reference in reference_spans:
            overlap = min(candidate["token_end"], reference["token_end"]) - max(candidate["token_start"], reference["token_start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_reference = reference
        if not best_reference:
            continue
        start_drift = abs(candidate["start"] - best_reference["start"])
        end_drift = abs(candidate["end"] - best_reference["end"])
        drift = max(start_drift, end_drift)
        drifts.append(drift)
        if drift > drift_threshold and len(examples) < 5:
            examples.append({
                "text": candidate["text"][:80],
                "candidate_start": round(candidate["start"], 3),
                "reference_start": round(best_reference["start"], 3),
                "start_drift": round(start_drift, 3),
                "end_drift": round(end_drift, 3),
            })

    if not drifts:
        return {
            "checked_segments": 0,
            "max_start_drift": 0.0,
            "bad_segment_count": 0,
            "accept": True,
        }

    bad_count = sum(1 for drift in drifts if drift > drift_threshold)
    max_drift = max(drifts)
    sorted_drifts = sorted(drifts)
    p90 = sorted_drifts[min(len(sorted_drifts) - 1, int(len(sorted_drifts) * 0.9))]
    return {
        "checked_segments": len(drifts),
        "max_start_drift": round(max_drift, 3),
        "p90_drift": round(p90, 3),
        "bad_segment_count": bad_count,
        "drift_threshold": drift_threshold,
        "accept": not (max_drift > drift_threshold and bad_count >= 2),
        "examples": examples,
    }


def sanitize_timing_anchor_segments(timing_segments, min_duration=0.05):
    cleaned = []
    for segment in timing_segments or []:
        text = normalize_subtitle_text(segment.get("text", ""))
        if not text:
            continue
        try:
            start = float(segment["start"])
            end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end - start < min_duration:
            continue
        cleaned.append({"start": start, "end": end, "text": text})

    cleaned.sort(key=lambda item: (item["start"], item["end"]))
    sanitized = []
    skipped_overlap = 0
    trimmed_overlap = 0
    for segment in cleaned:
        if sanitized and segment["start"] < sanitized[-1]["end"]:
            if segment["end"] <= sanitized[-1]["end"] + 0.001:
                skipped_overlap += 1
                continue
            segment = dict(segment)
            segment["start"] = sanitized[-1]["end"]
            if segment["end"] - segment["start"] < min_duration:
                skipped_overlap += 1
                continue
            trimmed_overlap += 1
        sanitized.append(segment)

    return sanitized, {
        "input_segments": len(timing_segments or []),
        "usable_segments": len(cleaned),
        "sanitized_segments": len(sanitized),
        "overlap_segments_skipped": skipped_overlap,
        "overlap_segments_trimmed": trimmed_overlap,
    }


def timing_segments_to_word_anchors(timing_segments):
    native_input = [
        segment
        for segment in timing_segments or []
        if not segment.get("_supplemental_timing_evidence")
    ]
    supplemental_input = [
        segment
        for segment in timing_segments or []
        if segment.get("_supplemental_timing_evidence")
    ]
    segments, report = sanitize_timing_anchor_segments(native_input)
    supplemental_segments, supplemental_report = (
        sanitize_timing_anchor_segments(supplemental_input)
    )
    words = []
    for segment, supplemental in [
        *((segment, False) for segment in segments),
        *((segment, True) for segment in supplemental_segments),
    ]:
        segment_words = normalize_subtitle_text(segment["text"]).split()
        if not segment_words:
            continue
        duration = max(0.05, segment["end"] - segment["start"])
        total_weight = sum(max(1, len(compact_text_for_alignment(word))) for word in segment_words)
        cursor = segment["start"]
        for index, word in enumerate(segment_words):
            if index == len(segment_words) - 1:
                end = segment["end"]
            else:
                weight = max(1, len(compact_text_for_alignment(word)))
                end = min(segment["end"], cursor + duration * weight / max(1, total_weight))
            words.append({
                "word": word,
                "start": cursor,
                "end": max(cursor + 0.01, end),
                "score": None,
            })
            if supplemental:
                words[-1]["_supplemental_timing_evidence"] = True
            cursor = end
    words.sort(
        key=lambda item: (
            float(item["start"]),
            float(item["end"]),
            bool(item.get("_supplemental_timing_evidence")),
        )
    )
    report.update({
        "input_segments": len(timing_segments or []),
        "native_input_segments": len(native_input),
        "supplemental_input_segments": len(supplemental_input),
        "supplemental_sanitized_segments": len(
            supplemental_segments
        ),
        "supplemental_overlap_segments_skipped": (
            supplemental_report.get("overlap_segments_skipped", 0)
        ),
        "supplemental_overlap_segments_trimmed": (
            supplemental_report.get("overlap_segments_trimmed", 0)
        ),
    })
    report["word_anchors"] = len(words)
    return words, report


def subtitle_text_integrity_report(transcript_text, segments, min_similarity=0.995, max_word_delta_ratio=0.005):
    transcript = normalize_subtitle_text(transcript_text)
    rendered = normalize_subtitle_text(" ".join(segment.get("text", "") for segment in segments or []))
    transcript_words = transcript.split()
    rendered_words = rendered.split()
    word_delta = abs(len(transcript_words) - len(rendered_words))
    allowed_word_delta = max(1, int(len(transcript_words) * max_word_delta_ratio))
    compact_match = compact_text_for_alignment(transcript) == compact_text_for_alignment(rendered)
    similarity = similarity_score(transcript, rendered)
    accept = bool(compact_match or (similarity >= min_similarity and word_delta <= allowed_word_delta))
    return {
        "accept": accept,
        "similarity": round(similarity, 4),
        "compact_match": compact_match,
        "transcript_words": len(transcript_words),
        "subtitle_words": len(rendered_words),
        "word_delta": word_delta,
        "allowed_word_delta": allowed_word_delta,
        "min_similarity": min_similarity,
    }


def subtitle_timing_integrity_report(candidate_segments, reference_segments=None, drift_threshold=2.25):
    problems = []
    previous_end = None
    for index, segment in enumerate(candidate_segments or []):
        try:
            start = max(0, int(float(segment["start"]) * 1000 + 0.5)) / 1000
            end = max(0, int(float(segment["end"]) * 1000 + 0.5)) / 1000
        except (KeyError, TypeError, ValueError):
            problems.append({"type": "invalid_time", "index": index})
            continue
        if end <= start:
            problems.append({"type": "non_positive_duration", "index": index, "start": start, "end": end})
        if previous_end is not None and start < previous_end:
            problems.append({
                "type": "non_monotonic",
                "index": index,
                "start": round(start, 3),
                "previous_end": round(previous_end, 3),
            })
        previous_end = max(previous_end if previous_end is not None else end, end)

    drift = timing_drift_report(candidate_segments, reference_segments, drift_threshold) if reference_segments else None
    accept = not problems and (not drift or drift.get("accept", True))
    return {
        "accept": accept,
        "monotonic": not problems,
        "problem_count": len(problems),
        "problems": problems[:10],
        "drift": drift,
    }


def source_subtitle_integrity_report(transcript_text, segments, reference_segments=None):
    text_report = subtitle_text_integrity_report(transcript_text, segments)
    timing_report = subtitle_timing_integrity_report(segments, reference_segments=reference_segments)
    return {
        "accept": bool(text_report["accept"] and timing_report["accept"]),
        "text": text_report,
        "timing": timing_report,
    }


def bridge_prompted_segment_speech_gaps(
    segments,
    canonical_metadata_segments,
    speech_map,
    *,
    max_uncovered_gap_seconds=1.5,
    max_uncovered_ratio=0.03,
    max_bridge_seconds=2.5,
    min_signal_overlap_ratio=0.80,
):
    """Bridge short word-anchor holes only inside confirmed active speech.

    A broad prompted segment can contain speech for which an independent ASR
    returned no exact word boundary.  In that case adjacent canonical cues may
    meet inside the active-speech interval.  Confirmed silence is never
    bridged, and no text is added, deleted, or reordered.
    """
    repaired = [dict(segment) for segment in segments or []]
    before = validate_longform_timed_segments(
        repaired,
        speech_map or {},
        max_uncovered_gap_seconds=max_uncovered_gap_seconds,
        max_uncovered_ratio=max_uncovered_ratio,
    )
    possible_audio = [
        (float(start), float(end))
        for start, end in (
            (speech_map or {}).get("possible_audio_intervals") or []
        )
    ]
    confirmed_silence = [
        (float(start), float(end))
        for start, end in (
            (speech_map or {}).get("confirmed_silence_intervals") or []
        )
    ]
    canonical_containers = [
        {
            "index": index,
            "start": float(segment["start"]),
            "end": float(segment["end"]),
        }
        for index, segment in enumerate(
            canonical_metadata_segments or []
        )
        if segment.get("start") is not None
        and segment.get("end") is not None
        and float(segment["end"]) > float(segment["start"])
    ]
    adjustments = []
    skipped = []
    for gap in before.get("largest_uncovered_intervals") or []:
        gap_start, gap_end = float(gap[0]), float(gap[1])
        gap_seconds = gap_end - gap_start
        if gap_seconds <= float(max_uncovered_gap_seconds) + 0.0005:
            continue
        if gap_seconds > float(max_bridge_seconds):
            skipped.append({
                "start": round(gap_start, 3),
                "end": round(gap_end, 3),
                "reason": "gap_exceeds_bridge_limit",
            })
            continue
        if interval_overlap_seconds(
            gap_start,
            gap_end,
            confirmed_silence,
        ) > 0.01:
            skipped.append({
                "start": round(gap_start, 3),
                "end": round(gap_end, 3),
                "reason": "confirmed_silence",
            })
            continue
        signal_ratio = (
            interval_overlap_seconds(
                gap_start,
                gap_end,
                possible_audio,
            )
            / max(0.001, gap_seconds)
        )
        if signal_ratio < float(min_signal_overlap_ratio):
            skipped.append({
                "start": round(gap_start, 3),
                "end": round(gap_end, 3),
                "reason": "insufficient_independent_audio_signal",
                "signal_overlap_ratio": round(signal_ratio, 4),
            })
            continue
        container = next(
            (
                item
                for item in canonical_containers
                if item["start"] <= gap_start + 0.001
                and item["end"] >= gap_end - 0.001
            ),
            None,
        )
        if container is None:
            skipped.append({
                "start": round(gap_start, 3),
                "end": round(gap_end, 3),
                "reason": "not_inside_one_canonical_prompted_segment",
            })
            continue
        left_index = next(
            (
                index
                for index in range(len(repaired) - 1, -1, -1)
                if float(repaired[index]["end"]) <= gap_start + 0.001
                and float(repaired[index]["end"])
                >= container["start"] - 0.5
            ),
            None,
        )
        right_index = next(
            (
                index
                for index in range(len(repaired))
                if float(repaired[index]["start"]) >= gap_end - 0.001
                and float(repaired[index]["start"])
                <= container["end"] + 0.5
            ),
            None,
        )
        if (
            left_index is None
            or right_index is None
            or left_index >= right_index
        ):
            skipped.append({
                "start": round(gap_start, 3),
                "end": round(gap_end, 3),
                "reason": "adjacent_canonical_cues_unavailable",
            })
            continue
        left_end = float(repaired[left_index]["end"])
        right_start = float(repaired[right_index]["start"])
        if right_start <= left_end:
            continue
        boundary = (gap_start + gap_end) / 2.0
        boundary = min(right_start, max(left_end, boundary))
        repaired[left_index]["end"] = boundary
        repaired[right_index]["start"] = boundary
        adjustments.append({
            "start": round(gap_start, 3),
            "end": round(gap_end, 3),
            "boundary": round(boundary, 3),
            "left_segment_index": left_index,
            "right_segment_index": right_index,
            "canonical_segment_index": container["index"],
            "signal_overlap_ratio": round(signal_ratio, 4),
        })
    after = validate_longform_timed_segments(
        repaired,
        speech_map or {},
        max_uncovered_gap_seconds=max_uncovered_gap_seconds,
        max_uncovered_ratio=max_uncovered_ratio,
    )
    return repaired, {
        "schema": "subgen_prompted_speech_gap_bridge_v1",
        "applied": bool(adjustments),
        "adjustments": adjustments,
        "skipped": skipped,
        "validation_before": before,
        "validation_after": after,
    }


def clip_longform_speech_map(speech_map, start, end, *, relative=False):
    """Return independent signal intervals clipped to one bounded window."""
    start = float(start)
    end = float(end)
    clipped = {
        key: value
        for key, value in dict(speech_map or {}).items()
        if key
        not in {
            "speech_intervals",
            "possible_audio_intervals",
            "confirmed_silence_intervals",
            "duration_seconds",
        }
    }
    offset = start if relative else 0.0
    for key in (
        "speech_intervals",
        "possible_audio_intervals",
        "confirmed_silence_intervals",
    ):
        clipped[key] = [
            [
                max(start, float(interval_start)) - offset,
                min(end, float(interval_end)) - offset,
            ]
            for interval_start, interval_end in (
                (speech_map or {}).get(key) or []
            )
            if min(end, float(interval_end))
            > max(start, float(interval_start))
        ]
    clipped["duration_seconds"] = (
        max(0.0, end - start)
        if relative
        else float((speech_map or {}).get("duration_seconds") or end)
    )
    return clipped


def merge_longform_alignment_recovery_segments(
    existing_segments,
    recovered_segments,
    target_gap,
    speech_map,
    *,
    min_signal_overlap_ratio=0.80,
    overlap_tolerance_seconds=0.05,
    max_uncovered_gap_seconds=1.5,
    max_uncovered_ratio=0.03,
):
    """Add independently supported recovery cues without replacing source text."""
    existing = [dict(segment) for segment in existing_segments or []]
    target_start = float(target_gap[0])
    target_end = float(target_gap[1])
    possible_audio = [
        (float(start), float(end))
        for start, end in (
            (speech_map or {}).get("possible_audio_intervals") or []
        )
    ]
    confirmed_silence = [
        (float(start), float(end))
        for start, end in (
            (speech_map or {}).get("confirmed_silence_intervals") or []
        )
    ]
    accepted = []
    rejected = []
    occupied = [
        (float(segment["start"]), float(segment["end"]))
        for segment in existing
    ]
    for index, segment in enumerate(recovered_segments or []):
        text = normalize_subtitle_text(segment.get("text", ""))
        try:
            start = float(segment["start"])
            end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            rejected.append({
                "segment_index": index,
                "reason": "invalid_timing",
            })
            continue
        if not text or end <= start:
            rejected.append({
                "segment_index": index,
                "start": round(start, 3),
                "end": round(end, 3),
                "reason": "empty_or_non_positive",
            })
            continue
        center = (start + end) / 2.0
        if not (target_start <= center <= target_end):
            rejected.append({
                "segment_index": index,
                "start": round(start, 3),
                "end": round(end, 3),
                "reason": "outside_target_gap",
            })
            continue
        duration = end - start
        signal_ratio = (
            interval_overlap_seconds(start, end, possible_audio)
            / max(0.001, duration)
        )
        if signal_ratio < float(min_signal_overlap_ratio):
            rejected.append({
                "segment_index": index,
                "start": round(start, 3),
                "end": round(end, 3),
                "reason": "insufficient_independent_audio_signal",
                "signal_overlap_ratio": round(signal_ratio, 4),
            })
            continue
        if interval_overlap_seconds(start, end, confirmed_silence) >= 1.0:
            rejected.append({
                "segment_index": index,
                "start": round(start, 3),
                "end": round(end, 3),
                "reason": "spans_confirmed_silence",
            })
            continue
        occupied_overlap = interval_overlap_seconds(start, end, occupied)
        if occupied_overlap > float(overlap_tolerance_seconds):
            rejected.append({
                "segment_index": index,
                "start": round(start, 3),
                "end": round(end, 3),
                "reason": "overlaps_existing_canonical_cue",
                "overlap_seconds": round(occupied_overlap, 3),
            })
            continue
        candidate = {
            **dict(segment),
            "start": start,
            "end": end,
            "text": text,
            "_canonical_coverage_recovery": True,
        }
        accepted.append(candidate)
        occupied.append((start, end))

    before = validate_longform_timed_segments(
        existing,
        speech_map or {},
        max_uncovered_gap_seconds=max_uncovered_gap_seconds,
        max_uncovered_ratio=max_uncovered_ratio,
    )
    combined = sorted(
        [*existing, *accepted],
        key=lambda item: (
            float(item["start"]),
            float(item["end"]),
        ),
    )
    after = validate_longform_timed_segments(
        combined,
        speech_map or {},
        max_uncovered_gap_seconds=max_uncovered_gap_seconds,
        max_uncovered_ratio=max_uncovered_ratio,
    )
    improved = bool(
        accepted
        and float(after.get("uncovered_speech_seconds") or 0.0)
        < float(before.get("uncovered_speech_seconds") or 0.0) - 0.01
        and not after.get("hallucination_candidates")
        and not after.get("confirmed_silence_spans")
    )
    if not improved:
        combined = existing
        accepted = []
        after = before
    return combined, {
        "schema": "subgen_longform_alignment_recovery_merge_v1",
        "changed": improved,
        "target_gap": [
            round(target_start, 3),
            round(target_end, 3),
        ],
        "accepted_count": len(accepted),
        "accepted": [
            {
                "start": round(float(segment["start"]), 3),
                "end": round(float(segment["end"]), 3),
                "text_sha256": hashlib.sha256(
                    str(segment.get("text") or "").encode("utf-8")
                ).hexdigest(),
            }
            for segment in accepted
        ],
        "rejected_count": len(rejected),
        "rejected": rejected[:100],
        "rejected_truncated": max(0, len(rejected) - 100),
        "validation_before": before,
        "validation_after": after,
    }


def enforce_source_subtitle_integrity(transcript_text, segments, reference_segments=None, label="source subtitles"):
    report = source_subtitle_integrity_report(
        transcript_text,
        segments,
        reference_segments=reference_segments,
    )
    if not report["accept"]:
        raise RuntimeError(
            f"{label} failed integrity check "
            f"(text similarity={report['text']['similarity']}, "
            f"compact_match={report['text']['compact_match']}, "
            f"transcript_words={report['text']['transcript_words']}, "
            f"subtitle_words={report['text']['subtitle_words']}, "
            f"monotonic={report['timing']['monotonic']}, "
            f"timing_problem_count={report['timing']['problem_count']})."
        )
    return report


def enforce_subtitle_timing_integrity(segments, label="subtitles"):
    report = subtitle_timing_integrity_report(segments)
    if not report["accept"]:
        raise RuntimeError(
            f"{label} failed timing integrity check "
            f"(monotonic={report['monotonic']}, "
            f"timing_problem_count={report['problem_count']})."
        )
    return report


def merge_time_intervals(intervals):
    cleaned = sorted(
        (
            (float(start), float(end))
            for start, end in intervals
            if end is not None and start is not None and float(end) > float(start)
        ),
        key=lambda item: (item[0], item[1]),
    )
    merged = []
    for start, end in cleaned:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def uncovered_intervals(reference_intervals, coverage_intervals):
    missing = []
    coverage_intervals = merge_time_intervals(coverage_intervals)
    for ref_start, ref_end in merge_time_intervals(reference_intervals):
        cursor = ref_start
        for cov_start, cov_end in coverage_intervals:
            if cov_end <= cursor:
                continue
            if cov_start >= ref_end:
                break
            if cov_start > cursor:
                missing.append((cursor, min(cov_start, ref_end)))
            cursor = max(cursor, cov_end)
            if cursor >= ref_end:
                break
        if cursor < ref_end:
            missing.append((cursor, ref_end))
    return merge_time_intervals(missing)


def source_speech_coverage_report(
    source_segments,
    reference_speech_segments,
    max_uncovered_seconds=8.0,
    max_uncovered_gap_seconds=None,
    max_uncovered_ratio=0.08,
    padding_seconds=0.5,
):
    reference_intervals = [
        (segment.get("start"), segment.get("end"))
        for segment in reference_speech_segments or []
        if normalize_subtitle_text(segment.get("text", ""))
        and segment.get("start") is not None
        and segment.get("end") is not None
        and float(segment["end"]) > float(segment["start"])
    ]
    source_intervals = [
        (
            max(0.0, float(segment["start"]) - float(padding_seconds)),
            float(segment["end"]) + float(padding_seconds),
        )
        for segment in source_segments or []
        if normalize_subtitle_text(segment.get("text", ""))
        and segment.get("start") is not None
        and segment.get("end") is not None
        and float(segment["end"]) > float(segment["start"])
    ]
    reference_intervals = merge_time_intervals(reference_intervals)
    source_intervals = merge_time_intervals(source_intervals)
    missing = uncovered_intervals(reference_intervals, source_intervals)
    reference_duration = sum(end - start for start, end in reference_intervals)
    uncovered_duration = sum(end - start for start, end in missing)
    uncovered_ratio = uncovered_duration / reference_duration if reference_duration > 0 else 0.0
    max_gap = max((end - start for start, end in missing), default=0.0)
    if max_uncovered_gap_seconds is None:
        max_uncovered_gap_seconds = max_uncovered_seconds
    accept = bool(
        reference_duration <= 0
        or (
            max_gap <= float(max_uncovered_gap_seconds)
            and uncovered_ratio <= float(max_uncovered_ratio)
        )
    )
    return {
        "accept": accept,
        "engine": "reference_anchor_window_coverage",
        "reference_kind": "transcription_segments_not_independent_vad",
        "reference_speech_seconds": round(reference_duration, 3),
        "subtitle_covered_speech_seconds": round(max(0.0, reference_duration - uncovered_duration), 3),
        "uncovered_speech_seconds": round(uncovered_duration, 3),
        "uncovered_speech_ratio": round(uncovered_ratio, 4),
        "max_uncovered_gap_seconds": round(max_gap, 3),
        "max_uncovered_seconds": float(max_uncovered_seconds),
        "max_allowed_uncovered_gap_seconds": float(max_uncovered_gap_seconds),
        "max_uncovered_ratio": float(max_uncovered_ratio),
        "padding_seconds": float(padding_seconds),
        "reference_interval_count": len(reference_intervals),
        "source_interval_count": len(source_intervals),
        "examples": [
            {"start": round(start, 3), "end": round(end, 3), "duration": round(end - start, 3)}
            for start, end in missing[:8]
        ],
    }


def enforce_source_speech_coverage(report, policy="stop", label="source subtitles"):
    policy = (policy or "stop").lower()
    if policy in {"off", "none", "disabled"} or report.get("accept", False):
        return report
    message = (
        f"{label} failed speech coverage verification "
        f"(uncovered_speech_seconds={report.get('uncovered_speech_seconds')}, "
        f"max_uncovered_gap_seconds={report.get('max_uncovered_gap_seconds')}, "
        f"uncovered_speech_ratio={report.get('uncovered_speech_ratio')}, "
        f"uncovered_intervals={report.get('examples')})."
    )
    if policy == "warn":
        print(f"Warning: {message}")
        return report
    raise RuntimeError(message)


def source_timing_anchor_report(
    transcript_text,
    source_segments,
    timing_segments,
    max_uncovered_seconds=8.0,
    max_uncovered_gap_seconds=8.0,
    max_uncovered_ratio=0.08,
    padding_seconds=0.5,
):
    integrity = source_subtitle_integrity_report(transcript_text, source_segments)
    coverage = source_speech_coverage_report(
        source_segments,
        timing_segments,
        max_uncovered_seconds=max_uncovered_seconds,
        max_uncovered_gap_seconds=max_uncovered_gap_seconds,
        max_uncovered_ratio=max_uncovered_ratio,
        padding_seconds=padding_seconds,
    )
    accept = bool(integrity.get("accept") and coverage.get("accept"))
    return {
        "accept": accept,
        "engine": "source_timing_anchor_coverage",
        "verification_source": "transcription_timing_anchor_segments",
        "token_coverage": (integrity.get("text") or {}).get("similarity"),
        "max_abs_start_drift": 0.0 if accept else None,
        "bad_segment_count": 0 if accept else None,
        "checked_segments": len(source_segments or []),
        "segment_count": len(source_segments or []),
        "source_integrity": integrity,
        "source_speech_coverage": coverage,
    }


def source_timing_verification_strategy(aligned_words, timing_anchor_segments):
    """Choose the strongest available verifier without rerunning an already-failed aligner."""
    if aligned_words:
        return "aligned_words"
    if timing_anchor_segments:
        return "timing_anchors"
    return "whisperx"


def source_srt_audio_timing_report(
    transcript_text,
    source_segments,
    aligned_words,
    min_token_coverage=0.75,
    min_checked_segment_ratio=0.75,
    max_start_drift_seconds=0.85,
    max_early_end_seconds=0.30,
    max_late_end_seconds=2.50,
    max_bad_segment_ratio=0.05,
    max_bad_segments=2,
):
    tokens, token_report = align_words_to_canonical_tokens(transcript_text, aligned_words)
    spans = segments_with_token_spans(source_segments)
    checked = 0
    skipped = 0
    bad = []
    severe_start_drift_count = 0
    examples = []
    start_drifts = []
    early_end_drifts = []
    late_end_drifts = []

    for cue_index, span in enumerate(spans, start=1):
        span_tokens = tokens[span["token_start"]:span["token_end"]]
        timed_tokens = [
            token
            for token in span_tokens
            if token.get("start") is not None and token.get("end") is not None
        ]
        if not timed_tokens:
            skipped += 1
            continue

        checked += 1
        first_word_start = float(timed_tokens[0]["start"])
        last_word_end = float(timed_tokens[-1]["end"])
        cue_start = float(span["start"])
        cue_end = float(span["end"])
        start_drift = cue_start - first_word_start
        early_end = max(0.0, last_word_end - cue_end)
        late_end = max(0.0, cue_end - last_word_end)
        start_drifts.append(abs(start_drift))
        early_end_drifts.append(early_end)
        late_end_drifts.append(late_end)

        reasons = []
        if abs(start_drift) > max_start_drift_seconds:
            reasons.append("start_drift")
        if early_end > max_early_end_seconds:
            reasons.append("ends_before_audio_word")
        if late_end > max_late_end_seconds:
            reasons.append("ends_too_late_after_audio_word")

        if reasons:
            if abs(start_drift) > max_start_drift_seconds * 2:
                severe_start_drift_count += 1
            item = {
                "index": cue_index,
                "text": span["text"][:90],
                "cue_start": round(cue_start, 3),
                "audio_word_start": round(first_word_start, 3),
                "start_drift": round(start_drift, 3),
                "cue_end": round(cue_end, 3),
                "audio_word_end": round(last_word_end, 3),
                "early_end": round(early_end, 3),
                "late_end": round(late_end, 3),
                "reasons": reasons,
            }
            bad.append(item)
            if len(examples) < 8:
                examples.append(item)

    checked_ratio = checked / len(spans) if spans else 0.0
    token_coverage = float(token_report.get("direct_token_coverage") or 0.0)
    allowed_bad = max(int(max_bad_segments), int(checked * float(max_bad_segment_ratio)))
    accept = bool(
        spans
        and token_coverage >= float(min_token_coverage)
        and checked_ratio >= float(min_checked_segment_ratio)
        and len(bad) <= allowed_bad
        and severe_start_drift_count == 0
    )

    return {
        "accept": accept,
        "engine": "source_srt_audio_word_timing_verifier",
        "token_coverage": round(token_coverage, 4),
        "min_token_coverage": float(min_token_coverage),
        "segment_count": len(spans),
        "checked_segments": checked,
        "checked_segment_ratio": round(checked_ratio, 4),
        "min_checked_segment_ratio": float(min_checked_segment_ratio),
        "skipped_segments_without_word_timing": skipped,
        "bad_segment_count": len(bad),
        "severe_start_drift_count": severe_start_drift_count,
        "allowed_bad_segments": allowed_bad,
        "max_abs_start_drift": round(max(start_drifts), 3) if start_drifts else 0.0,
        "max_early_end": round(max(early_end_drifts), 3) if early_end_drifts else 0.0,
        "max_late_end": round(max(late_end_drifts), 3) if late_end_drifts else 0.0,
        "thresholds": {
            "max_start_drift_seconds": float(max_start_drift_seconds),
            "max_early_end_seconds": float(max_early_end_seconds),
            "max_late_end_seconds": float(max_late_end_seconds),
            "max_bad_segment_ratio": float(max_bad_segment_ratio),
            "max_bad_segments": int(max_bad_segments),
        },
        "word_alignment": token_report,
        "examples": examples,
    }


def enforce_source_srt_audio_timing(report, policy="stop", label="source subtitles"):
    policy = (policy or "stop").lower()
    if policy in {"off", "none", "disabled"}:
        return report
    if report.get("accept", False):
        return report

    if report.get("engine") == "source_timing_anchor_coverage":
        coverage = report.get("source_speech_coverage") or {}
        integrity = report.get("source_integrity") or {}
        message = (
            f"{label} failed API timing-anchor verification "
            f"(text_integrity={integrity.get('accept')}, "
            f"uncovered_speech_seconds={coverage.get('uncovered_speech_seconds')}, "
            f"max_uncovered_gap_seconds={coverage.get('max_uncovered_gap_seconds')}, "
            f"uncovered_speech_ratio={coverage.get('uncovered_speech_ratio')})."
        )
    else:
        message = (
            f"{label} failed audio-grounded timing verification "
            f"(token_coverage={report.get('token_coverage')}, "
            f"checked_segments={report.get('checked_segments')}/{report.get('segment_count')}, "
            f"bad_segment_count={report.get('bad_segment_count')}, "
            f"max_abs_start_drift={report.get('max_abs_start_drift')}s)."
        )
    if policy == "warn":
        print(f"Warning: {message}")
        return report
    raise RuntimeError(message)


def source_timing_candidate_sort_key(report):
    if not report:
        return (1, 1_000_000, 1_000_000, 1_000_000.0, 0.0)
    return (
        0 if report.get("accept") else 1,
        int(report.get("severe_start_drift_count") or 0),
        int(report.get("bad_segment_count") or 0),
        float(report.get("max_abs_start_drift") or 0.0),
        -float(report.get("token_coverage") or 0.0),
    )


def choose_audio_verified_timing_candidate(
    transcript_text,
    whisperx_segments,
    anchor_segments,
    audio_aligned_words,
    reference_speech_segments=None,
):
    whisperx_report = source_srt_audio_timing_report(
        transcript_text,
        whisperx_segments,
        audio_aligned_words,
    )
    anchor_report = source_srt_audio_timing_report(
        transcript_text,
        anchor_segments,
        audio_aligned_words,
    )
    if reference_speech_segments:
        whisperx_report["source_speech_coverage"] = source_speech_coverage_report(
            whisperx_segments,
            reference_speech_segments,
        )
        anchor_report["source_speech_coverage"] = source_speech_coverage_report(
            anchor_segments,
            reference_speech_segments,
        )
        def coverage_sort_key(report):
            coverage = report.get("source_speech_coverage") or {}
            return (
                0 if coverage.get("accept") else 1,
                float(coverage.get("max_uncovered_gap_seconds") or 0.0),
                float(coverage.get("uncovered_speech_ratio") or 0.0),
                float(coverage.get("uncovered_speech_seconds") or 0.0),
            )
        whisperx_coverage_key = coverage_sort_key(whisperx_report)
        anchor_coverage_key = coverage_sort_key(anchor_report)
        if anchor_coverage_key < whisperx_coverage_key:
            return "anchor", whisperx_report, anchor_report
        if whisperx_coverage_key < anchor_coverage_key:
            return "whisperx", whisperx_report, anchor_report
    if source_timing_candidate_sort_key(whisperx_report) <= source_timing_candidate_sort_key(anchor_report):
        return "whisperx", whisperx_report, anchor_report
    return "anchor", whisperx_report, anchor_report


def run_source_srt_audio_timing_verifier(
    audio_path,
    transcript_text,
    source_segments,
    source_language,
    device,
    duration_seconds,
    pipeline_config,
):
    policy = pipeline_config.get(
        "source_timing_verifier_policy",
        config_default("source_timing_verifier_policy", "stop"),
    )
    if (policy or "stop").lower() in {"off", "none", "disabled"}:
        return None

    rough_segments = build_source_timing_verification_rough_segments(
        source_segments,
        duration_seconds=duration_seconds,
        block_seconds=pipeline_config.get(
            "source_timing_verifier_block_seconds",
            config_default("source_timing_verifier_block_seconds", 30),
        ),
        padding_seconds=pipeline_config.get(
            "source_timing_verifier_padding_seconds",
            config_default("source_timing_verifier_padding_seconds", 8),
        ),
    )
    aligned_words, verifier_alignment = whisperx_align_words_to_audio(
        str(audio_path),
        rough_segments,
        source_language,
        device=device,
    )
    report = source_srt_audio_timing_report(
        transcript_text,
        source_segments,
        aligned_words,
        min_token_coverage=pipeline_config.get(
            "source_timing_verifier_min_token_coverage",
            config_default("source_timing_verifier_min_token_coverage", 0.75),
        ),
        min_checked_segment_ratio=pipeline_config.get(
            "source_timing_verifier_min_checked_segment_ratio",
            config_default("source_timing_verifier_min_checked_segment_ratio", 0.75),
        ),
        max_start_drift_seconds=pipeline_config.get(
            "source_timing_verifier_max_start_drift_seconds",
            config_default("source_timing_verifier_max_start_drift_seconds", 0.85),
        ),
        max_early_end_seconds=pipeline_config.get(
            "source_timing_verifier_max_early_end_seconds",
            config_default("source_timing_verifier_max_early_end_seconds", 0.30),
        ),
        max_late_end_seconds=pipeline_config.get(
            "source_timing_verifier_max_late_end_seconds",
            config_default("source_timing_verifier_max_late_end_seconds", 2.50),
        ),
        max_bad_segment_ratio=pipeline_config.get(
            "source_timing_verifier_max_bad_segment_ratio",
            config_default("source_timing_verifier_max_bad_segment_ratio", 0.05),
        ),
        max_bad_segments=pipeline_config.get(
            "source_timing_verifier_max_bad_segments",
            config_default("source_timing_verifier_max_bad_segments", 2),
        ),
    )
    report["whisperx_alignment"] = verifier_alignment
    report["rough_segment_count"] = len(rough_segments)
    return enforce_source_srt_audio_timing(
        report,
        policy=policy,
        label="Source subtitles before translation",
    )


def subtitle_timing_signature(segments):
    signature = []
    for segment in segments or []:
        try:
            start_ms = max(0, int(float(segment["start"]) * 1000 + 0.5))
            end_ms = max(0, int(float(segment["end"]) * 1000 + 0.5))
        except (KeyError, TypeError, ValueError):
            signature.append((None, None))
            continue
        signature.append((start_ms, end_ms))
    return signature


def translated_subtitle_alignment_report(source_segments, translated_segments):
    source_signature = subtitle_timing_signature(source_segments)
    translated_signature = subtitle_timing_signature(translated_segments)
    timing_mismatches = []
    for index, (source_timing, translated_timing) in enumerate(
        zip(source_signature, translated_signature),
        start=1,
    ):
        if source_timing != translated_timing:
            timing_mismatches.append({
                "index": index,
                "source": source_timing,
                "translated": translated_timing,
            })
            if len(timing_mismatches) >= 10:
                break

    source_timing_report = subtitle_timing_integrity_report(source_segments)
    translated_timing_report = subtitle_timing_integrity_report(translated_segments)
    count_match = len(source_signature) == len(translated_signature)
    timing_match = count_match and not timing_mismatches
    accept = bool(
        count_match
        and timing_match
        and source_timing_report["accept"]
        and translated_timing_report["accept"]
    )
    return {
        "accept": accept,
        "count_match": count_match,
        "source_count": len(source_signature),
        "translated_count": len(translated_signature),
        "timing_match": timing_match,
        "timing_mismatches": timing_mismatches,
        "source_timing": source_timing_report,
        "translated_timing": translated_timing_report,
    }


def enforce_translated_subtitle_alignment(source_segments, translated_segments, label="translated subtitles"):
    report = translated_subtitle_alignment_report(source_segments, translated_segments)
    if not report["accept"]:
        raise RuntimeError(
            f"{label} failed source/translation alignment check "
            f"(source_count={report['source_count']}, "
            f"translated_count={report['translated_count']}, "
            f"timing_match={report['timing_match']}, "
            f"source_monotonic={report['source_timing']['monotonic']}, "
            f"translated_monotonic={report['translated_timing']['monotonic']})."
        )
    return report


class TranscriptAnchorCoverageError(RuntimeError):
    pass


class TranscriptPlausibilityError(RuntimeError):
    pass


class ChunkArtifactValidationError(RuntimeError):
    def __init__(self, message, artifact):
        super().__init__(message)
        self.diagnostics = {
            "rejected_artifact": artifact.to_dict(),
        }


class TranslationSemanticQAError(RuntimeError):
    def __init__(self, message, report, translated_segments):
        super().__init__(message)
        self.report = report
        self.translated_segments = [
            dict(segment)
            for segment in translated_segments
        ]
        self.diagnostics = {
            "translation_qa_report": report,
            "translated_segments": self.translated_segments,
        }


def transcript_repetitive_suffix_report(
    words,
    min_repetitions=2,
    min_repeated_words=2,
    max_unit_words=None,
):
    """Describe an exact token cycle that repeats contiguously at transcript end."""
    comparable_words = []
    for word in words:
        comparable = unicodedata.normalize("NFKC", str(word)).casefold()
        comparable = re.sub(r"(^[^\w]+|[^\w]+$)", "", comparable, flags=re.UNICODE)
        comparable_words.append(comparable or str(word).casefold())

    best = None
    word_count = len(comparable_words)
    configured_maximum = int(max_unit_words or 0)
    possible_maximum = word_count // max(2, int(min_repetitions))
    max_unit = (
        min(configured_maximum, possible_maximum)
        if configured_maximum > 0
        else possible_maximum
    )
    for unit_word_count in range(1, max_unit + 1):
        unit = comparable_words[-unit_word_count:]
        repetition_count = 1
        cursor = word_count - (2 * unit_word_count)
        while cursor >= 0 and comparable_words[cursor:cursor + unit_word_count] == unit:
            repetition_count += 1
            cursor -= unit_word_count
        repeated_word_count = repetition_count * unit_word_count
        if repetition_count < int(min_repetitions) or repeated_word_count < int(min_repeated_words):
            continue
        candidate = {
            "detected": True,
            "unit_text": " ".join(words[-unit_word_count:]),
            "unit_word_count": unit_word_count,
            "repetition_count": repetition_count,
            "repeated_word_count": repeated_word_count,
            "start_word_index": word_count - repeated_word_count,
            "end_word_index": word_count,
        }
        if best is None or (
            candidate["repeated_word_count"],
            candidate["repetition_count"],
            -candidate["unit_word_count"],
        ) > (
            best["repeated_word_count"],
            best["repetition_count"],
            -best["unit_word_count"],
        ):
            best = candidate

    return best or {
        "detected": False,
        "unit_text": None,
        "unit_word_count": 0,
        "repetition_count": 0,
        "repeated_word_count": 0,
        "start_word_index": None,
        "end_word_index": word_count,
    }


def transcript_plausibility_report(
    transcript_text,
    duration_seconds,
    usage=None,
    max_words_per_second=8.0,
    max_chars_per_second=80.0,
    min_repetitive_suffix_repetitions=2,
    min_repetitive_suffix_words=2,
    max_repetitive_unit_words=None,
):
    """Reject physically impossible model output before spending work on alignment."""
    normalized = normalize_subtitle_text(transcript_text)
    words = normalized.split()
    non_space_chars = sum(1 for char in normalized if not char.isspace())
    duration = max(0.001, float(duration_seconds or 0.0))
    words_per_second = len(words) / duration if duration_seconds else 0.0
    chars_per_second = non_space_chars / duration if duration_seconds else 0.0
    usage = dict(usage or {})
    finish_reason = str(usage.get("finish_reason") or "").upper()
    output_tokens = number_or_zero(
        usage.get("output_tokens")
        or usage.get("candidatesTokenCount")
        or usage.get("candidates_token_count")
    )
    max_output_tokens = number_or_zero(usage.get("max_output_tokens"))
    repetitive_suffix = transcript_repetitive_suffix_report(
        words,
        min_repetitions=min_repetitive_suffix_repetitions,
        min_repeated_words=min_repetitive_suffix_words,
        max_unit_words=max_repetitive_unit_words,
    )

    problems = []
    if not normalized:
        problems.append("empty_transcript")
    if duration_seconds and words_per_second > float(max_words_per_second):
        problems.append("impossible_word_rate")
    if duration_seconds and chars_per_second > float(max_chars_per_second):
        problems.append("impossible_character_rate")
    if finish_reason in {"MAX_TOKENS", "LENGTH"}:
        problems.append("output_token_limit_reached")
    elif max_output_tokens and output_tokens >= max_output_tokens:
        problems.append("output_token_limit_reached")
    repetition_candidates = []
    if repetitive_suffix["detected"]:
        repetition_candidates.append("contiguous_repetitive_suffix")

    return {
        "accept": not problems,
        "engine": "duration_normalized_transcript_plausibility",
        "duration_seconds": round(float(duration_seconds or 0.0), 3),
        "word_count": len(words),
        "non_space_character_count": non_space_chars,
        "words_per_second": round(words_per_second, 3),
        "characters_per_second": round(chars_per_second, 3),
        "max_words_per_second": float(max_words_per_second),
        "max_chars_per_second": float(max_chars_per_second),
        "finish_reason": usage.get("finish_reason"),
        "output_tokens": output_tokens,
        "max_output_tokens": max_output_tokens,
        "usage": usage,
        "repetitive_suffix": repetitive_suffix,
        "repetition_candidates": repetition_candidates,
        "repetition_requires_confirmation": bool(repetition_candidates),
        "text_only_repetition_decision_allowed": False,
        "problems": problems,
    }


def enforce_transcript_plausibility(report, label="transcript"):
    if report.get("accept"):
        return report
    raise TranscriptPlausibilityError(
        f"{label} failed plausibility verification "
        f"(duration={report.get('duration_seconds')}s, words={report.get('word_count')}, "
        f"words_per_second={report.get('words_per_second')}, "
        f"characters_per_second={report.get('characters_per_second')}, "
        f"problems={report.get('problems')})."
    )


def verify_transcript_plausibility_with_retry(
    transcript_text,
    transcription_usage,
    *,
    transcription_provider,
    transcription_model,
    pipeline_config,
    api_audio_path,
    video_duration_seconds,
    output_root,
    base,
    allow_rejected_for_review=False,
):
    """Verify canonical text and perform the bounded whole-audio Gemini retry."""
    plausibility_retry_count = 0
    plausibility_usage = transcription_usage
    attempt_usages = [dict(transcription_usage or {})]
    plausibility_retries = int(
        pipeline_config.get(
            "transcript_plausibility_retries",
            config_default("transcript_plausibility_retries", 1),
        )
        or 0
    )
    while True:
        transcript_plausibility = transcript_plausibility_report(
            transcript_text,
            video_duration_seconds,
            usage=plausibility_usage,
            max_words_per_second=pipeline_config.get(
                "transcript_plausibility_max_words_per_second",
                config_default("transcript_plausibility_max_words_per_second", 8.0),
            ),
            max_chars_per_second=pipeline_config.get(
                "transcript_plausibility_max_chars_per_second",
                config_default("transcript_plausibility_max_chars_per_second", 80.0),
            ),
            min_repetitive_suffix_repetitions=pipeline_config.get(
                "transcript_plausibility_min_repetitive_suffix_repetitions",
                config_default("transcript_plausibility_min_repetitive_suffix_repetitions", 2),
            ),
            min_repetitive_suffix_words=pipeline_config.get(
                "transcript_plausibility_min_repetitive_suffix_words",
                config_default("transcript_plausibility_min_repetitive_suffix_words", 2),
            ),
            max_repetitive_unit_words=pipeline_config.get(
                "transcript_plausibility_max_repetitive_unit_words",
                config_default("transcript_plausibility_max_repetitive_unit_words", 0),
            ),
        )
        transcript_plausibility["attempt_number"] = plausibility_retry_count + 1
        if transcript_plausibility.get("accept"):
            if transcript_plausibility.get("repetition_requires_confirmation"):
                transcript_plausibility.update({
                    "contained_for_review": True,
                    "terminal_repetition_candidate": True,
                    "original_wording_preserved": True,
                    "original_gemini_wording_preserved": (
                        transcription_provider == "google"
                    ),
                })
            return transcript_text, transcription_usage, transcript_plausibility

        rejected_number = plausibility_retry_count + 1
        rejected_report_path = Path(output_root) / f"{base}.rejected_transcript_{rejected_number}.json"
        rejected_text_path = Path(output_root) / f"{base}.rejected_transcript_{rejected_number}.txt"
        rejected_report_path.write_text(
            json.dumps(transcript_plausibility, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        rejected_text_path.write_text(transcript_text, encoding="utf-8")

        if (transcription_usage or {}).get("longform"):
            # Every long-form chunk has already passed its own bounded request,
            # plausibility, speech, silence, and resume gates.  Never replace
            # that evidence with an unsafe whole-audio retry.
            enforce_transcript_plausibility(
                transcript_plausibility,
                label=f"{transcription_model} long-form transcription",
            )

        if transcription_provider != "google" or plausibility_retry_count >= plausibility_retries:
            if transcription_provider == "google" and allow_rejected_for_review:
                transcript_plausibility["contained_for_review"] = True
                return transcript_text, transcription_usage, transcript_plausibility
            enforce_transcript_plausibility(
                transcript_plausibility,
                label=f"{transcription_model} transcription",
            )

        plausibility_retry_count += 1
        print(
            "\nWarning: Gemini returned a physically implausible or incomplete transcript. "
            f"Retrying whole-audio transcription ({plausibility_retry_count}/"
            f"{plausibility_retries}) before requesting timing anchors."
        )
        transcript_text, retry_usage = call_google_transcription(
            pipeline_config,
            str(api_audio_path),
            model=transcription_model,
            language=pipeline_config.get("source_language"),
            prompt=pipeline_config.get("transcription_prompt", ""),
            duration_seconds=video_duration_seconds,
            temperature=pipeline_config.get(
                "google_transcription_retry_temperature",
                config_default("google_transcription_retry_temperature", 0.8),
            ),
            prompt_version=PROFESSIONAL_PROMPT_VERSION,
        )
        attempt_usages.append(dict(retry_usage or {}))
        transcription_usage = sum_numeric_usage_dicts(attempt_usages)
        transcription_usage["attempts"] = attempt_usages
        for metadata_key in (
            "finish_reason",
            "max_output_tokens",
            "output_token_limit_strategy",
            "temperature",
        ):
            if metadata_key in retry_usage:
                transcription_usage[metadata_key] = retry_usage[metadata_key]
        plausibility_usage = retry_usage


def build_automatic_acoustic_repetition_report(
    media_path,
    audio_path,
    primary_timing_segments,
    pipeline_config,
    *,
    output_root,
    base,
    duration_seconds,
    device="cpu",
    candidate_region=None,
    artifact_suffix=None,
):
    """Run a localized timing pass and independent signal analysis.

    Neither this API nor the acoustic engine accepts an expected repetition
    count.  The extracted candidate and automatic evidence are retained beside
    the other pipeline diagnostics.
    """
    if duration_seconds is None:
        duration_seconds = get_video_duration(str(media_path))
    candidate = (
        dict(candidate_region)
        if candidate_region is not None
        else discover_terminal_candidate_region(
            duration_seconds,
            primary_timing_segments,
            config=pipeline_config.get("acoustic_repetition_analysis"),
        )
    )
    if not candidate.get("available"):
        return {
            "schema": "subgen_automatic_acoustic_repetition_evidence_v2",
            "inference_origin": "automatic_acoustic_engine",
            "expected_count_argument_supported": False,
            "human_ground_truth_used": False,
            "candidate_region": candidate,
            "predicted_count": None,
            "events": [],
            "methods_agree": False,
            "count_inference_confident": False,
            "safe_for_automatic_trim": False,
            "ambiguity_flags": [candidate.get("reason") or "candidate_region_unavailable"],
        }

    region_start = float(candidate["region_start"])
    region_end = float(candidate["region_end"])
    label = f".{artifact_suffix}" if artifact_suffix else ""
    candidate_path = Path(output_root) / f"{base}{label}.automatic_repetition_candidate.mp3"
    extract_audio_chunk_for_transcription(
        str(audio_path),
        str(candidate_path),
        region_start,
        region_end - region_start,
        sample_rate=pipeline_config.get("audio_sample_rate", 16000),
        bitrate=pipeline_config.get("api_audio_bitrate", "64k"),
    )
    preferred = str(
        pipeline_config.get(
            "api_timing_anchor_provider",
            pipeline_config.get(
                "timing_anchor_provider",
                config_default("timing_anchor_provider", "openai"),
            ),
        )
    ).strip().lower()
    providers = [preferred] if preferred in {"openai", "local"} else []
    local_fallback_model = configured_local_timing_anchor_fallback_model(
        pipeline_config
    )
    if not providers:
        providers = ["openai"]
    if (
        preferred == "openai"
        and local_fallback_model
        and "local" not in providers
    ):
        providers.append("local")
    attempts = []
    best = None
    for provider in providers:
        try:
            if provider == "openai":
                openai_env_key = pipeline_config.get("openai_api_env_key", "OPENAI_API_KEY")
                if not os.environ.get(openai_env_key):
                    raise RuntimeError(f"Timing credential {openai_env_key} is unavailable.")
                _, localized_segments, usage = transcribe_audio_openai(
                    str(candidate_path),
                    model=pipeline_config.get("api_timing_anchor_model", "whisper-1"),
                    language=None,
                    prompt="",
                    api_env_key=openai_env_key,
                    chunking="off",
                    word_timestamps=True,
                )
            else:
                localized_segments, info = transcribe_audio(
                    str(candidate_path),
                    model_size=(
                        local_fallback_model
                        or pipeline_config.get("model_size")
                        or "small"
                    ),
                    device=device,
                    beam_size=1,
                    word_timestamps=True,
                )
                usage = {
                    "language": getattr(info, "language", None),
                    "language_probability": getattr(info, "language_probability", None),
                }
            report = infer_repetition_evidence(
                media_path,
                audio_path,
                primary_timing_segments=primary_timing_segments,
                localized_timing_segments=localized_segments,
                localized_timing_source_offset_seconds=region_start,
                candidate_region=candidate,
                config=pipeline_config.get("acoustic_repetition_analysis"),
            )
            attempt = {
                "provider": provider,
                "status": "completed",
                "usage": usage,
                "localized_timing_observation_count": len(localized_segments),
                "methods_agree": report.get("methods_agree"),
                "count_inference_confident": report.get("count_inference_confident"),
            }
            attempts.append(attempt)
            report["localized_timing_attempts"] = list(attempts)
            report["selected_timing_provider"] = provider
            report["candidate_audio_path"] = str(candidate_path)
            if best is None or report.get("count_confidence", 0) > best.get("count_confidence", 0):
                best = report
            if report.get("count_inference_confident"):
                break
        except Exception as error:
            attempts.append({
                "provider": provider,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            })
    if best is None:
        return {
            "schema": "subgen_automatic_acoustic_repetition_evidence_v2",
            "inference_origin": "automatic_acoustic_engine",
            "expected_count_argument_supported": False,
            "human_ground_truth_used": False,
            "candidate_region": candidate,
            "candidate_audio_path": str(candidate_path),
            "localized_timing_attempts": attempts,
            "predicted_count": None,
            "events": [],
            "methods_agree": False,
            "count_inference_confident": False,
            "safe_for_automatic_trim": False,
            "ambiguity_flags": ["all_localized_timing_methods_failed"],
        }
    best["localized_timing_attempts"] = attempts
    return best


def independent_acoustic_timing_segments(timing_segments):
    """Exclude canonical-provider gap fillers from acoustic inference.

    Long-form timing may admit provider segments solely to keep subtitle
    alignment complete when Whisper-1 has a blind spot. Those segments are
    useful for fail-closed coverage, but they are not independent evidence and
    must never move or count a repetition candidate.
    """
    return [
        item
        for item in (timing_segments or [])
        if isinstance(item, dict)
        and not item.get("_supplemental_timing_evidence")
    ]


def replace_corrected_repetition_timing_anchors(
    timing_segments,
    automatic_report,
    trim_report,
):
    """Bind retained exact copies to their independently inferred intervals."""
    original = [dict(item) for item in (timing_segments or [])]
    failures = []
    if not (trim_report or {}).get("applied"):
        failures.append("repetition_trim_not_applied")
    if not (automatic_report or {}).get("count_inference_confident"):
        failures.append("acoustic_count_not_confident")
    if (automatic_report or {}).get("ambiguity_flags"):
        failures.append("acoustic_evidence_ambiguous")
    unit_text = normalize_subtitle_text((trim_report or {}).get("unit_text") or "")
    if not unit_text:
        failures.append("repeated_unit_text_unavailable")
    events = []
    for event in (automatic_report or {}).get("events") or []:
        try:
            start = float(event.get("start_seconds"))
            end = float(event.get("end_seconds"))
        except (AttributeError, TypeError, ValueError):
            failures.append("invalid_acoustic_event")
            continue
        if end <= start:
            failures.append("invalid_acoustic_event")
            continue
        events.append((start, end))
    retained = int((trim_report or {}).get("retained_occurrences") or 0)
    if not events or len(events) != retained:
        failures.append("retained_occurrence_timing_count_mismatch")
    events.sort()
    if any(events[index][0] < events[index - 1][1] for index in range(1, len(events))):
        failures.append("overlapping_acoustic_events")
    failures = list(dict.fromkeys(failures))
    if failures:
        return original, {
            "applied": False,
            "reason_codes": failures,
            "original_timing_segment_count": len(original),
        }

    region_start = events[0][0]
    region_end = events[-1][1]
    retained_segments = []
    removed_supplemental = []
    for segment in original:
        overlaps = (
            float(segment.get("end", 0.0)) > region_start
            and float(segment.get("start", 0.0)) < region_end
        )
        if overlaps and segment.get("_supplemental_timing_evidence"):
            removed_supplemental.append(segment)
            continue
        retained_segments.append(segment)
    anchors = [
        {
            "start": start,
            "end": end,
            "text": unit_text,
            "_automatic_acoustic_repetition_timing": True,
            "_timing_evidence_only": True,
        }
        for start, end in events
    ]
    combined = sorted(
        retained_segments + anchors,
        key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))),
    )
    return combined, {
        "applied": True,
        "reason_codes": [],
        "unit_text": unit_text,
        "retained_occurrences": retained,
        "removed_supplemental_segments": len(removed_supplemental),
        "automatic_anchor_intervals": events,
        "original_timing_segment_count": len(original),
        "corrected_timing_segment_count": len(combined),
    }


def apply_automatic_terminal_repetition_trim(
    transcript_text,
    plausibility_report,
    automatic_report,
    *,
    media_sha256=None,
):
    """Apply only evidence produced by the automatic acoustic engine."""
    if not isinstance(automatic_report, dict):
        return transcript_text, {
            "engine": "exact_terminal_repetition_trim_v1",
            "applied": False,
            "reason_codes": ["automatic_acoustic_evidence_not_available"],
            "corrected_text": transcript_text,
        }
    bound_hash = automatic_report.get("media_sha256")
    if not media_sha256 or bound_hash != media_sha256:
        return transcript_text, {
            "engine": "exact_terminal_repetition_trim_v1",
            "applied": False,
            "reason_codes": ["evidence_media_hash_mismatch"],
            "corrected_text": transcript_text,
            "expected_media_sha256": media_sha256,
            "evidence_media_sha256": bound_hash,
        }
    evidence = trim_evidence_from_automatic_report(automatic_report)
    report = terminal_repetition_trim_report(
        transcript_text,
        (plausibility_report or {}).get("repetitive_suffix") or {},
        evidence,
        require_automatic_inference=True,
    )
    return report["corrected_text"], report


def recover_with_adaptive_bounded_windows(
    *,
    pipeline_config,
    transcription_model,
    api_audio_path,
    output_root,
    base,
    media_identity,
    media_duration_seconds,
    timing_segments,
    contained_segments,
    coverage_report,
):
    """Run a bounded, non-recursive recovery plan after suffix recovery fails."""
    plan = plan_adaptive_recovery_windows(
        coverage_report,
        timing_segments,
        media_duration=media_duration_seconds,
        maximum_windows=int(
            pipeline_config.get("recovery_maximum_bounded_windows", 4)
        ),
        context_seconds=float(
            pipeline_config.get("recovery_window_context_seconds", 0.12)
        ),
    )
    attempts = []
    usages = []
    current_segments = list(contained_segments or [])
    current_coverage = dict(coverage_report or {})
    diagnostics_root = Path(output_root) / f"{base}.bounded_window_recovery"
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    for window in plan["windows"]:
        index = int(window["index"])
        source_offset = float(window["start_seconds"])
        audio_end = float(window["end_seconds"])
        audio_duration = audio_end - source_offset
        audio_path = diagnostics_root / f"window_{index}.mp3"
        call_root = diagnostics_root / f"window_{index}_request"
        attempt = {
            **window,
            "strategy": "adaptive_bounded_window",
            "audio_path": str(audio_path),
            "request_diagnostics_path": str(call_root),
            "accepted": False,
            "merge_decision": None,
        }
        try:
            extract_audio_chunk_for_transcription(
                str(api_audio_path),
                str(audio_path),
                source_offset,
                audio_duration,
                sample_rate=pipeline_config.get("audio_sample_rate", 16000),
                bitrate=pipeline_config.get("api_audio_bitrate", "64k"),
            )
            response = call_google_structured_transcription(
                pipeline_config,
                str(audio_path),
                model=transcription_model,
                language=pipeline_config.get("source_language"),
                prompt=pipeline_config.get("transcription_prompt", ""),
                duration_seconds=audio_duration,
                generation_scope="bounded_window",
                source_offset_seconds=source_offset,
                diagnostic_output_dir=call_root,
            )
            usages.append(response.get("usage") or {})
            absolute = offset_suffix_segments(
                response["segments"],
                source_offset,
                generation_scope="bounded_window",
            )
            timestamps = timestamp_proposal_report(
                media_identity,
                absolute,
                media_duration=media_duration_seconds,
                independent_speech_intervals=timing_segments,
            )
            repetition = repetition_anomaly_report(
                media_identity, timestamps["segments"]
            )
            eligible = [
                {**cue, "canonical_source": "gemini"}
                for cue in timestamps["segments"]
                if cue.get("independently_verified")
            ]
            merged = merge_validated_gemini_regions(current_segments, eligible)
            attempt.update({
                "prompt_version": response["prompt_version"],
                "request_sha256": (response.get("usage") or {}).get("request_sha256"),
                "response_sha256": (response.get("usage") or {}).get("response_sha256"),
                "timestamp_report": timestamps,
                "repetition_report": repetition,
                "merge_decision": {
                    "added": len(merged["added"]),
                    "rejected": merged["rejected"],
                },
                "accepted": bool(merged["added"]),
            })
            if merged["added"]:
                current_segments = merged["segments"]
                current_coverage = independent_speech_coverage(
                    media_identity,
                    timing_segments,
                    [
                        (
                            cue.get("start", cue.get("gemini_proposed_start")),
                            cue.get("end", cue.get("gemini_proposed_end")),
                        )
                        for cue in current_segments
                    ],
                    media_duration=media_duration_seconds,
                    evidence_sources=[
                        "timing_anchor",
                        "validated_gemini_regions",
                        "bounded_window",
                    ],
                )
                attempt["coverage_after_merge"] = current_coverage
        except Exception as error:
            attempt.update({
                "error_type": type(error).__name__,
                "error": str(error),
            })
        attempts.append(attempt)
    return {
        "plan": plan,
        "attempts": attempts,
        "segments": current_segments,
        "coverage_report": current_coverage,
        "usages": usages,
        "accept": bool(current_coverage.get("accept")),
    }


def retranslate_review_selected_cues(
    review,
    cue_ids,
    pipeline_config,
    *,
    device="cpu",
    actor="system",
):
    """Translate only explicitly selected source cues and update their history."""
    source_cues = selected_source_cues(review, cue_ids)
    source_language = review.get("source_language")
    target_language = review.get("target_language")
    if not target_language or target_language == source_language:
        raise ValueError("This review has no distinct target language to retranslate")
    provider = pipeline_config.get("translation_provider", "local")
    backend = "transformers" if provider == "local" else provider
    model = (
        pipeline_config.get("translation_model")
        or get_translation_model_label(source_language, target_language, pipeline_config)
    )
    translated = translate_segments(
        source_cues,
        src_lang=source_language,
        tgt_lang=target_language,
        device=device,
        batch_size=min(
            len(source_cues),
            int(pipeline_config.get("translation_batch_size", 8)),
        ),
        backend=backend,
        llm_model=model,
        context_window=int(pipeline_config.get("translation_context_window", 2)),
        source_dialect=pipeline_config.get("source_dialect", "auto"),
        target_dialect=pipeline_config.get("target_dialect", "natural"),
        translator_notes=pipeline_config.get("translator_notes", ""),
        provider_config=pipeline_config,
        glossary=pipeline_config.get("translation_glossary", []),
    )
    if len(translated) != len(source_cues):
        raise RuntimeError("Selected-cue translator returned a different cue count")
    apply_selected_cue_retranslation(
        review,
        cue_ids,
        translated,
        actor=actor,
        provider=provider,
        model=model,
    )
    return {
        "selected_source_cue_ids": [str(value) for value in cue_ids],
        "translated_cue_count": len(translated),
        "provider": provider,
        "model": model,
        "review": review,
    }


def align_transcript_to_timing_anchors(
    transcript_text,
    timing_segments,
    min_direct_coverage=0.55,
    canonical_metadata_segments=None,
    speech_map=None,
):
    aligned_words, anchor_report = timing_segments_to_word_anchors(timing_segments)
    if not aligned_words:
        raise RuntimeError("No usable text timing anchors were available.")
    segments, reconciliation_info = reconcile_transcript_with_aligned_words(
        transcript_text,
        aligned_words,
        min_direct_coverage=min_direct_coverage,
        canonical_metadata_segments=canonical_metadata_segments,
        possible_audio_intervals=(
            (speech_map or {}).get("possible_audio_intervals")
        ),
    )
    if (
        canonical_metadata_segments
        and speech_map
        and segments
    ):
        segments, bridge_report = bridge_prompted_segment_speech_gaps(
            segments,
            canonical_metadata_segments,
            speech_map,
        )
        reconciliation_info["prompted_speech_gap_bridge"] = (
            bridge_report
        )
    canonical_tokens = max(1, int(reconciliation_info.get("canonical_tokens") or 0))
    timed_words = int(reconciliation_info.get("timed_words") or 0)
    matched_tokens = int(reconciliation_info.get("matched_tokens") or 0)
    direct_coverage = matched_tokens / canonical_tokens
    ignored_timed = len(reconciliation_info.get("ignored_timed_words") or [])
    timed_to_canonical_ratio = timed_words / canonical_tokens
    ignored_timed_ratio = ignored_timed / max(1, timed_words)
    if (
        timed_words >= canonical_tokens + 80
        and timed_to_canonical_ratio > 1.35
        and ignored_timed_ratio > 0.30
        and direct_coverage < 0.88
    ):
        raise TranscriptAnchorCoverageError(
            "Timing anchors contain substantially more speech than the canonical transcript "
            f"({timed_words} anchor words vs {canonical_tokens} transcript tokens; "
            f"{ignored_timed} anchor words unmatched; direct coverage {direct_coverage:.1%}). "
            "The source transcript is likely incomplete."
        )
    integrity = source_subtitle_integrity_report(transcript_text, segments)
    if not integrity["accept"]:
        raise RuntimeError(
            "Canonical timing-anchor reconciliation failed subtitle integrity "
            f"(similarity={integrity['text']['similarity']}, "
            f"monotonic={integrity['timing']['monotonic']})."
        )
    reconciliation_info.update({
        "engine": "anchor_canonical_reconciliation",
        "anchor_report": anchor_report,
        "source_integrity": integrity,
    })
    return segments, reconciliation_info


def recover_longform_canonical_alignment_coverage(
    *,
    media_path,
    output_root,
    base,
    media_identity,
    media_duration_seconds,
    segments,
    timing_segments,
    speech_map,
    validation,
    pipeline_config,
    pipeline_plan,
    transcription_model,
    source_language,
):
    """Recover canonical text only where final independent alignment proves a gap."""
    max_uncovered_gap_seconds = float(
        pipeline_config.get(
            "longform_max_uncovered_gap_seconds",
            config_default("longform_max_uncovered_gap_seconds", 1.5),
        )
    )
    max_uncovered_ratio = float(
        pipeline_config.get(
            "longform_max_uncovered_ratio",
            config_default("longform_max_uncovered_ratio", 0.03),
        )
    )
    context_seconds = float(
        pipeline_config.get(
            "longform_coverage_recovery_context_seconds",
            config_default("longform_coverage_recovery_context_seconds", 6),
        )
    )
    max_window_seconds = float(
        pipeline_config.get(
            "longform_coverage_recovery_max_window_seconds",
            config_default("longform_coverage_recovery_max_window_seconds", 180),
        )
    )
    max_attempts = int(
        pipeline_config.get(
            "longform_coverage_recovery_max_total_attempts",
            config_default(
                "longform_coverage_recovery_max_total_attempts",
                12,
            ),
        )
    )
    gaps, selection = select_coverage_recovery_gaps(
        validation,
        max_uncovered_gap_seconds=max_uncovered_gap_seconds,
        join_seconds=max(
            context_seconds * 2.0,
            max_uncovered_gap_seconds,
        ),
        max_window_seconds=max_window_seconds,
        max_attempts=max_attempts,
    )
    current_segments = [dict(segment) for segment in segments or []]
    current_validation = dict(validation or {})
    attempts = []
    usages = []
    recovery_root = (
        Path(output_root)
        / f"{base}_longform_alignment_coverage_recovery"
    )
    recovery_root.mkdir(parents=True, exist_ok=True)

    for attempt_index, gap in enumerate(gaps, 1):
        if current_validation.get("accept"):
            break
        gap_start = float(gap[0])
        gap_end = float(gap[1])
        window_start = max(0.0, gap_start - context_seconds)
        window_end = min(
            float(media_duration_seconds),
            gap_end + context_seconds,
        )
        if window_end - window_start > max_window_seconds:
            window_end = min(
                float(media_duration_seconds),
                window_start + max_window_seconds,
            )
        audio_path = (
            recovery_root / f"window-{attempt_index:04d}.mp3"
        )
        checkpoint_dir = (
            recovery_root / f"window-{attempt_index:04d}"
        )
        attempt = {
            "index": attempt_index,
            "target_gap": [
                round(gap_start, 3),
                round(gap_end, 3),
            ],
            "window": [
                round(window_start, 3),
                round(window_end, 3),
            ],
            "provider": "google",
            "model": transcription_model,
            "status": "failed",
            "audio_path": str(audio_path),
            "checkpoint_dir": str(checkpoint_dir),
        }
        keep_failed_audio = True
        try:
            extract_audio_chunk_for_transcription(
                str(media_path),
                str(audio_path),
                window_start,
                window_end - window_start,
                sample_rate=pipeline_config.get(
                    "audio_sample_rate",
                    16000,
                ),
                bitrate=pipeline_config.get(
                    "api_audio_bitrate",
                    "64k",
                ),
            )
            relative_speech_map = clip_longform_speech_map(
                speech_map,
                window_start,
                window_end,
                relative=True,
            )
            recovered_artifact = transcribe_provider_longform(
                str(audio_path),
                provider="google",
                model=transcription_model,
                language=source_language,
                prompt=pipeline_config.get(
                    "transcription_prompt",
                    "",
                ),
                pipeline_config=pipeline_config,
                pipeline_plan=pipeline_plan,
                output_dir=checkpoint_dir,
                precomputed_speech_map=relative_speech_map,
            )
            usages.append(dict(recovered_artifact.usage or {}))
            absolute_metadata = [
                {
                    **dict(segment),
                    "start": float(segment["start"]) + window_start,
                    "end": float(segment["end"]) + window_start,
                }
                for segment in recovered_artifact.segments or []
                if segment.get("start") is not None
                and segment.get("end") is not None
            ]
            window_timing = [
                dict(segment)
                for segment in timing_segments or []
                if float(segment["end"]) > window_start
                and float(segment["start"]) < window_end
            ]
            if not window_timing:
                raise RuntimeError(
                    "Canonical coverage recovery has no independent timing "
                    "anchors in its bounded window."
                )
            absolute_speech_map = clip_longform_speech_map(
                speech_map,
                window_start,
                window_end,
            )
            recovered_segments, recovered_alignment = (
                align_transcript_to_timing_anchors(
                    recovered_artifact.text,
                    window_timing,
                    canonical_metadata_segments=absolute_metadata,
                    speech_map=absolute_speech_map,
                )
            )
            merged_segments, merge_report = (
                merge_longform_alignment_recovery_segments(
                    current_segments,
                    recovered_segments,
                    gap,
                    speech_map,
                    min_signal_overlap_ratio=float(
                        pipeline_config.get(
                            "longform_coverage_recovery_min_unscored_speech_overlap_ratio",
                            config_default(
                                "longform_coverage_recovery_min_unscored_speech_overlap_ratio",
                                0.80,
                            ),
                        )
                    ),
                    max_uncovered_gap_seconds=(
                        max_uncovered_gap_seconds
                    ),
                    max_uncovered_ratio=max_uncovered_ratio,
                )
            )
            attempt.update({
                "status": (
                    "accepted"
                    if merge_report.get("changed")
                    else "rejected"
                ),
                "language": recovered_artifact.language,
                "usage": recovered_artifact.usage,
                "longform_manifest": (
                    (recovered_artifact.metadata or {}).get(
                        "longform_manifest"
                    )
                ),
                "alignment": recovered_alignment,
                "merge": merge_report,
            })
            if merge_report.get("changed"):
                current_segments = merged_segments
                current_validation = dict(
                    merge_report["validation_after"]
                )
            keep_failed_audio = False
        except Exception as error:
            attempt.update({
                "error_type": type(error).__name__,
                "error": str(error),
            })
        finally:
            if not keep_failed_audio:
                audio_path.unlink(missing_ok=True)
        attempts.append(attempt)

    return current_segments, {
        "schema": "subgen_longform_canonical_alignment_recovery_v1",
        "media_identity": media_identity,
        "selection": selection,
        "attempt_count": len(attempts),
        "attempts": attempts,
        "usage": sum_numeric_usage_dicts(usages),
        "validation_before": validation,
        "validation_after": current_validation,
        "accept": bool(current_validation.get("accept")),
    }


def forced_align_transcript_to_audio(
    audio_path,
    transcript_text,
    rough_segments,
    language_code,
    device="cpu",
    min_coverage=0.55,
):
    """
    Align exact API transcript text back to the waveform with WhisperX.

    rough_segments provide approximate windows only. The returned words and
    subtitle text come from the API transcript after forced audio alignment.
    """
    whisperx = import_whisperx()
    language_code = language_code or "en"
    align_model, metadata, model_info = load_whisperx_alignment_model(
        whisperx,
        language_code,
        device,
    )
    forced_input = [
        {
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "text": normalize_subtitle_text(segment["text"]),
        }
        for segment in rough_segments
        if normalize_subtitle_text(segment.get("text"))
    ]
    if not forced_input:
        raise RuntimeError("Forced alignment has no rough transcript segments to align.")

    try:
        aligned_result = whisperx.align(
            forced_input,
            align_model,
            metadata,
            audio_path,
            device,
            return_char_alignments=False,
        )
    finally:
        try:
            import gc
            del align_model
            gc.collect()
            if device != "cpu":
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception:
            pass

    aligned_words = collect_aligned_words(aligned_result)
    segments, reconciliation_info = reconcile_transcript_with_aligned_words(
        transcript_text,
        aligned_words,
        min_direct_coverage=min_coverage,
    )
    coverage = reconciliation_info["direct_token_coverage"]
    if coverage < min_coverage:
        raise RuntimeError(
            f"Forced alignment coverage is too low ({coverage:.2%}). "
            "Audio/transcript mismatch is too large for reliable timing."
        )
    if not segments:
        raise RuntimeError("Forced alignment did not produce timed subtitle segments.")
    return segments, {
        "engine": "whisperx",
        "coverage": round(coverage, 4),
        "aligned_words": len(aligned_words),
        "alignment_model": model_info,
        "_aligned_word_segments": aligned_words,
        "reconciliation": reconciliation_info,
    }


def get_translation_model_name(src_lang, tgt_lang):
    """Choose known-good translation models where the generic name is fragile."""
    if src_lang == tgt_lang:
        return None

    model_overrides = {
        ("en", "ar"): "Helsinki-NLP/opus-mt-tc-big-en-ar",
        ("he", "en"): "Helsinki-NLP/opus-mt-tc-big-he-en",
        ("ar", "en"): "Helsinki-NLP/opus-mt-ar-en",
        ("en", "fa"): "SeyedAli/English-to-Persian-Translation-mT5-V1",
    }

    return model_overrides.get((src_lang, tgt_lang), f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}")


def get_language_name(language_code):
    return get_supported_languages().get(language_code, language_code or "unknown")


def normalize_glossary(glossary):
    normalized = []
    for entry in glossary or []:
        if isinstance(entry, dict):
            source = normalize_subtitle_text(entry.get("source"))
            target = normalize_subtitle_text(entry.get("target"))
            forbidden = entry.get("forbidden", [])
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            source = normalize_subtitle_text(entry[0])
            target = normalize_subtitle_text(entry[1])
            forbidden = []
        else:
            continue

        if source and target:
            normalized.append({
                "source": source,
                "target": target,
                "forbidden": [normalize_subtitle_text(item) for item in forbidden if normalize_subtitle_text(item)],
            })

    return normalized


def format_glossary_for_prompt(glossary):
    entries = normalize_glossary(glossary)
    if not entries:
        return "None."

    lines = []
    for entry in entries:
        line = f"- {entry['source']} => {entry['target']}"
        if entry["forbidden"]:
            line += f" (never translate as: {', '.join(entry['forbidden'])})"
        lines.append(line)
    return "\n".join(lines)


def apply_glossary_to_translation(source_text, translated_text, glossary):
    source_text = normalize_subtitle_text(source_text)
    translated_text = normalize_subtitle_text(translated_text)
    for entry in normalize_glossary(glossary):
        if entry["source"] not in source_text:
            continue
        for forbidden in sorted(entry["forbidden"], key=len, reverse=True):
            translated_text = translated_text.replace(forbidden, entry["target"])
            translated_text = translated_text.replace(forbidden.capitalize(), entry["target"])
    return translated_text


def nested_value(data, *paths):
    for path in paths:
        current = data or {}
        for key in path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if current is not None:
            return current
    return None


def number_or_zero(value):
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def extract_token_usage(usage):
    usage = usage or {}
    input_tokens = number_or_zero(nested_value(
        usage,
        ("input_tokens",),
        ("prompt_tokens",),
        ("inputTokens",),
        ("promptTokenCount",),
    ))
    output_tokens = number_or_zero(nested_value(
        usage,
        ("output_tokens",),
        ("completion_tokens",),
        ("outputTokens",),
        ("candidatesTokenCount",),
    ))
    cached_input_tokens = number_or_zero(nested_value(
        usage,
        ("input_tokens_details", "cached_tokens"),
        ("prompt_tokens_details", "cached_tokens"),
        ("cache_read_input_tokens",),
        ("cachedContentTokenCount",),
    ))
    total_tokens = number_or_zero(nested_value(
        usage,
        ("total_tokens",),
        ("totalTokenCount",),
    ))
    if not total_tokens:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": int(input_tokens),
        "cached_input_tokens": int(cached_input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(total_tokens),
    }


def lookup_model_pricing(pricing_config, provider_id, model):
    provider_pricing = (pricing_config or {}).get(provider_id, {})
    if model in provider_pricing:
        return provider_pricing[model]

    # Allow date-suffixed or deployment-suffixed model names to inherit a base price.
    candidates = sorted(provider_pricing, key=len, reverse=True)
    for candidate in candidates:
        if model and str(model).startswith(candidate):
            return provider_pricing[candidate]
    return None


def estimate_usage_cost(provider_id, model, task, usage=None, duration_seconds=None, pricing_config=None):
    token_usage = extract_token_usage(usage)
    pricing = lookup_model_pricing(pricing_config, provider_id, model)
    if not pricing:
        return token_usage, None, "pricing_not_configured"

    if task == "transcription" and pricing.get("per_minute") is not None and duration_seconds:
        minutes = float(duration_seconds) / 60.0
        return token_usage, round(minutes * float(pricing["per_minute"]), 6), "estimated_per_minute"

    input_tokens = token_usage["input_tokens"]
    cached_tokens = min(token_usage["cached_input_tokens"], input_tokens)
    uncached_input_tokens = max(0, input_tokens - cached_tokens)
    output_tokens = token_usage["output_tokens"]

    if input_tokens == 0 and output_tokens == 0:
        return token_usage, None, "usage_not_returned"

    input_cost = uncached_input_tokens * float(pricing.get("input_per_1m", 0)) / 1_000_000
    cached_cost = cached_tokens * float(
        pricing.get("cached_input_per_1m", pricing.get("input_per_1m", 0))
    ) / 1_000_000
    output_cost = output_tokens * float(pricing.get("output_per_1m", 0)) / 1_000_000
    return token_usage, round(input_cost + cached_cost + output_cost, 6), "token_pricing"


def build_usage_event(provider_id, model, task, usage=None, duration_seconds=None, pricing_config=None):
    token_usage, cost_usd, cost_basis = estimate_usage_cost(
        provider_id,
        model,
        task,
        usage=usage,
        duration_seconds=duration_seconds,
        pricing_config=pricing_config,
    )
    return {
        "provider": provider_id,
        "model": model,
        "task": task,
        "usage": usage,
        "tokens": token_usage,
        "duration_seconds": round(float(duration_seconds), 3) if duration_seconds else None,
        "cost_usd": cost_usd,
        "cost_basis": cost_basis,
    }


def record_usage_event(config, provider_id, model, task, usage=None, duration_seconds=None):
    if config is None:
        return None
    event = build_usage_event(
        provider_id,
        model,
        task,
        usage=usage,
        duration_seconds=duration_seconds,
        pricing_config=config.get("api_pricing", {}),
    )
    config.setdefault("_usage_events", []).append(event)
    return event


def summarize_usage_events(events):
    events = list(events or [])
    known_cost = sum(number_or_zero(event.get("cost_usd")) for event in events if event.get("cost_usd") is not None)
    unknown_cost_events = [event for event in events if event.get("cost_usd") is None]
    totals = {
        "input_tokens": sum(event.get("tokens", {}).get("input_tokens", 0) for event in events),
        "cached_input_tokens": sum(event.get("tokens", {}).get("cached_input_tokens", 0) for event in events),
        "output_tokens": sum(event.get("tokens", {}).get("output_tokens", 0) for event in events),
        "total_tokens": sum(event.get("tokens", {}).get("total_tokens", 0) for event in events),
    }
    return {
        "events": events,
        "totals": totals,
        "known_cost_usd": round(known_cost, 6),
        "unknown_cost_event_count": len(unknown_cost_events),
    }


def format_usd(value):
    if value is None:
        return "unknown"
    return f"${float(value):.6f}"


def print_usage_report(report):
    events = report.get("events", [])
    if not events:
        print("\nAPI usage: no paid API usage recorded for this run.")
        return

    print("\nAPI usage and estimated cost")
    for event in events:
        tokens = event.get("tokens", {})
        duration = event.get("duration_seconds")
        duration_text = f", {duration / 60.0:.2f} min" if duration else ""
        print(
            f"- {event.get('task')} | {event.get('provider')}:{event.get('model')}"
            f"{duration_text} | in {tokens.get('input_tokens', 0)}, "
            f"out {tokens.get('output_tokens', 0)}, total {tokens.get('total_tokens', 0)} "
            f"| {format_usd(event.get('cost_usd'))} ({event.get('cost_basis')})"
        )

    totals = report.get("totals", {})
    print(
        f"Total tokens: in {totals.get('input_tokens', 0)}, "
        f"out {totals.get('output_tokens', 0)}, total {totals.get('total_tokens', 0)}"
    )
    print(f"Known estimated API cost: {format_usd(report.get('known_cost_usd', 0))}")
    if report.get("unknown_cost_event_count"):
        print("Some costs are unknown because pricing for that provider/model is not configured.")


def get_translation_model_label(src_lang, tgt_lang, pipeline_config):
    provider_id = pipeline_config.get("translation_provider")
    if not provider_id:
        provider_id = "openai" if pipeline_config.get("translation_backend") == "openai" else "local"
    if provider_id != "local":
        provider = get_provider(pipeline_config, provider_id, "translation")
        return pipeline_config.get("translation_model") or provider.get("translation_model")
    return get_translation_model_name(src_lang, tgt_lang)


def build_llm_translation_instructions(
    src_lang,
    tgt_lang,
    source_dialect="auto",
    target_dialect="natural",
    translator_notes="",
    glossary=None,
):
    source_language = get_language_name(src_lang)
    target_language = get_language_name(tgt_lang)
    notes = translator_notes.strip() if translator_notes else "None."

    return (
        "You are a professional subtitle translator for natural spoken language.\n"
        f"Translate from {source_language} ({src_lang}) to {target_language} ({tgt_lang}).\n"
        f"Source dialect: {source_dialect or 'auto'}.\n"
        f"Target dialect/register: {target_dialect or 'natural'}.\n"
        "Prioritize intended meaning over literal word mapping. Resolve idioms, slang, "
        "dialectal expressions, sarcasm, religious/political references, pronouns, and "
        "ambiguous words from context.\n"
        "Hard subtitle-boundary rule: each output item must translate only the source "
        "text of that same item index. Use context only to understand meaning; never "
        "pull words, clauses, names, titles, blessings, or sentence completions from "
        "neighboring context items into the current output. If the source item is a "
        "sentence fragment, translate that fragment as a fragment.\n"
        "If the speaker repeats a word or phrase in the source, preserve that repetition "
        "naturally in the translation. Do not deduplicate repeated speech. If there is a "
        "silent gap or pause between subtitle items, do not add filler words or bridge "
        "the pause with text that was not spoken.\n"
        "Preserve names, places, numbers, dates, and named organizations unless they have "
        "a standard translated form.\n"
        "Keep each translation suitable for subtitles: concise, natural, and readable. "
        "Do not add explanations or commentary.\n"
        "Return exactly one translated text for each requested subtitle index.\n"
        f"Mandatory glossary:\n{format_glossary_for_prompt(glossary)}\n"
        f"Additional translator notes: {notes}"
    )


def build_llm_translation_prompt(context_segments, batch_start, batch_end):
    items_to_translate = [
        {
            "index": item["index"],
            "start": item["start"],
            "end": item["end"],
            "text": item["text"],
        }
        for item in context_segments
        if batch_start <= item["index"] < batch_end
    ]
    context_before = [
        {
            "index": item["index"],
            "start": item["start"],
            "end": item["end"],
            "text": item["text"],
        }
        for item in context_segments
        if item["index"] < batch_start
    ]
    context_after = [
        {
            "index": item["index"],
            "start": item["start"],
            "end": item["end"],
            "text": item["text"],
        }
        for item in context_segments
        if item["index"] >= batch_end
    ]
    payload = {
        "instructions": (
            "Translate every item in items_to_translate. Do not translate context_before "
            "or context_after. Do not move context text into any translated item."
        ),
        "context_before": context_before,
        "items_to_translate": items_to_translate,
        "context_after": context_after,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def source_qa_schema():
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "accept": {"type": "boolean"},
            "severity": {"type": "string", "enum": ["ok", "warning", "fail"]},
            "summary": {"type": "string"},
            "problems": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": [
                                "missing_text",
                                "duplicated_text",
                                "bad_split",
                                "too_long",
                                "too_short",
                                "fallback_alignment",
                                "low_alignment_coverage",
                                "mixed_language_risk",
                                "idiom_or_dialect_risk",
                                "named_entity_risk",
                                "other",
                            ],
                        },
                        "segment_indexes": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                        "reason": {"type": "string"},
                        "recommended_action": {
                            "type": "string",
                            "enum": [
                                "continue",
                                "retry_forced_alignment",
                                "retry_api_transcription",
                                "split_chunk_smaller",
                                "add_glossary",
                                "manual_review",
                            ],
                        },
                    },
                    "required": ["type", "segment_indexes", "reason", "recommended_action"],
                },
            },
        },
        "required": ["accept", "severity", "summary", "problems"],
    }


def build_source_qa_instructions(source_language, target_language, source_dialect="auto", tiktok_style=False):
    style_info = "NOTE: The user has requested rapid-fire, word-by-word TikTok-style subtitles. Therefore, the source is intentionally split into hundreds of micro-segments (often 1-3 words) to pop up instantly. Do NOT fail or warn about 'bad_split' or 'too_short' segments, as this is the desired formatting." if tiktok_style else ""
    return (
        "You are a senior subtitle QA reviewer for natural spoken language.\n"
        "You do not create timestamps. You judge whether source subtitles are good enough "
        "to translate without wasting tokens.\n"
        f"Source language: {get_language_name(source_language)} ({source_language}).\n"
        f"Target language: {get_language_name(target_language)} ({target_language or 'same-language subtitles'}).\n"
        f"Source dialect hint: {source_dialect or 'auto'}.\n"
        f"{style_info}\n"
        "Evaluate meaning preservation, missing/duplicated transcript text, unnatural phrase splits, "
        "idioms, slang, dialectal expressions, named entities, mixed-language passages, and whether "
        "the alignment metadata suggests unreliable timing.\n"
        "Human speakers may repeat themselves. Do not treat repeated words or repeated phrases as an "
        "error when the full transcript indicates the repetition is actually spoken. Treat it as an "
        "error only when the subtitle duplicates text that is not present in the transcript/audio-derived "
        "source.\n"
        "Long pauses and silence are valid in lectures. Do not fail merely because there is a silent gap. "
        "Fail when subtitle text is stretched across silence, when speech after a pause is shifted late or "
        "early, or when text is omitted/merged because of the pause.\n"
        "Deterministic reports in alignment_metadata are authoritative for exact text preservation and "
        "audio-grounded cue timing. If source_integrity says the source subtitle text exactly matches the "
        "transcript and source_timing_verifier says timing passes, do not fail for missing_text, "
        "fallback_alignment, low_alignment_coverage, or timing concerns. You may still warn about readability "
        "or translation risks.\n"
        "The payload may contain one bounded batch from a longer subtitle document. In that case, review only "
        "aligned_source_segments. Use context_before and context_after only to understand phrase boundaries. "
        "Report only the global segment indexes present in aligned_source_segments.\n"
        "If alignment metadata says WhisperX was rejected by a drift guardrail and the current engine "
        "uses anchor timing, do not fail solely because WhisperX was rejected. Judge the actual aligned "
        "source segments against the transcript and fail only if those current segments are materially "
        "mistimed, missing text, duplicated, or unusable.\n"
        "Accept warnings only when translation can safely proceed. Fail only when the source subtitles "
        "are likely to cause materially wrong translation or unusable subtitle timing.\n"
        "Return JSON only."
    )


def _compact_source_qa_metadata_value(value, depth=0):
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= 500 else value[:497] + "..."
    if isinstance(value, dict):
        if depth >= 4:
            return {"field_count": len(value), "detail": "omitted_at_depth_limit"}
        compacted = {}
        for key, nested_value in value.items():
            key = str(key)
            if key.startswith("_"):
                continue
            compacted[key] = _compact_source_qa_metadata_value(
                nested_value,
                depth + 1,
            )
        return compacted
    if isinstance(value, (list, tuple)):
        values = list(value)
        if all(
            item is None or isinstance(item, (bool, int, float, str))
            for item in values
        ):
            limit = 20 if depth <= 2 else 8
            compacted = [
                _compact_source_qa_metadata_value(item, depth + 1)
                for item in values[:limit]
            ]
            if len(values) <= limit:
                return compacted
            return {
                "item_count": len(values),
                "sample": compacted,
                "sample_truncated": len(values) - limit,
            }
        sample_limit = 3 if depth <= 2 else 1
        return {
            "item_count": len(values),
            "sample": [
                _compact_source_qa_metadata_value(item, depth + 1)
                for item in values[:sample_limit]
            ],
            "sample_truncated": max(0, len(values) - sample_limit),
        }
    return str(value)[:500]


def compact_source_qa_alignment_metadata(alignment_info, max_chars=16000):
    if not isinstance(alignment_info, dict) or not alignment_info:
        return {}

    max_chars = max(2000, int(max_chars or 16000))
    priority_keys = [
        "engine",
        "timestamp_kind",
        "timestamp_provider",
        "timestamp_model",
        "timing_mode",
        "timing_anchor_provider",
        "timing_alignment_version",
        "forced_alignment_scope",
        "rough_timing_fallback_allowed",
        "canonical_tokens",
        "timing_tokens",
        "matched_tokens",
        "native_matched_tokens",
        "supplemental_matched_tokens",
        "direct_match_coverage",
        "coverage",
        "minimum_required_coverage",
        "source_integrity",
        "source_timing_verifier",
        "longform_source_validation",
        "longform_canonical_alignment_recovery",
        "prompted_speech_gap_bridge",
        "anchor_drift_guardrail",
        "whisperx_audio_timing_report",
        "anchor_audio_timing_report",
    ]
    ordered_keys = []
    for key in priority_keys:
        if key in alignment_info and key not in ordered_keys:
            ordered_keys.append(key)
    for key in alignment_info:
        if key not in ordered_keys and not str(key).startswith("_"):
            ordered_keys.append(key)

    compacted = {}
    omitted_keys = []
    for key in ordered_keys:
        candidate_value = _compact_source_qa_metadata_value(
            alignment_info[key],
        )
        candidate = dict(compacted)
        candidate[key] = candidate_value
        candidate["_metadata_compaction"] = {
            "original_top_level_field_count": len(alignment_info),
            "included_top_level_field_count": len(candidate) - 1,
            "omitted_top_level_field_count": 0,
            "raw_arrays_replaced_with_counts_and_bounded_samples": True,
        }
        if len(json.dumps(candidate, ensure_ascii=False)) <= max_chars:
            compacted[key] = candidate_value
        else:
            omitted_keys.append(key)

    compacted["_metadata_compaction"] = {
        "original_top_level_field_count": len(alignment_info),
        "included_top_level_field_count": len(compacted),
        "omitted_top_level_field_count": len(omitted_keys),
        "omitted_top_level_fields": omitted_keys[:20],
        "raw_arrays_replaced_with_counts_and_bounded_samples": True,
        "max_serialized_chars": max_chars,
    }
    while (
        len(json.dumps(compacted, ensure_ascii=False)) > max_chars
        and len(compacted) > 1
    ):
        removable = next(
            (
                key
                for key in reversed(list(compacted))
                if key != "_metadata_compaction"
                and key not in priority_keys
            ),
            None,
        )
        if removable is None:
            removable = next(
                (
                    key
                    for key in reversed(list(compacted))
                    if key != "_metadata_compaction"
                ),
                None,
            )
        if removable is None:
            break
        compacted.pop(removable, None)
        omitted_keys.append(removable)
        compacted["_metadata_compaction"].update({
            "included_top_level_field_count": len(compacted) - 1,
            "omitted_top_level_field_count": len(omitted_keys),
            "omitted_top_level_fields": omitted_keys[:20],
        })
    return compacted


def _source_qa_segment_payload(segment, index):
    return {
        "index": index,
        "start": round(float(segment["start"]), 3),
        "end": round(float(segment["end"]), 3),
        "duration": round(
            float(segment["end"]) - float(segment["start"]),
            3,
        ),
        "text": normalize_subtitle_text(segment["text"]),
    }


def build_source_qa_prompt(
    transcript_text,
    segments,
    alignment_info,
    source_language,
    target_language,
    max_alignment_metadata_chars=16000,
):
    payload = {
        "full_api_transcript": normalize_subtitle_text(transcript_text),
        "source_language": source_language,
        "target_language": target_language,
        "alignment_metadata": compact_source_qa_alignment_metadata(
            alignment_info,
            max_chars=max_alignment_metadata_chars,
        ),
        "aligned_source_segments": [
            _source_qa_segment_payload(segment, index)
            for index, segment in enumerate(segments)
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_source_qa_batch_prompt(
    segments,
    alignment_info,
    source_language,
    target_language,
    batch_start,
    batch_end,
    context_segments=2,
    max_alignment_metadata_chars=16000,
):
    total_segments = len(segments)
    batch_start = max(0, int(batch_start))
    batch_end = min(total_segments, int(batch_end))
    context_segments = max(0, min(10, int(context_segments or 0)))
    context_start = max(0, batch_start - context_segments)
    context_end = min(total_segments, batch_end + context_segments)
    review_items = [
        _source_qa_segment_payload(segments[index], index)
        for index in range(batch_start, batch_end)
    ]
    context_before = [
        _source_qa_segment_payload(segments[index], index)
        for index in range(context_start, batch_start)
    ]
    context_after = [
        _source_qa_segment_payload(segments[index], index)
        for index in range(batch_end, context_end)
    ]
    transcript_excerpt = normalize_subtitle_text(
        " ".join(
            segment.get("text", "")
            for segment in segments[context_start:context_end]
        )
    )
    payload = {
        "document_scope": {
            "total_segments": total_segments,
            "batch_start_index": batch_start,
            "batch_end_index_exclusive": batch_end,
            "review_segment_count": len(review_items),
            "transcript_exactness_checked_deterministically": True,
        },
        "source_language": source_language,
        "target_language": target_language,
        "transcript_excerpt_with_neighbor_context": transcript_excerpt,
        "alignment_metadata": compact_source_qa_alignment_metadata(
            alignment_info,
            max_chars=max_alignment_metadata_chars,
        ),
        "context_before": context_before,
        "aligned_source_segments": review_items,
        "context_after": context_after,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_source_qa_prompt_batches(
    transcript_text,
    segments,
    alignment_info,
    source_language,
    target_language,
    max_prompt_chars=48000,
    max_segments_per_batch=200,
    context_segments=2,
    max_alignment_metadata_chars=16000,
):
    max_prompt_chars = max(8000, min(120000, int(max_prompt_chars or 48000)))
    max_segments_per_batch = max(
        1,
        min(1000, int(max_segments_per_batch or 200)),
    )
    max_alignment_metadata_chars = max(
        2000,
        min(
            max_prompt_chars // 2,
            int(max_alignment_metadata_chars or 16000),
        ),
    )
    full_prompt = build_source_qa_prompt(
        transcript_text,
        segments,
        alignment_info,
        source_language,
        target_language,
        max_alignment_metadata_chars=max_alignment_metadata_chars,
    )
    if (
        len(full_prompt) <= max_prompt_chars
        and len(segments) <= max_segments_per_batch
    ):
        return [{
            "start": 0,
            "end": len(segments),
            "prompt": full_prompt,
            "prompt_chars": len(full_prompt),
            "full_document": True,
        }]

    batches = []
    start = 0
    while start < len(segments):
        low = start + 1
        high = min(len(segments), start + max_segments_per_batch)
        best = None
        while low <= high:
            end = (low + high) // 2
            prompt = build_source_qa_batch_prompt(
                segments,
                alignment_info,
                source_language,
                target_language,
                start,
                end,
                context_segments=context_segments,
                max_alignment_metadata_chars=max_alignment_metadata_chars,
            )
            if len(prompt) <= max_prompt_chars:
                best = (end, prompt)
                low = end + 1
            else:
                high = end - 1
        if best is None:
            single_prompt = build_source_qa_batch_prompt(
                segments,
                alignment_info,
                source_language,
                target_language,
                start,
                start + 1,
                context_segments=0,
                max_alignment_metadata_chars=min(
                    4000,
                    max_alignment_metadata_chars,
                ),
            )
            if len(single_prompt) > max_prompt_chars:
                raise RuntimeError(
                    "One source subtitle cue exceeds the bounded QA prompt "
                    f"limit of {max_prompt_chars} characters."
                )
            best = (start + 1, single_prompt)
        end, prompt = best
        batches.append({
            "start": start,
            "end": end,
            "prompt": prompt,
            "prompt_chars": len(prompt),
            "full_document": False,
        })
        start = end
    return batches


def aggregate_source_qa_batch_reports(batch_reports):
    severity_rank = {"ok": 0, "warning": 1, "fail": 2}
    aggregate_severity = "ok"
    aggregate_accept = True
    aggregate_problems = []
    seen_problems = set()
    summaries = []
    batch_summaries = []
    usages = []

    for batch, report in batch_reports:
        severity = report.get("severity", "fail")
        if severity_rank.get(severity, 2) > severity_rank[aggregate_severity]:
            aggregate_severity = severity
        aggregate_accept = aggregate_accept and bool(
            report.get("accept", False)
        )
        summary = normalize_subtitle_text(report.get("summary", ""))
        if summary and summary not in summaries:
            summaries.append(summary)
        for problem in report.get("problems") or []:
            indexes = problem.get("segment_indexes") or []
            invalid_indexes = [
                index
                for index in indexes
                if (
                    not isinstance(index, int)
                    or index < batch["start"]
                    or index >= batch["end"]
                )
            ]
            if invalid_indexes:
                raise RuntimeError(
                    "OpenAI source subtitle QA returned segment indexes "
                    f"outside its assigned batch: {invalid_indexes}."
                )
            identity = json.dumps(
                problem,
                ensure_ascii=False,
                sort_keys=True,
            )
            if identity not in seen_problems:
                seen_problems.add(identity)
                aggregate_problems.append(problem)
        usage = report.pop("_usage", None)
        if usage:
            usages.append(usage)
        batch_summaries.append({
            "start_segment_index": batch["start"],
            "end_segment_index_exclusive": batch["end"],
            "prompt_chars": batch["prompt_chars"],
            "accept": bool(report.get("accept", False)),
            "severity": severity,
            "summary": summary,
            "problem_count": len(report.get("problems") or []),
        })

    summary_text = (
        f"Reviewed {len(batch_reports)} bounded source-QA batch"
        f"{'es' if len(batch_reports) != 1 else ''}; "
        f"{sum(1 for _, report in batch_reports if report.get('accept'))}/"
        f"{len(batch_reports)} accepted."
    )
    if summaries:
        summary_text += " " + " | ".join(summaries[:6])
    return {
        "accept": aggregate_accept,
        "severity": aggregate_severity,
        "summary": summary_text[:3000],
        "problems": aggregate_problems,
        "batch_count": len(batch_reports),
        "batched": len(batch_reports) > 1,
        "batches": batch_summaries,
        "_usage": sum_numeric_usage_dicts(usages),
    }


def run_source_subtitle_qa(transcript_text, segments, alignment_info, source_lang, target_language, pipeline_config):
    if not pipeline_config.get("qa_enabled", True):
        return {"enabled": False, "accept": True, "severity": "ok", "summary": "QA disabled.", "problems": []}

    provider_id = pipeline_config.get("qa_provider", "openai")
    if provider_id != "openai":
        return {
            "enabled": False,
            "accept": True,
            "severity": "warning",
            "summary": f"QA provider '{provider_id}' is not implemented; QA skipped.",
            "problems": [],
        }

    model = pipeline_config.get("qa_model") or pipeline_config.get("translation_model") or pipeline_config.get("llm_model", "gpt-4o-mini")
    instructions = build_source_qa_instructions(
        source_lang,
        target_language,
        source_dialect=pipeline_config.get("source_dialect", "auto"),
        tiktok_style=bool(pipeline_config.get("tiktok_style", False)),
    )
    batches = build_source_qa_prompt_batches(
        transcript_text,
        segments,
        alignment_info,
        source_lang,
        target_language,
        max_prompt_chars=pipeline_config.get(
            "source_qa_max_prompt_chars",
            config_default("source_qa_max_prompt_chars", 48000),
        ),
        max_segments_per_batch=pipeline_config.get(
            "source_qa_max_segments_per_batch",
            config_default("source_qa_max_segments_per_batch", 200),
        ),
        context_segments=pipeline_config.get(
            "source_qa_context_segments",
            config_default("source_qa_context_segments", 2),
        ),
        max_alignment_metadata_chars=pipeline_config.get(
            "source_qa_max_alignment_metadata_chars",
            config_default(
                "source_qa_max_alignment_metadata_chars",
                16000,
            ),
        ),
    )
    batch_reports = []
    for batch_number, batch in enumerate(batches, start=1):
        response_json = call_openai_structured_json(
            batch["prompt"],
            instructions,
            model,
            source_qa_schema(),
            "subtitle_source_qa",
            api_env_key=pipeline_config.get(
                "openai_api_env_key",
                "OPENAI_API_KEY",
            ),
            error_label=(
                "OpenAI source subtitle QA request "
                f"(batch {batch_number}/{len(batches)})"
            ),
        )
        batch_reports.append((batch, response_json))
    response_json = aggregate_source_qa_batch_reports(batch_reports)
    record_usage_event(
        pipeline_config,
        provider_id,
        model,
        "source_qa",
        usage=response_json.pop("_usage", None),
    )
    response_json["enabled"] = True
    response_json["model"] = model
    response_json["provider"] = provider_id
    return response_json


def reconcile_source_qa_with_deterministic_gates(
    qa_report,
    source_integrity_report=None,
    source_timing_verifier_report=None,
):
    if not qa_report:
        return qa_report

    deterministic_text_ok = bool((source_integrity_report or {}).get("text", {}).get("accept"))
    deterministic_timing_ok = bool((source_timing_verifier_report or {}).get("accept"))
    if not (deterministic_text_ok and deterministic_timing_ok):
        return qa_report

    if bool(qa_report.get("accept", True)) and qa_report.get("severity", "ok") != "fail":
        return qa_report

    reconciled = dict(qa_report)
    original_summary = qa_report.get("summary", "")
    reconciled["accept"] = True
    reconciled["severity"] = "warning"
    reconciled["summary"] = (
        "LLM source QA raised a concern, but deterministic source text integrity "
        "and audio-grounded timing verification both passed. Proceeding with QA as advisory. "
        f"Original QA summary: {original_summary}"
    ).strip()
    reconciled["deterministic_override"] = {
        "source_text_integrity_accept": True,
        "source_timing_verifier_accept": True,
        "original_accept": bool(qa_report.get("accept", True)),
        "original_severity": qa_report.get("severity", "ok"),
        "original_summary": original_summary,
    }
    return reconciled


def enforce_source_qa_policy(qa_report, pipeline_config):
    severity = (qa_report or {}).get("severity", "ok")
    accept = bool((qa_report or {}).get("accept", True))
    policy = pipeline_config.get("qa_policy", config_default("qa_policy", "stop"))
    if severity == "ok" or accept:
        return
    if policy == "warn":
        print(f"Warning: source QA failed but qa_policy=warn. {qa_report.get('summary', '')}")
        return
    raise RuntimeError(f"Source subtitle QA failed: {qa_report.get('summary', '')}")


def source_qa_failed(qa_report):
    if not qa_report:
        return False
    return qa_report.get("severity") == "fail" and not bool(qa_report.get("accept", True))


def translation_semantic_qa_schema():
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "accept": {"type": "boolean"},
            "summary": {"type": "string"},
            "document_meaning_preserved": {"type": "boolean"},
            "problems": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "index": {"type": "integer"},
                        "severity": {"type": "string", "enum": ["warning", "fail"]},
                        "reason": {"type": "string"},
                        "corrected_text": {"type": "string"},
                    },
                    "required": ["index", "severity", "reason", "corrected_text"],
                },
            },
        },
        "required": ["accept", "summary", "document_meaning_preserved", "problems"],
    }


def translation_semantic_prefilter(source_segments, translated_segments):
    problems = []
    for index, (source, translated) in enumerate(zip(source_segments, translated_segments)):
        source_text = normalize_subtitle_text(source.get("text", ""))
        translated_text = normalize_subtitle_text(translated.get("text", ""))
        if source_text and not translated_text:
            problems.append({"index": index, "type": "empty_translation"})
        if index > 0:
            previous_source = normalize_subtitle_text(source_segments[index - 1].get("text", ""))
            previous_translation = normalize_subtitle_text(translated_segments[index - 1].get("text", ""))
            if (
                translated_text
                and translated_text == previous_translation
                and source_text != previous_source
            ):
                problems.append({"index": index, "type": "duplicate_target_for_different_source"})
    return problems


def build_translation_semantic_qa_prompt(source_segments, translated_segments, source_language, target_language):
    return json.dumps({
        "source_language": source_language,
        "target_language": target_language,
        "deterministic_suspicions": translation_semantic_prefilter(source_segments, translated_segments),
        "paired_cues": [
            {
                "index": index,
                "start": round(float(source["start"]), 3),
                "end": round(float(source["end"]), 3),
                "source": normalize_subtitle_text(source.get("text", "")),
                "translation": normalize_subtitle_text(translated.get("text", "")),
            }
            for index, (source, translated) in enumerate(zip(source_segments, translated_segments))
        ],
    }, ensure_ascii=False, indent=2)


def run_translation_semantic_qa(
    source_segments,
    translated_segments,
    source_language,
    target_language,
    pipeline_config,
):
    if not pipeline_config.get("translation_qa_enabled", True):
        return {
            "enabled": False,
            "accept": True,
            "summary": "Translation semantic QA disabled.",
            "document_meaning_preserved": True,
            "problems": [],
        }

    provider_id = pipeline_config.get("qa_provider", "openai")
    if provider_id != "openai":
        return {
            "enabled": False,
            "accept": True,
            "summary": f"Semantic QA provider '{provider_id}' is not implemented; QA skipped.",
            "document_meaning_preserved": True,
            "problems": [],
        }

    model = (
        pipeline_config.get("qa_model")
        or pipeline_config.get("translation_model")
        or pipeline_config.get("llm_model", "gpt-4o-mini")
    )
    instructions = (
        "You are a senior bilingual subtitle translation verifier and repairer. "
        "Judge semantic equivalence, not literal word matching. Idioms, dialect, slang, names, "
        "and implied meaning must be rendered naturally in the target language. Spoken repetitions "
        "must remain repetitions; do not deduplicate them. Evaluate every target cue against the "
        "source cue with the same index while also using the complete sequence for document context. "
        "Subtitle cues may be sentence fragments whose grammar or meaning is completed by an adjacent "
        "cue. Judge the concatenated source and target sequence across those boundaries; never fail a "
        "cue merely because it is an incomplete fragment when the adjacent sequence is correct. "
        "Do not treat a natural stylistic synonym as a semantic defect, and do not replace a precise "
        "correct term with a less precise alternative. "
        "Do not permit meaning to be omitted, invented, duplicated, or moved to the wrong cue. "
        "A deterministic suspicion is evidence to inspect, not automatic proof. For every material "
        "problem, return a complete corrected target-language text for that same cue. Never propose "
        "timestamp, cue-count, split, merge, insertion, or deletion changes. A correction must never "
        "erase a spoken repetition that the current translation preserved. Return JSON only."
    )
    response = call_openai_structured_json(
        build_translation_semantic_qa_prompt(
            source_segments,
            translated_segments,
            source_language,
            target_language,
        ),
        instructions,
        model,
        translation_semantic_qa_schema(),
        "subtitle_translation_semantic_qa",
        api_env_key=pipeline_config.get("openai_api_env_key", "OPENAI_API_KEY"),
        error_label="OpenAI translation semantic QA request",
    )
    record_usage_event(
        pipeline_config,
        provider_id,
        model,
        "translation_qa",
        usage=response.pop("_usage", None),
    )
    response.update({"enabled": True, "provider": provider_id, "model": model})
    return response


def apply_translation_semantic_repairs(source_segments, translated_segments, qa_report):
    repaired = [dict(segment) for segment in translated_segments]
    changed_indexes = []
    for problem in (qa_report or {}).get("problems", []):
        index = problem.get("index")
        corrected_text = normalize_subtitle_text(problem.get("corrected_text", ""))
        if not isinstance(index, int) or not (0 <= index < len(repaired)) or not corrected_text:
            continue
        source_tokens = [
            compact_text_for_alignment(token)
            for token in normalize_subtitle_text(
                source_segments[index].get("text", "")
            ).split()
            if compact_text_for_alignment(token)
        ]
        current_tokens = [
            compact_text_for_alignment(token)
            for token in normalize_subtitle_text(
                repaired[index].get("text", "")
            ).split()
            if compact_text_for_alignment(token)
        ]
        corrected_tokens = [
            compact_text_for_alignment(token)
            for token in corrected_text.split()
            if compact_text_for_alignment(token)
        ]
        source_has_repetition = (
            len(set(source_tokens)) < len(source_tokens)
        )
        current_has_repetition = (
            len(set(current_tokens)) < len(current_tokens)
        )
        corrected_has_repetition = (
            len(set(corrected_tokens)) < len(corrected_tokens)
        )
        if (
            source_has_repetition
            and current_has_repetition
            and not corrected_has_repetition
        ):
            continue
        if corrected_text != normalize_subtitle_text(repaired[index].get("text", "")):
            repaired[index]["text"] = corrected_text
            changed_indexes.append(index)

    # Source cue boundaries are the immutable timing contract.
    for index, source in enumerate(source_segments):
        repaired[index]["start"] = source["start"]
        repaired[index]["end"] = source["end"]
    enforce_translated_subtitle_alignment(
        source_segments,
        repaired,
        label="Semantically repaired translated subtitles",
    )
    return repaired, changed_indexes


def verify_and_repair_translation_semantics(
    source_segments,
    translated_segments,
    source_language,
    target_language,
    pipeline_config,
):
    current = [dict(segment) for segment in translated_segments]
    max_repairs = int(pipeline_config.get("translation_qa_max_repairs", 1) or 0)
    attempts = []
    for repair_attempt in range(max_repairs + 1):
        report = run_translation_semantic_qa(
            source_segments,
            current,
            source_language,
            target_language,
            pipeline_config,
        )
        attempts.append(report)
        if report.get("accept") and report.get("document_meaning_preserved", True):
            final_report = dict(report)
            final_report["attempts"] = [dict(item) for item in attempts]
            final_report["repair_count"] = repair_attempt
            return current, final_report
        if repair_attempt >= max_repairs:
            break
        current, changed_indexes = apply_translation_semantic_repairs(
            source_segments,
            current,
            report,
        )
        report["applied_repair_indexes"] = changed_indexes
        if not changed_indexes:
            break

    final_report = dict(attempts[-1])
    final_report["attempts"] = [dict(item) for item in attempts]
    final_report["repair_count"] = max(0, len(attempts) - 1)
    policy = str(pipeline_config.get("translation_qa_policy", "stop") or "stop").lower()
    if policy == "warn":
        print(f"Warning: translation semantic QA failed: {final_report.get('summary', '')}")
        return current, final_report
    raise TranslationSemanticQAError(
        "Translation semantic QA failed after repair: "
        f"{final_report.get('summary', '')}",
        final_report,
        current,
    )


def timing_only_segments(segments):
    return [
        {
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "text": "",
        }
        for segment in segments or []
        if segment.get("start") is not None
        and segment.get("end") is not None
        and float(segment["end"]) > float(segment["start"])
    ]


def segments_cache_signature(segments):
    normalized = [
        {
            "start": round(float(segment.get("start", 0.0)), 3),
            "end": round(float(segment.get("end", 0.0)), 3),
            "text": normalize_subtitle_text(segment.get("text", "")),
        }
        for segment in segments or []
        if normalize_subtitle_text(segment.get("text", ""))
    ]
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def align_api_transcript_for_timing(
    transcript_text,
    timing_segments,
    timing_mode,
    api_audio_path,
    source_lang,
    device,
    duration_seconds=None,
    model_name=None,
):
    alignment_info = None
    anchor_segments = None
    anchor_alignment_info = None
    if timing_segments and timing_mode in {"precise", "forced", "fuzzy"}:
        try:
            anchor_segments, anchor_alignment_info = align_transcript_to_timing_anchors(
                transcript_text,
                timing_segments,
            )
        except TranscriptAnchorCoverageError:
            raise
        except Exception as e:
            print(f"Warning: canonical timing-anchor reconciliation failed: {e}")
            anchor_segments = align_transcript_to_timing_segments(transcript_text, timing_segments)
            anchor_alignment_info = {
                "engine": "anchor_fuzzy_legacy",
                "canonical_anchor_error": str(e),
                "canonical_anchor_error_type": type(e).__name__,
            }

    if timing_mode in {"precise", "forced"}:
        if anchor_segments:
            rough_segments = anchor_segments
        elif timing_segments:
            rough_segments = align_transcript_to_timing_segments(transcript_text, timing_segments)
        else:
            rough_segments = build_rough_transcript_segments(transcript_text, duration_seconds)
        try:
            model_label = f"'{model_name}' " if model_name else ""
            print(
                f"Forced-aligning {model_label}API transcript text to the audio waveform "
                "with WhisperX."
            )
            segments, alignment_info = forced_align_transcript_to_audio(
                str(api_audio_path),
                transcript_text,
                rough_segments,
                source_lang,
                device=device,
            )
            if anchor_segments:
                drift_report = timing_drift_report(segments, anchor_segments)
                alignment_info["anchor_drift_guardrail"] = drift_report
                if anchor_alignment_info:
                    alignment_info["anchor_alignment"] = anchor_alignment_info
                if not drift_report.get("accept", True):
                    audio_words = alignment_info.get("_aligned_word_segments") or []
                    selected_candidate = None
                    whisperx_audio_report = None
                    anchor_audio_report = None
                    if audio_words:
                        (
                            selected_candidate,
                            whisperx_audio_report,
                            anchor_audio_report,
                        ) = choose_audio_verified_timing_candidate(
                            transcript_text,
                            segments,
                            anchor_segments,
                            audio_words,
                            reference_speech_segments=timing_segments,
                        )
                        alignment_info["whisperx_audio_timing_report"] = whisperx_audio_report
                        alignment_info["anchor_audio_timing_report"] = anchor_audio_report
                    if selected_candidate == "whisperx":
                        print(
                            "Warning: WhisperX timing disagrees with native audio timestamp anchors "
                            f"(max drift {drift_report.get('max_start_drift')}s), but audio-grounded "
                            "verification prefers WhisperX timing. Keeping WhisperX timing."
                        )
                    else:
                        if selected_candidate == "anchor":
                            print(
                                "Warning: WhisperX timing disagrees with native audio timestamp anchors "
                                f"(max drift {drift_report.get('max_start_drift')}s). "
                                "Audio-grounded verification prefers anchor timing."
                            )
                        else:
                            print(
                                "Warning: WhisperX timing disagrees with native audio timestamp anchors "
                                f"(max drift {drift_report.get('max_start_drift')}s). "
                                "Using anchor-timestamp alignment because no audio word report was available."
                            )
                        rejected_alignment = dict(alignment_info)
                        rejected_alignment.pop("_aligned_word_segments", None)
                        if "whisperx_audio_timing_report" in rejected_alignment:
                            rejected_alignment["whisperx_audio_timing_report"] = whisperx_audio_report
                        if "anchor_audio_timing_report" in rejected_alignment:
                            rejected_alignment["anchor_audio_timing_report"] = anchor_audio_report
                        segments = anchor_segments
                        alignment_info = {
                            "engine": "anchor_canonical_audio_verified"
                            if selected_candidate == "anchor"
                            else (
                                "anchor_canonical_guardrail"
                                if anchor_alignment_info and anchor_alignment_info.get("engine") == "anchor_canonical_reconciliation"
                                else "anchor_fuzzy_guardrail"
                            ),
                            "rejected_engine": "whisperx",
                            "rejected_alignment": rejected_alignment,
                            "anchor_alignment": anchor_alignment_info,
                            "anchor_drift_guardrail": drift_report,
                            "anchor_audio_timing_report": anchor_audio_report,
                            "whisperx_audio_timing_report": whisperx_audio_report,
                        }
        except Exception as e:
            if timing_mode == "forced" or not timing_segments:
                raise
            print(f"Warning: forced alignment unavailable or failed: {e}")
            print(
                "Falling back to API-text alignment on audio timestamp anchors."
            )
            segments = anchor_segments or rough_segments
            alignment_info = {
                "engine": (
                    "anchor_canonical_fallback"
                    if anchor_alignment_info and anchor_alignment_info.get("engine") == "anchor_canonical_reconciliation"
                    else "fuzzy_fallback"
                ),
                "forced_alignment_error": str(e),
                "forced_alignment_error_type": type(e).__name__,
                "anchor_alignment": anchor_alignment_info,
            }
    elif timing_mode == "fuzzy":
        print(
            "Aligning API transcript text to timing anchors with canonical "
            "token reconciliation."
        )
        segments = anchor_segments or align_transcript_to_timing_segments(transcript_text, timing_segments)
        alignment_info = anchor_alignment_info or {"engine": "fuzzy"}
    elif timing_mode == "proportional":
        print(
            "Warning: proportional API transcript timing can drift. "
            "Use precise for better text/timing alignment."
        )
        segments = align_transcript_to_timing_segments_proportional(transcript_text, timing_segments)
        alignment_info = {"engine": "proportional"}
    elif timing_mode == "local_whisper":
        print(
            "Using local Whisper segment text for timing accuracy; "
            "API transcript saved as reference."
        )
        segments = timing_segments
        alignment_info = {"engine": "local_whisper_text"}
    else:
        raise RuntimeError(
            "Unsupported api_transcript_timing_mode. Use precise, forced, fuzzy, local_whisper, or proportional."
        )
    return segments, alignment_info


HEX_COLOR_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}$")


def clamp_int(value, minimum, maximum, fallback):
    try:
        number = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(minimum, min(maximum, number))


def clamp_float(value, minimum, maximum, fallback):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(minimum, min(maximum, number))


def valid_hex_color(value, fallback):
    if isinstance(value, str) and HEX_COLOR_PATTERN.match(value.strip()):
        return value.strip().upper()
    return fallback


def probe_video_metadata(video_path):
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height:format=duration",
        "-of", "json",
        str(video_path),
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
        data = json.loads(result.stdout)
        stream = (data.get("streams") or [{}])[0]
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        duration = float((data.get("format") or {}).get("duration") or 0)
    except Exception:
        width = height = 0
        try:
            duration = get_video_duration(str(video_path)) or 0
        except Exception:
            duration = 0

    if width and height:
        aspect_ratio = round(width / height, 4)
        if aspect_ratio < 0.9:
            orientation = "vertical_short_form"
        elif aspect_ratio > 1.45:
            orientation = "widescreen"
        else:
            orientation = "square_or_standard"
    else:
        aspect_ratio = None
        orientation = "unknown"

    return {
        "width": width,
        "height": height,
        "duration_seconds": round(duration, 3) if duration else None,
        "aspect_ratio": aspect_ratio,
        "orientation": orientation,
    }


SUBTITLE_MODES = {"auto", "normal", "tiktok"}


def normalize_subtitle_mode(value=None, legacy_tiktok_style=False):
    """Normalize the new mode while accepting the former checkbox value."""
    mode = str(value or "").strip().lower()
    if mode in SUBTITLE_MODES:
        return mode
    return "tiktok" if bool(legacy_tiktok_style) else "normal"


def resolve_subtitle_mode(video_path, pipeline_config=None, video_metadata=None):
    """Resolve Auto per video; portrait videos use short-form subtitle formatting."""
    config = pipeline_config or {}
    requested = normalize_subtitle_mode(
        config.get("subtitle_mode"),
        legacy_tiktok_style=config.get("tiktok_style", False),
    )
    metadata = video_metadata or probe_video_metadata(video_path)
    effective = requested
    if requested == "auto":
        effective = (
            "tiktok"
            if metadata.get("orientation") == "vertical_short_form"
            else "normal"
        )
    return requested, effective, metadata


def sample_video_frames(video_path, output_root, sample_count=5, frame_width=960):
    output_dir = Path(output_root) / "_visual_style_frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = probe_video_metadata(video_path)
    duration = metadata.get("duration_seconds") or get_video_duration(str(video_path)) or 0
    sample_count = clamp_int(sample_count, 1, 8, 5)
    frame_width = clamp_int(frame_width, 320, 1280, 960)
    if duration <= 0:
        times = [0]
    else:
        times = [duration * (index + 1) / (sample_count + 1) for index in range(sample_count)]

    frame_paths = []
    for index, timestamp in enumerate(times, 1):
        frame_path = output_dir / f"frame_{index:02d}.jpg"
        command = [
            "ffmpeg",
            "-y",
            "-ss", f"{timestamp:.3f}",
            "-i", str(video_path),
            "-frames:v", "1",
            "-vf", f"scale={frame_width}:-2:force_original_aspect_ratio=decrease",
            "-q:v", "3",
            str(frame_path),
        ]
        try:
            subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=True,
                **hidden_subprocess_kwargs(),
            )
        except subprocess.CalledProcessError:
            continue
        if frame_path.exists() and frame_path.stat().st_size > 0:
            frame_paths.append(frame_path)
    return frame_paths, metadata


def encode_image_data_url(path):
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"


def visual_style_schema():
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "accept": {"type": "boolean"},
            "summary": {"type": "string"},
            "style": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "font_name": {"type": "string"},
                    "font_scale": {"type": "string", "enum": ["small", "normal", "large", "extra_large"]},
                    "primary_color": {"type": "string"},
                    "outline_color": {"type": "string"},
                    "back_color": {"type": "string"},
                    "outline_width": {"type": "integer"},
                    "shadow": {"type": "integer"},
                    "border_style": {"type": "integer", "enum": [1, 3, 4]},
                },
                "required": [
                    "font_name",
                    "font_scale",
                    "primary_color",
                    "outline_color",
                    "back_color",
                    "outline_width",
                    "shadow",
                    "border_style",
                ],
            },
            "reasons": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["accept", "summary", "style", "reasons"],
    }


def visual_style_render_review_schema():
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "accept": {"type": "boolean"},
            "summary": {"type": "string"},
            "selected_candidate_id": {"type": "string"},
            "problems": {
                "type": "array",
                "items": {"type": "string"},
            },
            "reasons": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["accept", "summary", "selected_candidate_id", "problems", "reasons"],
    }


def build_visual_style_prompt(video_metadata, target_language, base_style, sample_count):
    bounds = visual_style_bounds(video_metadata, tiktok_style=bool((base_style or {}).get("tiktok_style", False)))
    payload = {
        "task": "Choose subtitle visual styling for this specific video.",
        "video_metadata": video_metadata,
        "target_language": target_language,
        "current_style_baseline": base_style or {},
        "hard_safety_bounds": {
            "font_size_min": bounds["min_font"],
            "font_size_max": bounds["max_font"],
            "geometry_policy": "Do not choose subtitle position, alignment, margins, or raw pixel geometry. SubGen will place subtitles bottom-center in the lower half using deterministic safe-area rules.",
        },
        "sampled_frame_count": sample_count,
        "style_rules": [
            "Optimize readability over aesthetics.",
            "Use high contrast against the actual video backgrounds in the sampled frames.",
            "For vertical/short-form video, use larger text and enough bottom margin for phone UI overlays.",
            "For widescreen video, avoid overly large text; keep subtitles readable without covering faces or important content.",
            "Prefer white text with black outline or box unless the frames make another high-contrast choice clearly better.",
            "Choose font_scale, contrast colors, outline, shadow, and background box only.",
            "Do not recommend top placement, center placement, margins, coordinates, or ASS alignment codes.",
            "Return only supported style fields. Geometry is intentionally excluded from the schema.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_visual_render_review_prompt(video_metadata, candidates, sample_count):
    payload = {
        "task": "Select the subtitle style candidate whose rendered preview is fully visible and readable.",
        "video_metadata": video_metadata,
        "sampled_preview_count": sample_count,
        "hard_requirements": [
            "Reject any candidate with subtitle text above the vertical center of the frame.",
            "Reject any candidate with clipped, cropped, or partially off-screen text.",
            "Reject any candidate where words are hidden by frame edges or UI overlays.",
            "Prefer the lowest readable subtitle position that keeps all words visible.",
            "Choose only one selected_candidate_id from the supplied candidates.",
        ],
        "candidates": [
            {
                "candidate_id": item["candidate_id"],
                "style": item["style"],
            }
            for item in candidates
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def visual_style_scale_neighbors(font_scale):
    scale_order = ["small", "normal", "large", "extra_large"]
    if font_scale not in scale_order:
        font_scale = "normal"
    index = scale_order.index(font_scale)
    selected = [font_scale]
    if index > 0:
        selected.append(scale_order[index - 1])
    if index < len(scale_order) - 1:
        selected.append(scale_order[index + 1])
    return selected


def visual_style_bounds(video_metadata=None, tiktok_style=False):
    metadata = video_metadata or {}
    width = int(metadata.get("width") or 1920)
    height = int(metadata.get("height") or 1080)
    short_side = max(1, min(width, height))
    vertical = height > width

    if tiktok_style or vertical:
        min_font = max(22, round(short_side * 0.030))
        max_font = min(64, max(32, round(short_side * 0.060)))
        max_margin_v = max(32, round(height * 0.120))
    else:
        min_font = max(18, round(short_side * 0.022))
        max_font = min(42, max(28, round(short_side * 0.042)))
        max_margin_v = max(24, round(height * 0.105))

    return {
        "width": width,
        "height": height,
        "min_font": min_font,
        "max_font": max_font,
        "min_margin_v": max(12, round(height * 0.025)),
        "max_margin_v": max_margin_v,
        "max_margin_h": max(20, round(width * 0.080)),
        "allowed_alignments": {1, 2, 3},
    }


ARABIC_SCRIPT_SUBTITLE_LANGUAGES = {"ar", "fa", "ur", "ps", "sd"}


def uses_arabic_script_subtitle_font(lang_code):
    return str(lang_code or "").lower().split("-")[0] in ARABIC_SCRIPT_SUBTITLE_LANGUAGES


def bundled_subtitle_fonts_dir():
    return Path(__file__).resolve().parent


def escape_ffmpeg_filter_path(path):
    return str(Path(path)).replace("\\", "/").replace(":", "\\:")


def apply_script_safe_subtitle_style(style_config, lang_code, video_metadata=None):
    style = dict(style_config or {})
    if not uses_arabic_script_subtitle_font(lang_code):
        return style

    font_path = bundled_subtitle_fonts_dir() / "Amiri-Regular.ttf"
    if font_path.exists():
        style["font_name"] = "Amiri"

    metadata = video_metadata or {}
    width = int(metadata.get("width") or 1920)
    height = int(metadata.get("height") or 1080)
    short_side = max(1, min(width, height))
    tiktok_style = bool(style.get("tiktok_style", False))
    if tiktok_style:
        max_rtl_font = max(26, min(56, round(short_side * 0.065)))
        default_font = min(max_rtl_font, max(28, round(short_side * 0.050)))
    else:
        max_rtl_font = max(24, min(34, round(height * 0.030)))
        default_font = min(28, max_rtl_font)
    style["font_size"] = clamp_int(style.get("font_size"), 18, max_rtl_font, default_font)
    style["outline_width"] = clamp_int(style.get("outline_width"), 1, 3, 1)
    style["shadow"] = clamp_int(style.get("shadow"), 0, 2, 1)
    style["alignment"] = 2
    return style


def validate_visual_style(
    raw_style,
    base_style=None,
    video_metadata=None,
    tiktok_style=False,
    allow_position_change=False,
):
    base = dict(base_style or {})
    style = dict(base)
    raw_style = raw_style or {}
    bounds = visual_style_bounds(video_metadata, tiktok_style=tiktok_style)
    style["font_name"] = str(raw_style.get("font_name") or base.get("font_name") or "Arial").strip() or "Arial"
    font_scale = str(raw_style.get("font_scale") or base.get("font_scale") or ("large" if tiktok_style else "normal")).strip().lower()
    scale_fraction = {
        "small": 0.20,
        "normal": 0.42,
        "large": 0.68,
        "extra_large": 0.88,
    }.get(font_scale, 0.42)
    scaled_font_size = round(bounds["min_font"] + (bounds["max_font"] - bounds["min_font"]) * scale_fraction)
    style["font_size"] = clamp_int(
        raw_style.get("font_size") if "font_size" in raw_style else scaled_font_size,
        bounds["min_font"],
        bounds["max_font"],
        clamp_int(base.get("font_size"), bounds["min_font"], bounds["max_font"], bounds["min_font"]),
    )
    style["font_scale"] = font_scale
    style["primary_color"] = valid_hex_color(raw_style.get("primary_color"), base.get("primary_color", "#FFFFFF"))
    style["outline_color"] = valid_hex_color(raw_style.get("outline_color"), base.get("outline_color", "#000000"))
    style["back_color"] = valid_hex_color(raw_style.get("back_color"), base.get("back_color", "#000000"))
    style["outline_width"] = clamp_float(
        raw_style.get("outline_width"),
        0.0,
        8.0,
        clamp_float(base.get("outline_width"), 0.0, 8.0, 1.0),
    )
    style["shadow"] = clamp_int(raw_style.get("shadow"), 0, 4, int(base.get("shadow", 1)))
    style["bg_opacity"] = clamp_int(
        raw_style.get("bg_opacity"),
        0,
        100,
        clamp_int(base.get("bg_opacity"), 0, 100, 70),
    )
    raw_border_style = raw_style.get("border_style")
    style["border_style"] = clamp_int(raw_border_style, 1, 4, int(base.get("border_style", 3)))
    if raw_border_style not in {1, 3, 4}:
        style["border_style"] = int(base.get("border_style", 3))
    raw_alignment = raw_style.get("alignment")
    base_alignment = int(base.get("alignment", 2) or 2)
    proposed_alignment = clamp_int(raw_alignment, 1, 9, base_alignment)
    if allow_position_change and proposed_alignment in bounds["allowed_alignments"]:
        style["alignment"] = proposed_alignment
    elif base_alignment in bounds["allowed_alignments"]:
        style["alignment"] = base_alignment
    else:
        style["alignment"] = 2
    style["margin_v"] = clamp_int(
        raw_style.get("margin_v"),
        bounds["min_margin_v"],
        bounds["max_margin_v"],
        clamp_int(base.get("margin_v"), bounds["min_margin_v"], bounds["max_margin_v"], bounds["min_margin_v"]),
    )
    style["margin_l"] = clamp_int(
        raw_style.get("margin_l"),
        0,
        bounds["max_margin_h"],
        clamp_int(base.get("margin_l"), 0, bounds["max_margin_h"], 20),
    )
    style["margin_r"] = clamp_int(
        raw_style.get("margin_r"),
        0,
        bounds["max_margin_h"],
        clamp_int(base.get("margin_r"), 0, bounds["max_margin_h"], 20),
    )
    return style


def make_visual_style_candidates(raw_style, base_style, video_metadata=None, tiktok_style=False):
    """Build a small set of concrete ASS styles for renderer-backed visual review."""
    bounds = visual_style_bounds(video_metadata, tiktok_style=tiktok_style)
    base_candidate = validate_visual_style(
        raw_style,
        base_style,
        video_metadata,
        tiktok_style=tiktok_style,
        allow_position_change=False,
    )
    font_scales = visual_style_scale_neighbors(base_candidate.get("font_scale", "normal"))
    margin_candidates = [
        bounds["min_margin_v"],
        clamp_int(round(bounds["height"] * 0.045), bounds["min_margin_v"], bounds["max_margin_v"], bounds["min_margin_v"]),
        clamp_int(round(bounds["height"] * 0.070), bounds["min_margin_v"], bounds["max_margin_v"], bounds["min_margin_v"]),
    ]

    candidates = []
    seen = set()
    for font_scale in font_scales:
        for margin_v in margin_candidates:
            candidate_input = dict(raw_style or {})
            candidate_input["font_scale"] = font_scale
            candidate_input["margin_v"] = margin_v
            candidate_input["alignment"] = 2
            style = validate_visual_style(
                candidate_input,
                base_candidate,
                video_metadata,
                tiktok_style=tiktok_style,
                allow_position_change=True,
            )
            style["alignment"] = 2
            style["margin_v"] = margin_v
            key = (
                style.get("font_name"),
                style.get("font_size"),
                style.get("primary_color"),
                style.get("outline_color"),
                style.get("back_color"),
                style.get("outline_width"),
                style.get("shadow"),
                style.get("border_style"),
                style.get("alignment"),
                style.get("margin_v"),
                style.get("margin_l"),
                style.get("margin_r"),
            )
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                "candidate_id": f"style_{len(candidates) + 1}",
                "style": style,
            })
            if len(candidates) >= 5:
                return candidates
    return candidates or [{"candidate_id": "style_1", "style": base_candidate}]


def select_subtitle_preview_times(srt_path, max_samples=2):
    return [
        (float(seg["start"]) + float(seg["end"])) / 2
        for seg in select_subtitle_preview_segments(srt_path, max_samples=max_samples)
    ]


def select_subtitle_preview_segments(srt_path, max_samples=2):
    try:
        segments = parse_srt(srt_path)
    except Exception:
        return []
    usable = [
        seg for seg in segments
        if str(seg.get("text", "")).strip() and (float(seg.get("end", 0)) - float(seg.get("start", 0))) >= 0.35
    ]
    if not usable:
        return []
    positions = [0.35, 0.65] if max_samples <= 2 else [0.25, 0.50, 0.75]
    chosen = []
    for position in positions[:max_samples]:
        index = min(len(usable) - 1, max(0, round((len(usable) - 1) * position)))
        seg = usable[index]
        midpoint = (float(seg["start"]) + float(seg["end"])) / 2
        if not any(abs(midpoint - ((float(old["start"]) + float(old["end"])) / 2)) < 1.0 for old in chosen):
            chosen.append(seg)
    return chosen


def subtitle_filter_for_style(srt_filename_relative, style_config, video_metadata=None, lang_code=None):
    metadata = video_metadata or {}
    original_size = f"{int(metadata.get('width') or 1920)}x{int(metadata.get('height') or 1080)}"
    style_config = apply_script_safe_subtitle_style(style_config, lang_code, metadata)
    style_config = validate_visual_style(
        style_config,
        style_config,
        metadata,
        tiktok_style=bool((style_config or {}).get("tiktok_style", False)),
        allow_position_change=False,
    )
    font_name = style_config.get("font_name", "Arial")
    font_size = style_config.get("font_size", 20)
    primary_color = hex_to_ass_color(style_config.get("primary_color", "#FFFFFF"))
    text_outline_color = hex_to_ass_color(style_config.get("outline_color", "#000000"))
    background_color = hex_to_ass_color(
        style_config.get("back_color", "#000000"),
        style_config.get("bg_opacity", 70),
    )
    outline_width = style_config.get("outline_width", 1)
    shadow = style_config.get("shadow", 1)
    border_style = style_config.get("border_style", 3)
    alignment = style_config.get("alignment", 2)
    margin_v = style_config.get("margin_v", 30)
    margin_l = style_config.get("margin_l", 20)
    margin_r = style_config.get("margin_r", 20)
    fonts_dir = escape_ffmpeg_filter_path(bundled_subtitle_fonts_dir())
    fonts_part = f"fontsdir='{fonts_dir}':" if font_name == "Amiri" else ""
    return (
        f"subtitles=filename='{srt_filename_relative}':"
        f"original_size={original_size}:"
        f"charenc=UTF-8:"
        f"{fonts_part}"
        f"force_style='FontName={font_name},"
        f"FontSize={font_size},"
        f"PrimaryColour={primary_color},"
        f"OutlineColour={text_outline_color},"
        f"BackColour={background_color},"
        f"Shadow={shadow},"
        f"Outline={outline_width},"
        f"BorderStyle={border_style},"
        f"Alignment={alignment},"
        f"MarginV={margin_v},"
        f"MarginL={margin_l},"
        f"MarginR={margin_r}'"
    )


def render_visual_style_previews(video_path, srt_path, output_root, candidates, video_metadata=None, max_samples=2, lang_code=None):
    video_path_obj = Path(video_path).resolve()
    srt_path_obj = Path(srt_path).resolve()
    preview_root = Path(output_root).resolve() / "_visual_style_previews"
    preview_root.mkdir(parents=True, exist_ok=True)
    cwd_dir = video_path_obj.parent
    preview_segments = select_subtitle_preview_segments(srt_path_obj, max_samples=max_samples)
    if not preview_segments:
        return []

    rendered = []
    for candidate in candidates:
        paths = []
        for index, segment in enumerate(preview_segments, 1):
            timestamp = (float(segment["start"]) + float(segment["end"])) / 2
            duration = max(2.0, min(5.0, float(segment["end"]) - float(segment["start"])))
            preview_srt_path = preview_root / f"{candidate['candidate_id']}_{index}.srt"
            write_srt(
                [{
                    "start": 0.0,
                    "end": duration,
                    "text": segment["text"],
                }],
                preview_srt_path,
            )
            try:
                preview_srt_relative = preview_srt_path.relative_to(cwd_dir).as_posix()
            except ValueError:
                preview_srt_relative = str(preview_srt_path).replace("\\", "/").replace(":", "\\:")
            vf_arg = subtitle_filter_for_style(
                preview_srt_relative,
                candidate["style"],
                video_metadata,
                lang_code=lang_code,
            )
            output_path = preview_root / f"{candidate['candidate_id']}_{index}.jpg"
            command = [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-ss",
                f"{max(0.0, timestamp):.3f}",
                "-i",
                video_path_obj.name,
                "-frames:v",
                "1",
                "-vf",
                vf_arg,
                str(output_path),
            ]
            result = subprocess.run(
                command,
                cwd=cwd_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                **hidden_subprocess_kwargs(),
            )
            if result.returncode == 0 and output_path.exists():
                paths.append(output_path)
        if paths:
            rendered.append({
                "candidate_id": candidate["candidate_id"],
                "style": candidate["style"],
                "preview_paths": paths,
            })
    return rendered


def run_visual_style_render_review(video_path, srt_path, output_root, video_metadata, candidates, pipeline_config, lang_code=None):
    rendered_candidates = render_visual_style_previews(
        video_path,
        srt_path,
        output_root,
        candidates,
        video_metadata=video_metadata,
        max_samples=pipeline_config.get("visual_style_preview_samples", 2),
        lang_code=lang_code,
    )
    if not rendered_candidates:
        return None

    provider_id = pipeline_config.get("visual_style_provider", "openai")
    api_env_key = pipeline_config.get("openai_api_env_key", "OPENAI_API_KEY")
    model = pipeline_config.get("visual_style_model") or pipeline_config.get("qa_model") or pipeline_config.get("llm_model", "gpt-4o-mini")
    prompt = build_visual_render_review_prompt(video_metadata, rendered_candidates, len(rendered_candidates[0]["preview_paths"]))
    content = [{"type": "input_text", "text": prompt}]
    for candidate in rendered_candidates:
        content.append({"type": "input_text", "text": f"candidate_id: {candidate['candidate_id']}"})
        content.extend({"type": "input_image", "image_url": encode_image_data_url(path)} for path in candidate["preview_paths"])

    response_json = call_openai_structured_json(
        prompt,
        (
            "You are reviewing actual ffmpeg/libass rendered subtitle previews. "
            "Select the candidate that is fully visible, readable, and lowest on screen without covering important content. "
            "Reject candidates whose subtitles are above the vertical center or clipped. Return strict JSON only."
        ),
        model,
        visual_style_render_review_schema(),
        "subtitle_visual_style_render_review",
        api_env_key=api_env_key,
        error_label="OpenAI rendered subtitle style review request",
        input_payload=[{"role": "user", "content": content}],
    )
    record_usage_event(
        pipeline_config,
        provider_id,
        model,
        "visual_style_render_review",
        usage=response_json.pop("_usage", None),
    )
    response_json["rendered_candidates"] = [
        {
            "candidate_id": item["candidate_id"],
            "style": item["style"],
            "preview_paths": [str(path) for path in item["preview_paths"]],
        }
        for item in rendered_candidates
    ]
    return response_json


def run_visual_style_qa(video_path, output_root, target_language, base_style, pipeline_config, srt_path=None):
    fallback_metadata = {}
    try:
        fallback_metadata = probe_video_metadata(video_path)
    except Exception:
        fallback_metadata = {}
    base_style = apply_script_safe_subtitle_style(base_style, target_language, fallback_metadata)

    if not pipeline_config.get("visual_style_enabled", True):
        return {
            "enabled": False,
            "accept": True,
            "summary": "Visual style QA disabled.",
            "style": validate_visual_style(
                base_style,
                base_style,
                fallback_metadata,
                tiktok_style=bool(pipeline_config.get("tiktok_style", False)),
                allow_position_change=False,
            ),
            "reasons": [],
        }

    provider_id = pipeline_config.get("visual_style_provider", "openai")
    if provider_id != "openai":
        return {
            "enabled": False,
            "accept": True,
            "summary": f"Visual style provider '{provider_id}' is not implemented; using configured style.",
            "style": validate_visual_style(
                base_style,
                base_style,
                fallback_metadata,
                tiktok_style=bool(pipeline_config.get("tiktok_style", False)),
                allow_position_change=False,
            ),
            "reasons": [],
        }

    api_env_key = pipeline_config.get("openai_api_env_key", "OPENAI_API_KEY")
    if not os.environ.get(api_env_key):
        return {
            "enabled": False,
            "accept": True,
            "summary": f"{api_env_key} is not configured; using configured style.",
            "style": validate_visual_style(
                base_style,
                base_style,
                fallback_metadata,
                tiktok_style=bool(pipeline_config.get("tiktok_style", False)),
                allow_position_change=False,
            ),
            "reasons": [],
        }

    frame_paths, metadata = sample_video_frames(
        video_path,
        output_root,
        sample_count=pipeline_config.get("visual_style_sample_count", 5),
        frame_width=pipeline_config.get("visual_style_frame_width", 960),
    )
    base_style = apply_script_safe_subtitle_style(base_style, target_language, metadata or fallback_metadata)
    if not frame_paths:
        return {
            "enabled": False,
            "accept": True,
            "summary": "Could not sample video frames; using configured style.",
            "style": validate_visual_style(
                base_style,
                base_style,
                metadata,
                tiktok_style=bool(pipeline_config.get("tiktok_style", False)),
                allow_position_change=False,
            ),
            "reasons": [],
        }

    model = pipeline_config.get("visual_style_model") or pipeline_config.get("qa_model") or pipeline_config.get("llm_model", "gpt-4o-mini")
    prompt = build_visual_style_prompt(metadata, target_language, base_style, len(frame_paths))
    content = [{"type": "input_text", "text": prompt}]
    content.extend({"type": "input_image", "image_url": encode_image_data_url(path)} for path in frame_paths)
    response_json = call_openai_structured_json(
        prompt,
        (
            "You are a senior subtitle readability and accessibility reviewer. "
            "View the sampled video frames and choose subtitle visual settings that will remain readable "
            "across the video. Return strict JSON only."
        ),
        model,
        visual_style_schema(),
        "subtitle_visual_style",
        api_env_key=api_env_key,
        error_label="OpenAI visual subtitle style request",
        input_payload=[{"role": "user", "content": content}],
    )
    record_usage_event(
        pipeline_config,
        provider_id,
        model,
        "visual_style",
        usage=response_json.pop("_usage", None),
    )
    response_json["enabled"] = True
    response_json["provider"] = provider_id
    response_json["model"] = model
    response_json["video_metadata"] = metadata
    response_json["style"] = validate_visual_style(
        apply_script_safe_subtitle_style(response_json.get("style"), target_language, metadata),
        base_style,
        metadata,
        tiktok_style=bool(pipeline_config.get("tiktok_style", False)),
        allow_position_change=False,
    )
    response_json["style_candidates"] = make_visual_style_candidates(
        response_json.get("style"),
        base_style,
        metadata,
        tiktok_style=bool(pipeline_config.get("tiktok_style", False)),
    )
    if srt_path and pipeline_config.get("visual_style_preview_review", True):
        render_review = run_visual_style_render_review(
            video_path,
            srt_path,
            output_root,
            metadata,
            response_json["style_candidates"],
            pipeline_config,
            lang_code=target_language,
        )
        if render_review:
            response_json["render_review"] = render_review
            selected_id = render_review.get("selected_candidate_id")
            selected_candidate = next(
                (item for item in response_json["style_candidates"] if item["candidate_id"] == selected_id),
                None,
            )
            if render_review.get("accept", False) and selected_candidate:
                response_json["style"] = selected_candidate["style"]
                response_json["summary"] = render_review.get("summary") or response_json.get("summary", "")
            elif not render_review.get("accept", True):
                response_json["accept"] = False
                response_json["summary"] = render_review.get("summary") or response_json.get("summary", "")
    response_json["sampled_frames"] = [str(path) for path in frame_paths]
    return response_json


def resolve_visual_style_for_video(video_path, output_root, target_language, base_style, pipeline_config, srt_path=None):
    if not pipeline_config.get("visual_style_enabled", True):
        return apply_script_safe_subtitle_style(base_style, target_language), None

    print("\nVisual style QA: sampling video frames and choosing subtitle readability settings...")
    report = run_visual_style_qa(
        video_path,
        output_root,
        target_language,
        base_style,
        pipeline_config,
        srt_path=srt_path,
    )
    print(f"Visual style QA: {report.get('summary', '')}")
    if report.get("accept", True):
        return apply_script_safe_subtitle_style(report.get("style") or base_style, target_language), report
    return apply_script_safe_subtitle_style(base_style, target_language), report


def extract_response_text(response_data):
    if response_data.get("output_text"):
        return response_data["output_text"]

    output_parts = []
    for item in response_data.get("output", []):
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("text"):
                output_parts.append(content["text"])

    return "\n".join(output_parts)


def call_openai_structured_json(
    prompt,
    instructions,
    model,
    schema,
    schema_name,
    api_env_key="OPENAI_API_KEY",
    error_label="OpenAI structured request",
    input_payload=None,
):
    api_key = require_openai_api_key(api_env_key)
    request_payload = {
        "model": model,
        "instructions": instructions,
        "input": input_payload if input_payload is not None else prompt,
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": schema,
                "strict": True,
            }
        },
    }

    import time
    import random

    max_retries = 3
    base_delay = 2.0
    response_data = None

    for attempt in range(max_retries + 1):
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(request_payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                response_data = json.loads(response.read().decode("utf-8"))
                break
        except urllib.error.HTTPError as e:
            if (e.code == 429 or 500 <= e.code < 600) and attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0.1, 1.0)
                print(f"OpenAI translation HTTP {e.code} error. Retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(delay)
                continue
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{error_label} failed: HTTP {e.code}: {error_body}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0.1, 1.0)
                print(f"OpenAI translation network error: {e}. Retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(delay)
                continue
            raise RuntimeError(f"{error_label} failed: {e}") from e

    response_text = extract_response_text(response_data)
    if not response_text:
        raise RuntimeError(f"{error_label} did not contain text output.")

    try:
        parsed = json.loads(response_text)
        parsed["_usage"] = response_data.get("usage")
        return parsed
    except json.JSONDecodeError as e:
        raise RuntimeError(f"{error_label} was not valid JSON: {response_text}") from e


def call_openai_structured_translation(prompt, instructions, model, api_env_key="OPENAI_API_KEY"):
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "translations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "index": {"type": "integer"},
                        "text": {"type": "string"},
                    },
                    "required": ["index", "text"],
                },
            }
        },
        "required": ["translations"],
    }
    return call_openai_structured_json(
        prompt,
        instructions,
        model,
        schema,
        "subtitle_translation_batch",
        api_env_key=api_env_key,
        error_label="OpenAI translation request",
    )


def translate_segments_openai(
    segments,
    src_lang,
    tgt_lang,
    model="gpt-4o",
    batch_size=8,
    context_window=2,
    source_dialect="auto",
    target_dialect="natural",
    translator_notes="",
    provider_id="openai",
    provider_config=None,
    glossary=None,
):
    if src_lang == tgt_lang:
        return segments
    requested_batch_size = max(1, int(batch_size))
    minimum_semantic_batch_size = max(
        1,
        int(
            (provider_config or {}).get(
                "translation_min_semantic_batch_size",
                config_default(
                    "translation_min_semantic_batch_size",
                    4,
                ),
            )
        ),
    )
    batch_size = max(
        requested_batch_size,
        minimum_semantic_batch_size,
    )

    instructions = build_llm_translation_instructions(
        src_lang,
        tgt_lang,
        source_dialect=source_dialect,
        target_dialect=target_dialect,
        translator_notes=translator_notes,
        glossary=glossary,
    )
    indexed_segments = [
        {
            "index": index,
            "start": seg["start"],
            "end": seg["end"],
            "text": normalize_subtitle_text(seg["text"]),
            **{
                key: seg[key]
                for key in ("speaker", "language", "overlap", "overlap_turns")
                if seg.get(key) is not None
            },
        }
        for index, seg in enumerate(segments)
    ]

    translated_by_index = {}
    print(
        f"Translating {len(segments)} segments with provider "
        f"'{provider_id}', model '{model}' "
        f"(requested batch={requested_batch_size}, "
        f"effective semantic batch={batch_size})..."
    )

    # 1. Define all batches
    batches = []
    for batch_start in range(0, len(indexed_segments), batch_size):
        batch_end = min(len(indexed_segments), batch_start + batch_size)
        context_start = max(0, batch_start - context_window)
        context_end = min(len(indexed_segments), batch_end + context_window)
        prompt = build_llm_translation_prompt(
            indexed_segments[context_start:context_end],
            batch_start,
            batch_end,
        )
        batches.append((batch_start, batch_end, prompt))

    # 2. Translate in parallel
    from concurrent.futures import ThreadPoolExecutor
    
    def translate_batch(batch_info):
        batch_start, batch_end, prompt = batch_info
        if provider_id == "openai":
            response_json = call_openai_structured_translation(
                prompt,
                instructions,
                model,
                api_env_key=(provider_config or {}).get("openai_api_env_key", "OPENAI_API_KEY"),
            )
        else:
            response_json = call_provider_translation(
                provider_config or {},
                provider_id,
                prompt,
                instructions,
                model=model,
            )
        return batch_start, batch_end, response_json

    print(f"Translating {len(batches)} batches in parallel...")
    results = []
    with ThreadPoolExecutor(max_workers=min(len(batches), 8)) as executor:
        results = list(executor.map(translate_batch, batches))

    # 3. Process results
    for batch_start, batch_end, response_json in results:
        record_usage_event(
            provider_config,
            provider_id,
            model,
            "translation",
            usage=response_json.pop("_usage", None),
        )

        for item in response_json.get("translations", []):
            index = item.get("index")
            if batch_start <= index < batch_end:
                translated_by_index[index] = apply_glossary_to_translation(
                    indexed_segments[index]["text"],
                    item.get("text"),
                    glossary,
                )

        missing = [index for index in range(batch_start, batch_end) if index not in translated_by_index]
        if missing:
            raise RuntimeError(f"OpenAI translation response missed subtitle indices: {missing}")

    return [
        {
            "start": seg["start"],
            "end": seg["end"],
            "text": translated_by_index[index],
            **{
                key: seg[key]
                for key in ("speaker", "language", "overlap", "overlap_turns")
                if seg.get(key) is not None
            },
        }
        for index, seg in enumerate(segments)
    ]


def parse_srt_time(value):
    hours, minutes, rest = value.split(":")
    seconds, millis = rest.split(",")
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(millis) / 1000
    )


def parse_srt(srt_path):
    """Read an SRT file into pipeline segment dictionaries."""
    content = Path(srt_path).read_text(encoding="utf-8-sig").strip()
    if not content:
        return []

    segments = []
    for block in content.replace("\r\n", "\n").split("\n\n"):
        lines = [line for line in block.split("\n") if line.strip()]
        if len(lines) < 3 or "-->" not in lines[1]:
            continue

        start_text, end_text = [part.strip() for part in lines[1].split("-->", 1)]
        segments.append({
            "start": parse_srt_time(start_text),
            "end": parse_srt_time(end_text),
            "text": " ".join(lines[2:]).strip(),
        })

    return segments


def load_manifest(manifest_path):
    if not Path(manifest_path).exists():
        return {}

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_manifest(manifest_path, manifest):
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def normalize_subtitle_text(text):
    return " ".join((text or "").strip().split())


def wrap_subtitle_text(text, max_chars_per_line=42):
    text = normalize_subtitle_text(text)
    if not text:
        return ""

    return "\n".join(
        textwrap.wrap(
            text,
            width=max_chars_per_line,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def split_text_for_subtitle(text, max_chars_per_line=42, max_lines=2):
    text = normalize_subtitle_text(text)
    if not text:
        return []

    max_chars = max(1, max_chars_per_line * max_lines)
    words = text.split()
    chunks = []
    current = []

    for word in words:
        candidate = " ".join(current + [word])
        if current and len(candidate) > max_chars:
            chunks.append(" ".join(current))
            current = [word]
        else:
            current.append(word)

    if current:
        chunks.append(" ".join(current))

    return chunks


def prepare_segments_for_srt(
    segments,
    max_chars_per_line=42,
    max_lines=2,
    min_duration=0.8,
):
    """
    Clean subtitle text, merge very short neighboring cues, and split long cues.
    Timings are kept monotonic and proportional to text length.
    """
    cleaned = [
        {
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": normalize_subtitle_text(seg["text"]),
            **{
                key: seg[key]
                for key in (
                    "speaker",
                    "language",
                    "confidence",
                    "probability",
                    "overlap",
                    "overlap_turns",
                )
                if seg.get(key) is not None
            },
        }
        for seg in segments
        if normalize_subtitle_text(seg.get("text"))
    ]

    merged = []
    for seg in cleaned:
        if (
            merged
            and seg["end"] > merged[-1]["end"]
            and merged[-1]["end"] - merged[-1]["start"] < min_duration
            and seg["start"] - merged[-1]["end"] <= 0.5
            and seg.get("speaker") == merged[-1].get("speaker")
            and seg.get("language") == merged[-1].get("language")
        ):
            merged[-1]["end"] = seg["end"]
            merged[-1]["text"] = f"{merged[-1]['text']} {seg['text']}"
        else:
            merged.append(seg)

    prepared = []
    for seg in merged:
        chunks = split_text_for_subtitle(seg["text"], max_chars_per_line, max_lines)
        if not chunks:
            continue

        duration = max(0.1, seg["end"] - seg["start"])
        total_chars = sum(max(1, len(chunk)) for chunk in chunks)
        cursor = seg["start"]

        for index, chunk in enumerate(chunks):
            if index == len(chunks) - 1:
                chunk_end = seg["end"]
            else:
                chunk_duration = duration * (max(1, len(chunk)) / total_chars)
                chunk_end = min(seg["end"], cursor + chunk_duration)

            safe_end = min(seg["end"], max(cursor + 0.001, chunk_end))
            prepared.append({
                "start": cursor,
                "end": safe_end,
                "text": wrap_subtitle_text(chunk, max_chars_per_line),
                **{
                    key: seg[key]
                    for key in (
                        "speaker",
                        "language",
                        "confidence",
                        "probability",
                        "overlap",
                        "overlap_turns",
                    )
                    if seg.get(key) is not None
                },
            })
            cursor = chunk_end

    return prepared


def wrap_segments_for_srt_without_resplitting(segments, max_chars_per_line=42):
    return [
        {
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "text": wrap_subtitle_text(
                normalize_subtitle_text(segment.get("text", "")),
                max_chars_per_line=max_chars_per_line,
            ),
            **{
                key: segment[key]
                for key in (
                    "speaker",
                    "language",
                    "confidence",
                    "probability",
                    "overlap",
                    "overlap_turns",
                )
                if segment.get(key) is not None
            },
        }
        for segment in segments or []
        if normalize_subtitle_text(segment.get("text", ""))
        and segment.get("start") is not None
        and segment.get("end") is not None
        and float(segment["end"]) > float(segment["start"])
    ]


# -----------------------------
# Step 1: Extract audio
# -----------------------------
def extract_audio(video_path, audio_path, sample_rate=16000):
    command = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ac", "1",
        "-ar", str(sample_rate),
        audio_path
    ]
    run_ffmpeg_with_progress(command, video_path, "Extracting audio")


def extract_api_audio(video_path, audio_path, sample_rate=16000, bitrate="64k"):
    command = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-b:a", bitrate,
        audio_path,
    ]
    run_ffmpeg_with_progress(command, video_path, "Preparing API audio")


# -----------------------------
# Step 2: Transcribe audio
# -----------------------------
def normalize_whisper_word_text(text):
    return normalize_subtitle_text(str(text or "").strip())


def build_word_timing_segments(
    whisper_segments,
    max_chars=84,
    max_duration=5.0,
    max_gap=0.65,
):
    """Build subtitle-sized timing anchors from faster-whisper word timestamps."""
    anchors = []
    current_words = []
    current_start = None
    current_end = None

    def flush_current():
        nonlocal current_words, current_start, current_end
        text = normalize_subtitle_text(" ".join(current_words))
        if text and current_start is not None and current_end is not None and current_end > current_start:
            anchors.append({
                "start": float(current_start),
                "end": float(current_end),
                "text": text,
            })
        current_words = []
        current_start = None
        current_end = None

    for segment in whisper_segments:
        for word in getattr(segment, "words", None) or []:
            text = normalize_whisper_word_text(getattr(word, "word", ""))
            start = getattr(word, "start", None)
            end = getattr(word, "end", None)
            if not text or start is None or end is None:
                continue

            gap = 0.0 if current_end is None else float(start) - float(current_end)
            candidate_text = normalize_subtitle_text(" ".join(current_words + [text]))
            duration = 0.0 if current_start is None else float(end) - float(current_start)
            should_flush = bool(current_words) and (
                len(candidate_text) > max_chars
                or duration > max_duration
                or gap > max_gap
            )
            if should_flush:
                flush_current()

            if current_start is None:
                current_start = float(start)
            current_words.append(text)
            current_end = float(end)

            if text[-1:] in {".", "?", "!", "؟", "。"} and len(" ".join(current_words)) >= 24:
                flush_current()

    flush_current()
    return anchors


def transcribe_audio(audio_path, model_size="medium", device="cpu", beam_size=5, word_timestamps=False):
    WhisperModel = import_faster_whisper()
    compute_type = "float16" if device == "cuda" else "int8"
    model_reference = resolve_faster_whisper_model_reference(model_size)
    model = WhisperModel(model_reference, device=device, compute_type=compute_type)
    segments_iterator, info = model.transcribe(
        audio_path,
        beam_size=beam_size,
        vad_filter=True,
        word_timestamps=word_timestamps,
    )

    try:
        raw_segments = list(segments_iterator)
        word_segments = build_word_timing_segments(raw_segments) if word_timestamps else []
        segments_list = word_segments or [
            {"start": s.start, "end": s.end, "text": s.text}
            for s in raw_segments
        ]

        print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
        return segments_list, info
    finally:
        if device != "cpu":
            try:
                import torch
                import gc
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


def make_transcription_chunk_plan(duration_seconds, chunk_seconds=180, overlap_seconds=8):
    duration_seconds = max(0.0, float(duration_seconds or 0.0))
    chunk_seconds = max(30.0, float(chunk_seconds or 180.0))
    overlap_seconds = max(0.0, min(float(overlap_seconds or 0.0), chunk_seconds / 3.0))
    if duration_seconds <= 0.0:
        return []

    plan = []
    start = 0.0
    index = 1
    while start < duration_seconds - 0.001:
        end = min(duration_seconds, start + chunk_seconds)
        plan.append({
            "index": index,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
        })
        if end >= duration_seconds - 0.001:
            break
        start = max(0.0, end - overlap_seconds)
        index += 1
    return plan


def transcription_merge_token(token):
    return re.sub(r"[^\w]+", "", str(token or ""), flags=re.UNICODE).casefold()


def transcript_word_spans(text):
    spans = []
    for match in re.finditer(r"\S+", text or ""):
        normalized = transcription_merge_token(match.group(0))
        if normalized:
            spans.append((normalized, match.start(), match.end()))
    return spans


def find_exact_token_overlap(left_tokens, right_tokens, max_overlap_words=80, min_overlap_words=3):
    max_size = min(len(left_tokens), len(right_tokens), int(max_overlap_words))
    min_size = min(int(min_overlap_words), max_size) if max_size else 0
    for size in range(max_size, min_size - 1, -1):
        if left_tokens[-size:] == right_tokens[:size]:
            return size
    return 0


def find_diff_token_overlap(
    left_tokens,
    right_tokens,
    max_overlap_words=120,
    min_overlap_words=6,
    min_ratio=0.72,
    size_slop=14,
):
    max_right = min(len(right_tokens), int(max_overlap_words))
    if not left_tokens or not right_tokens or max_right < min_overlap_words:
        return 0, 0.0, 0

    left_window = left_tokens[-int(max_overlap_words):]
    best = {"cut": 0, "ratio": 0.0, "matches": 0}
    for right_size in range(max_right, int(min_overlap_words) - 1, -1):
        right_prefix = right_tokens[:right_size]
        min_left_size = max(int(min_overlap_words), right_size - int(size_slop))
        max_left_size = min(len(left_window), right_size + int(size_slop))
        for left_size in range(max_left_size, min_left_size - 1, -1):
            left_suffix = left_window[-left_size:]
            matcher = SequenceMatcher(None, left_suffix, right_prefix, autojunk=False)
            blocks = [block for block in matcher.get_matching_blocks() if block.size]
            matching_words = sum(block.size for block in blocks)
            ratio = matcher.ratio()
            if matching_words < min_overlap_words or ratio < min_ratio:
                continue

            cut = 0
            for block in blocks:
                if block.a + block.size >= len(left_suffix) - 2:
                    cut = max(cut, block.b + block.size)
            if cut < min_overlap_words:
                continue

            if cut > best["cut"] or (cut == best["cut"] and ratio > best["ratio"]):
                best = {"cut": cut, "ratio": ratio, "matches": matching_words}
                break
        if best["cut"]:
            break

    return best["cut"], round(best["ratio"], 4), best["matches"]


def merge_transcript_chunks(chunk_texts, max_overlap_words=120, min_overlap_words=6, min_overlap_ratio=0.72):
    cleaned_chunks = [normalize_subtitle_text(text) for text in chunk_texts if normalize_subtitle_text(text)]
    if not cleaned_chunks:
        return "", []

    merged_text = cleaned_chunks[0]
    merge_info = []
    for index, next_text in enumerate(cleaned_chunks[1:], start=2):
        left_tokens = [item[0] for item in transcript_word_spans(merged_text)]
        right_spans = transcript_word_spans(next_text)
        right_tokens = [item[0] for item in right_spans]
        overlap = find_exact_token_overlap(
            left_tokens,
            right_tokens,
            max_overlap_words=max_overlap_words,
            min_overlap_words=min_overlap_words,
        )
        overlap_method = "exact" if overlap else None
        overlap_ratio = 1.0 if overlap else 0.0
        overlap_matches = overlap
        if not overlap:
            overlap, overlap_ratio, overlap_matches = find_diff_token_overlap(
                left_tokens,
                right_tokens,
                max_overlap_words=max_overlap_words,
                min_overlap_words=min_overlap_words,
                min_ratio=min_overlap_ratio,
            )
            overlap_method = "diff" if overlap else None

        if overlap and len(right_spans) >= overlap:
            addition_start = right_spans[overlap - 1][2]
            addition = next_text[addition_start:].lstrip()
        else:
            addition = next_text

        if addition:
            merged_text = normalize_subtitle_text(f"{merged_text} {addition}")
        merge_info.append({
            "chunk_index": index,
            "overlap_words_removed": overlap,
            "overlap_method": overlap_method,
            "overlap_ratio": overlap_ratio,
            "overlap_matched_words": overlap_matches,
        })

    return merged_text, merge_info


def merge_timed_transcript_segments(chunk_segment_groups, overlap_seconds=8):
    accepted_segments = []
    merge_info = []

    for group in chunk_segment_groups or []:
        segments = group.get("segments") or []
        chunk_index = int(group.get("index") or 1)
        duplicate_cutoff = float(group.get("start") or 0.0)
        if chunk_index > 1:
            duplicate_cutoff += max(0.0, float(overlap_seconds or 0.0))

        kept = []
        skipped = []
        trimmed = []
        boundary_replacements = []
        for segment in segments:
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", start))
            text = normalize_subtitle_text(segment.get("text", ""))
            granularity = segment.get("_granularity")
            if not text:
                continue
            if chunk_index > 1 and end <= duplicate_cutoff:
                skipped.append(segment)
                continue

            if (
                chunk_index > 1
                and granularity == "word"
                and start < duplicate_cutoff < end
            ):
                replaced = []
                if accepted_segments and accepted_segments[-1]["end"] > start:
                    previous_text = normalize_subtitle_text(accepted_segments[-1].get("text", ""))
                    is_same_boundary_word = bool(
                        previous_text
                        and text
                        and (
                            text.startswith(previous_text)
                            or previous_text.startswith(text)
                        )
                    )
                    if is_same_boundary_word:
                        replaced.append(accepted_segments.pop())
                if replaced:
                    boundary_replacements.append({
                        "replacement_start": round(start, 3),
                        "replacement_end": round(end, 3),
                        "replacement_text": text,
                        "removed": list(reversed(replaced)),
                    })

            if chunk_index > 1 and start < duplicate_cutoff:
                previous_end = accepted_segments[-1]["end"] if accepted_segments else start
                adjusted_start = max(start, min(previous_end, end - 0.1))
                if adjusted_start > start:
                    trimmed.append({
                        "original_start": round(start, 3),
                        "adjusted_start": round(adjusted_start, 3),
                        "end": round(end, 3),
                    })
                    start = adjusted_start

            kept.append({
                "start": start,
                "end": end,
                "text": text,
            })

        accepted_segments.extend(kept)
        if chunk_index > 1:
            merge_info.append({
                "chunk_index": chunk_index,
                "overlap_method": "timestamp",
                "duplicate_cutoff": round(duplicate_cutoff, 3),
                "segments_removed": len(skipped),
                "segments_kept": len(kept),
                "segments_trimmed": len(trimmed),
                "trimmed_segments": trimmed[:5],
                "boundary_word_replacements": boundary_replacements[:5],
            })

    accepted_segments.sort(key=lambda segment: (segment["start"], segment["end"]))
    transcript_text = normalize_subtitle_text(" ".join(segment["text"] for segment in accepted_segments))
    return transcript_text, accepted_segments, merge_info


def sum_numeric_usage_dicts(usages):
    latest_identity_keys = {
        "request_sha256",
        "response_sha256",
        "model_version",
        "prompt_version",
        "generation_scope",
        "source_offset_seconds",
        "thinking_config_version",
        "thinking_config",
    }

    def merge_values(left, right, key=None):
        if key in latest_identity_keys and right is not None:
            return right
        if key == "hidden_thinking_exhaustion_detected":
            return bool(left) or bool(right)
        if key == "zero_thinking_usage_verified":
            values = [value for value in (left, right) if value is not None]
            return all(values) if values else None
        if isinstance(left, dict) or isinstance(right, dict):
            merged = dict(left or {})
            for nested_key, value in dict(right or {}).items():
                merged[nested_key] = merge_values(
                    merged.get(nested_key), value, nested_key
                )
            return merged
        if isinstance(right, (int, float)) and not isinstance(right, bool):
            return number_or_zero(left) + right
        return left if left is not None else right

    total = {}
    for usage in usages or []:
        if isinstance(usage, dict):
            total = merge_values(total, usage)
    return total


def transcript_cache_is_suspect(manifest, pipeline_config=None):
    config = pipeline_config or CONFIG
    plausibility = (manifest or {}).get("transcript_plausibility_report") or {}
    if plausibility and not plausibility.get("accept", False):
        return True
    if plausibility.get("repetition_requires_confirmation"):
        return True
    if (manifest or {}).get("transcription_containment"):
        return True
    if config.get("api_transcription_reuse_suspect_transcripts", False):
        return False

    usage = (manifest or {}).get("transcription_usage") or {}
    if usage.get("chunked") or usage.get("chunk_count"):
        return False

    threshold = int(config.get("api_transcription_suspect_output_tokens", 2048) or 0)
    output_tokens = number_or_zero(usage.get("output_tokens"))
    return bool(threshold and output_tokens >= threshold)


def transcribe_audio_openai_single(
    audio_path,
    model="whisper-1",
    language=None,
    prompt="",
    api_env_key="OPENAI_API_KEY",
    allow_empty=False,
    word_timestamps=False,
):
    max_upload_bytes = 25 * 1024 * 1024
    audio_path_obj = Path(audio_path)
    if audio_path_obj.stat().st_size > max_upload_bytes:
        raise RuntimeError(
            f"OpenAI transcription upload is larger than 25 MB: {audio_path_obj}. "
            "Use a shorter video or lower api_audio_bitrate."
        )

    fields = {
        "model": model,
        "response_format": "verbose_json",
        "language": language,
        "prompt": prompt,
    }
    if word_timestamps and model == "whisper-1":
        fields["timestamp_granularities[]"] = ["word", "segment"]
    print(f"Transcribing with OpenAI model '{model}'...")
    response_json = post_openai_multipart(
        "https://api.openai.com/v1/audio/transcriptions",
        fields=fields,
        files={"file": audio_path_obj},
        api_env_key=api_env_key,
    )
    transcript_text = normalize_subtitle_text(response_json.get("text", ""))
    if not transcript_text and not allow_empty:
        raise RuntimeError("OpenAI transcription response did not contain transcript text.")

    segments = response_json.get("segments", [])
    used_word_timestamps = False
    if word_timestamps:
        timed_words = []
        for word in response_json.get("words") or []:
            text = normalize_subtitle_text(word.get("word", ""))
            start = word.get("start")
            end = word.get("end")
            if not text or start is None or end is None or float(end) <= float(start):
                continue
            timed_words.append({
                "start": float(start),
                "end": float(end),
                "text": text,
                "_granularity": "word",
            })
        if timed_words:
            segments = timed_words
            used_word_timestamps = True
        elif transcript_text:
            raise RuntimeError(
                "OpenAI Whisper-1 returned transcript text without the requested word timestamps."
            )
    usage = dict(response_json.get("usage") or {})
    usage["timing_granularity"] = "word" if used_word_timestamps else "segment"
    return transcript_text, segments, usage


def transcript_looks_like_prompt_echo(text, prompt, min_tokens=3, min_ratio=0.82):
    text_tokens = [transcription_merge_token(token) for token in str(text or "").split()]
    text_tokens = [token for token in text_tokens if token]
    if len(text_tokens) < int(min_tokens):
        return False

    prompt_tokens = [transcription_merge_token(token) for token in str(prompt or "").split()]
    prompt_tokens = [token for token in prompt_tokens if token]
    if not prompt_tokens:
        return False

    normalized_text = " ".join(text_tokens)
    normalized_prompt = " ".join(prompt_tokens)
    if normalized_text and normalized_text in normalized_prompt:
        return True

    prompt_token_set = set(prompt_tokens)
    prompt_overlap = sum(1 for token in text_tokens if token in prompt_token_set) / len(text_tokens)
    if prompt_overlap >= float(min_ratio):
        return True

    instruction_phrases = {
        "transcribe spoken words",
        "do not translate speech",
        "do not invent words",
        "preserve repeated words",
        "multiple spoken languages",
        "timing anchors",
    }
    text_casefold = str(text or "").casefold()
    return any(phrase in text_casefold for phrase in instruction_phrases)


def extract_audio_chunk_for_transcription(audio_path, chunk_path, start, duration, sample_rate=16000, bitrate="64k"):
    command = [
        "ffmpeg", "-y",
        "-ss", f"{float(start):.3f}",
        "-t", f"{float(duration):.3f}",
        "-i", str(audio_path),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-b:a", bitrate,
        str(chunk_path),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        **hidden_subprocess_kwargs(),
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create transcription chunk {chunk_path}: {result.stderr}")


def audio_speech_activity_report(
    audio_path,
    duration_seconds=None,
    silence_noise_db=-45,
    min_speech_seconds=0.25,
    min_speech_ratio=0.01,
):
    """Estimate whether a chunk contains speech-like non-silence using FFmpeg."""
    audio_path_obj = Path(audio_path)
    try:
        duration = float(duration_seconds) if duration_seconds is not None else get_video_duration(str(audio_path_obj))
    except Exception:
        duration = 0.0

    if duration <= 0.0:
        return {
            "engine": "ffmpeg_silencedetect",
            "accept": False,
            "has_speech": True,
            "reason": "unknown_duration",
            "duration": round(duration, 3),
            "non_silent_seconds": None,
            "non_silent_ratio": None,
            "silence_intervals": [],
        }

    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(audio_path_obj),
        "-af",
        f"silencedetect=noise={float(silence_noise_db):g}dB:d=0.2",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        **hidden_subprocess_kwargs(),
    )
    if result.returncode != 0:
        return {
            "engine": "ffmpeg_silencedetect",
            "accept": False,
            "has_speech": True,
            "reason": "silencedetect_failed",
            "duration": round(duration, 3),
            "non_silent_seconds": None,
            "non_silent_ratio": None,
            "silence_intervals": [],
            "error": result.stderr[-1000:],
        }

    intervals = []
    open_start = None
    for line in (result.stderr or "").splitlines():
        start_match = re.search(r"silence_start:\s*([0-9.]+)", line)
        if start_match:
            open_start = max(0.0, float(start_match.group(1)))
            continue

        end_match = re.search(r"silence_end:\s*([0-9.]+)\s*\|\s*silence_duration:\s*([0-9.]+)", line)
        if end_match:
            end = min(duration, max(0.0, float(end_match.group(1))))
            start = open_start
            if start is None:
                silence_duration = max(0.0, float(end_match.group(2)))
                start = max(0.0, end - silence_duration)
            if end > start:
                intervals.append((start, end))
            open_start = None

    if open_start is not None and open_start < duration:
        intervals.append((open_start, duration))

    clipped = []
    for start, end in sorted(intervals):
        start = max(0.0, min(duration, start))
        end = max(start, min(duration, end))
        if not clipped or start > clipped[-1][1]:
            clipped.append([start, end])
        else:
            clipped[-1][1] = max(clipped[-1][1], end)

    silent_seconds = sum(end - start for start, end in clipped)
    non_silent_seconds = max(0.0, duration - silent_seconds)
    non_silent_ratio = non_silent_seconds / duration if duration else 0.0
    has_speech = (
        non_silent_seconds >= float(min_speech_seconds)
        and non_silent_ratio >= float(min_speech_ratio)
    )
    return {
        "engine": "ffmpeg_silencedetect",
        "accept": True,
        "has_speech": bool(has_speech),
        "reason": "speech_detected" if has_speech else "mostly_silence",
        "duration": round(duration, 3),
        "non_silent_seconds": round(non_silent_seconds, 3),
        "non_silent_ratio": round(non_silent_ratio, 4),
        "silence_noise_db": float(silence_noise_db),
        "silence_intervals": [
            {"start": round(start, 3), "end": round(end, 3)}
            for start, end in clipped[:20]
        ],
    }


def transcribe_audio_openai_chunked(
    audio_path,
    model="whisper-1",
    language=None,
    prompt="",
    api_env_key="OPENAI_API_KEY",
    chunk_seconds=180,
    overlap_seconds=8,
    chunk_output_dir=None,
    sample_rate=16000,
    bitrate="64k",
    tolerate_empty_chunks=True,
    retry_empty_speech_chunks=True,
    empty_chunk_retry_context_seconds=6,
    silence_noise_db=-45,
    min_speech_seconds=0.25,
    min_speech_ratio=0.01,
    word_timestamps=False,
):
    audio_path_obj = Path(audio_path)
    duration_seconds = get_video_duration(str(audio_path_obj))
    plan = make_transcription_chunk_plan(duration_seconds, chunk_seconds, overlap_seconds)
    if len(plan) <= 1:
        return transcribe_audio_openai_single(
            audio_path,
            model=model,
            language=language,
            prompt=prompt,
            api_env_key=api_env_key,
            word_timestamps=word_timestamps,
        )

    chunk_dir = Path(chunk_output_dir) if chunk_output_dir else audio_path_obj.parent / f"{audio_path_obj.stem}_transcription_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Transcribing with OpenAI model '{model}' in {len(plan)} chunks "
        f"({chunk_seconds}s chunks, {overlap_seconds}s overlap)..."
    )

    # 1. Extract all chunks first
    chunk_paths = []
    for item in plan:
        chunk_path = chunk_dir / f"{audio_path_obj.stem}.chunk_{item['index']:03d}.mp3"
        extract_audio_chunk_for_transcription(
            audio_path_obj,
            chunk_path,
            item["start"],
            item["duration"],
            sample_rate=sample_rate,
            bitrate=bitrate,
        )
        chunk_paths.append((item, chunk_path))

    # 2. Transcribe in parallel
    print(f"Transcribing {len(plan)} chunks in parallel...")
    from concurrent.futures import ThreadPoolExecutor
    
    def transcribe_one(chunk_info):
        item, chunk_path = chunk_info
        print(
            f"  Starting OpenAI transcription chunk {item['index']}/{len(plan)}: "
            f"{item['start']:.1f}s-{item['end']:.1f}s"
        )
        speech_activity = audio_speech_activity_report(
            chunk_path,
            duration_seconds=item["duration"],
            silence_noise_db=silence_noise_db,
            min_speech_seconds=min_speech_seconds,
            min_speech_ratio=min_speech_ratio,
        )
        text, segments, usage = transcribe_audio_openai_single(
            str(chunk_path),
            model=model,
            language=language,
            prompt=prompt,
            api_env_key=api_env_key,
            allow_empty=tolerate_empty_chunks,
            word_timestamps=word_timestamps,
        )
        prompt_echo = transcript_looks_like_prompt_echo(text, prompt)
        if prompt_echo:
            text = ""
            segments = []
        retry_report = None
        effective_start = item["start"]
        effective_chunk_path = chunk_path
        status = "prompt_echo" if prompt_echo else ("transcribed" if text else "empty")

        if not text and tolerate_empty_chunks:
            if not speech_activity.get("has_speech", True):
                status = "accepted_prompt_echo_silence" if prompt_echo else "accepted_silence"
            elif retry_empty_speech_chunks:
                retry_context = max(0.0, float(empty_chunk_retry_context_seconds or 0.0))
                retry_start = max(0.0, item["start"] - retry_context)
                retry_end = min(float(duration_seconds), item["end"] + retry_context)
                retry_duration = max(0.0, retry_end - retry_start)
                retry_chunk_path = chunk_dir / f"{audio_path_obj.stem}.chunk_{item['index']:03d}.retry.mp3"
                extract_audio_chunk_for_transcription(
                    audio_path_obj,
                    retry_chunk_path,
                    retry_start,
                    retry_duration,
                    sample_rate=sample_rate,
                    bitrate=bitrate,
                )
                retry_text, retry_segments, retry_usage = transcribe_audio_openai_single(
                    str(retry_chunk_path),
                    model=model,
                    language=language,
                    prompt=prompt,
                    api_env_key=api_env_key,
                    allow_empty=True,
                    word_timestamps=word_timestamps,
                )
                retry_prompt_echo = transcript_looks_like_prompt_echo(retry_text, prompt)
                if retry_prompt_echo:
                    retry_text = ""
                    retry_segments = []
                retry_report = {
                    "start": round(retry_start, 3),
                    "end": round(retry_end, 3),
                    "duration": round(retry_duration, 3),
                    "text_chars": len(retry_text),
                    "prompt_echo": bool(retry_prompt_echo),
                    "usage": retry_usage,
                    "path": str(retry_chunk_path),
                }
                if retry_text:
                    text = retry_text
                    segments = retry_segments
                    usage = sum_numeric_usage_dicts([usage or {}, retry_usage or {}])
                    effective_start = retry_start
                    effective_chunk_path = retry_chunk_path
                    status = "retried_with_context"
                else:
                    usage = sum_numeric_usage_dicts([usage or {}, retry_usage or {}])
                    status = "prompt_echo_speech_after_retry" if prompt_echo or retry_prompt_echo else "empty_speech_after_retry"

        if not text and not tolerate_empty_chunks:
            raise RuntimeError("OpenAI transcription response did not contain transcript text.")

        return item, text, segments, usage, effective_chunk_path, effective_start, speech_activity, status, retry_report, prompt_echo

    results = []
    with ThreadPoolExecutor(max_workers=min(len(plan), 8)) as executor:
        results = list(executor.map(transcribe_one, chunk_paths))

    # Sort results by chunk index to keep order
    results.sort(key=lambda x: x[0]["index"])

    chunk_texts = []
    chunk_usages = []
    chunk_reports = []
    all_segments = []
    timed_chunk_groups = []

    unresolved_speech_chunks = []

    for item, text, segments, usage, chunk_path, effective_start, speech_activity, status, retry_report, prompt_echo in results:
        chunk_transcript_path = chunk_dir / f"{audio_path_obj.stem}.chunk_{item['index']:03d}.txt"
        chunk_segments_path = chunk_dir / f"{audio_path_obj.stem}.chunk_{item['index']:03d}.segments.json"
        chunk_transcript_path.write_text(text, encoding="utf-8")
        chunk_texts.append(text)
        chunk_usages.append(usage or {})
        report = {
            "index": item["index"],
            "start": item["start"],
            "end": item["end"],
            "duration": item["duration"],
            "status": status,
            "prompt_echo": bool(prompt_echo),
            "text_chars": len(text),
            "transcript_path": str(chunk_transcript_path),
            "segments_path": str(chunk_segments_path),
            "effective_audio_path": str(chunk_path),
            "effective_start": round(float(effective_start), 3),
            "speech_activity": speech_activity,
            "usage": usage,
        }
        if retry_report:
            report["retry"] = retry_report
        chunk_reports.append(report)
        if status in {"empty_speech_after_retry", "prompt_echo_speech_after_retry"}:
            unresolved_speech_chunks.append(report)

        global_segments = []
        for seg in segments:
            global_segment = {
                "start": seg["start"] + effective_start,
                "end": seg["end"] + effective_start,
                "text": seg["text"],
            }
            if seg.get("_granularity"):
                global_segment["_granularity"] = seg["_granularity"]
            all_segments.append(global_segment)
            global_segments.append(global_segment)
        chunk_segments_path.write_text(
            json.dumps(global_segments, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if global_segments:
            timed_chunk_groups.append({
                "index": item["index"],
                "start": item["start"],
                "end": item["end"],
                "segments": global_segments,
            })

    timed_transcript_text, timed_segments, timed_merge_info = merge_timed_transcript_segments(
        timed_chunk_groups,
        overlap_seconds=overlap_seconds,
    )
    if timed_transcript_text:
        transcript_text = timed_transcript_text
        all_segments = timed_segments
        merge_info = timed_merge_info
        merged_segments_path = chunk_dir / f"{audio_path_obj.stem}.merged_timing_segments.json"
        merged_segments_path.write_text(
            json.dumps(timed_segments, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        transcript_text, merge_info = merge_transcript_chunks(chunk_texts)
    usage = sum_numeric_usage_dicts(chunk_usages)
    usage.update({
        "chunked": True,
        "chunk_count": len(plan),
        "empty_chunk_policy": "speech_aware_retry" if tolerate_empty_chunks else "strict",
        "accepted_silence_chunk_count": sum(
            1 for item in chunk_reports
            if item.get("status") in {"accepted_silence", "accepted_prompt_echo_silence"}
        ),
        "retried_empty_speech_chunk_count": sum(1 for item in chunk_reports if item.get("status") == "retried_with_context"),
        "unresolved_empty_speech_chunk_count": len(unresolved_speech_chunks),
        "prompt_echo_chunk_count": sum(1 for item in chunk_reports if item.get("prompt_echo")),
        "chunk_seconds": float(chunk_seconds),
        "overlap_seconds": float(overlap_seconds),
        "source_duration_seconds": round(float(duration_seconds), 3),
        "billed_duration_seconds": round(sum(item["duration"] for item in plan), 3),
        "chunk_merge": merge_info,
        "chunks": chunk_reports,
    })
    if unresolved_speech_chunks and not all_segments:
        indexes = ", ".join(str(item["index"]) for item in unresolved_speech_chunks[:10])
        raise RuntimeError(
            "OpenAI transcription returned empty text for speech-containing chunks "
            f"after retry, and no timing segments were recovered. Affected chunk(s): {indexes}."
        )
    return transcript_text, all_segments, usage


def transcribe_audio_openai_llm(
    audio_path,
    model="gpt-4o-audio-preview",
    language=None,
    prompt="",
    api_env_key="OPENAI_API_KEY",
):
    import base64
    api_key = os.environ.get(api_env_key)
    if not api_key:
        raise RuntimeError(f"OpenAI API key not found in environment variable: {api_env_key}")

    audio_path_obj = Path(audio_path)
    if audio_path_obj.stat().st_size > 18 * 1024 * 1024:
        raise RuntimeError(
            "OpenAI LLM audio request exceeds 18 MB safety limit. "
            "Lower api_audio_bitrate for this file."
        )

    audio_bytes = audio_path_obj.read_bytes()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    language_hint = language or "auto-detect"
    instruction = (
        f"Transcribe this audio verbatim. Source language: {language_hint}. "
        "If the audio contains multiple spoken languages, transcribe each spoken passage "
        "in the language actually spoken. Do not translate speech during transcription. "
        "Do not force all speech into the declared source language. "
        "Preserve the exact spoken sequence as much as possible, including dialect, "
        "filler words, hesitations, false starts, repeated words, repeated phrases, "
        "names, numbers, idioms, and natural wording. Do not summarize, polish, "
        "deduplicate, normalize away dialect, or remove repetitions. "
        "If there is silence or a long pause, do not invent filler text; continue only "
        "when speech resumes. "
        "Return only the transcript text, without explanations."
    )
    if prompt:
        instruction += f" Vocabulary/context hints: {prompt}"

    ext = audio_path_obj.suffix.lower().lstrip('.')
    if ext not in ["wav", "mp3", "flac", "ogg"]:
        ext = "mp3"

    payload = {
        "model": model,
        "modalities": ["text"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_b64,
                            "format": ext
                        }
                    }
                ]
            }
        ]
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    print(f"Transcribing audio with OpenAI LLM '{model}' on their servers...")
    url = "https://api.openai.com/v1/chat/completions"
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            transcript_text = res_data["choices"][0]["message"]["content"].strip()
            usage = res_data.get("usage")
            return transcript_text, usage
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        raise RuntimeError(f"OpenAI LLM transcription failed: {e.code} - {err_body}")
    except Exception as e:
        raise RuntimeError(f"OpenAI LLM transcription failed: {e}")


def transcribe_audio_openai(
    audio_path,
    model="gpt-4o-audio-preview",
    language=None,
    prompt="",
    api_env_key="OPENAI_API_KEY",
    chunking="auto",
    chunk_seconds=180,
    chunk_overlap_seconds=8,
    chunk_output_dir=None,
    sample_rate=16000,
    bitrate="64k",
    tolerate_empty_chunks=True,
    retry_empty_speech_chunks=True,
    empty_chunk_retry_context_seconds=6,
    silence_noise_db=-45,
    min_speech_seconds=0.25,
    min_speech_ratio=0.01,
    word_timestamps=False,
):
    audio_path_obj = Path(audio_path)
    if model.startswith("gpt-"):
        transcript_text, usage = transcribe_audio_openai_llm(
            audio_path,
            model=model,
            language=language,
            prompt=prompt,
            api_env_key=api_env_key,
        )
        return transcript_text, [], usage
    max_upload_bytes = 25 * 1024 * 1024
    chunking_mode = str(chunking or "auto").strip().lower()
    chunking_enabled = chunking_mode not in {"0", "false", "off", "no", "disabled", "none"}

    duration_seconds = None
    try:
        duration_seconds = get_video_duration(str(audio_path_obj))
    except Exception:
        duration_seconds = None

    should_chunk = False
    if chunking_enabled:
        if chunking_mode != "auto" and duration_seconds and duration_seconds > float(chunk_seconds or 180):
            should_chunk = True
        elif audio_path_obj.stat().st_size > max_upload_bytes:
            should_chunk = True

    if not should_chunk:
        print(f"[PIPELINE] Audio size is {audio_path_obj.stat().st_size / (1024*1024):.2f}MB (under 25MB limit). Transcribing as a single file to preserve maximum context and quality.")

    if should_chunk:
        return transcribe_audio_openai_chunked(
            audio_path,
            model=model,
            language=language,
            prompt=prompt,
            api_env_key=api_env_key,
            chunk_seconds=chunk_seconds,
            overlap_seconds=chunk_overlap_seconds,
            chunk_output_dir=chunk_output_dir,
            sample_rate=sample_rate,
            bitrate=bitrate,
            tolerate_empty_chunks=tolerate_empty_chunks,
            retry_empty_speech_chunks=retry_empty_speech_chunks,
            empty_chunk_retry_context_seconds=empty_chunk_retry_context_seconds,
            silence_noise_db=silence_noise_db,
            min_speech_seconds=min_speech_seconds,
            min_speech_ratio=min_speech_ratio,
            word_timestamps=word_timestamps,
        )

    return transcribe_audio_openai_single(
        audio_path,
        model=model,
        language=language,
        prompt=prompt,
        api_env_key=api_env_key,
        word_timestamps=word_timestamps,
    )


def longform_runtime_options(
    pipeline_config,
    *,
    provider=None,
    timing_evidence_only=False,
):
    return {
        "target_seconds": pipeline_config.get(
            "longform_chunk_target_seconds",
            config_default("longform_chunk_target_seconds", 300),
        ),
        "min_seconds": pipeline_config.get(
            "longform_chunk_min_seconds",
            config_default("longform_chunk_min_seconds", 180),
        ),
        "max_seconds": pipeline_config.get(
            "longform_chunk_max_seconds",
            config_default("longform_chunk_max_seconds", 480),
        ),
        "overlap_seconds": pipeline_config.get(
            "longform_chunk_overlap_seconds",
            config_default("longform_chunk_overlap_seconds", 12),
        ),
        "boundary_search_seconds": pipeline_config.get(
            "longform_boundary_search_seconds",
            config_default("longform_boundary_search_seconds", 60),
        ),
        "min_boundary_silence_seconds": pipeline_config.get(
            "longform_min_boundary_silence_seconds",
            config_default("longform_min_boundary_silence_seconds", 1.0),
        ),
        "sample_rate": pipeline_config.get("audio_sample_rate", 16000),
        "bitrate": pipeline_config.get("api_audio_bitrate", "64k"),
        "retry_context_seconds": pipeline_config.get(
            "longform_retry_context_seconds",
            config_default("longform_retry_context_seconds", 15),
        ),
        "invalid_chunk_split_retry_enabled": pipeline_config.get(
            "longform_invalid_chunk_split_retry_enabled",
            config_default(
                "longform_invalid_chunk_split_retry_enabled",
                True,
            ),
        ),
        "invalid_chunk_split_retry_min_seconds": pipeline_config.get(
            "longform_invalid_chunk_split_retry_min_seconds",
            config_default(
                "longform_invalid_chunk_split_retry_min_seconds",
                60,
            ),
        ),
        "invalid_chunk_split_retry_overlap_seconds": pipeline_config.get(
            "longform_invalid_chunk_split_retry_overlap_seconds",
            config_default(
                "longform_invalid_chunk_split_retry_overlap_seconds",
                6,
            ),
        ),
        "coverage_recovery_enabled": pipeline_config.get(
            "longform_coverage_recovery_enabled",
            config_default("longform_coverage_recovery_enabled", True),
        ),
        "coverage_recovery_context_seconds": pipeline_config.get(
            "longform_coverage_recovery_context_seconds",
            config_default("longform_coverage_recovery_context_seconds", 6),
        ),
        "coverage_recovery_max_attempts_per_chunk": pipeline_config.get(
            "longform_coverage_recovery_max_attempts_per_chunk",
            config_default(
                "longform_coverage_recovery_max_attempts_per_chunk",
                6,
            ),
        ),
        "coverage_recovery_max_total_attempts": pipeline_config.get(
            "longform_coverage_recovery_max_total_attempts",
            config_default(
                "longform_coverage_recovery_max_total_attempts",
                12,
            ),
        ),
        "coverage_recovery_min_novel_probability": pipeline_config.get(
            "longform_coverage_recovery_min_novel_probability",
            config_default(
                "longform_coverage_recovery_min_novel_probability",
                0.60,
            ),
        ),
        "coverage_recovery_existing_match_tolerance_seconds": (
            pipeline_config.get(
                "longform_coverage_recovery_existing_match_tolerance_seconds",
                config_default(
                    "longform_coverage_recovery_existing_match_tolerance_seconds",
                    0.75,
                ),
            )
        ),
        "coverage_recovery_allow_unscored_timed_segments": bool(
            (
                provider == "google"
                and pipeline_config.get(
                    "longform_coverage_recovery_allow_unscored_google_segments",
                    config_default(
                        "longform_coverage_recovery_allow_unscored_google_segments",
                        True,
                    ),
                )
            )
            or (
                timing_evidence_only
                and pipeline_config.get(
                    "longform_coverage_recovery_allow_unscored_timing_evidence",
                    config_default(
                        "longform_coverage_recovery_allow_unscored_timing_evidence",
                        True,
                    ),
                )
            )
        ),
        "coverage_recovery_min_unscored_speech_overlap_ratio": (
            pipeline_config.get(
                "longform_coverage_recovery_min_unscored_speech_overlap_ratio",
                config_default(
                    "longform_coverage_recovery_min_unscored_speech_overlap_ratio",
                    0.80,
                ),
            )
        ),
        "coverage_recovery_max_window_seconds": pipeline_config.get(
            "longform_coverage_recovery_max_window_seconds",
            config_default(
                "longform_coverage_recovery_max_window_seconds",
                180,
            ),
        ),
        "max_uncovered_gap_seconds": pipeline_config.get(
            "longform_max_uncovered_gap_seconds",
            config_default("longform_max_uncovered_gap_seconds", 1.5),
        ),
        "max_uncovered_ratio": pipeline_config.get(
            "longform_max_uncovered_ratio",
            config_default("longform_max_uncovered_ratio", 0.03),
        ),
        "speech_map_options": {
            "engine": pipeline_config.get(
                "longform_speech_map_engine",
                config_default("longform_speech_map_engine", "auto"),
            ),
            "silence_noise_db": pipeline_config.get(
                "api_timing_anchor_silence_noise_db",
                config_default("api_timing_anchor_silence_noise_db", -45),
            ),
            "confirmed_silence_seconds": pipeline_config.get(
                "longform_confirmed_silence_seconds",
                config_default("longform_confirmed_silence_seconds", 1.0),
            ),
            "vad_analysis_window_seconds": pipeline_config.get(
                "longform_vad_analysis_window_seconds",
                config_default("longform_vad_analysis_window_seconds", 600),
            ),
            "vad_analysis_overlap_seconds": pipeline_config.get(
                "longform_vad_analysis_overlap_seconds",
                config_default("longform_vad_analysis_overlap_seconds", 2),
            ),
            "vad_threshold": pipeline_config.get(
                "longform_vad_threshold",
                config_default("longform_vad_threshold", 0.45),
            ),
            "vad_min_speech_ms": pipeline_config.get(
                "longform_vad_min_speech_ms",
                config_default("longform_vad_min_speech_ms", 150),
            ),
            "vad_min_silence_ms": pipeline_config.get(
                "longform_vad_min_silence_ms",
                config_default("longform_vad_min_silence_ms", 350),
            ),
            "vad_speech_pad_ms": pipeline_config.get(
                "longform_vad_speech_pad_ms",
                config_default("longform_vad_speech_pad_ms", 350),
            ),
        },
    }


def normalize_extreme_character_runs_in_text(
    text,
    *,
    max_character_run=11,
    replacement_length=3,
):
    value = str(text or "")
    maximum = max(1, int(max_character_run))
    retained = max(1, min(maximum, int(replacement_length)))
    occurrences = []
    pattern = re.compile(
        rf"([^\W_])\1{{{maximum},}}",
        flags=re.UNICODE,
    )

    def replace(match):
        character = match.group(1)
        occurrences.append({
            "character": character,
            "original_length": len(match.group(0)),
            "replacement_length": retained,
            "character_offset": match.start(),
        })
        return character * retained

    return pattern.sub(replace, value), occurrences


def normalize_artifact_extreme_character_runs(
    artifact,
    *,
    max_character_run=11,
    replacement_length=3,
    apply_changes=True,
):
    occurrences = []

    def normalize(scope, value):
        normalized, changes = normalize_extreme_character_runs_in_text(
            value,
            max_character_run=max_character_run,
            replacement_length=replacement_length,
        )
        for change in changes:
            occurrences.append({"scope": scope, **change})
        return normalized

    normalized_text = normalize("text", artifact.text)
    normalized_segments = [
        {
            **segment,
            "text": normalize(
                f"segments[{index}].text",
                segment.get("text"),
            ),
        }
        for index, segment in enumerate(artifact.segments or [])
    ]
    normalized_words = []
    for index, word in enumerate(artifact.words or []):
        item = dict(word)
        text_key = "text" if "text" in item else "word"
        item[text_key] = normalize(
            f"words[{index}].{text_key}",
            item.get(text_key),
        )
        normalized_words.append(item)
    if apply_changes:
        artifact.text = normalized_text
        artifact.segments = normalized_segments
        artifact.words = normalized_words
    return {
        "schema": "subgen_character_run_normalization_v1",
        "changed": bool(occurrences),
        "applied": bool(apply_changes and occurrences),
        "text_only_mutation_allowed": False,
        "max_character_run": max(1, int(max_character_run)),
        "replacement_length": max(
            1,
            min(int(max_character_run), int(replacement_length)),
        ),
        "occurrence_count": len(occurrences),
        "occurrences": occurrences[:20],
        "occurrences_truncated": max(0, len(occurrences) - 20),
    }


def _validate_chunk_artifact_plausibility(artifact, chunk, pipeline_config):
    maximum_character_run = int(
        pipeline_config.get(
            "local_max_character_run",
            config_default("local_max_character_run", 11),
        )
    )
    character_normalization = normalize_artifact_extreme_character_runs(
        artifact,
        max_character_run=maximum_character_run,
        replacement_length=int(
            pipeline_config.get(
                "local_character_run_normalization_length",
                config_default(
                    "local_character_run_normalization_length",
                    3,
                ),
            )
        ),
        apply_changes=False,
    )
    report = transcript_plausibility_report(
        artifact.text,
        chunk.get("duration"),
        usage=artifact.usage,
        max_words_per_second=pipeline_config.get(
            "transcript_plausibility_max_words_per_second",
            config_default("transcript_plausibility_max_words_per_second", 8.0),
        ),
        max_chars_per_second=pipeline_config.get(
            "transcript_plausibility_max_chars_per_second",
            config_default("transcript_plausibility_max_chars_per_second", 80.0),
        ),
        min_repetitive_suffix_repetitions=pipeline_config.get(
            "transcript_plausibility_min_repetitive_suffix_repetitions",
            config_default(
                "transcript_plausibility_min_repetitive_suffix_repetitions",
                2,
            ),
        ),
        min_repetitive_suffix_words=pipeline_config.get(
            "transcript_plausibility_min_repetitive_suffix_words",
            config_default("transcript_plausibility_min_repetitive_suffix_words", 2),
        ),
        max_repetitive_unit_words=pipeline_config.get(
            "transcript_plausibility_max_repetitive_unit_words",
            config_default("transcript_plausibility_max_repetitive_unit_words", 0),
        ),
    )
    artifact.metadata = dict(artifact.metadata or {})
    artifact.metadata["character_run_normalization"] = (
        character_normalization
    )
    artifact.metadata["chunk_plausibility"] = report
    loop_report = transcript_loop_anomaly_report(
        artifact.text,
        max_identical_token_run=int(
            pipeline_config.get(
                "local_max_identical_token_run",
                config_default("local_max_identical_token_run", 7),
            )
        ),
        max_character_run=maximum_character_run,
        phrase_loop_min_repetitions=int(
            pipeline_config.get(
                "local_phrase_loop_min_repetitions",
                config_default("local_phrase_loop_min_repetitions", 2),
            )
        ),
        phrase_loop_min_words=int(
            pipeline_config.get(
                "local_phrase_loop_min_words",
                config_default("local_phrase_loop_min_words", 2),
            )
        ),
        phrase_loop_max_unit_words=int(
            pipeline_config.get(
                "local_phrase_loop_max_unit_words",
                config_default("local_phrase_loop_max_unit_words", 0),
            )
        ),
    )
    artifact.metadata["chunk_loop_anomaly"] = loop_report
    blocking = [
        problem
        for problem in report.get("problems") or []
        if problem != "empty_transcript"
    ]
    if blocking:
        raise ChunkArtifactValidationError(
            "Provider returned an implausible bounded-chunk transcript "
            f"for chunk {chunk.get('index')}: {blocking}.",
            artifact,
        )
    return artifact


def transcript_loop_anomaly_report(
    text,
    *,
    max_identical_token_run=7,
    max_character_run=11,
    phrase_loop_min_repetitions=2,
    phrase_loop_min_words=2,
    phrase_loop_max_unit_words=None,
):
    tokens = [
        re.sub(r"[^\w]+", "", token.casefold(), flags=re.UNICODE)
        for token in normalize_subtitle_text(text).split()
    ]
    tokens = [token for token in tokens if token]
    longest_token_run = 0
    token_run_value = None
    current_value = None
    current_count = 0
    for token in tokens:
        if token == current_value:
            current_count += 1
        else:
            current_value = token
            current_count = 1
        if current_count > longest_token_run:
            longest_token_run = current_count
            token_run_value = token
    longest_character_run = 0
    character_run_value = None
    for token in tokens:
        match = max(
            (
                (len(group.group(0)), group.group(0)[0])
                for group in re.finditer(r"(.)\1+", token, flags=re.UNICODE)
            ),
            default=(1 if token else 0, token[:1] or None),
        )
        if match[0] > longest_character_run:
            longest_character_run = match[0]
            character_run_value = match[1]
    phrase_loop = {
        "detected": False,
        "unit_text": None,
        "unit_word_count": 0,
        "repetition_count": 0,
        "repeated_word_count": 0,
        "start_word_index": None,
        "end_word_index": None,
    }
    minimum_repetitions = max(2, int(phrase_loop_min_repetitions or 2))
    minimum_words = max(2, int(phrase_loop_min_words))
    exact_runs = contiguous_token_repetition_runs(
        tokens,
        min_repetitions=minimum_repetitions,
        min_repeated_tokens=minimum_words,
    )
    configured_maximum = int(phrase_loop_max_unit_words or 0)
    if configured_maximum > 0:
        exact_runs = [
            run
            for run in exact_runs
            if int(run["unit_tokens"]) <= configured_maximum
        ]
    if exact_runs:
        selected = max(
            exact_runs,
            key=lambda run: (
                run["repeated_token_count"],
                run["text_occurrences"],
                -run["unit_tokens"],
            ),
        )
        phrase_loop = {
            "detected": True,
            "unit_text": selected["unit_text"],
            "unit_word_count": selected["unit_tokens"],
            "repetition_count": selected["text_occurrences"],
            "repeated_word_count": selected["repeated_token_count"],
            "start_word_index": selected["start_token_index"],
            "end_word_index": selected["end_token_index"],
            "detection_method": selected["detection_method"],
        }
    candidate_codes = []
    if longest_token_run > int(max_identical_token_run):
        candidate_codes.append("extreme_identical_token_run")
    if longest_character_run > int(max_character_run):
        candidate_codes.append("extreme_single_character_run")
    if phrase_loop["detected"]:
        candidate_codes.append("contiguous_phrase_repetition")
    return {
        "schema": "subgen_transcript_repetition_candidates_v3",
        "accept": True,
        "problems": [],
        "candidate_codes": candidate_codes,
        "requires_confirmation": bool(candidate_codes),
        "text_only_rejection_allowed": False,
        "longest_identical_token_run": longest_token_run,
        "identical_token": token_run_value,
        "longest_character_run": longest_character_run,
        "repeated_character": character_run_value,
        "max_identical_token_run": int(max_identical_token_run),
        "max_character_run": int(max_character_run),
        "phrase_loop": phrase_loop,
        "phrase_loop_min_repetitions": minimum_repetitions,
        "phrase_loop_min_words": minimum_words,
        "phrase_loop_max_unit_words": (
            configured_maximum if configured_maximum > 0 else None
        ),
    }


def transcribe_provider_longform(
    media_path,
    *,
    provider,
    model,
    language,
    prompt,
    pipeline_config,
    pipeline_plan,
    output_dir,
    source_sha256=None,
    precomputed_speech_map=None,
    supplemental_timing_segments=None,
    force_expected_timing=None,
):
    """Run every remote provider through the same bounded coordinator."""
    timing_kind = pipeline_plan.transcription.adapter_timing_kind
    expected_timing = bool(
        force_expected_timing
        if force_expected_timing is not None
        else (
            provider in {"google", "xai"}
            or (
                provider == "openai"
                and model in {"whisper-1", "gpt-4o-transcribe-diarize"}
            )
            or timing_kind in {"native_word", "native_segment", "prompted_segment"}
        )
    )
    timing_evidence_only = bool(force_expected_timing is True)

    def transcribe_chunk(chunk_audio_path, chunk):
        if provider == "google":
            artifact = call_google_timestamped_transcription(
                pipeline_config,
                chunk_audio_path,
                model=model,
                language=language,
                prompt=prompt,
                allow_empty=True,
            )
        elif provider == "openai" and model in {
            "whisper-1",
            "gpt-4o-transcribe",
            "gpt-4o-mini-transcribe",
            "gpt-4o-transcribe-diarize",
        }:
            artifact = call_openai_audio_transcription(
                pipeline_config,
                chunk_audio_path,
                model=model,
                language=language,
                prompt=prompt,
                allow_empty=True,
            )
        elif provider == "openai":
            text, usage = transcribe_audio_openai_llm(
                chunk_audio_path,
                model=model,
                language=language,
                prompt=prompt,
                api_env_key=pipeline_config.get(
                    "openai_api_env_key",
                    "OPENAI_API_KEY",
                ),
            )
            artifact = TranscriptionArtifact(
                provider=provider,
                model=model,
                text=text,
                language=language,
                duration=chunk.get("duration"),
                timing_kind="none",
                usage=usage,
            )
        elif provider == "xai":
            artifact = call_xai_transcription(
                pipeline_config,
                chunk_audio_path,
                model=model,
                language=language,
                prompt=prompt,
                allow_empty=True,
            )
        elif provider == "cohere":
            artifact = call_cohere_transcription(
                pipeline_config,
                chunk_audio_path,
                model=model,
                language=language,
                allow_empty=True,
            )
        elif pipeline_plan.transcription.adapter == "openai_compatible_audio_transcription":
            artifact = call_openai_compatible_audio_transcription(
                pipeline_config,
                provider,
                chunk_audio_path,
                model=model,
                timing_kind=timing_kind,
                language=language,
                prompt=prompt,
                allow_empty=True,
            )
        else:
            raise RuntimeError(
                f"Long-form transcription is not implemented for {provider}/{model}."
            )
        if transcript_looks_like_prompt_echo(artifact.text, prompt):
            artifact.text = ""
            artifact.segments = []
            artifact.words = []
            artifact.metadata = dict(artifact.metadata or {})
            artifact.metadata["prompt_echo_rejected"] = True
        if timing_evidence_only and artifact.words:
            artifact.segments = [
                {
                    key: word[key]
                    for key in (
                        "start",
                        "end",
                        "text",
                        "speaker",
                        "language",
                        "confidence",
                        "probability",
                    )
                    if word.get(key) is not None
                }
                for word in artifact.words
            ]
            artifact.text = " ".join(
                str(segment.get("text") or "")
                for segment in artifact.segments
            ).strip()
            artifact.metadata = dict(artifact.metadata or {})
            artifact.metadata.update({
                "timing_evidence_only": True,
                "timing_segments_derived_from_words": True,
                "provider_segment_text_is_not_canonical": True,
            })
        return _validate_chunk_artifact_plausibility(
            artifact,
            chunk,
            pipeline_config,
        )

    result = run_longform_transcription(
        media_path,
        output_dir,
        provider=provider,
        model=model,
        language=language,
        prompt=prompt,
        transcribe_chunk=transcribe_chunk,
        expected_timing=expected_timing,
        timing_evidence_only=timing_evidence_only,
        transcription_options={
            "adapter": pipeline_plan.transcription.adapter,
            "adapter_timing_kind": timing_kind,
            "expected_timing": expected_timing,
            "max_words_per_second": pipeline_config.get(
                "transcript_plausibility_max_words_per_second",
                config_default("transcript_plausibility_max_words_per_second", 8.0),
            ),
            "max_chars_per_second": pipeline_config.get(
                "transcript_plausibility_max_chars_per_second",
                config_default("transcript_plausibility_max_chars_per_second", 80.0),
            ),
            "validation_gates": {
                "max_identical_token_run": pipeline_config.get(
                    "local_max_identical_token_run",
                    config_default("local_max_identical_token_run", 7),
                ),
                "max_character_run": pipeline_config.get(
                    "local_max_character_run",
                    config_default("local_max_character_run", 11),
                ),
                "character_run_normalization_length": pipeline_config.get(
                    "local_character_run_normalization_length",
                    config_default(
                        "local_character_run_normalization_length",
                        3,
                    ),
                ),
                "phrase_loop_min_repetitions": pipeline_config.get(
                    "local_phrase_loop_min_repetitions",
                    config_default("local_phrase_loop_min_repetitions", 2),
                ),
                "phrase_loop_min_words": pipeline_config.get(
                    "local_phrase_loop_min_words",
                    config_default("local_phrase_loop_min_words", 2),
                ),
                "phrase_loop_max_unit_words": pipeline_config.get(
                    "local_phrase_loop_max_unit_words",
                    config_default("local_phrase_loop_max_unit_words", 0),
                ),
            },
        },
        supplemental_timing_segments=supplemental_timing_segments,
        source_sha256=source_sha256,
        precomputed_speech_map=precomputed_speech_map,
        **longform_runtime_options(
            pipeline_config,
            provider=provider,
            timing_evidence_only=timing_evidence_only,
        ),
    )
    return TranscriptionArtifact(**{
        key: result.get(key)
        for key in TranscriptionArtifact.__dataclass_fields__
    })


def detect_local_language_windows(
    model,
    audio_path,
    duration_seconds,
    pipeline_config,
    decoded_audio=None,
):
    """Detect language on bounded time windows for same-script code switching."""
    if not hasattr(model, "detect_language"):
        return []
    from faster_whisper.audio import decode_audio

    sample_rate = 16000
    audio = (
        decoded_audio
        if decoded_audio is not None
        else decode_audio(audio_path, sampling_rate=sample_rate)
    )
    duration = min(
        float(duration_seconds or 0.0),
        len(audio) / sample_rate,
    )
    if duration <= 0:
        return []
    maximum_windows = max(
        1,
        int(
            pipeline_config.get(
                "local_language_max_windows_per_chunk",
                config_default("local_language_max_windows_per_chunk", 24),
            )
        ),
    )
    configured_window = max(
        5.0,
        float(
            pipeline_config.get(
                "local_language_window_seconds",
                config_default("local_language_window_seconds", 15),
            )
        ),
    )
    window_seconds = max(
        configured_window,
        duration / maximum_windows,
    )
    windows = []
    start = 0.0
    while start < duration - 0.001:
        end = min(duration, start + window_seconds)
        first_sample = int(start * sample_rate)
        last_sample = int(end * sample_rate)
        try:
            language_code, probability, _ = model.detect_language(
                audio=audio[first_sample:last_sample],
                vad_filter=True,
                language_detection_segments=1,
                language_detection_threshold=0.5,
            )
            windows.append({
                "start": round(start, 3),
                "end": round(end, 3),
                "language": language_code,
                "probability": round(float(probability), 5),
            })
        except Exception as exc:
            windows.append({
                "start": round(start, 3),
                "end": round(end, 3),
                "language": None,
                "probability": None,
                "error": f"{type(exc).__name__}: {exc}",
            })
        start = end
    return windows


def resolve_local_language_windows(
    language_windows,
    min_probability,
    isolated_override_probability=0.90,
):
    resolved = [dict(window) for window in language_windows or []]
    reliable = [
        (index, window)
        for index, window in enumerate(resolved)
        if window.get("language")
        and window.get("probability") is not None
        and float(window["probability"]) >= float(min_probability)
    ]
    if not reliable:
        return resolved
    for index, window in enumerate(resolved):
        if (
            window.get("language")
            and window.get("probability") is not None
            and float(window["probability"]) >= float(min_probability)
        ):
            continue
        nearest_index, nearest = min(
            reliable,
            key=lambda item: abs(item[0] - index),
        )
        window["raw_language"] = window.get("language")
        window["raw_probability"] = window.get("probability")
        window["language"] = nearest["language"]
        window["resolved_from_window"] = nearest_index
    for index in range(1, len(resolved) - 1):
        previous = resolved[index - 1]
        current = resolved[index]
        following = resolved[index + 1]
        if (
            previous.get("language")
            and previous.get("language") == following.get("language")
            and current.get("language") != previous.get("language")
            and (
                current.get("probability") is None
                or float(current["probability"])
                < float(isolated_override_probability)
            )
        ):
            current["isolated_language"] = current.get("language")
            current["isolated_probability"] = current.get("probability")
            current["language"] = previous["language"]
            current["resolved_from_adjacent_windows"] = True
    return resolved


def local_language_runs(language_windows):
    runs = []
    for window in language_windows or []:
        language = window.get("language")
        if not language:
            continue
        start = float(window["start"])
        end = float(window["end"])
        if runs and runs[-1]["language"] == language:
            runs[-1]["end"] = end
            runs[-1]["window_count"] += 1
            runs[-1]["probabilities"].append(window.get("probability"))
        else:
            runs.append({
                "start": start,
                "end": end,
                "language": language,
                "window_count": 1,
                "probabilities": [window.get("probability")],
            })
    return runs


def language_runs_are_safe_for_retranscription(
    language_runs,
    minimum_windows_per_run=2,
):
    runs = list(language_runs or [])
    languages = {
        run.get("language")
        for run in runs
        if run.get("language")
    }
    return bool(
        len(languages) > 1
        and all(
            int(run.get("window_count") or 0)
            >= max(1, int(minimum_windows_per_run))
            for run in runs
        )
    )


def language_runs_for_retranscription(
    language_runs,
    *,
    minimum_windows_per_run=2,
    single_window_min_probability=0.90,
):
    runs = [dict(run) for run in language_runs or []]
    languages = {
        run.get("language")
        for run in runs
        if run.get("language")
    }
    if len(languages) <= 1:
        return []
    selected = []
    for run in runs:
        probabilities = [
            float(value)
            for value in run.get("probabilities") or []
            if value is not None
        ]
        if (
            int(run.get("window_count") or 0)
            >= max(1, int(minimum_windows_per_run))
            or (
                int(run.get("window_count") or 0) == 1
                and probabilities
                and min(probabilities)
                >= float(single_window_min_probability)
            )
        ):
            selected.append(run)
    return selected


def explicit_language_run_acceptance_report(
    text,
    duration_seconds,
    word_probabilities,
    pipeline_config,
):
    plausibility = transcript_plausibility_report(
        text,
        duration_seconds,
        max_words_per_second=pipeline_config.get(
            "transcript_plausibility_max_words_per_second",
            config_default(
                "transcript_plausibility_max_words_per_second",
                8.0,
            ),
        ),
        max_chars_per_second=pipeline_config.get(
            "transcript_plausibility_max_chars_per_second",
            config_default(
                "transcript_plausibility_max_chars_per_second",
                80.0,
            ),
        ),
        min_repetitive_suffix_repetitions=pipeline_config.get(
            "transcript_plausibility_min_repetitive_suffix_repetitions",
            config_default(
                "transcript_plausibility_min_repetitive_suffix_repetitions",
                2,
            ),
        ),
        min_repetitive_suffix_words=pipeline_config.get(
            "transcript_plausibility_min_repetitive_suffix_words",
            config_default(
                "transcript_plausibility_min_repetitive_suffix_words",
                2,
            ),
        ),
        max_repetitive_unit_words=pipeline_config.get(
            "transcript_plausibility_max_repetitive_unit_words",
            config_default(
                "transcript_plausibility_max_repetitive_unit_words",
                0,
            ),
        ),
    )
    loop = transcript_loop_anomaly_report(
        text,
        max_identical_token_run=int(
            pipeline_config.get(
                "local_max_identical_token_run",
                config_default("local_max_identical_token_run", 7),
            )
        ),
        max_character_run=int(
            pipeline_config.get(
                "local_max_character_run",
                config_default("local_max_character_run", 11),
            )
        ),
        phrase_loop_min_repetitions=int(
            pipeline_config.get(
                "local_phrase_loop_min_repetitions",
                config_default("local_phrase_loop_min_repetitions", 2),
            )
        ),
        phrase_loop_min_words=int(
            pipeline_config.get(
                "local_phrase_loop_min_words",
                config_default("local_phrase_loop_min_words", 2),
            )
        ),
        phrase_loop_max_unit_words=int(
            pipeline_config.get(
                "local_phrase_loop_max_unit_words",
                config_default("local_phrase_loop_max_unit_words", 0),
            )
        ),
    )
    low_word_threshold = float(
        pipeline_config.get(
            "local_low_word_probability_threshold",
            config_default("local_low_word_probability_threshold", 0.35),
        )
    )
    probabilities = [
        float(value)
        for value in word_probabilities or []
        if value is not None
    ]
    low_word_ratio = (
        sum(value < low_word_threshold for value in probabilities)
        / len(probabilities)
        if probabilities
        else 0.0
    )
    problems = list(plausibility.get("problems") or [])
    problems.extend(loop.get("problems") or [])
    if (
        probabilities
        and low_word_ratio
        > float(
            pipeline_config.get(
                "local_max_low_word_probability_ratio",
                config_default(
                    "local_max_low_word_probability_ratio",
                    0.35,
                ),
            )
        )
    ):
        problems.append("excessive_low_confidence_words")
    return {
        "schema": "subgen_explicit_language_run_acceptance_v1",
        "accept": not problems,
        "problems": list(dict.fromkeys(problems)),
        "plausibility": plausibility,
        "loop_anomaly": loop,
        "word_probability_count": len(probabilities),
        "low_word_probability_ratio": round(low_word_ratio, 5),
    }


def local_segment_language(
    text,
    start,
    end,
    language_windows,
    default_language,
):
    script_language = infer_language_from_text(text, default="")
    if script_language:
        return script_language
    center = (float(start) + float(end)) / 2.0
    candidates = [
        window
        for window in language_windows or []
        if float(window["start"]) <= center < float(window["end"])
        and window.get("language")
    ]
    if candidates:
        return candidates[0]["language"]
    return default_language or "en"


def transcribe_local_longform(
    media_path,
    *,
    model_size,
    device,
    beam_size,
    language,
    pipeline_config,
    output_dir,
    source_sha256=None,
    precomputed_speech_map=None,
):
    """Transcribe local media sequentially while reusing one model instance."""
    compute_type = "float16" if device == "cuda" else "int8"
    resolved_model_reference = resolve_faster_whisper_model_reference(
        model_size
    )
    model = None

    def transcribe_chunk(chunk_audio_path, chunk):
        nonlocal model
        if model is None:
            WhisperModel = import_faster_whisper()
            model = WhisperModel(
                resolved_model_reference,
                device=device,
                compute_type=compute_type,
            )
        multilingual_auto = bool(
            language is None
            and pipeline_config.get(
                "local_multilingual_auto",
                config_default("local_multilingual_auto", True),
            )
        )
        condition_on_previous_text = bool(
            pipeline_config.get(
                "local_condition_on_previous_text",
                config_default("local_condition_on_previous_text", False),
            )
        )
        temperature = float(
            pipeline_config.get(
                "local_temperature",
                config_default("local_temperature", 0.0),
            )
        )
        hallucination_silence_threshold = pipeline_config.get(
            "local_hallucination_silence_threshold",
            config_default("local_hallucination_silence_threshold", 2.0),
        )
        language_detection_segments = int(
            pipeline_config.get(
                "local_language_detection_segments",
                config_default("local_language_detection_segments", 3),
            )
        )
        iterator, info = model.transcribe(
            chunk_audio_path,
            beam_size=beam_size,
            vad_filter=True,
            word_timestamps=True,
            language=language,
            multilingual=multilingual_auto,
            condition_on_previous_text=condition_on_previous_text,
            temperature=temperature,
            hallucination_silence_threshold=(
                float(hallucination_silence_threshold)
                if hallucination_silence_threshold is not None
                else None
            ),
            language_detection_segments=language_detection_segments,
        )
        raw_segments = list(iterator)
        detected_language = getattr(info, "language", None) or language
        language_windows = (
            detect_local_language_windows(
                model,
                chunk_audio_path,
                chunk.get("duration"),
                pipeline_config,
            )
            if multilingual_auto
            else []
        )
        min_language_probability = float(
            pipeline_config.get(
                "local_min_language_probability",
                config_default("local_min_language_probability", 0.35),
            )
        )
        language_windows = resolve_local_language_windows(
            language_windows,
            min_language_probability,
            pipeline_config.get(
                "local_isolated_language_window_override_probability",
                config_default(
                    "local_isolated_language_window_override_probability",
                    0.90,
                ),
            ),
        )
        language_runs = local_language_runs(language_windows)
        minimum_windows_per_run = int(
            pipeline_config.get(
                "local_language_run_min_windows_for_retranscription",
                config_default(
                    "local_language_run_min_windows_for_retranscription",
                    2,
                ),
            )
        )
        single_window_min_probability = float(
            pipeline_config.get(
                "local_language_run_single_window_min_probability",
                config_default(
                    "local_language_run_single_window_min_probability",
                    0.90,
                ),
            )
        )
        attempted_explicit_language_runs = (
            language_runs_for_retranscription(
                language_runs,
                minimum_windows_per_run=minimum_windows_per_run,
                single_window_min_probability=(
                    single_window_min_probability
                ),
            )
            if multilingual_auto and hasattr(model, "detect_language")
            else []
        )
        words = []
        fallback_segments = []
        accepted_explicit_language_runs = []
        rejected_explicit_language_runs = []
        if attempted_explicit_language_runs:
            from faster_whisper.audio import decode_audio

            sample_rate = 16000
            decoded_audio = decode_audio(
                chunk_audio_path,
                sampling_rate=sample_rate,
            )
            context_seconds = 1.0
            for run in attempted_explicit_language_runs:
                run_words = []
                run_fallback_segments = []
                ownership_start = float(run["start"])
                ownership_end = float(run["end"])
                run_start = max(0.0, ownership_start - context_seconds)
                run_end = min(
                    len(decoded_audio) / sample_rate,
                    ownership_end + context_seconds,
                )
                first_sample = int(run_start * sample_rate)
                last_sample = int(run_end * sample_rate)
                run_iterator, _ = model.transcribe(
                    decoded_audio[first_sample:last_sample],
                    beam_size=beam_size,
                    vad_filter=True,
                    word_timestamps=True,
                    language=run["language"],
                    multilingual=False,
                    condition_on_previous_text=condition_on_previous_text,
                    temperature=temperature,
                    hallucination_silence_threshold=(
                        float(hallucination_silence_threshold)
                        if hallucination_silence_threshold is not None
                        else None
                    ),
                    language_detection_segments=1,
                )
                for raw_segment in list(run_iterator):
                    absolute_start = run_start + float(raw_segment.start)
                    absolute_end = run_start + float(raw_segment.end)
                    center = (absolute_start + absolute_end) / 2.0
                    if not (
                        ownership_start <= center
                        < ownership_end + 0.001
                    ):
                        continue
                    run_fallback_segments.append({
                        "start": absolute_start,
                        "end": absolute_end,
                        "text": normalize_subtitle_text(raw_segment.text),
                        "language": run["language"],
                    })
                    for word in (
                        getattr(raw_segment, "words", None) or []
                    ):
                        text = normalize_whisper_word_text(
                            getattr(word, "word", "")
                        )
                        word_start = run_start + float(word.start)
                        word_end = run_start + float(word.end)
                        word_center = (word_start + word_end) / 2.0
                        if (
                            not text
                            or word_end <= word_start
                            or not (
                                ownership_start <= word_center
                                < ownership_end + 0.001
                            )
                        ):
                            continue
                        run_words.append({
                            "start": word_start,
                            "end": word_end,
                            "text": text,
                            "probability": getattr(
                                word,
                                "probability",
                                None,
                            ),
                            "language": run["language"],
                        })
                run_text = normalize_subtitle_text(
                    " ".join(
                        item.get("text", "")
                        for item in run_fallback_segments
                    )
                )
                run_report = explicit_language_run_acceptance_report(
                    run_text,
                    ownership_end - ownership_start,
                    [
                        word.get("probability")
                        for word in run_words
                    ],
                    pipeline_config,
                )
                run_identity = {
                    key: run.get(key)
                    for key in (
                        "start",
                        "end",
                        "language",
                        "window_count",
                        "probabilities",
                    )
                }
                if run_report["accept"]:
                    words.extend(run_words)
                    fallback_segments.extend(run_fallback_segments)
                    accepted_explicit_language_runs.append({
                        **run_identity,
                        "validation": run_report,
                    })
                else:
                    rejected_explicit_language_runs.append({
                        **run_identity,
                        "validation": run_report,
                    })
        used_explicit_language_runs = bool(
            accepted_explicit_language_runs
        )
        explicit_spans = [
            (float(item["start"]), float(item["end"]))
            for item in fallback_segments
            if item.get("text")
            and float(item["end"]) > float(item["start"])
        ]
        for raw_segment in raw_segments:
            segment_start = float(raw_segment.start)
            segment_end = float(raw_segment.end)
            segment_center = (segment_start + segment_end) / 2.0
            segment_language = (
                language
                or local_segment_language(
                    getattr(raw_segment, "text", ""),
                    segment_start,
                    segment_end,
                    language_windows,
                    detected_language or "en",
                )
            )
            if not any(
                start <= segment_center < end
                for start, end in explicit_spans
            ):
                fallback_segments.append({
                    "start": segment_start,
                    "end": segment_end,
                    "text": normalize_subtitle_text(raw_segment.text),
                    "language": segment_language,
                })
            for word in (
                getattr(raw_segment, "words", None) or []
            ):
                text = normalize_whisper_word_text(
                    getattr(word, "word", "")
                )
                if (
                    not text
                    or getattr(word, "start", None) is None
                    or getattr(word, "end", None) is None
                    or float(word.end) <= float(word.start)
                ):
                    continue
                word_start = float(word.start)
                word_end = float(word.end)
                word_center = (word_start + word_end) / 2.0
                if any(
                    start <= word_center < end
                    for start, end in explicit_spans
                ):
                    continue
                words.append({
                    "start": word_start,
                    "end": word_end,
                    "text": text,
                    "probability": getattr(
                        word,
                        "probability",
                        None,
                    ),
                    "language": segment_language,
                })
        segments = group_words_for_subtitles(words)
        if not segments:
            segments = [
                item
                for item in fallback_segments
                if item["text"] and item["end"] > item["start"]
            ]
        language_probability = getattr(info, "language_probability", None)
        segment_languages = {
            segment.get("language")
            for segment in segments
            if segment.get("language")
        }
        language_detection_uncertain = bool(
            language is None
            and language_probability is not None
            and len(segment_languages) <= 1
            and float(language_probability)
            < float(
                pipeline_config.get(
                    "local_min_language_probability",
                    config_default("local_min_language_probability", 0.35),
                )
            )
        )
        if language_detection_uncertain and not chunk.get(
            "coverage_recovery_gap"
        ):
            raise RuntimeError(
                "Local language auto-detection is too uncertain for a safe "
                f"transcript (chunk {chunk.get('index')}, "
                f"language={detected_language}, "
                f"probability={float(language_probability):.3f}). "
                "Choose the source language explicitly or use a stronger "
                "multilingual transcription provider."
            )
        word_probabilities = [
            float(word["probability"])
            for word in words
            if word.get("probability") is not None
        ]
        low_word_threshold = float(
            pipeline_config.get(
                "local_low_word_probability_threshold",
                config_default("local_low_word_probability_threshold", 0.35),
            )
        )
        low_word_ratio = (
            sum(
                1
                for probability in word_probabilities
                if probability < low_word_threshold
            )
            / len(word_probabilities)
            if word_probabilities
            else 0.0
        )
        if (
            word_probabilities
            and low_word_ratio
            > float(
                pipeline_config.get(
                    "local_max_low_word_probability_ratio",
                    config_default(
                        "local_max_low_word_probability_ratio",
                        0.35,
                    ),
                )
            )
        ):
            raise RuntimeError(
                "Local transcription contains too many low-confidence words "
                f"(chunk {chunk.get('index')}, "
                f"ratio={low_word_ratio:.1%} below probability "
                f"{low_word_threshold:.2f})."
            )
        artifact_language = (
            "mixed"
            if len(segment_languages) > 1
            else (
                next(iter(segment_languages))
                if segment_languages
                else detected_language
            )
        )
        text = normalize_subtitle_text(
            " ".join(segment.get("text", "") for segment in segments)
        )
        artifact = TranscriptionArtifact(
            provider="local",
            model=model_size,
            text=text,
            segments=segments,
            words=words,
            language=artifact_language,
            duration=chunk.get("duration"),
            timing_kind="native_word",
            metadata={
                "language_probability": language_probability,
                "word_probability_count": len(word_probabilities),
                "median_word_probability": (
                    sorted(word_probabilities)[len(word_probabilities) // 2]
                    if word_probabilities
                    else None
                ),
                "low_word_probability_threshold": low_word_threshold,
                "low_word_probability_ratio": round(low_word_ratio, 5),
                "multilingual_decoding": multilingual_auto,
                "language_detection_uncertain": (
                    language_detection_uncertain
                ),
                "segment_languages": sorted(segment_languages),
                "language_windows": language_windows,
                "language_runs": language_runs,
                "explicit_language_run_retranscription": (
                    used_explicit_language_runs
                ),
                "explicit_language_run_min_windows": (
                    minimum_windows_per_run
                ),
                "explicit_language_run_single_window_min_probability": (
                    single_window_min_probability
                ),
                "explicit_language_runs": [
                    run
                    for run in accepted_explicit_language_runs
                ],
                "rejected_explicit_language_runs": [
                    run
                    for run in rejected_explicit_language_runs
                ],
                "multilingual_raw_gap_fill": bool(
                    used_explicit_language_runs
                ),
                "speaker_labeling": "not_available_in_local_faster_whisper",
                "resolved_model_reference": resolved_model_reference,
            },
        )
        return _validate_chunk_artifact_plausibility(
            artifact,
            chunk,
            pipeline_config,
        )

    try:
        result = run_longform_transcription(
            media_path,
            output_dir,
            provider="local",
            model=model_size,
            language=language,
            prompt="",
            transcribe_chunk=transcribe_chunk,
            expected_timing=True,
            transcription_options={
                "beam_size": int(beam_size),
                "resolved_model_reference": resolved_model_reference,
                "device": device,
                "compute_type": compute_type,
                "vad_filter": True,
                "word_timestamps": True,
                "multilingual_auto": bool(
                    language is None
                    and pipeline_config.get(
                        "local_multilingual_auto",
                        config_default("local_multilingual_auto", True),
                    )
                ),
                "condition_on_previous_text": pipeline_config.get(
                    "local_condition_on_previous_text",
                    config_default(
                        "local_condition_on_previous_text",
                        False,
                    ),
                ),
                "temperature": pipeline_config.get(
                    "local_temperature",
                    config_default("local_temperature", 0.0),
                ),
                "hallucination_silence_threshold": pipeline_config.get(
                    "local_hallucination_silence_threshold",
                    config_default(
                        "local_hallucination_silence_threshold",
                        2.0,
                    ),
                ),
                "language_detection_segments": pipeline_config.get(
                    "local_language_detection_segments",
                    config_default(
                        "local_language_detection_segments",
                        3,
                    ),
                ),
                "language_window_seconds": pipeline_config.get(
                    "local_language_window_seconds",
                    config_default("local_language_window_seconds", 15),
                ),
                "language_max_windows_per_chunk": pipeline_config.get(
                    "local_language_max_windows_per_chunk",
                    config_default(
                        "local_language_max_windows_per_chunk",
                        24,
                    ),
                ),
                "language_detection_vad_filter": True,
                "isolated_language_window_override_probability": (
                    pipeline_config.get(
                        "local_isolated_language_window_override_probability",
                        config_default(
                            "local_isolated_language_window_override_probability",
                            0.90,
                        ),
                    )
                ),
                "explicit_language_run_retranscription": True,
                "language_run_min_windows_for_retranscription": (
                    pipeline_config.get(
                        "local_language_run_min_windows_for_retranscription",
                        config_default(
                            "local_language_run_min_windows_for_retranscription",
                            2,
                        ),
                    )
                ),
                "language_run_single_window_min_probability": (
                    pipeline_config.get(
                        "local_language_run_single_window_min_probability",
                        config_default(
                            "local_language_run_single_window_min_probability",
                            0.90,
                        ),
                    )
                ),
                "language_run_context_seconds": 1.0,
                "validation_gates": {
                    "min_language_probability": pipeline_config.get(
                        "local_min_language_probability",
                        config_default("local_min_language_probability", 0.35),
                    ),
                    "low_word_probability_threshold": pipeline_config.get(
                        "local_low_word_probability_threshold",
                        config_default("local_low_word_probability_threshold", 0.35),
                    ),
                    "max_low_word_probability_ratio": pipeline_config.get(
                        "local_max_low_word_probability_ratio",
                        config_default("local_max_low_word_probability_ratio", 0.35),
                    ),
                    "max_identical_token_run": pipeline_config.get(
                        "local_max_identical_token_run",
                        config_default("local_max_identical_token_run", 7),
                    ),
                    "max_character_run": pipeline_config.get(
                        "local_max_character_run",
                        config_default("local_max_character_run", 11),
                    ),
                    "character_run_normalization_length": (
                        pipeline_config.get(
                            "local_character_run_normalization_length",
                            config_default(
                                "local_character_run_normalization_length",
                                3,
                            ),
                        )
                    ),
                    "phrase_loop_min_repetitions": pipeline_config.get(
                        "local_phrase_loop_min_repetitions",
                        config_default("local_phrase_loop_min_repetitions", 2),
                    ),
                    "phrase_loop_min_words": pipeline_config.get(
                        "local_phrase_loop_min_words",
                        config_default("local_phrase_loop_min_words", 2),
                    ),
                    "phrase_loop_max_unit_words": pipeline_config.get(
                        "local_phrase_loop_max_unit_words",
                        config_default("local_phrase_loop_max_unit_words", 0),
                    ),
                },
            },
            source_sha256=source_sha256,
            precomputed_speech_map=precomputed_speech_map,
            **longform_runtime_options(pipeline_config),
        )
        return TranscriptionArtifact(**{
            key: result.get(key)
            for key in TranscriptionArtifact.__dataclass_fields__
        })
    finally:
        try:
            import gc
            if model is not None:
                del model
            gc.collect()
            if device != "cpu":
                torch = import_torch()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception:
            pass


# -----------------------------
# Step 3: Translate segments
# -----------------------------
def translate_segments(
    segments,
    src_lang,
    tgt_lang,
    device="cpu",
    batch_size=8,
    backend="transformers",
    llm_model="gpt-4o",
    context_window=2,
    source_dialect="auto",
    target_dialect="natural",
    translator_notes="",
    provider_config=None,
    glossary=None,
):
    """
    Translates subtitle segments using either the local transformer backend
    or a context-aware LLM backend.
    """
    if backend != "transformers":
        return translate_segments_openai(
            segments,
            src_lang,
            tgt_lang,
            model=llm_model,
            batch_size=batch_size,
            context_window=context_window,
            source_dialect=source_dialect,
            target_dialect=target_dialect,
            translator_notes=translator_notes,
            provider_id=backend,
            provider_config=provider_config,
            glossary=glossary,
        )

    model_name = get_translation_model_name(src_lang, tgt_lang)
    if not model_name:
        return segments

    print(f"Loading translation model: {model_name}")
    torch = import_torch()
    MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM = import_transformers_translation()

    try:
        if model_name == "SeyedAli/English-to-Persian-Translation-mT5-V1":
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
    except Exception as e:
        raise RuntimeError(
            f"Could not load translation model '{model_name}' for {src_lang}->{tgt_lang}. "
            "Try another target language or install/cache the model first."
        ) from e

    if device != "cpu":
        model = model.to(device)

    model.eval()

    # -----------------------------
    # Batch translation
    # -----------------------------
    original_texts = [seg["text"] for seg in segments]
    translated_texts = []

    print(f"Translating {len(original_texts)} segments to '{tgt_lang}'...")

    for i in tqdm(range(0, len(original_texts), batch_size), desc="Translating"):
        batch = original_texts[i:i + batch_size]

        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        if device != "cpu":
            encoded = {key: value.to(device) for key, value in encoded.items()}

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
            "text": apply_glossary_to_translation(
                seg["text"],
                translated_texts[i],
                glossary,
            ),
            **{
                key: seg[key]
                for key in ("speaker", "language", "overlap", "overlap_turns")
                if seg.get(key) is not None
            },
        })

    # Clean up PyTorch resources and release VRAM
    if device != "cpu":
        try:
            del model
            del tokenizer
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

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
    video_metadata = {}
    try:
        video_metadata = probe_video_metadata(str(video_path_obj))
    except Exception:
        video_metadata = {}
    original_size = f"{int(video_metadata.get('width') or 1920)}x{int(video_metadata.get('height') or 1080)}"
    
    video_filename_relative = video_path_obj.name
    try:
        srt_filename_relative = srt_path_obj.relative_to(cwd_dir).as_posix()
    except ValueError:
        srt_filename_relative = str(srt_path_obj).replace("\\", "/").replace(":", "\\:")
    if output_path_obj.parent == cwd_dir:
        output_filename_argument = output_path_obj.name
    else:
        output_filename_argument = str(output_path_obj)

    # Base subtitle filter
    
    # Use custom style config if provided, otherwise use defaults
    if style_config:
        requested_style = dict(style_config)
        style_config = apply_script_safe_subtitle_style(requested_style, lang_code, video_metadata)
        style_config = validate_visual_style(
            style_config,
            requested_style,
            video_metadata,
            tiktok_style=bool(style_config.get("tiktok_style", False)),
            allow_position_change=False,
        )
        print(f"Manual subtitle style requested: {json.dumps(requested_style, ensure_ascii=False, sort_keys=True)}")
        print(f"Effective FFmpeg subtitle style: {json.dumps(style_config, ensure_ascii=False, sort_keys=True)}")
        vf_arg = subtitle_filter_for_style(srt_filename_relative, style_config, video_metadata, lang_code=lang_code)
    else:
        # Default styles based on language
        fonts_dir = escape_ffmpeg_filter_path(bundled_subtitle_fonts_dir())

        if uses_arabic_script_subtitle_font(lang_code):
            vf_arg = (
                f"subtitles=filename='{srt_filename_relative}':"
                f"original_size={original_size}:"
                f"charenc=UTF-8:"
                f"fontsdir='{fonts_dir}':"
                "force_style='FontName=Amiri,"
                "FontSize=28,"
                "PrimaryColour=&HFFFFFF&,"  # White
                "OutlineColour=&H000000&,"  # Black
                "BackColour=&H000000&,"     # Black
                "Shadow=1,"
                "Outline=1,"
                "BorderStyle=3,"
                "Alignment=2'"
            )
        else:
            vf_arg = (
                f"subtitles=filename='{srt_filename_relative}':"
                f"original_size={original_size}:"
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
        output_filename_argument,
    ]
    
    run_ffmpeg_with_progress(command, str(video_path_obj), "Burning subtitles", cwd=cwd_dir)
    return style_config


# -----------------------------
# Main pipeline
# -----------------------------
def main(
    video_path,
    srt_path_arg=None,
    target_language=None,
    style_config=None,
    pipeline_config=None,
    keep_files=False,
    force=False,
    cache_action=None,
    no_burn=False,
    output_dir=None,
    approved_review_path=None,
    source_location=None,
):
    pipeline_config = (pipeline_config or CONFIG).copy()
    video_path_obj = Path(video_path).resolve()
    if not video_path_obj.exists():
        print(f"Error: Video file not found at {video_path}")
        sys.exit(1)

    check_required_tools()
    requested_subtitle_mode, effective_subtitle_mode, video_metadata = resolve_subtitle_mode(
        str(video_path_obj),
        pipeline_config,
    )
    pipeline_config["subtitle_mode"] = requested_subtitle_mode
    pipeline_config["effective_subtitle_mode"] = effective_subtitle_mode
    pipeline_config["tiktok_style"] = effective_subtitle_mode == "tiktok"
    print(
        "[PIPELINE] Subtitle format: "
        f"requested={requested_subtitle_mode}, effective={effective_subtitle_mode}, "
        f"video={video_metadata.get('orientation', 'unknown')}"
    )

    pipeline_plan = build_pipeline_plan(pipeline_config, target_language=target_language)
    pipeline_config["_pipeline_plan"] = pipeline_plan.to_dict()
    print(
        f"[PIPELINE] Plan: {pipeline_plan.title} "
        f"({pipeline_plan.route}, id={pipeline_plan.plan_id})"
    )
    for stage_number, stage in enumerate(pipeline_plan.stages, start=1):
        print(f"[PIPELINE]   {stage_number}. {stage.label}")
    cache_action, cache_policy = resolve_cache_action(cache_action, force=force)
    reuse_gemini_text = cache_policy["reuse_gemini_text"]
    reuse_source_timing = cache_policy["reuse_source_timing"]
    reuse_translation = cache_policy["reuse_translation"]
    force = not reuse_translation
    style_config = dict(style_config or {})
    if effective_subtitle_mode == "tiktok":
        style_config["tiktok_style"] = True
        pipeline_config["max_chars_per_line"] = 18
        pipeline_config["max_lines"] = 1
        pipeline_config["min_subtitle_duration"] = 0.1
    else:
        style_config.pop("tiktok_style", None)
    pipeline_config["_usage_events"] = []

    # Initialize the database
    init_db()

    # Calculate both the legacy fast cache identity and a full cryptographic
    # content hash. Register the original source before any provider can fail.
    video_hash = None
    content_sha256 = None
    try:
        video_hash = calculate_video_hash(video_path_obj)
        print(f"[DB] Calculated video hash: {video_hash}")
    except Exception as e:
        print(f"[WARNING] Could not calculate video hash: {e}")
    try:
        content_sha256 = sha256_full_file(video_path_obj)
        print(f"[DB] Calculated full media SHA-256: {content_sha256}")
    except Exception as e:
        print(f"[WARNING] Could not calculate full media SHA-256: {e}")

    resolved_source_location = normalize_source_location(
        source_location
        or {"kind": "local", "path": str(video_path_obj), "name": video_path_obj.name}
    )
    if video_hash:
        register_media_source(
            video_hash,
            video_path_obj.stat().st_size,
            resolved_source_location,
            content_sha256=content_sha256,
        )

    base = video_path_obj.stem
    output_root = Path(output_dir).resolve() if output_dir else video_path_obj.parent
    output_root.mkdir(parents=True, exist_ok=True)
    approved_review = load_review(approved_review_path) if approved_review_path else None
    
    # Load manifest data from SQLite database instead of manifest.json
    manifest = {
        "pipeline_plan": pipeline_plan.to_dict(),
        "pipeline_plan_id": pipeline_plan.plan_id,
        "source_pipeline_plan_id": pipeline_plan.source_plan_id,
        "pipeline_plan_version": pipeline_plan.plan_version,
        "subtitle_mode_requested": requested_subtitle_mode,
        "subtitle_mode_effective": effective_subtitle_mode,
        "video_orientation": video_metadata.get("orientation", "unknown"),
        "content_sha256": content_sha256,
        "source_location": resolved_source_location,
    }
    cached_tx = None
    cached_text_tx = get_cached_transcription(video_hash) if video_hash else None
    if video_hash and reuse_source_timing:
        cached_tx = cached_text_tx
        if cached_tx:
            print("[DB] Found cached transcription in SQLite database.")
            cached_alignment_info = cached_tx["alignment_info"] or {}
            manifest.update({
                "detected_language": cached_tx["detected_language"],
                "transcription_provider": cached_tx["transcription_provider"],
                "transcription_model": cached_tx["transcription_model"],
                "raw_segments": cached_tx["segments"],
                "alignment_info": cached_alignment_info,
                "api_transcript_timing_mode": cached_alignment_info.get("timing_mode"),
                "timing_anchor_provider": cached_alignment_info.get("timing_anchor_provider"),
                "timing_alignment_version": cached_alignment_info.get("timing_alignment_version"),
                "thinking_config_version": cached_tx.get("request_config_version"),
            })
            
    device = resolve_device(pipeline_config.get("device", "auto"))
    output_path = None
    video_duration_seconds = cached_tx["duration"] if cached_tx else None
    
    if srt_path_arg:
        # A specific SRT file was provided, so only run the burn-in step.
        print("SRT file provided. Skipping transcription and translation...")
        srt_path = Path(srt_path_arg).resolve()
        if not srt_path.exists():
            print(f"Error: Provided SRT file not found at {srt_path_arg}")
            sys.exit(1)
        if pipeline_config.get("review_before_burn", True):
            if not approved_review:
                raise RuntimeError(
                    "Burn is fail-closed: provide an approved review manifest for this exact subtitle draft."
                )
            burn_gate = assert_burn_allowed(approved_review, video_path_obj)
            if hashlib.sha256(srt_path.read_bytes()).hexdigest() != burn_gate["approved_draft_hash"]:
                raise RuntimeError("Provided SRT bytes do not match the approved draft hash.")

        is_tiktok = bool(pipeline_config.get("tiktok_style", False))
        mp4_suffix = "_tiktok" if is_tiktok else ""
        burn_lang_code = target_language or "en"
        output_path = output_root / f"{base}{mp4_suffix}_subtitled_{burn_lang_code}.mp4"
        effective_style_config = style_config or {}
        
        # Check if the subtitle styling has changed compared to the last run
        style_changed = False
        manifest_style = get_burned_style(video_hash) if video_hash else {}
        for key in [
            "font_name", "font_size", "font_scale", "primary_color", "outline_color",
            "back_color", "bg_opacity", "outline_width", "shadow", "border_style",
            "alignment", "margin_v", "margin_l", "margin_r", "tiktok_style",
        ]:
            if manifest_style.get(key) != style_config.get(key):
                style_changed = True
                break

        # Re-burn if the video doesn't exist, if force is True, if the SRT file was updated, or if the styling changed
        if output_path.exists() and not force and not style_changed and srt_path.exists() and (srt_path.stat().st_mtime <= output_path.stat().st_mtime):
            print(f"Existing output video found, and subtitles/styling are up-to-date. Skipping burn: {output_path}")
        else:
            if style_changed:
                print("Subtitle styling changed. Re-burning video...")
            elif srt_path.exists() and output_path.exists() and (srt_path.stat().st_mtime > output_path.stat().st_mtime):
                print("Subtitles file updated. Re-burning video...")
            
            if style_config is not None:
                effective_style_config = dict(style_config)
                visual_style_report = {
                    "enabled": False,
                    "accept": True,
                    "summary": "Using the manually configured subtitle style without AI replacement.",
                }
                print("Visual style QA: manual style selected; preserving the user's configuration.")
            else:
                effective_style_config, visual_style_report = resolve_visual_style_for_video(
                    str(video_path_obj),
                    output_root,
                    burn_lang_code,
                    style_config,
                    pipeline_config,
                    srt_path=str(srt_path),
                )
            rendered_style_config = burn_subtitles(
                str(video_path_obj),
                str(srt_path),
                str(output_path),
                lang_code=burn_lang_code,
                style_config=effective_style_config,
            )
            if rendered_style_config:
                effective_style_config = rendered_style_config

        if video_hash and not no_burn:
            save_burned_style(video_hash, effective_style_config if not no_burn else style_config)
        if approved_review and output_path and output_path.exists():
            complete_burn(approved_review, srt_path, output_path)
            save_review(approved_review_path, approved_review)
            save_review_manifest(approved_review)

    else:
        # No SRT file provided, run the full pipeline.
        audio_path = output_root / f"{base}_audio.wav"
        api_audio_path = output_root / f"{base}_api_audio.mp3"
        api_transcript_path = output_root / f"{base}.api_transcript.txt"
        transcription_usage = None
        transcription_containment = None
        recovery_attempts = []
        longform_enabled = bool(
            pipeline_config.get(
                "longform_enabled",
                config_default("longform_enabled", True),
            )
        )
        longform_speech_map = None
        longform_source_sha256 = None
        canonical_provider_segments = []
        automatic_acoustic_report = None

        def ensure_source_verifier_audio(for_api_transcription):
            if for_api_transcription:
                if not api_audio_path.exists():
                    print("Preparing compact audio for source timing verification...")
                    extract_api_audio(
                        str(video_path_obj),
                        str(api_audio_path),
                        sample_rate=pipeline_config.get("audio_sample_rate", 16000),
                        bitrate=pipeline_config.get("api_audio_bitrate", "64k"),
                    )
                return api_audio_path

            if not audio_path.exists():
                print("Extracting audio for source timing verification...")
                extract_audio(
                    str(video_path_obj),
                    str(audio_path),
                    sample_rate=pipeline_config.get("audio_sample_rate", 16000),
                )
            return audio_path

        def verify_source_timing_before_translation(
            transcript_text,
            source_segments,
            language,
            current_alignment_info,
            timing_anchor_segments,
        ):
            if not transcript_text or not source_segments:
                return current_alignment_info, None
            policy = pipeline_config.get(
                "source_timing_verifier_policy",
                config_default("source_timing_verifier_policy", "stop"),
            )
            if (policy or "stop").lower() in {"off", "none", "disabled"}:
                return current_alignment_info, None

            print("\nStep 2d: Verifying source subtitle timing against audio word alignment...")
            current_alignment_info = current_alignment_info or {}
            aligned_words = current_alignment_info.get("_aligned_word_segments") or []
            verification_strategy = source_timing_verification_strategy(
                aligned_words,
                timing_anchor_segments,
            )
            if verification_strategy == "timing_anchors":
                report = source_timing_anchor_report(
                    transcript_text,
                    source_segments,
                    timing_anchor_segments,
                    max_uncovered_seconds=pipeline_config.get(
                        "source_speech_coverage_max_uncovered_seconds",
                        config_default("source_speech_coverage_max_uncovered_seconds", 8.0),
                    ),
                    max_uncovered_gap_seconds=pipeline_config.get(
                        "source_speech_coverage_max_uncovered_gap_seconds",
                        config_default("source_speech_coverage_max_uncovered_gap_seconds", 8.0),
                    ),
                    max_uncovered_ratio=pipeline_config.get(
                        "source_speech_coverage_max_uncovered_ratio",
                        config_default("source_speech_coverage_max_uncovered_ratio", 0.08),
                    ),
                    padding_seconds=pipeline_config.get(
                        "source_speech_coverage_padding_seconds",
                        config_default("source_speech_coverage_padding_seconds", 0.5),
                    ),
                )
                report["verification_source"] = (
                    "openai_whisper_1_timing_anchors"
                    if timing_anchor_provider == "openai"
                    else f"{timing_anchor_provider}_timing_anchors"
                )
                manifest["source_timing_verifier_report"] = report
                current_alignment_info["source_timing_verifier"] = report
                report = enforce_source_srt_audio_timing(
                    report,
                    policy=policy,
                    label="Source subtitles before translation",
                )
            elif verification_strategy == "aligned_words":
                report = source_srt_audio_timing_report(
                    transcript_text,
                    source_segments,
                    aligned_words,
                    min_token_coverage=pipeline_config.get(
                        "source_timing_verifier_min_token_coverage",
                        config_default("source_timing_verifier_min_token_coverage", 0.75),
                    ),
                    min_checked_segment_ratio=pipeline_config.get(
                        "source_timing_verifier_min_checked_segment_ratio",
                        config_default("source_timing_verifier_min_checked_segment_ratio", 0.75),
                    ),
                    max_start_drift_seconds=pipeline_config.get(
                        "source_timing_verifier_max_start_drift_seconds",
                        config_default("source_timing_verifier_max_start_drift_seconds", 0.85),
                    ),
                    max_early_end_seconds=pipeline_config.get(
                        "source_timing_verifier_max_early_end_seconds",
                        config_default("source_timing_verifier_max_early_end_seconds", 0.30),
                    ),
                    max_late_end_seconds=pipeline_config.get(
                        "source_timing_verifier_max_late_end_seconds",
                        config_default("source_timing_verifier_max_late_end_seconds", 2.50),
                    ),
                    max_bad_segment_ratio=pipeline_config.get(
                        "source_timing_verifier_max_bad_segment_ratio",
                        config_default("source_timing_verifier_max_bad_segment_ratio", 0.05),
                    ),
                    max_bad_segments=pipeline_config.get(
                        "source_timing_verifier_max_bad_segments",
                        config_default("source_timing_verifier_max_bad_segments", 2),
                    ),
                )
                report["whisperx_alignment"] = {
                    "source": "selected_alignment_word_segments",
                    "aligned_words": len(aligned_words),
                    "language": language,
                }
                report = enforce_source_srt_audio_timing(
                    report,
                    policy=policy,
                    label="Source subtitles before translation",
                )
            else:
                verifier_audio = ensure_source_verifier_audio(transcription_provider != "local")
                report = run_source_srt_audio_timing_verifier(
                    verifier_audio,
                    transcript_text,
                    source_segments,
                    language,
                    device,
                    video_duration_seconds,
                    pipeline_config,
                )
            if report:
                speech_coverage_report = None
                if timing_anchor_segments:
                    speech_coverage_report = source_speech_coverage_report(
                        source_segments,
                        timing_anchor_segments,
                        max_uncovered_seconds=pipeline_config.get(
                            "source_speech_coverage_max_uncovered_seconds",
                            config_default("source_speech_coverage_max_uncovered_seconds", 8.0),
                        ),
                        max_uncovered_gap_seconds=pipeline_config.get(
                            "source_speech_coverage_max_uncovered_gap_seconds",
                            config_default("source_speech_coverage_max_uncovered_gap_seconds", 8.0),
                        ),
                        max_uncovered_ratio=pipeline_config.get(
                            "source_speech_coverage_max_uncovered_ratio",
                            config_default("source_speech_coverage_max_uncovered_ratio", 0.08),
                        ),
                        padding_seconds=pipeline_config.get(
                            "source_speech_coverage_padding_seconds",
                            config_default("source_speech_coverage_padding_seconds", 0.5),
                        ),
                    )
                    report["source_speech_coverage"] = speech_coverage_report
                    manifest["source_timing_verifier_report"] = report
                    current_alignment_info["source_timing_verifier"] = report
                    enforce_source_speech_coverage(
                        speech_coverage_report,
                        policy=policy,
                        label="Source subtitles before translation",
                    )
                print(
                    "Source timing verifier: "
                    f"{'pass' if report.get('accept') else 'fail'} - "
                    f"max start drift {report.get('max_abs_start_drift')}s, "
                    f"bad cues {report.get('bad_segment_count')}/"
                    f"{report.get('checked_segments')}"
                )
                if speech_coverage_report:
                    print(
                        "Source speech coverage: "
                        f"{'pass' if speech_coverage_report.get('accept') else 'fail'} - "
                        f"uncovered {speech_coverage_report.get('uncovered_speech_seconds')}s, "
                        f"max gap {speech_coverage_report.get('max_uncovered_gap_seconds')}s"
                    )
                current_alignment_info["source_timing_verifier"] = report
                current_alignment_info.pop("_aligned_word_segments", None)
                manifest["source_timing_verifier_report"] = report
            return current_alignment_info, report

        is_tiktok = bool(pipeline_config.get("tiktok_style", False))
        srt_suffix = ".tiktok" if is_tiktok else ""
        mp4_suffix = "_tiktok" if is_tiktok else ""

        # Recreate source SRT from SQLite cache if it's missing on disk but exists in DB.
        # A forced downstream run should not discard a compatible source transcript/timing cache.
        if video_hash and reuse_source_timing:
            cached_tx = get_cached_transcription(video_hash)
            if cached_tx:
                cached_alignment_info = cached_tx.get("alignment_info") or {}
                cached_alignment_is_current = (
                    cached_alignment_info.get("timing_alignment_version")
                    == CURRENT_TIMING_ALIGNMENT_VERSION
                )
                source_lang = cached_tx["detected_language"]
                source_srt = output_root / f"{base}{srt_suffix}.{source_lang}.srt"
                if not source_srt.exists() and cached_alignment_is_current:
                    print("[DB] Recreating source SRT file from database cache...")
                    if cached_tx.get("transcription_provider") != "local":
                        if pipeline_config.get("tiktok_style", False):
                            formatted_source_segments = prepare_segments_for_srt(
                                cached_tx["segments"],
                                max_chars_per_line=int(pipeline_config.get("max_chars_per_line", 42)),
                                max_lines=int(pipeline_config.get("max_lines", 2)),
                                min_duration=float(pipeline_config.get("min_subtitle_duration", 0.8)),
                            )
                        else:
                            formatted_source_segments = wrap_segments_for_srt_without_resplitting(
                                cached_tx["segments"],
                                max_chars_per_line=int(pipeline_config.get("max_chars_per_line", 42)),
                            )
                        enforce_source_subtitle_integrity(
                            cached_tx.get("transcript_text", ""),
                            formatted_source_segments,
                            label="Cached source subtitles before DB SRT recreation",
                        )
                    else:
                        formatted_source_segments = prepare_segments_for_srt(
                            cached_tx["segments"],
                            max_chars_per_line=int(pipeline_config.get("max_chars_per_line", 42)),
                            max_lines=int(pipeline_config.get("max_lines", 2)),
                            min_duration=float(pipeline_config.get("min_subtitle_duration", 0.8)),
                        )
                        enforce_subtitle_timing_integrity(
                            formatted_source_segments,
                            label="Cached source subtitles before DB SRT recreation",
                        )
                    write_srt(formatted_source_segments, source_srt)
                
                # Also recreate the api_transcript.txt file from DB cache if missing
                if not api_transcript_path.exists() and cached_tx.get("transcript_text"):
                    print("[DB] Recreating API transcript text file from database cache...")
                    api_transcript_path.write_text(cached_tx["transcript_text"], encoding="utf-8")

        source_lang = manifest.get("detected_language")
        source_srt = output_root / f"{base}{srt_suffix}.{source_lang}.srt" if source_lang else None
        final_srt = output_root / f"{base}{srt_suffix}.{target_language}.srt" if target_language else None
        transcription_provider = pipeline_config.get("transcription_provider")
        if not transcription_provider:
            transcription_provider = (
                "openai"
                if pipeline_config.get("transcription_backend") == "openai"
                else "local"
            )
        timing_anchor_provider = pipeline_config.get("timing_anchor_provider", "local")
        manifest_transcription_provider = manifest.get("transcription_provider")
        manifest_transcription_model = manifest.get("transcription_model")
        timing_mode = (
            "direct"
            if pipeline_plan.route == ROUTE_DIRECT_TIMESTAMPED_TRANSCRIPT
            else pipeline_config.get("api_transcript_timing_mode", "precise")
        )
        if transcription_provider != "local" and timing_mode == "local_whisper":
            print(
                "Warning: local_whisper uses local Whisper text and is disabled for API transcription. "
                "Using precise instead."
            )
            timing_mode = "precise"
        manifest_timing_mode = manifest.get("api_transcript_timing_mode")
        manifest_timing_anchor_provider = manifest.get("timing_anchor_provider")
        current_timing_alignment_version = CURRENT_TIMING_ALIGNMENT_VERSION
        manifest_timing_alignment_version = manifest.get("timing_alignment_version")
        cached_source_pipeline_plan_id = (
            (manifest.get("alignment_info") or {}).get("source_pipeline_plan_id")
            or (manifest.get("alignment_info") or {}).get("pipeline_plan_id")
        )
        manifest_transcription_suspect = (
            transcription_provider != "local"
            and transcript_cache_is_suspect(manifest, pipeline_config)
        )
        manifest_source_qa_report = manifest.get("source_qa_report") or {}
        manifest_source_qa_failed = (
            bool(manifest_source_qa_report)
            and pipeline_config.get("qa_enabled", True)
            and pipeline_config.get("qa_policy", "stop") == "stop"
            and not bool(manifest_source_qa_report.get("accept", True))
            and manifest_source_qa_report.get("severity") == "fail"
        )
        cache_metadata_compatible = (
            reuse_source_timing
            and
            source_lang
            and not manifest_source_qa_failed
            and not manifest_transcription_suspect
            and (
                transcription_provider != "google"
                or (
                    manifest.get("thinking_config_version")
                    == GEMINI_TRANSCRIPTION_THINKING_CONFIG_VERSION
                    and (
                        not cached_tx
                        or cached_tx.get("request_config_version")
                        == GEMINI_TRANSCRIPTION_THINKING_CONFIG_VERSION
                    )
                )
            )
            and (
                manifest_transcription_provider == transcription_provider
                or (transcription_provider == "local" and manifest_transcription_provider is None)
            )
            and (
                not manifest_transcription_model
                or manifest_transcription_model == pipeline_plan.transcription.model
            )
            and (
                cached_source_pipeline_plan_id == pipeline_plan.source_plan_id
                or (
                    not cached_source_pipeline_plan_id
                    and pipeline_plan.route != ROUTE_DIRECT_TIMESTAMPED_TRANSCRIPT
                )
            )
            and (
                transcription_provider == "local"
                or manifest_timing_mode == timing_mode
            )
            and (
                transcription_provider == "local"
                or not manifest_timing_anchor_provider
                or manifest_timing_anchor_provider == timing_anchor_provider
            )
            and (
                transcription_provider == "local"
                or manifest_timing_alignment_version == current_timing_alignment_version
            )
        )
        can_reuse_transcription = (
            cache_metadata_compatible
            and source_srt
            and source_srt.exists()
        )
        can_reuse_cached_raw_segments = (
            cache_metadata_compatible
            and bool(manifest.get("raw_segments"))
        )
        timing_segments = (
            manifest.get("raw_segments", [])
            if reuse_source_timing and cache_metadata_compatible
            else []
        )
        transcription_rebuilt = False
        alignment_info = manifest.get("alignment_info") or None
        source_qa_report = manifest.get("source_qa_report")
        source_integrity_report = manifest.get("source_integrity_report")
        source_timing_verifier_report = manifest.get("source_timing_verifier_report")
        transcript_for_qa = None
        
        if can_reuse_transcription:
            print(f"Step 1-2: Reusing existing transcription: {source_srt}")
            segments = parse_srt(source_srt)
            if transcription_provider != "local":
                cached_transcript_text = (
                    (cached_tx or {}).get("transcript_text")
                    or (
                        api_transcript_path.read_text(encoding="utf-8")
                        if api_transcript_path.exists()
                        else ""
                    )
                )
                enforce_source_subtitle_integrity(
                    cached_transcript_text,
                    segments,
                    label="Cached source subtitles before reuse",
                )
                alignment_info, source_timing_verifier_report = verify_source_timing_before_translation(
                    cached_transcript_text,
                    segments,
                    source_lang,
                    alignment_info,
                    timing_segments,
                )
            else:
                enforce_subtitle_timing_integrity(
                    segments,
                    label="Cached source subtitles before reuse",
                )
                alignment_info, source_timing_verifier_report = verify_source_timing_before_translation(
                    " ".join(segment.get("text", "") for segment in segments),
                    segments,
                    source_lang,
                    alignment_info,
                    timing_segments,
                )
        elif can_reuse_cached_raw_segments:
            print("\nReusing compatible raw word-level segments from database cache...")
            segments = manifest.get("raw_segments")
            source_lang = manifest.get("detected_language")
            source_srt = output_root / f"{base}{srt_suffix}.{source_lang}.srt"

            if transcription_provider != "local":
                if pipeline_config.get("tiktok_style", False):
                    formatted_source_segments = prepare_segments_for_srt(
                        segments,
                        max_chars_per_line=int(pipeline_config.get("max_chars_per_line", 42)),
                        max_lines=int(pipeline_config.get("max_lines", 2)),
                        min_duration=float(pipeline_config.get("min_subtitle_duration", 0.8)),
                    )
                else:
                    formatted_source_segments = wrap_segments_for_srt_without_resplitting(
                        segments,
                        max_chars_per_line=int(pipeline_config.get("max_chars_per_line", 42)),
                    )
                cached_transcript_text = (cached_tx or {}).get("transcript_text", "")
                enforce_source_subtitle_integrity(
                    cached_transcript_text,
                    formatted_source_segments,
                    label="Cached raw source subtitles before write",
                )
                alignment_info, source_timing_verifier_report = verify_source_timing_before_translation(
                    cached_transcript_text,
                    formatted_source_segments,
                    source_lang,
                    alignment_info,
                    timing_segments,
                )
            else:
                formatted_source_segments = prepare_segments_for_srt(
                    segments,
                    max_chars_per_line=int(pipeline_config.get("max_chars_per_line", 42)),
                    max_lines=int(pipeline_config.get("max_lines", 2)),
                    min_duration=float(pipeline_config.get("min_subtitle_duration", 0.8)),
                )
                enforce_subtitle_timing_integrity(
                    formatted_source_segments,
                    label="Cached raw source subtitles before write",
                )
                alignment_info, source_timing_verifier_report = verify_source_timing_before_translation(
                    " ".join(segment.get("text", "") for segment in formatted_source_segments),
                    formatted_source_segments,
                    source_lang,
                    alignment_info,
                    timing_segments,
                )
            print(f"\nWriting source subtitles to '{source_srt}'...")
            write_srt(formatted_source_segments, source_srt)
            segments = formatted_source_segments
        else:
            if manifest.get("raw_segments") and not cache_metadata_compatible:
                print(
                    "Cached timing segments are not compatible with the current "
                    "transcription/timing settings, so transcription alignment will be regenerated."
                )
            if manifest_transcription_suspect and not force:
                usage = manifest.get("transcription_usage") or {}
                print(
                    "Cached API transcript is not being reused because the previous transcription "
                    f"hit the suspect output-token ceiling ({usage.get('output_tokens')} tokens)."
                )
            alignment_info = None
            source_qa_report = None
            source_integrity_report = None
            source_timing_verifier_report = None
            transcript_for_qa = None
            artifact_language = None
            if transcription_provider != "local":
                if longform_enabled:
                    print(
                        "Step 1: Long-form mode will extract, process, checkpoint, "
                        "and remove one bounded audio chunk at a time."
                    )
                elif api_audio_path.exists() and not force:
                    print(f"Step 1: Reusing existing API audio: {api_audio_path}")
                else:
                    print("Step 1: Preparing compact audio for API transcription...")
                    extract_api_audio(
                        str(video_path_obj),
                        str(api_audio_path),
                        sample_rate=pipeline_config.get("audio_sample_rate", 16000),
                        bitrate=pipeline_config.get("api_audio_bitrate", "64k"),
                    )

                print(f"\nStep 2b: Transcribing text with provider '{transcription_provider}'...")
                if transcription_provider == "openai":
                    transcription_model = pipeline_config.get("transcription_model")
                    if not transcription_model or not (transcription_model.startswith("gpt-") or transcription_model.startswith("whisper-")):
                        transcription_model = "gpt-4o-mini-audio-preview"
                elif transcription_provider == "google":
                    google_provider = get_provider(pipeline_config, "google", "transcription")
                    transcription_model = pipeline_config.get("transcription_model")
                    if not transcription_model or not transcription_model.startswith("gemini-"):
                        transcription_model = google_provider["transcription_model"]
                elif transcription_provider == "xai":
                    transcription_model = pipeline_config.get("transcription_model") or "speech-to-text"
                elif transcription_provider == "cohere":
                    transcription_model = (
                        pipeline_config.get("transcription_model")
                        or "cohere-transcribe-03-2026"
                    )
                else:
                    transcription_model = pipeline_config.get("transcription_model") or "whisper-1"

                api_transcript_cache_suspect = (
                    api_transcript_path.exists()
                    and transcript_cache_is_suspect(manifest, pipeline_config)
                )
                force_reusable_transcript_text = None
                if (
                    reuse_gemini_text
                    and transcription_provider == "google"
                ):
                    if (
                        cached_text_tx
                        and cached_text_tx.get("transcription_provider") == transcription_provider
                        and cached_text_tx.get("transcription_model") == transcription_model
                        and cached_text_tx.get("transcript_text")
                    ):
                        force_reusable_transcript_text = cached_text_tx["transcript_text"]
                        print(
                            "Reusing Gemini transcript text from SQLite cache; "
                            + (
                                "source timing will be reused when compatible."
                                if reuse_source_timing
                                else "source timing will be regenerated."
                            )
                        )
                    elif api_transcript_path.exists():
                        force_reusable_transcript_text = api_transcript_path.read_text(encoding="utf-8")
                        print(
                            "Reusing existing Gemini transcript text from disk; "
                            + (
                                "source timing will be reused when compatible."
                                if reuse_source_timing
                                else "source timing will be regenerated."
                            )
                        )
                allow_srt_timing_recovery = (
                    not manifest.get("raw_segments")
                    and not manifest_source_qa_failed
                    and not manifest_transcription_suspect
                )
                if not timing_segments and not force and allow_srt_timing_recovery:
                    # Attempt to reconstruct timing anchors from existing SRT files to prevent re-transcription
                    srt_to_recover = None
                    recover_timing_only = False
                    if source_srt and source_srt.exists():
                        srt_to_recover = source_srt
                    elif final_srt and Path(final_srt).exists():
                        srt_to_recover = Path(final_srt)
                        recover_timing_only = True
                    elif target_language:
                        possible_srt = output_root / f"{base}{srt_suffix}.{target_language}.srt"
                        if possible_srt.exists():
                            srt_to_recover = possible_srt
                            recover_timing_only = True
                            
                    if srt_to_recover:
                        if recover_timing_only:
                            print(
                                "[CACHE] Reconstructing timing windows from translated SRT file "
                                f"without using its target-language text: {srt_to_recover.name}"
                            )
                        else:
                            print(f"[CACHE] Reconstructing timing anchors from existing source SRT file: {srt_to_recover.name}")
                        try:
                            recovered_segments = parse_srt(srt_to_recover)
                            timing_segments = (
                                timing_only_segments(recovered_segments)
                                if recover_timing_only
                                else recovered_segments
                            )
                            manifest["raw_segments"] = timing_segments
                        except Exception as e:
                            print(f"[WARNING] Failed to parse SRT for timing anchors: {e}")
                            timing_segments = []

                if force_reusable_transcript_text:
                    transcript_text = force_reusable_transcript_text
                    transcription_usage = (cached_text_tx or {}).get("transcription_usage")
                elif api_transcript_path.exists() and not force and not api_transcript_cache_suspect and timing_segments:
                    print(f"Reusing existing API transcript and timing anchors: {api_transcript_path}")
                    transcript_text = api_transcript_path.read_text(encoding="utf-8")
                    transcription_usage = manifest.get("transcription_usage")
                else:
                    if api_transcript_path.exists() and not timing_segments:
                        print("Re-transcribing because cached timing anchors are missing from manifest.")
                    if api_transcript_cache_suspect and not force:
                        usage = manifest.get("transcription_usage") or {}
                        print(
                            "Re-transcribing instead of reusing cached API transcript "
                            f"because output_tokens={usage.get('output_tokens')} indicates a capped response."
                        )
                    direct_artifact = None
                    if longform_enabled:
                        direct_artifact = transcribe_provider_longform(
                            str(video_path_obj),
                            provider=transcription_provider,
                            model=transcription_model,
                            language=pipeline_config.get("source_language"),
                            prompt=pipeline_config.get("transcription_prompt", ""),
                            pipeline_config=pipeline_config,
                            pipeline_plan=pipeline_plan,
                            output_dir=(
                                output_root
                                / f"{base}_longform_transcription"
                            ),
                        )
                        transcript_text = direct_artifact.text
                        canonical_provider_segments = direct_artifact.segments
                        transcription_usage = direct_artifact.usage
                        artifact_language = direct_artifact.language
                        longform_speech_map = (
                            direct_artifact.metadata or {}
                        ).get("speech_map")
                        longform_source_sha256 = (
                            direct_artifact.metadata or {}
                        ).get("source_sha256")
                        if pipeline_plan.route == ROUTE_DIRECT_TIMESTAMPED_TRANSCRIPT:
                            timing_segments = canonical_provider_segments
                        else:
                            # Prompted timestamps are sufficient to assign
                            # overlap ownership, but the final source timing
                            # still comes from an independent ASR anchor pass.
                            timing_segments = []
                        manifest["longform_transcription"] = (
                            direct_artifact.metadata or {}
                        )
                        manifest["canonical_provider_segments"] = (
                            canonical_provider_segments
                        )
                    elif pipeline_plan.route == ROUTE_DIRECT_TIMESTAMPED_TRANSCRIPT:
                        if transcription_provider == "openai":
                            direct_artifact = call_openai_audio_transcription(
                                pipeline_config,
                                str(api_audio_path),
                                model=transcription_model,
                                language=pipeline_config.get("source_language"),
                                prompt=pipeline_config.get("transcription_prompt", ""),
                            )
                        elif transcription_provider == "google":
                            direct_artifact = call_google_timestamped_transcription(
                                pipeline_config,
                                str(api_audio_path),
                                model=transcription_model,
                                language=pipeline_config.get("source_language"),
                                prompt=pipeline_config.get("transcription_prompt", ""),
                            )
                        elif transcription_provider == "xai":
                            direct_artifact = call_xai_transcription(
                                pipeline_config,
                                str(api_audio_path),
                                model=transcription_model,
                                language=pipeline_config.get("source_language"),
                                prompt=pipeline_config.get("transcription_prompt", ""),
                            )
                        elif pipeline_plan.transcription.adapter == "openai_compatible_audio_transcription":
                            direct_artifact = call_openai_compatible_audio_transcription(
                                pipeline_config,
                                transcription_provider,
                                str(api_audio_path),
                                model=transcription_model,
                                timing_kind=pipeline_plan.transcription.adapter_timing_kind,
                                language=pipeline_config.get("source_language"),
                                prompt=pipeline_config.get("transcription_prompt", ""),
                            )
                        else:
                            raise RuntimeError(
                                f"Direct timestamped transcription is not implemented for {transcription_provider}."
                            )
                        transcript_text = direct_artifact.text
                        timing_segments = direct_artifact.segments
                        transcription_usage = direct_artifact.usage
                        artifact_language = direct_artifact.language
                    elif transcription_provider == "openai":
                        if transcription_model in {
                            "whisper-1",
                            "gpt-4o-transcribe",
                            "gpt-4o-mini-transcribe",
                            "gpt-4o-transcribe-diarize",
                        }:
                            openai_artifact = call_openai_audio_transcription(
                                pipeline_config,
                                str(api_audio_path),
                                model=transcription_model,
                                language=pipeline_config.get("source_language"),
                                prompt=pipeline_config.get("transcription_prompt", ""),
                            )
                            transcript_text = openai_artifact.text
                            timing_segments = openai_artifact.segments
                            transcription_usage = openai_artifact.usage
                            artifact_language = openai_artifact.language
                        else:
                            transcript_text, timing_segments, transcription_usage = transcribe_audio_openai(
                                str(api_audio_path),
                                model=transcription_model,
                                language=pipeline_config.get("source_language"),
                                prompt=pipeline_config.get("transcription_prompt", ""),
                                api_env_key=pipeline_config.get("openai_api_env_key", "OPENAI_API_KEY"),
                                chunking=pipeline_config.get("api_transcription_chunking", "auto"),
                                chunk_seconds=pipeline_config.get("api_transcription_chunk_seconds", 180),
                                chunk_overlap_seconds=pipeline_config.get("api_transcription_chunk_overlap_seconds", 8),
                                chunk_output_dir=output_root / f"{base}_api_transcription_chunks",
                                sample_rate=pipeline_config.get("audio_sample_rate", 16000),
                                bitrate=pipeline_config.get("api_audio_bitrate", "64k"),
                            )
                    elif transcription_provider == "google":
                        if video_duration_seconds is None:
                            try:
                                video_duration_seconds = get_video_duration(str(video_path_obj))
                            except Exception:
                                video_duration_seconds = None
                        transcript_text, transcription_usage = call_google_transcription(
                            pipeline_config,
                            str(api_audio_path),
                            model=transcription_model,
                            language=pipeline_config.get("source_language"),
                            prompt=pipeline_config.get("transcription_prompt", ""),
                            duration_seconds=video_duration_seconds,
                            prompt_version=pipeline_config.get(
                                "google_transcription_prompt_version",
                                CURRENT_PRODUCTION_PROMPT_VERSION,
                            ),
                        )
                        timing_segments = []
                    elif transcription_provider == "cohere":
                        cohere_artifact = call_cohere_transcription(
                            pipeline_config,
                            str(api_audio_path),
                            model=transcription_model,
                            language=pipeline_config.get("source_language"),
                        )
                        transcript_text = cohere_artifact.text
                        timing_segments = []
                        transcription_usage = cohere_artifact.usage
                        artifact_language = cohere_artifact.language
                    elif pipeline_plan.transcription.adapter == "openai_compatible_audio_transcription":
                        compatible_artifact = call_openai_compatible_audio_transcription(
                            pipeline_config,
                            transcription_provider,
                            str(api_audio_path),
                            model=transcription_model,
                            timing_kind="none",
                            language=pipeline_config.get("source_language"),
                            prompt=pipeline_config.get("transcription_prompt", ""),
                        )
                        transcript_text = compatible_artifact.text
                        timing_segments = []
                        transcription_usage = compatible_artifact.usage
                        artifact_language = compatible_artifact.language
                    else:
                        raise RuntimeError(
                            f"Canonical text transcription is not implemented for {transcription_provider}."
                        )

                if video_duration_seconds is None:
                    try:
                        video_duration_seconds = get_video_duration(str(video_path_obj))
                    except Exception:
                        video_duration_seconds = None

                transcription_containment = None
                recovery_attempts = []

                transcript_text, transcription_usage, transcript_plausibility = (
                    verify_transcript_plausibility_with_retry(
                        transcript_text,
                        transcription_usage,
                        transcription_provider=transcription_provider,
                        transcription_model=transcription_model,
                        pipeline_config=pipeline_config,
                        api_audio_path=api_audio_path,
                        video_duration_seconds=video_duration_seconds,
                        output_root=output_root,
                        base=base,
                        allow_rejected_for_review=bool(
                            pipeline_config.get("review_before_burn", True)
                        ),
                    )
                )
                manifest["transcript_plausibility_report"] = transcript_plausibility

                if not timing_segments and pipeline_plan.uses_separate_timing_anchors:
                    timing_anchor_language = normalize_language_label(
                        pipeline_config.get("source_language")
                        or artifact_language
                    )
                    local_timing_anchor_fallback_model = None
                    if timing_anchor_language == "mixed":
                        timing_anchor_language = None
                    manifest["timing_anchor_language_hint"] = (
                        timing_anchor_language
                    )
                    if timing_anchor_provider == "openai":
                        try:
                            timing_anchor_prompt = pipeline_config.get("timing_anchor_prompt", "") or ""
                            if longform_enabled:
                                print(
                                    "Running OpenAI Whisper-1 through the bounded "
                                    "long-form coordinator for independent word anchors..."
                                )
                                anchor_artifact = transcribe_provider_longform(
                                    str(video_path_obj),
                                    provider="openai",
                                    model="whisper-1",
                                    language=timing_anchor_language,
                                    prompt=timing_anchor_prompt,
                                    pipeline_config=pipeline_config,
                                    pipeline_plan=pipeline_plan,
                                    output_dir=(
                                        output_root
                                        / f"{base}_longform_timing_anchors"
                                    ),
                                    source_sha256=longform_source_sha256,
                                    precomputed_speech_map=longform_speech_map,
                                    supplemental_timing_segments=(
                                        canonical_provider_segments
                                        if pipeline_config.get(
                                            "timing_anchor_canonical_segment_fallback_enabled",
                                            config_default(
                                                "timing_anchor_canonical_segment_fallback_enabled",
                                                True,
                                            ),
                                        )
                                        else None
                                    ),
                                    force_expected_timing=True,
                                )
                                timing_segments = anchor_artifact.segments
                                manifest["timing_anchor_usage"] = anchor_artifact.usage
                                manifest["longform_timing_anchors"] = (
                                    anchor_artifact.metadata or {}
                                )
                                longform_speech_map = (
                                    longform_speech_map
                                    or (anchor_artifact.metadata or {}).get("speech_map")
                                )
                                record_usage_event(
                                    pipeline_config,
                                    "openai",
                                    "whisper-1",
                                    "timing_anchor",
                                    usage=anchor_artifact.usage,
                                    duration_seconds=video_duration_seconds,
                                )
                            else:
                                anchor_chunk_seconds = float(
                                    pipeline_config.get(
                                        "api_timing_anchor_chunk_seconds",
                                        config_default("api_timing_anchor_chunk_seconds", 30),
                                    )
                                    or config_default("api_timing_anchor_chunk_seconds", 30)
                                )
                                anchor_overlap_seconds = float(
                                    pipeline_config.get(
                                        "api_timing_anchor_chunk_overlap_seconds",
                                        config_default("api_timing_anchor_chunk_overlap_seconds", 4),
                                    )
                                    or config_default("api_timing_anchor_chunk_overlap_seconds", 4)
                                )
                                # Whisper-1 prompt is a limited context/style hint,
                                # not an instruction channel.
                                _, timing_segments, _ = transcribe_audio_openai(
                                    str(api_audio_path),
                                    model="whisper-1",
                                    language=timing_anchor_language,
                                    prompt=timing_anchor_prompt,
                                    api_env_key=pipeline_config.get("openai_api_env_key", "OPENAI_API_KEY"),
                                    chunking="on",
                                    chunk_seconds=anchor_chunk_seconds,
                                    chunk_overlap_seconds=anchor_overlap_seconds,
                                    chunk_output_dir=output_root / f"{base}_timing_anchor_chunks",
                                    sample_rate=pipeline_config.get("audio_sample_rate", 16000),
                                    bitrate=pipeline_config.get("api_audio_bitrate", "64k"),
                                    tolerate_empty_chunks=True,
                                    retry_empty_speech_chunks=True,
                                    empty_chunk_retry_context_seconds=pipeline_config.get(
                                        "api_timing_anchor_empty_chunk_retry_context_seconds",
                                        config_default("api_timing_anchor_empty_chunk_retry_context_seconds", 6),
                                    ),
                                    silence_noise_db=pipeline_config.get(
                                        "api_timing_anchor_silence_noise_db",
                                        config_default("api_timing_anchor_silence_noise_db", -45),
                                    ),
                                    min_speech_seconds=pipeline_config.get(
                                        "api_timing_anchor_min_speech_seconds",
                                        config_default("api_timing_anchor_min_speech_seconds", 0.25),
                                    ),
                                    min_speech_ratio=pipeline_config.get(
                                        "api_timing_anchor_min_speech_ratio",
                                        config_default("api_timing_anchor_min_speech_ratio", 0.01),
                                    ),
                                    word_timestamps=True,
                                )
                            manifest["raw_segments"] = timing_segments
                        except Exception as e:
                            manifest["timing_anchor_failure"] = {
                                "provider": "openai",
                                "model": "whisper-1",
                                "error_type": type(e).__name__,
                                "error": str(e),
                            }
                            local_timing_anchor_fallback_model = (
                                resolve_local_timing_anchor_fallback_after_failure(
                                    pipeline_config,
                                    e,
                                )
                            )
                            print(
                                "[WARNING] OpenAI Whisper-1 timing anchors "
                                f"failed: {e}. Running the explicitly enabled "
                                "local timing-only fallback "
                                f"({local_timing_anchor_fallback_model})."
                            )
                            timing_anchor_provider = "local"

                    if timing_anchor_provider == "local":
                        anchor_model_size = (
                            local_timing_anchor_fallback_model
                            or pipeline_config.get("model_size")
                            or "small"
                        )
                        print(f"Running a local Whisper pass ({anchor_model_size}) with word timestamps to obtain precise timing anchors...")
                        try:
                            if longform_enabled:
                                anchor_artifact = transcribe_local_longform(
                                    str(video_path_obj),
                                    model_size=anchor_model_size,
                                    device=device,
                                    beam_size=1,
                                    language=pipeline_config.get("source_language"),
                                    pipeline_config=pipeline_config,
                                    output_dir=(
                                        output_root
                                        / f"{base}_longform_local_timing_anchors"
                                    ),
                                    source_sha256=longform_source_sha256,
                                    precomputed_speech_map=longform_speech_map,
                                )
                                timing_segments = anchor_artifact.segments
                                manifest["longform_timing_anchors"] = (
                                    anchor_artifact.metadata or {}
                                )
                                longform_speech_map = (
                                    longform_speech_map
                                    or (anchor_artifact.metadata or {}).get("speech_map")
                                )
                            else:
                                timing_segments, _ = transcribe_audio(
                                    str(api_audio_path),
                                    model_size=anchor_model_size,
                                    device=device,
                                    beam_size=1,
                                    word_timestamps=True
                                )
                            # Store in manifest so we don't have to run it again if cached
                            manifest["raw_segments"] = timing_segments
                        except Exception as e:
                            print(
                                f"[WARNING] Local Whisper timing anchor pass failed: {e}. "
                                "Fail-closed timing validation will reject missing anchors."
                            )
                            timing_segments = []

                terminal_trim_report = None
                middle_deletion_report = None
                middle_plain_plausibility_report = None
                automatic_acoustic_report = None
                acoustic_primary_timing_segments = (
                    independent_acoustic_timing_segments(timing_segments)
                )
                manifest["acoustic_timing_independence"] = {
                    "input_segments": len(timing_segments or []),
                    "independent_segments": len(
                        acoustic_primary_timing_segments
                    ),
                    "supplemental_segments_excluded": (
                        len(timing_segments or [])
                        - len(acoustic_primary_timing_segments)
                    ),
                }
                if (
                    transcript_plausibility.get("terminal_repetition_candidate")
                    and pipeline_config.get(
                        "automatic_repetition_deletion_enabled",
                        config_default("automatic_repetition_deletion_enabled", True),
                    )
                ):
                    automatic_repetition_audio_path = ensure_source_verifier_audio(
                        transcription_provider != "local"
                    )
                    automatic_acoustic_report = build_automatic_acoustic_repetition_report(
                        str(video_path_obj),
                        str(automatic_repetition_audio_path),
                        acoustic_primary_timing_segments,
                        pipeline_config,
                        output_root=output_root,
                        base=base,
                        duration_seconds=video_duration_seconds,
                        device=device,
                    )
                    manifest["automatic_acoustic_repetition_report"] = automatic_acoustic_report
                    acoustic_report_path = output_root / f"{base}.automatic_acoustic_repetition.json"
                    acoustic_report_path.write_text(
                        json.dumps(automatic_acoustic_report, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    transcript_text, terminal_trim_report = apply_automatic_terminal_repetition_trim(
                        transcript_text,
                        transcript_plausibility,
                        automatic_acoustic_report,
                        media_sha256=content_sha256,
                    )
                    manifest["terminal_repetition_trim_report"] = terminal_trim_report
                    if terminal_trim_report.get("applied"):
                        print(
                            "Trimmed only the independently unsupported terminal copies; "
                            "the original Gemini wording and prefix were preserved exactly."
                        )
                        transcript_plausibility = transcript_plausibility_report(
                            transcript_text,
                            video_duration_seconds,
                            usage=transcription_usage,
                            max_words_per_second=pipeline_config.get(
                                "transcript_plausibility_max_words_per_second",
                                config_default("transcript_plausibility_max_words_per_second", 8.0),
                            ),
                            max_chars_per_second=pipeline_config.get(
                                "transcript_plausibility_max_chars_per_second",
                                config_default("transcript_plausibility_max_chars_per_second", 80.0),
                            ),
                        )
                        transcript_plausibility.update({
                            "terminal_repetition_corrected": True,
                            "original_gemini_wording_preserved": True,
                            "trim_report_sha256": hashlib.sha256(
                                json.dumps(
                                    terminal_trim_report,
                                    ensure_ascii=False,
                                    sort_keys=True,
                                ).encode("utf-8")
                            ).hexdigest(),
                        })
                        timing_segments, repetition_timing_report = (
                            replace_corrected_repetition_timing_anchors(
                                timing_segments,
                                automatic_acoustic_report,
                                terminal_trim_report,
                            )
                        )
                        manifest["automatic_repetition_timing_report"] = (
                            repetition_timing_report
                        )
                        manifest["raw_segments"] = timing_segments
                        manifest["transcript_plausibility_report"] = transcript_plausibility
                    elif "middle_loop_requires_recovery" in terminal_trim_report.get("reason_codes", []):
                        if (
                            automatic_acoustic_report
                            and automatic_acoustic_report.get("media_sha256") == content_sha256
                        ):
                            evidence = trim_evidence_from_automatic_report(
                                automatic_acoustic_report
                            )
                            middle_deletion_report = middle_emission_text_trim_report(
                                transcript_text,
                                transcript_plausibility.get("repetitive_suffix") or {},
                                evidence,
                                require_automatic_inference=True,
                            )
                            manifest["middle_repetition_deletion_report"] = middle_deletion_report
                            if middle_deletion_report.get("applied"):
                                middle_plain_plausibility_report = dict(transcript_plausibility)
                                transcript_text = middle_deletion_report["corrected_text"]
                                transcript_plausibility["terminal_repetition_candidate"] = False
                                transcript_plausibility["middle_repetition_candidate"] = True
                                transcript_plausibility["original_gemini_wording_preserved"] = True

                if (
                    transcription_provider == "google"
                    and middle_deletion_report
                    and middle_deletion_report.get("applied")
                    and timing_segments
                    and video_duration_seconds
                ):
                    source_offset = float(middle_deletion_report["post_trigger_seconds"])
                    suffix_path = output_root / f"{base}.post_middle_loop.mp3"
                    attempt = {
                        "strategy": "post_middle_loop_only",
                        "generation_scope": "recovered_suffix",
                        "source_offset_seconds": source_offset,
                        "audio_end_seconds": video_duration_seconds,
                        "accepted": False,
                    }
                    try:
                        extract_audio_chunk_for_transcription(
                            str(api_audio_path), str(suffix_path), source_offset,
                            video_duration_seconds - source_offset,
                            sample_rate=pipeline_config.get("audio_sample_rate", 16000),
                            bitrate=pipeline_config.get("api_audio_bitrate", "64k"),
                        )
                        suffix = call_google_structured_transcription(
                            pipeline_config,
                            str(suffix_path),
                            model=transcription_model,
                            language=pipeline_config.get("source_language"),
                            prompt=pipeline_config.get("transcription_prompt", ""),
                            duration_seconds=video_duration_seconds - source_offset,
                            generation_scope="recovered_suffix",
                            source_offset_seconds=source_offset,
                        )
                        absolute = offset_suffix_segments(suffix["segments"], source_offset)
                        timestamp_report = timestamp_proposal_report(
                            video_hash or base,
                            absolute,
                            media_duration=video_duration_seconds,
                            independent_speech_intervals=timing_segments,
                        )
                        eligible = [
                            item for item in timestamp_report["segments"]
                            if item.get("independently_verified")
                            and float(item.get("gemini_proposed_start", 0)) >= source_offset
                        ]
                        suffix_repetition = repetition_anomaly_report(video_hash or base, eligible)
                        if any(issue.get("blocking") for issue in suffix_repetition.get("issues") or []):
                            raise RuntimeError("Post-loop recovery repeated or looped and was rejected.")
                        if not eligible:
                            raise RuntimeError("Post-loop recovery produced no independently supported later speech.")
                        recovered_text = " ".join(item.get("text", "") for item in eligible).strip()
                        if not recovered_text:
                            raise RuntimeError("Post-loop recovery contained no usable text.")
                        transcript_text = f"{transcript_text.rstrip()} {recovered_text}".strip()
                        attempt.update({
                            "accepted": True,
                            "prompt_version": suffix["prompt_version"],
                            "timestamp_report": timestamp_report,
                            "repetition_report": suffix_repetition,
                            "recovered_text": recovered_text,
                        })
                        transcription_usage = sum_numeric_usage_dicts([
                            transcription_usage, suffix.get("usage") or {},
                        ])
                        transcript_plausibility = transcript_plausibility_report(
                            transcript_text, video_duration_seconds, usage=transcription_usage
                        )
                        transcript_plausibility.update({
                            "middle_repetition_corrected": True,
                            "original_gemini_wording_preserved": True,
                            "later_speech_recovered": True,
                        })
                        manifest["transcript_plausibility_report"] = transcript_plausibility
                        transcription_containment = {
                            "trigger": "middle_repetition_with_later_speech",
                            "plain_plausibility_report": middle_plain_plausibility_report,
                            "middle_deletion_report": middle_deletion_report,
                            "recovery_attempts": [attempt],
                        }
                    except Exception as recovery_error:
                        attempt["error"] = str(recovery_error)
                        transcription_containment = {
                            "trigger": "middle_repetition_with_later_speech",
                            "plain_plausibility_report": middle_plain_plausibility_report,
                            "middle_deletion_report": middle_deletion_report,
                            "recovery_attempts": [attempt],
                            "error": str(recovery_error),
                        }
                    finally:
                        suffix_path.unlink(missing_ok=True)

                if (
                    transcription_provider == "google"
                    and not transcript_plausibility.get("accept")
                    and not transcript_plausibility.get("terminal_repetition_candidate")
                    and not transcript_plausibility.get("middle_repetition_candidate")
                    and timing_segments
                    and video_duration_seconds
                ):
                    print(
                        "Gemini plain output remains malformed. Requesting professional structured "
                        "cues for independently validated containment; timestamps remain advisory."
                    )
                    try:
                        structured = call_google_structured_transcription(
                            pipeline_config,
                            str(api_audio_path),
                            model=transcription_model,
                            language=pipeline_config.get("source_language"),
                            prompt=pipeline_config.get("transcription_prompt", ""),
                            duration_seconds=video_duration_seconds,
                            generation_scope="full_audio",
                        )
                        timestamp_report = timestamp_proposal_report(
                            video_hash or base,
                            structured["segments"],
                            media_duration=video_duration_seconds,
                            independent_speech_intervals=timing_segments,
                        )
                        contained_segments = []
                        for proposal in timestamp_report["segments"]:
                            if not proposal.get("independently_verified"):
                                continue
                            contained_segments.append({
                                **proposal,
                                "start": proposal["gemini_proposed_start"],
                                "end": proposal["gemini_proposed_end"],
                                "canonical_source": "gemini",
                            })
                        repetition_report = repetition_anomaly_report(
                            video_hash or base, contained_segments
                        )
                        coverage_report = independent_speech_coverage(
                            video_hash or base,
                            timing_segments,
                            [(cue["start"], cue["end"]) for cue in contained_segments],
                            media_duration=video_duration_seconds,
                            evidence_sources=["timing_anchor", "gemini_structured_proposal"],
                        )
                        boundary_report = estimate_recovery_boundaries(
                            coverage_report,
                            repetition_report,
                            uncertainty_seconds=float(
                                pipeline_config.get("recovery_boundary_uncertainty_seconds", 0.30)
                            ),
                        )
                        transcription_containment = {
                            "trigger": "rejected_plain_gemini_output",
                            "plain_plausibility_report": transcript_plausibility,
                            "professional_full_audio": {
                                "prompt_version": structured["prompt_version"],
                                "usage": structured["usage"],
                                "timestamp_report": timestamp_report,
                                "repetition_report": repetition_report,
                                "coverage_report": coverage_report,
                                "boundary_report": boundary_report,
                            },
                        }

                        meaningful_gaps = coverage_report.get("meaningful_uncovered_intervals") or []
                        recovery_usages = []
                        attempted_offsets = set()
                        candidate_offsets = []
                        first_boundary = boundary_report["t_first_uncovered"]
                        if meaningful_gaps and first_boundary.get("safe_for_automatic_recovery"):
                            context_margin = float(pipeline_config.get("recovery_context_margin_seconds", 0.20))
                            candidate_offsets.append((
                                "longest_safe_suffix",
                                max(0.0, first_boundary["estimated_seconds"] - context_margin),
                                "First meaningful independently detected uncovered speech.",
                            ))
                        post_boundary = boundary_report["t_post_trigger"]
                        if post_boundary.get("safe_for_automatic_recovery"):
                            candidate_offsets.append((
                                "post_trigger_suffix",
                                min(video_duration_seconds, post_boundary["estimated_seconds"] + 0.08),
                                "Start after the independently localized suspicious repetition region.",
                            ))

                        for strategy, source_offset, reason in candidate_offsets[:2]:
                            rounded_offset = round(source_offset, 3)
                            if rounded_offset in attempted_offsets or source_offset >= video_duration_seconds - 0.25:
                                continue
                            attempted_offsets.add(rounded_offset)
                            suffix_path = output_root / f"{base}.{strategy}.mp3"
                            attempt = {
                                "strategy": strategy,
                                "generation_scope": "recovered_suffix",
                                "source_offset_seconds": source_offset,
                                "audio_end_seconds": video_duration_seconds,
                                "reason": reason,
                                "accepted": False,
                            }
                            try:
                                extract_audio_chunk_for_transcription(
                                    str(api_audio_path),
                                    str(suffix_path),
                                    source_offset,
                                    video_duration_seconds - source_offset,
                                    sample_rate=pipeline_config.get("audio_sample_rate", 16000),
                                    bitrate=pipeline_config.get("api_audio_bitrate", "64k"),
                                )
                                suffix = call_google_structured_transcription(
                                    pipeline_config,
                                    str(suffix_path),
                                    model=transcription_model,
                                    language=pipeline_config.get("source_language"),
                                    prompt=pipeline_config.get("transcription_prompt", ""),
                                    duration_seconds=video_duration_seconds - source_offset,
                                    generation_scope="recovered_suffix",
                                    source_offset_seconds=source_offset,
                                    diagnostic_output_dir=(
                                        output_root
                                        / f"{base}.recovery_diagnostics"
                                        / strategy
                                    ),
                                )
                                recovery_usages.append(suffix.get("usage") or {})
                                absolute = offset_suffix_segments(suffix["segments"], source_offset)
                                suffix_timestamps = timestamp_proposal_report(
                                    video_hash or base,
                                    absolute,
                                    media_duration=video_duration_seconds,
                                    independent_speech_intervals=timing_segments,
                                )
                                suffix_repetition = repetition_anomaly_report(
                                    video_hash or base, suffix_timestamps["segments"]
                                )
                                eligible = [
                                    {**cue, "canonical_source": "gemini"}
                                    for cue in suffix_timestamps["segments"]
                                    if cue.get("independently_verified")
                                ]
                                merged = merge_validated_gemini_regions(contained_segments, eligible)
                                attempt.update({
                                    "prompt_version": suffix["prompt_version"],
                                    "request_sha256": (suffix.get("usage") or {}).get("request_sha256"),
                                    "response_sha256": (suffix.get("usage") or {}).get("response_sha256"),
                                    "timestamp_report": suffix_timestamps,
                                    "repetition_report": suffix_repetition,
                                    "merge": {"added": len(merged["added"]), "rejected": merged["rejected"]},
                                    "accepted": bool(merged["added"]),
                                })
                                if merged["added"]:
                                    contained_segments = merged["segments"]
                                    coverage_report = independent_speech_coverage(
                                        video_hash or base,
                                        timing_segments,
                                        [(cue.get("start", cue.get("gemini_proposed_start")), cue.get("end", cue.get("gemini_proposed_end"))) for cue in contained_segments],
                                        media_duration=video_duration_seconds,
                                        evidence_sources=["timing_anchor", "validated_gemini_regions"],
                                    )
                                    attempt["coverage_after_merge"] = coverage_report
                                    if coverage_report.get("accept"):
                                        recovery_attempts.append(attempt)
                                        break
                            except Exception as recovery_error:
                                attempt["error"] = str(recovery_error)
                            finally:
                                suffix_path.unlink(missing_ok=True)
                            recovery_attempts.append(attempt)

                        if not coverage_report.get("accept"):
                            bounded = recover_with_adaptive_bounded_windows(
                                pipeline_config=pipeline_config,
                                transcription_model=transcription_model,
                                api_audio_path=api_audio_path,
                                output_root=output_root,
                                base=base,
                                media_identity=video_hash or base,
                                media_duration_seconds=video_duration_seconds,
                                timing_segments=timing_segments,
                                contained_segments=contained_segments,
                                coverage_report=coverage_report,
                            )
                            transcription_containment["bounded_window_plan"] = bounded["plan"]
                            recovery_attempts.extend(bounded["attempts"])
                            recovery_usages.extend(bounded["usages"])
                            contained_segments = bounded["segments"]
                            coverage_report = bounded["coverage_report"]

                        if not contained_segments:
                            raise RuntimeError("Professional Gemini containment produced no independently supported cues.")
                        segments = contained_segments
                        transcript_text = " ".join(cue.get("text", "") for cue in segments).strip()
                        transcription_usage = sum_numeric_usage_dicts([
                            transcription_usage,
                            structured.get("usage") or {},
                            *recovery_usages,
                        ])
                        transcription_usage["prompt_version"] = structured["prompt_version"]
                        transcription_containment["recovery_attempts"] = recovery_attempts
                        transcription_containment["final_coverage_report"] = coverage_report
                        transcription_containment["boundary_report"] = boundary_report
                    except Exception as containment_error:
                        transcription_containment = {
                            "trigger": "rejected_plain_gemini_output",
                            "error": str(containment_error),
                            "plain_plausibility_report": transcript_plausibility,
                            "recovery_attempts": recovery_attempts,
                        }
                        raise RuntimeError(
                            "Malformed Gemini output could not be contained into independently supported "
                            "Gemini regions; refusing to create an incomplete automatic draft."
                        ) from containment_error

                if video_duration_seconds is None:
                    try:
                        video_duration_seconds = get_video_duration(str(video_path_obj))
                    except Exception:
                        video_duration_seconds = None
                record_usage_event(
                    pipeline_config,
                    transcription_provider,
                    transcription_model,
                    "transcription",
                    usage=transcription_usage,
                    duration_seconds=(
                        (transcription_usage or {}).get("billed_duration_seconds")
                        or video_duration_seconds
                    ),
                )
                transcript_for_qa = transcript_text
                source_lang = (
                    pipeline_config.get("source_language")
                    or artifact_language
                    or infer_language_from_text(transcript_text)
                )
                completeness_retries = int(
                    pipeline_config.get(
                        "api_transcription_completeness_retries",
                        config_default("api_transcription_completeness_retries", 1),
                    )
                    or 0
                )
                if reuse_gemini_text:
                    completeness_retries = 0
                completeness_retry_count = 0
                while True:
                    try:
                        if transcription_containment and transcription_containment.get("professional_full_audio"):
                            alignment_info = {
                                "engine": "gemini_proposals_independently_supported",
                                "timestamp_kind": "advisory_then_acoustically_validated",
                                "timestamp_provider": "google",
                                "timestamp_model": transcription_model,
                                "containment": transcription_containment,
                            }
                        elif pipeline_plan.route == ROUTE_DIRECT_TIMESTAMPED_TRANSCRIPT:
                            segments = timing_segments
                            alignment_info = {
                                "engine": "direct_timestamped_transcription",
                                "timestamp_kind": pipeline_plan.transcription.adapter_timing_kind,
                                "timestamp_provider": transcription_provider,
                                "timestamp_model": transcription_model,
                            }
                        elif (transcription_usage or {}).get("longform"):
                            if not timing_segments:
                                raise RuntimeError(
                                    "Long-form canonical text has no independent word "
                                    "timing anchors; refusing proportional timing."
                                )
                            segments, alignment_info = align_transcript_to_timing_anchors(
                                transcript_text,
                                timing_segments,
                                canonical_metadata_segments=canonical_provider_segments,
                                speech_map=longform_speech_map,
                            )
                            alignment_info.update({
                                "engine": "longform_canonical_text_to_word_anchors",
                                "forced_alignment_scope": "bounded_chunks",
                                "rough_timing_fallback_allowed": False,
                                "longform_pipeline_version": LONGFORM_PIPELINE_VERSION,
                            })
                        else:
                            segments, alignment_info = align_api_transcript_for_timing(
                                transcript_text,
                                timing_segments,
                                timing_mode,
                                api_audio_path,
                                source_lang,
                                device,
                                duration_seconds=video_duration_seconds,
                                model_name=transcription_model,
                            )
                        if longform_speech_map and segments:
                            longform_validation = validate_longform_timed_segments(
                                segments,
                                longform_speech_map,
                                max_uncovered_gap_seconds=pipeline_config.get(
                                    "longform_max_uncovered_gap_seconds",
                                    config_default(
                                        "longform_max_uncovered_gap_seconds",
                                        1.5,
                                    ),
                                ),
                                max_uncovered_ratio=pipeline_config.get(
                                    "longform_max_uncovered_ratio",
                                    config_default(
                                        "longform_max_uncovered_ratio",
                                        0.03,
                                    ),
                                ),
                            )
                            manifest["longform_source_validation"] = longform_validation
                            alignment_info["longform_source_validation"] = longform_validation
                            if (
                                not longform_validation.get("accept")
                                and transcription_provider == "google"
                                and (transcription_usage or {}).get("longform")
                                and bool(
                                    pipeline_config.get(
                                        "longform_coverage_recovery_enabled",
                                        config_default(
                                            "longform_coverage_recovery_enabled",
                                            True,
                                        ),
                                    )
                                )
                                and any(
                                    problem
                                    in {
                                        "uncovered_speech_gap",
                                        "low_speech_coverage",
                                    }
                                    for problem in (
                                        longform_validation.get("problems")
                                        or []
                                    )
                                )
                            ):
                                print(
                                    "Final canonical alignment exposed "
                                    "independently confirmed speech missing "
                                    "from the canonical transcript. Running "
                                    "bounded Gemini recovery windows..."
                                )
                                segments, alignment_recovery = (
                                    recover_longform_canonical_alignment_coverage(
                                        media_path=video_path_obj,
                                        output_root=output_root,
                                        base=base,
                                        media_identity=(
                                            video_hash or base
                                        ),
                                        media_duration_seconds=(
                                            video_duration_seconds
                                        ),
                                        segments=segments,
                                        timing_segments=timing_segments,
                                        speech_map=longform_speech_map,
                                        validation=longform_validation,
                                        pipeline_config=pipeline_config,
                                        pipeline_plan=pipeline_plan,
                                        transcription_model=(
                                            transcription_model
                                        ),
                                        source_language=source_lang,
                                    )
                                )
                                manifest[
                                    "longform_canonical_alignment_recovery"
                                ] = alignment_recovery
                                alignment_info[
                                    "longform_canonical_alignment_recovery"
                                ] = alignment_recovery
                                recovery_usage = (
                                    alignment_recovery.get("usage") or {}
                                )
                                if recovery_usage:
                                    transcription_usage = (
                                        sum_numeric_usage_dicts([
                                            transcription_usage,
                                            recovery_usage,
                                        ])
                                    )
                                    record_usage_event(
                                        pipeline_config,
                                        "google",
                                        transcription_model,
                                        "canonical_alignment_recovery",
                                        usage=recovery_usage,
                                        duration_seconds=(
                                            recovery_usage.get(
                                                "billed_duration_seconds"
                                            )
                                        ),
                                    )
                                if alignment_recovery.get("accept"):
                                    transcript_text = (
                                        normalize_subtitle_text(
                                            " ".join(
                                                segment.get("text", "")
                                                for segment in segments
                                            )
                                        )
                                    )
                                    transcript_for_qa = transcript_text
                                longform_validation = (
                                    alignment_recovery.get(
                                        "validation_after"
                                    )
                                    or longform_validation
                                )
                                manifest[
                                    "longform_source_validation"
                                ] = longform_validation
                                alignment_info[
                                    "longform_source_validation"
                                ] = longform_validation
                            if not longform_validation.get("accept"):
                                raise RuntimeError(
                                    "Final source cues failed the independent long-form "
                                    f"speech/silence gate: {longform_validation.get('problems')}."
                                )
                        break
                    except TranscriptAnchorCoverageError as e:
                        if (
                            (transcription_usage or {}).get("longform")
                            or transcription_provider != "google"
                            or completeness_retry_count >= completeness_retries
                        ):
                            raise
                        completeness_retry_count += 1
                        print(
                            "\nWarning: Gemini transcript appears incomplete against audio timing anchors. "
                            f"Retrying whole-audio Gemini transcription ({completeness_retry_count}/"
                            f"{completeness_retries}) before producing subtitles."
                        )
                        transcript_text, retry_usage = call_google_transcription(
                            pipeline_config,
                            str(api_audio_path),
                            model=transcription_model,
                            language=pipeline_config.get("source_language"),
                            prompt=pipeline_config.get("transcription_prompt", ""),
                            duration_seconds=video_duration_seconds,
                            prompt_version=PROFESSIONAL_PROMPT_VERSION,
                        )
                        transcription_usage = sum_numeric_usage_dicts([transcription_usage, retry_usage])
                        transcript_for_qa = transcript_text
                        source_lang = pipeline_config.get("source_language") or infer_language_from_text(transcript_text)
                        continue
                    except Exception as e:
                        if (transcription_usage or {}).get("longform"):
                            raise
                        if timing_mode == "forced":
                            raise
                        print(f"\nWarning: WhisperX forced alignment failed: {e}")
                        if transcription_provider != "local" and timing_segments:
                            print("Falling back to canonical transcript-to-anchor reconciliation.")
                            segments, alignment_info = align_transcript_to_timing_anchors(
                                transcript_text,
                                timing_segments,
                            )
                            alignment_info.update({
                                "forced_alignment_error": str(e),
                                "forced_alignment_error_type": type(e).__name__,
                            })
                        elif transcription_provider != "local":
                            raise RuntimeError(
                                "Forced alignment failed and no timing anchors were available; "
                                "refusing to produce source subtitles from non-canonical timing text."
                            ) from e
                        else:
                            print("Falling back to rough transcript timing.")
                            segments = build_rough_transcript_segments(transcript_text, video_duration_seconds)
                            alignment_info = {
                                "engine": "rough_transcript_fallback",
                                "forced_alignment_error": str(e),
                                "forced_alignment_error_type": type(e).__name__,
                            }
                        break
                if transcription_containment:
                    contained_path = output_root / f"{base}.contained_transcript_for_review.txt"
                    contained_path.write_text(transcript_text, encoding="utf-8")
                    manifest["contained_transcript_for_review"] = str(contained_path)
                    manifest["transcription_containment"] = transcription_containment
                else:
                    api_transcript_path.write_text(transcript_text, encoding="utf-8")
                if alignment_info is None:
                    alignment_info = {}
                alignment_info.update({
                    "timing_mode": (
                        "direct" if pipeline_plan.route == ROUTE_DIRECT_TIMESTAMPED_TRANSCRIPT else timing_mode
                    ),
                    "timing_anchor_provider": (
                        None
                        if pipeline_plan.route == ROUTE_DIRECT_TIMESTAMPED_TRANSCRIPT
                        else timing_anchor_provider
                    ),
                    "timing_alignment_version": current_timing_alignment_version,
                    "pipeline_plan_id": pipeline_plan.plan_id,
                    "source_pipeline_plan_id": pipeline_plan.source_plan_id,
                    "pipeline_plan_version": pipeline_plan.plan_version,
                })
                source_srt = output_root / f"{base}{srt_suffix}.{source_lang}.srt"
            elif transcription_provider == "local":
                if longform_enabled:
                    print(
                        "Step 1: Local long-form mode will process one bounded "
                        "audio chunk at a time."
                    )
                    print("\nStep 2: Transcribing locally with word timestamps...")
                    local_artifact = transcribe_local_longform(
                        str(video_path_obj),
                        model_size=pipeline_config.get("model_size", "small"),
                        device=device,
                        beam_size=int(pipeline_config.get("beam_size", 5)),
                        language=pipeline_config.get("source_language"),
                        pipeline_config=pipeline_config,
                        output_dir=(
                            output_root
                            / f"{base}_longform_local_transcription"
                        ),
                    )
                    segments = local_artifact.segments
                    timing_segments = [dict(segment) for segment in segments]
                    transcript_for_qa = local_artifact.text
                    source_lang = (
                        pipeline_config.get("source_language")
                        or local_artifact.language
                        or infer_language_from_text(local_artifact.text)
                    )
                    source_srt = output_root / f"{base}{srt_suffix}.{source_lang}.srt"
                    transcription_usage = local_artifact.usage
                    longform_speech_map = (
                        local_artifact.metadata or {}
                    ).get("speech_map")
                    longform_source_sha256 = (
                        local_artifact.metadata or {}
                    ).get("source_sha256")
                    manifest["longform_transcription"] = (
                        local_artifact.metadata or {}
                    )
                    alignment_info = {
                        "engine": "local_longform_native_word_timestamps",
                        "timing_kind": local_artifact.timing_kind,
                        "longform_pipeline_version": LONGFORM_PIPELINE_VERSION,
                    }
                else:
                    if audio_path.exists() and not force:
                        print(f"Step 1: Reusing existing audio: {audio_path}")
                    else:
                        print("Step 1: Extracting audio...")
                        extract_audio(
                            str(video_path_obj),
                            str(audio_path),
                            sample_rate=pipeline_config.get("audio_sample_rate", 16000),
                        )

                    print("\nStep 2: Transcribing audio locally with word timestamps...")
                    segments, info = transcribe_audio(
                        str(audio_path),
                        model_size=pipeline_config.get("model_size", "small"),
                        device=device,
                        beam_size=int(pipeline_config.get("beam_size", 5)),
                        word_timestamps=True,
                    )
                    source_lang = info.language
                    source_srt = output_root / f"{base}{srt_suffix}.{source_lang}.srt"
                    transcription_usage = None
            else:
                provider = get_provider(pipeline_config, transcription_provider)
                raise RuntimeError(f"{provider['name']} does not support transcription in SubGen.")

            raw_segments = [dict(s) for s in segments]
            if transcription_provider != "local" and transcript_for_qa:
                if pipeline_config.get("tiktok_style", False):
                    formatted_source_segments = prepare_segments_for_srt(
                        segments,
                        max_chars_per_line=int(pipeline_config.get("max_chars_per_line", 42)),
                        max_lines=int(pipeline_config.get("max_lines", 2)),
                        min_duration=float(pipeline_config.get("min_subtitle_duration", 0.8)),
                    )
                else:
                    formatted_source_segments = wrap_segments_for_srt_without_resplitting(
                        segments,
                        max_chars_per_line=int(pipeline_config.get("max_chars_per_line", 42)),
                    )
                source_integrity_report = enforce_source_subtitle_integrity(
                    transcript_for_qa,
                    formatted_source_segments,
                    label="Source subtitles before translation",
                )
                if alignment_info is None:
                    alignment_info = {}
                alignment_info["source_integrity"] = source_integrity_report
            else:
                formatted_source_segments = prepare_segments_for_srt(
                    segments,
                    max_chars_per_line=int(pipeline_config.get("max_chars_per_line", 42)),
                    max_lines=int(pipeline_config.get("max_lines", 2)),
                    min_duration=float(pipeline_config.get("min_subtitle_duration", 0.8)),
                )
                enforce_subtitle_timing_integrity(
                    formatted_source_segments,
                    label="Source subtitles before write",
                )
            if longform_speech_map:
                formatted_longform_validation = validate_longform_timed_segments(
                    formatted_source_segments,
                    longform_speech_map,
                    max_uncovered_gap_seconds=pipeline_config.get(
                        "longform_max_uncovered_gap_seconds",
                        config_default("longform_max_uncovered_gap_seconds", 1.5),
                    ),
                    max_uncovered_ratio=pipeline_config.get(
                        "longform_max_uncovered_ratio",
                        config_default("longform_max_uncovered_ratio", 0.03),
                    ),
                )
                manifest["longform_formatted_source_validation"] = (
                    formatted_longform_validation
                )
                if not formatted_longform_validation.get("accept"):
                    raise RuntimeError(
                        "Formatted source subtitles failed the independent "
                        "speech/silence gate: "
                        f"{formatted_longform_validation.get('problems')}."
                    )
            verifier_transcript_text = (
                transcript_for_qa
                if transcription_provider != "local"
                else " ".join(segment.get("text", "") for segment in formatted_source_segments)
            )
            alignment_info, source_timing_verifier_report = verify_source_timing_before_translation(
                verifier_transcript_text,
                formatted_source_segments,
                source_lang,
                alignment_info,
                timing_segments,
            )

            print(f"\nWriting source subtitles to '{source_srt}'...")
            write_srt(formatted_source_segments, source_srt)
            segments = formatted_source_segments
            transcription_rebuilt = True

            if transcription_provider != "local" and transcript_for_qa:
                print("\nStep 2c: Reviewing source subtitles with LLM QA...")
                source_qa_report = run_source_subtitle_qa(
                    transcript_for_qa,
                    formatted_source_segments,
                    alignment_info,
                    source_lang,
                    target_language,
                    pipeline_config,
                )
                source_qa_report = reconcile_source_qa_with_deterministic_gates(
                    source_qa_report,
                    source_integrity_report=source_integrity_report,
                    source_timing_verifier_report=source_timing_verifier_report,
                )
                print(
                    "Source QA: "
                    f"{source_qa_report.get('severity', 'ok')} - "
                    f"{source_qa_report.get('summary', '')}"
                )
                if False: # Disabled automatic override to respect user-selected timing mode
                    print(
                        "Source QA failed after a weak timing/text mode. "
                        "Retrying with precise API-transcript alignment before stopping."
                    )
                    timing_mode = "precise"
                    segments, alignment_info = align_api_transcript_for_timing(
                        transcript_for_qa,
                        timing_segments,
                        timing_mode,
                        api_audio_path,
                        source_lang,
                        device,
                        model_name=transcription_model,
                    )
                    formatted_source_segments = prepare_segments_for_srt(
                        segments,
                        max_chars_per_line=int(pipeline_config.get("max_chars_per_line", 42)),
                        max_lines=int(pipeline_config.get("max_lines", 2)),
                        min_duration=float(pipeline_config.get("min_subtitle_duration", 0.8)),
                    )
                    print(f"Rewriting source subtitles after QA retry: '{source_srt}'...")
                    write_srt(formatted_source_segments, source_srt)
                    segments = formatted_source_segments
                    source_qa_report = run_source_subtitle_qa(
                        transcript_for_qa,
                        formatted_source_segments,
                        alignment_info,
                        source_lang,
                        target_language,
                        pipeline_config,
                    )
                    print(
                        "Source QA after retry: "
                        f"{source_qa_report.get('severity', 'ok')} - "
                        f"{source_qa_report.get('summary', '')}"
                    )

            manifest.update({
                "input": str(video_path_obj),
                "detected_language": source_lang,
                "transcription_provider": transcription_provider,
                "transcription_model": (
                    pipeline_config.get("transcription_model", "whisper-1")
                    if transcription_provider != "local"
                    else pipeline_config.get("model_size", "small")
                ),
                "transcription_usage": transcription_usage,
                "thinking_config_version": (
                    (transcription_usage or {}).get("thinking_config_version")
                    if transcription_provider == "google"
                    else None
                ),
                "device": device,
                "source_srt": str(source_srt),
                "api_transcript": str(api_transcript_path) if transcription_provider != "local" else None,
                "api_transcript_timing_mode": timing_mode if transcription_provider != "local" else None,
                "alignment_info": alignment_info if (transcription_provider != "local" or alignment_info) else None,
                "source_integrity_report": source_integrity_report if transcription_provider != "local" else None,
                "source_timing_verifier_report": source_timing_verifier_report,
                "source_qa_report": source_qa_report if transcription_provider != "local" else None,
                "openai_profile": (
                    pipeline_config.get("openai_profile")
                    if transcription_provider == "openai"
                    else None
                ),
                "raw_segments": raw_segments,
            })
            if source_qa_report:
                enforce_source_qa_policy(source_qa_report, pipeline_config)
            cache_eligible = (
                transcription_provider == "local"
                or (
                    not transcription_containment
                    and bool((transcript_plausibility or {}).get("accept"))
                )
            )
            manifest["transcription_cache_eligible"] = cache_eligible
            if video_hash and cache_eligible:
                save_transcription(
                    video_hash,
                    file_size=video_path_obj.stat().st_size,
                    duration=video_duration_seconds,
                    detected_language=source_lang,
                    provider=transcription_provider,
                    model=manifest["transcription_model"],
                    transcript_text=transcript_text if transcription_provider != "local" else " ".join(s["text"] for s in raw_segments),
                    segments=raw_segments,
                    alignment_info=alignment_info,
                    prompt_version=(transcription_usage or {}).get("prompt_version"),
                    request_config_version=(
                        (transcription_usage or {}).get("thinking_config_version")
                    ),
                )

        # Decide the final segments and SRT path.
        final_segments = segments
        srt_lang_code = source_lang
        srt_path = source_srt
        source_segments_signature = segments_cache_signature(segments)
        translation_rebuilt = False

        if target_language and target_language != source_lang:
            srt_path = output_root / f"{base}{srt_suffix}.{target_language}.srt"
            srt_lang_code = target_language
            # Check if translation exists in DB cache first, even if SRT was deleted from disk
            cached_tl = (
                get_cached_translation(video_hash, target_language, source_segments_signature)
                if video_hash
                else None
            )
            translated_srt_is_current = (
                srt_path.exists()
                and not force
                and not transcription_rebuilt
                and (
                    not source_srt
                    or not Path(source_srt).exists()
                    or srt_path.stat().st_mtime >= Path(source_srt).stat().st_mtime
                )
            )
            if translated_srt_is_current:
                try:
                    candidate_final_segments = parse_srt(srt_path)
                    enforce_translated_subtitle_alignment(
                        segments,
                        candidate_final_segments,
                        label="Existing translated subtitles before reuse",
                    )
                    print(f"Step 3: Reusing existing translated subtitles: {srt_path}")
                    final_segments = candidate_final_segments
                except Exception as e:
                    print(
                        "Step 3: Existing translated subtitles failed alignment "
                        f"validation and will be regenerated: {e}"
                    )
                    translated_srt_is_current = False
            elif srt_path.exists() and not force:
                print(
                    "Step 3: Existing translated subtitles are stale relative to "
                    "the current source timing."
                )
            if translated_srt_is_current:
                pass
            elif cached_tl and not force and not transcription_rebuilt:
                print(f"Step 3: Reusing translated subtitles from database cache...")
                final_segments = cached_tl["segments"]
                enforce_translated_subtitle_alignment(
                    segments,
                    final_segments,
                    label="Cached translated subtitles before write",
                )
                write_srt(final_segments, srt_path)
            else:
                print(f"\nStep 3: Translating from '{source_lang}' to '{target_language}'...")
                translated_segments = translate_segments(
                    segments,
                    src_lang=source_lang,
                    tgt_lang=target_language,
                    device=device,
                    batch_size=int(pipeline_config.get("translation_batch_size", 8)),
                    backend=(
                        "transformers"
                        if pipeline_config.get("translation_provider", "local") == "local"
                        else pipeline_config.get("translation_provider")
                    ),
                    llm_model=pipeline_config.get("translation_model") or pipeline_config.get("llm_model", "gpt-4o"),
                    context_window=int(pipeline_config.get("translation_context_window", 2)),
                    source_dialect=pipeline_config.get("source_dialect", "auto"),
                    target_dialect=pipeline_config.get("target_dialect", "natural"),
                    translator_notes=pipeline_config.get("translator_notes", ""),
                    provider_config=pipeline_config,
                    glossary=pipeline_config.get("translation_glossary", []),
                )
                final_segments = wrap_segments_for_srt_without_resplitting(
                    translated_segments,
                    max_chars_per_line=int(pipeline_config.get("max_chars_per_line", 42)),
                )
                translation_rebuilt = True
                enforce_translated_subtitle_alignment(
                    segments,
                    final_segments,
                    label="Translated subtitles before write",
                )
        else:
            print("\nStep 3: Skipping translation.")

        translation_qa_report = None
        if target_language and target_language != source_lang and translation_rebuilt:
            print("\nStep 3b: Verifying cue-level and document-level translation semantics...")
            final_segments, translation_qa_report = verify_and_repair_translation_semantics(
                segments,
                final_segments,
                source_lang,
                target_language,
                pipeline_config,
            )
            enforce_translated_subtitle_alignment(
                segments,
                final_segments,
                label="Semantically verified translated subtitles",
            )
            print(f"\nStep 4: Writing semantically verified translated subtitles to '{srt_path}'...")
            write_srt(final_segments, srt_path)
            print(
                "Translation semantic QA: "
                f"{'pass' if translation_qa_report.get('accept') else 'fail'} - "
                f"{translation_qa_report.get('summary', '')}"
            )
            if video_hash:
                save_translation(
                    video_hash,
                    target_language,
                    pipeline_config.get("translation_provider", "local"),
                    get_translation_model_label(source_lang, target_language, pipeline_config),
                    final_segments,
                    source_signature=source_segments_signature,
                )

        manifest.update({
            "target_language": target_language,
            "final_srt": str(srt_path),
            "translation_provider": pipeline_config.get("translation_provider", "local"),
            "translation_model": get_translation_model_label(source_lang, target_language, pipeline_config) if target_language else None,
            "source_dialect": pipeline_config.get("source_dialect", "auto"),
            "target_dialect": pipeline_config.get("target_dialect", "natural"),
            "translation_glossary": pipeline_config.get("translation_glossary", []),
            "translation_qa_report": translation_qa_report,
            "openai_profile": (
                pipeline_config.get("openai_profile")
                if pipeline_config.get("translation_provider") == "openai"
                else manifest.get("openai_profile")
            ),
        })

        # Every full preparation produces a durable review manifest.  The exact
        # source and translated drafts remain separate even when the SRT selected
        # for display is the translated one.
        source_hash = (
            (manifest.get("longform_transcription") or {}).get("source_sha256")
            or content_sha256
            or sha256_full_file(video_path_obj)
        )
        prompt_version = (
            (transcription_usage or {}).get("prompt_version")
            or (cached_text_tx or {}).get("prompt_version")
            or (
                pipeline_config.get("google_transcription_prompt_version", CURRENT_PRODUCTION_PROMPT_VERSION)
                if transcription_provider == "google"
                else None
            )
        )
        source_review_cues = []
        independently_verified = bool((source_timing_verifier_report or {}).get("accept"))
        for cue_index, source_cue in enumerate(segments, 1):
            cue = dict(source_cue)
            cue.update({
                "id": str(cue.get("id") or f"source-{cue_index}"),
                "canonical_source": "gemini" if transcription_provider == "google" else transcription_provider,
                "prompt_version": prompt_version,
                "generation_scope": "full_audio",
                "audio_start": float(cue.get("start", 0.0)),
                "audio_end": float(cue.get("end", 0.0)),
                "independently_verified": bool(cue.get("independently_verified", independently_verified)),
                "manually_edited": False,
            })
            source_review_cues.append(cue)
        translation_review_cues = []
        if target_language and target_language != source_lang:
            for cue_index, translated_cue in enumerate(final_segments, 1):
                cue = dict(translated_cue)
                cue.update({
                    "id": str(cue.get("id") or f"translation-{cue_index}"),
                    "canonical_source": "automatic_translation",
                    "manually_edited": False,
                })
                translation_review_cues.append(cue)
        review = new_review(
            video_hash or source_hash,
            source_hash,
            source_language=source_lang,
            target_language=target_language,
            provider=transcription_provider,
            model=manifest.get("transcription_model"),
            prompt_version=prompt_version,
            source_draft=source_review_cues,
            translation_draft=translation_review_cues,
            source_location=resolved_source_location,
        )
        if transcription_containment:
            review["recovery_attempts"] = transcription_containment.get("recovery_attempts") or []
            review["timestamp_report"] = (
                (transcription_containment.get("professional_full_audio") or {}).get("timestamp_report")
            )
            review["containment_report"] = transcription_containment
            professional_report = transcription_containment.get("professional_full_audio") or {}
            suspicious_intervals = [
                (issue.get("start_seconds"), issue.get("end_seconds"))
                for issue in ((professional_report.get("repetition_report") or {}).get("issues") or [])
                if issue.get("start_seconds") is not None and issue.get("end_seconds") is not None
            ]
            suspicious_intervals.extend(
                (start, end)
                for start, end in (
                    (transcription_containment.get("final_coverage_report") or {}).get(
                        "meaningful_uncovered_intervals"
                    ) or []
                )
            )
            if suspicious_intervals:
                review_start = min(start for start, _ in suspicious_intervals)
                review_end = max(end for _, end in suspicious_intervals)
            else:
                review_start = 0.0
                review_end = video_duration_seconds
            add_review_issue(review, make_review_issue(
                review["video_id"],
                "malformed_gemini_region_requires_review",
                "The production Gemini transcript was rejected. Professional structured and suffix results are contained evidence, not proof of lexical correctness.",
                severity="critical",
                start_seconds=review_start,
                end_seconds=review_end,
                evidence_source=["transcript_plausibility", "gemini_structured", "independent_timing"],
                evidence={
                    "plain_problems": (transcription_containment.get("plain_plausibility_report") or {}).get("problems") or [],
                    "recovery_attempt_count": len(review["recovery_attempts"]),
                    "boundary_report": transcription_containment.get("boundary_report"),
                },
            ))
            for issue in ((professional_report.get("timestamp_report") or {}).get("issues") or []):
                add_review_issue(review, issue)
            for issue in ((professional_report.get("repetition_report") or {}).get("issues") or []):
                add_review_issue(review, issue)
        if timing_segments:
            coverage_report = independent_speech_coverage(
                review["video_id"],
                timing_segments,
                [(cue.get("start"), cue.get("end")) for cue in source_review_cues],
                media_duration=video_duration_seconds or get_video_duration(str(video_path_obj)),
                evidence_sources=["timing_anchor", "alignment"],
                vad_boundary_uncertainty=float(pipeline_config.get("review_vad_boundary_uncertainty", 0.20)),
                alignment_padding=float(pipeline_config.get("review_alignment_padding", 0.20)),
                ignore_shorter_than=float(pipeline_config.get("review_ignore_short_speech_seconds", 0.25)),
                critical_gap_seconds=float(pipeline_config.get("review_critical_gap_seconds", 0.75)),
                min_coverage_ratio=float(pipeline_config.get("review_min_speech_coverage", 0.97)),
            )
            review["coverage_report"] = coverage_report
            for issue in coverage_report.get("issues") or []:
                add_review_issue(review, issue)
        preliminary_repetition_report = repetition_anomaly_report(
            review["video_id"], source_review_cues
        )
        acoustic_reports = []
        acoustic_primary_timing_segments = (
            independent_acoustic_timing_segments(timing_segments)
        )
        if automatic_acoustic_report:
            acoustic_reports.append(automatic_acoustic_report)
        if (
            preliminary_repetition_report.get("runs")
            and pipeline_config.get(
                "automatic_repetition_verification_enabled",
                config_default("automatic_repetition_verification_enabled", True),
            )
        ):
            for candidate_index, repetition_run in enumerate(
                preliminary_repetition_report.get("runs") or [],
                start=1,
            ):
                run_start = repetition_run.get("start_seconds")
                run_end = repetition_run.get("end_seconds")
                already_observed = False
                if run_start is not None and run_end is not None:
                    for existing_report in acoustic_reports:
                        region = existing_report.get("candidate_region") or {}
                        try:
                            region_start = float(
                                region.get(
                                    "region_start",
                                    existing_report.get("region_start"),
                                )
                            )
                            region_end = float(
                                region.get(
                                    "region_end",
                                    existing_report.get("region_end"),
                                )
                            )
                        except (TypeError, ValueError):
                            continue
                        if min(float(run_end), region_end) > max(float(run_start), region_start):
                            already_observed = True
                            break
                if already_observed:
                    continue
                candidate_region = discover_text_guided_candidate_region(
                    video_duration_seconds
                    or get_video_duration(str(video_path_obj)),
                    repetition_run,
                    primary_timing_segments=acoustic_primary_timing_segments,
                    config=pipeline_config.get("acoustic_repetition_analysis"),
                )
                repetition_audio_path = (
                    ensure_source_verifier_audio(transcription_provider != "local")
                    if candidate_region.get("available")
                    else (
                        api_audio_path
                        if transcription_provider != "local"
                        else audio_path
                    )
                )
                report = build_automatic_acoustic_repetition_report(
                    str(video_path_obj),
                    str(repetition_audio_path),
                    acoustic_primary_timing_segments,
                    pipeline_config,
                    output_root=output_root,
                    base=base,
                    duration_seconds=video_duration_seconds,
                    device=device,
                    candidate_region=candidate_region,
                    artifact_suffix=f"repetition_{candidate_index}",
                )
                report["text_candidate"] = repetition_run
                acoustic_reports.append(report)
        confident_reports = [
            report
            for report in acoustic_reports
            if report.get("count_inference_confident")
            and report.get("media_sha256") == content_sha256
        ]
        if acoustic_reports:
            review["acoustic_repetition_reports"] = acoustic_reports
            acoustic_review_path = output_root / f"{base}.acoustic_repetition_review.json"
            acoustic_review_path.write_text(
                json.dumps(acoustic_reports, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        repetition_report = repetition_anomaly_report(
            review["video_id"],
            source_review_cues,
            acoustic_reports=confident_reports,
        )
        review["repetition_report"] = repetition_report
        for issue in repetition_report.get("issues") or []:
            add_review_issue(review, issue)
        if source_qa_report and not source_qa_report.get("accept", True):
            add_review_issue(review, make_review_issue(
                review["video_id"],
                "source_llm_qa_failed",
                source_qa_report.get("summary") or "Source subtitle QA requires review.",
                severity="critical",
                evidence_source=["source_llm_qa"],
                evidence={"problems": source_qa_report.get("problems") or []},
            ))
        if translation_qa_report and not translation_qa_report.get("accept", True):
            add_review_issue(review, make_review_issue(
                review["video_id"],
                "translation_llm_qa_failed",
                translation_qa_report.get("summary") or "Translation QA requires review.",
                severity="critical",
                evidence_source=["translation_llm_qa"],
                evidence={"problems": translation_qa_report.get("problems") or []},
            ))
        set_ready_for_review(review)
        review_manifest_path = output_root / f"{base}.review.json"
        save_review(review_manifest_path, review)
        save_review_manifest(review)
        manifest["review_manifest"] = str(review_manifest_path)
        manifest["review_state"] = review["state"]
        manifest["prompt_version"] = prompt_version
        print(f"Review manifest: {review_manifest_path} ({review['state']})")

        visual_style_report = None
        preparation_only = bool(no_burn or pipeline_config.get("review_before_burn", True))
        if preparation_only:
            print("\nStep 5: Preparation complete. Explicit hash-bound approval is required before burn.")
            output_path = srt_path
        else:
            print("\nStep 5: Burning subtitles into video...")
            output_path = output_root / f"{base}{mp4_suffix}_subtitled_{srt_lang_code}.mp4"
            if output_path.exists() and not force:
                print(f"Existing output video found. Skipping burn: {output_path}")
            else:
                style_config, visual_style_report = resolve_visual_style_for_video(
                    str(video_path_obj),
                    output_root,
                    srt_lang_code,
                    style_config,
                    pipeline_config,
                    srt_path=str(srt_path),
                )
                burn_subtitles(
                    str(video_path_obj),
                    str(srt_path),
                    str(output_path),
                    lang_code=srt_lang_code,
                    style_config=style_config,
                )
        
        if video_hash and not preparation_only:
            save_burned_style(video_hash, style_config)

        if not keep_files:
            print("\nCleaning up temporary files...")
            import shutil
            # 1. Clean up temporary audio files
            for temp_audio in [
                output_root / f"{base}_audio.wav",
                output_root / f"{base}_api_audio.mp3"
            ]:
                if temp_audio.exists():
                    try:
                        temp_audio.unlink()
                        print(f"Deleted temporary audio: {temp_audio.name}")
                    except Exception as e:
                        print(f"Warning: Could not delete {temp_audio.name}: {e}")

            # 2. Clean up temporary directories
            for temp_dir in [
                output_root / f"{base}_api_transcription_chunks",
                output_root / f"{base}_timing_anchor_chunks",
                output_root / f"{base}_visual_style_frames",
                output_root / f"{base}_visual_style_previews",
            ]:
                if temp_dir.exists():
                    try:
                        shutil.rmtree(temp_dir)
                        print(f"Deleted temporary directory: {temp_dir.name}")
                    except Exception as e:
                        print(f"Warning: Could not delete directory {temp_dir.name}: {e}")

        return output_path


if __name__ == "__main__":
    import json
    supported_langs = get_supported_languages()
    lang_help = ", ".join(f"{code} ({name})" for code, name in supported_langs.items())

    parser = argparse.ArgumentParser(description="SubGen Subtitle Generator Pipeline")
    parser.add_argument("video_path", type=str, help="Path to input video file.")
    parser.add_argument("srt_path", type=str, help="Path to output SRT file.")
    parser.add_argument("target_language", type=str, nargs="?", default=None, help="Target language code.")
    parser.add_argument("--config", type=str, default=None, help="Path to config file.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu or cuda).")
    parser.add_argument("--model-size", type=str, default="small", help="Whisper model size.")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for transcription.")
    parser.add_argument("--transcription-backend", type=str, default="local", help="Transcription backend.")
    parser.add_argument("--transcription-model", type=str, default="small", help="Transcription model.")
    parser.add_argument("--source-language", type=str, default=None, help="Source language.")
    parser.add_argument("--transcription-prompt", type=str, default="", help="Transcription prompt.")
    parser.add_argument("--api-audio-bitrate", type=str, default="64k", help="Audio bitrate for API.")
    parser.add_argument("--translation-backend", type=str, default="local", help="Translation backend.")
    parser.add_argument("--llm-model", type=str, default="gpt-4o", help="LLM model.")
    parser.add_argument("--translation-batch-size", type=int, default=8, help="Translation batch size.")
    parser.add_argument("--translation-context-window", type=int, default=2, help="Translation context window.")
    parser.add_argument("--source-dialect", type=str, default="auto", help="Source dialect.")
    parser.add_argument("--target-dialect", type=str, default="natural", help="Target dialect.")
    parser.add_argument("--translator-notes", type=str, default="", help="Translator notes.")
    parser.add_argument("--max-chars-per-line", type=int, default=42, help="Max chars per line.")
    parser.add_argument("--max-lines", type=int, default=None, help="Target maximum subtitle lines per cue before splitting.")
    parser.add_argument("--keep-files", action="store_true", help="Keep intermediate files.")
    parser.add_argument("--force", action="store_true", help="Force run.")
    parser.add_argument("--no-burn", action="store_true", help="Do not burn subtitles.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--approved-review", type=str, default=None, help="Approved review manifest required for burn.")

    args = parser.parse_args(normalize_language_shortcuts(sys.argv[1:], supported_langs))

    pipeline_config, style_config = merge_config(args.config)
    cli_overrides = {
        "device": args.device,
        "model_size": args.model_size,
        "beam_size": args.beam_size,
        "transcription_backend": args.transcription_backend,
        "transcription_model": args.transcription_model,
        "source_language": args.source_language,
        "transcription_prompt": args.transcription_prompt,
        "api_audio_bitrate": args.api_audio_bitrate,
        "translation_backend": args.translation_backend,
        "llm_model": args.llm_model,
        "translation_batch_size": args.translation_batch_size,
        "translation_context_window": args.translation_context_window,
        "source_dialect": args.source_dialect,
        "target_dialect": args.target_dialect,
        "translator_notes": args.translator_notes,
        "max_chars_per_line": args.max_chars_per_line,
        "max_lines": args.max_lines,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            pipeline_config[key] = value

    if not args.target_language and args.config:
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                args.target_language = json.load(f).get("target_language")
        except Exception:
            pass

    if args.target_language and args.target_language not in supported_langs:
        print(f"Error: Unsupported target language '{args.target_language}'.")
        print("Please use one of the following language codes:")
        print(lang_help)
        sys.exit(1)
        
    try:
        main(
            args.video_path,
            args.srt_path,
            args.target_language,
            style_config,
            pipeline_config=pipeline_config,
            keep_files=args.keep_files,
            force=args.force,
            no_burn=args.no_burn,
            output_dir=args.output_dir,
            approved_review_path=args.approved_review,
        )
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
