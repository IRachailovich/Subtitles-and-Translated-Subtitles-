"""Normalized audio-transcription adapters used by capability pipeline plans."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from subgen_providers import post_json, provider_retry_delay, require_provider_key
from subgen_transcription import (
    CURRENT_PRODUCTION_PROMPT_VERSION,
    gemini_transcription_generation_config,
    gemini_transcription_request_identity,
    gemini_transcription_response_metadata,
)


@dataclass
class TranscriptionArtifact:
    provider: str
    model: str
    text: str
    segments: List[Dict[str, Any]] = field(default_factory=list)
    words: List[Dict[str, Any]] = field(default_factory=list)
    language: Optional[str] = None
    duration: Optional[float] = None
    timing_kind: str = "none"
    usage: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_language_label(value: Any) -> Optional[str]:
    """Normalize provider language metadata for downstream ASR hints."""
    text = normalize_text(value).casefold().replace("_", "-")
    if not text:
        return None
    aliases = {
        "arabic": "ar",
        "chinese": "zh",
        "english": "en",
        "farsi": "fa",
        "french": "fr",
        "german": "de",
        "hebrew": "he",
        "italian": "it",
        "japanese": "ja",
        "persian": "fa",
        "portuguese": "pt",
        "russian": "ru",
        "spanish": "es",
        "multilingual": "mixed",
        "multiple": "mixed",
        "multiple languages": "mixed",
    }
    normalized = aliases.get(text, text)
    if normalized in {"auto", "auto-detect", "unknown", "und"}:
        return None
    if normalized == "mixed":
        return normalized
    language_code = normalized.split("-", 1)[0]
    if re.fullmatch(r"[a-z]{2,3}", language_code):
        return language_code
    return None


def reconcile_language_with_text_script(
    language: Optional[str],
    text: Any,
) -> tuple[Optional[str], Optional[str]]:
    """Correct provider labels that contradict an unambiguous writing script."""
    value = str(text or "")
    hebrew = len(re.findall(r"[\u0590-\u05FF]", value))
    arabic_script = len(re.findall(r"[\u0600-\u06FF]", value))
    original = language
    if hebrew and not arabic_script:
        language = "he"
    elif arabic_script and not hebrew:
        if language not in {"ar", "fa", "ur", "ps", "sd"}:
            language = "ar"
    return language, original if language != original else None


def parse_timestamp(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip().replace(",", ".")
    if not text:
        raise ValueError("Timestamp is empty.")
    parts = text.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        minutes, seconds = parts
        return float(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
    raise ValueError(f"Unsupported timestamp: {value}")


def normalize_timed_items(items: Iterable[Mapping[str, Any]], text_key: str = "text") -> List[Dict[str, Any]]:
    normalized = []
    for item in items or []:
        text = normalize_text(item.get(text_key) or item.get("word"))
        if not text:
            continue
        try:
            start = max(0.0, parse_timestamp(item.get("start")))
            end = parse_timestamp(item.get("end"))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        normalized_item = {"start": start, "end": end, "text": text}
        speaker = item.get("speaker")
        if speaker is not None:
            speaker = str(speaker).strip()
            if speaker.casefold() not in {
                "",
                "none",
                "null",
                "unknown",
                "n/a",
                "na",
                "not specified",
                "unspecified",
            }:
                normalized_item["speaker"] = speaker
        language = normalize_language_label(item.get("language"))
        language, contradicted_language = (
            reconcile_language_with_text_script(language, text)
        )
        if language:
            normalized_item["language"] = language
        provider_language = (
            normalize_language_label(item.get("provider_language"))
            or contradicted_language
        )
        if provider_language:
            normalized_item["provider_language"] = (
                provider_language
            )
        for confidence_key in ("confidence", "probability"):
            if item.get(confidence_key) is not None:
                normalized_item[confidence_key] = item.get(confidence_key)
        normalized.append(normalized_item)
    normalized.sort(key=lambda item: (item["start"], item["end"]))
    return normalized


def coalesce_overlapping_speaker_segments(
    segments: Iterable[Mapping[str, Any]],
    *,
    tolerance: float = 0.02,
) -> List[Dict[str, Any]]:
    """Represent real speaker overlap without manufacturing repeated text.

    Explicitly different speakers are retained as one SRT-compatible overlap
    cue with lossless ``overlap_turns`` metadata.  Same-speaker or unlabeled
    overlaps are treated as adjacent ASR observations and split at their
    midpoint so both texts survive without occupying the same time range.
    """
    ordered = [dict(item) for item in normalize_timed_items(segments)]
    if not ordered:
        return []
    clusters: List[List[Dict[str, Any]]] = []
    for segment in ordered:
        if (
            not clusters
            or float(segment["start"])
            >= max(float(item["end"]) for item in clusters[-1]) - float(tolerance)
        ):
            clusters.append([segment])
        else:
            clusters[-1].append(segment)

    resolved = []
    for cluster in clusters:
        if len(cluster) == 1:
            resolved.append(cluster[0])
            continue
        speakers = []
        languages = []
        for item in cluster:
            speaker = item.get("speaker")
            language = item.get("language")
            if speaker is not None and speaker not in speakers:
                speakers.append(speaker)
            if language and language not in languages:
                languages.append(language)
        if len(speakers) < 2:
            adjacent = [dict(item) for item in cluster]
            for index in range(len(adjacent) - 1):
                left = adjacent[index]
                right = adjacent[index + 1]
                if float(right["start"]) >= float(left["end"]):
                    continue
                boundary = (
                    max(float(left["start"]), float(right["start"]))
                    + min(float(left["end"]), float(right["end"]))
                ) / 2.0
                left["end"] = max(float(left["start"]) + 0.001, boundary)
                right["start"] = min(float(right["end"]) - 0.001, boundary)
            resolved.extend(
                item
                for item in adjacent
                if float(item["end"]) > float(item["start"])
            )
            continue
        combined = {
            "start": min(float(item["start"]) for item in cluster),
            "end": max(float(item["end"]) for item in cluster),
            "text": normalize_text(" ".join(item["text"] for item in cluster)),
            "overlap": True,
            "overlap_turns": [
                {
                    key: item[key]
                    for key in ("start", "end", "text", "speaker", "language")
                    if item.get(key) is not None
                }
                for item in cluster
            ],
        }
        if speakers:
            combined["speaker"] = "+".join(str(item) for item in speakers)
        if languages:
            combined["language"] = (
                languages[0] if len(languages) == 1 else "mixed"
            )
        resolved.append(combined)
    return resolved


def group_words_for_subtitles(
    words: Iterable[Mapping[str, Any]],
    max_chars: int = 84,
    max_duration: float = 5.0,
    max_gap: float = 0.65,
) -> List[Dict[str, Any]]:
    normalized_words = normalize_timed_items(words)
    segments = []
    current = []

    def flush():
        nonlocal current
        if not current:
            return
        segment = {
            "start": current[0]["start"],
            "end": current[-1]["end"],
            "text": normalize_text(" ".join(item["text"] for item in current)),
        }
        speakers = {item.get("speaker") for item in current if item.get("speaker") is not None}
        if len(speakers) == 1:
            segment["speaker"] = next(iter(speakers))
        languages = {item.get("language") for item in current if item.get("language")}
        if len(languages) == 1:
            segment["language"] = next(iter(languages))
        elif len(languages) > 1:
            segment["language"] = "mixed"
        segments.append(segment)
        current = []

    for word in normalized_words:
        candidate = normalize_text(" ".join([item["text"] for item in current] + [word["text"]]))
        duration = 0.0 if not current else word["end"] - current[0]["start"]
        gap = 0.0 if not current else word["start"] - current[-1]["end"]
        speaker_changed = bool(
            current
            and word.get("speaker") is not None
            and current[-1].get("speaker") is not None
            and word.get("speaker") != current[-1].get("speaker")
        )
        language_changed = bool(
            current
            and word.get("language")
            and current[-1].get("language")
            and word.get("language") != current[-1].get("language")
        )
        if current and (
            len(candidate) > max_chars
            or duration > max_duration
            or gap > max_gap
            or speaker_changed
            or language_changed
        ):
            flush()
        current.append(word)
        if word["text"][-1:] in {".", "?", "!", "؟", "。"} and len(candidate) >= 24:
            flush()
    flush()
    return segments


def artifact_from_response(
    provider: str,
    model: str,
    response: Mapping[str, Any],
    timing_kind: str,
    *,
    allow_empty: bool = False,
) -> TranscriptionArtifact:
    raw_words = response.get("words") or []
    words = normalize_timed_items(raw_words)
    raw_segments = response.get("segments") or []
    segments = normalize_timed_items(raw_segments)
    if not segments and words:
        segments = group_words_for_subtitles(words)
    segments = coalesce_overlapping_speaker_segments(segments)
    segment_text = normalize_text(" ".join(item["text"] for item in segments))
    response_text = normalize_text(response.get("text"))
    text = segment_text or response_text
    if not text and not allow_empty:
        raise RuntimeError(f"{provider} transcription response did not contain transcript text.")
    if text and timing_kind != "none" and not segments:
        raise RuntimeError(
            f"{provider}/{model} was selected for timestamped transcription but returned no usable timestamps."
        )
    response_language = normalize_language_label(response.get("language"))
    segment_languages = list(dict.fromkeys(
        item.get("language")
        for item in segments
        if item.get("language")
    ))
    if len(segment_languages) > 1:
        detected_language = "mixed"
    elif response_language == "mixed":
        detected_language = "mixed"
    elif len(segment_languages) == 1:
        detected_language = segment_languages[0]
    else:
        detected_language = response_language
    return TranscriptionArtifact(
        provider=provider,
        model=model,
        text=text,
        segments=segments,
        words=words,
        language=detected_language,
        duration=float(response["duration"]) if response.get("duration") is not None else None,
        timing_kind=timing_kind,
        usage=response.get("usage") or response.get("usageMetadata"),
        metadata={"response_text": response_text} if response_text and response_text != text else {},
    )


def _multipart_body(fields: Mapping[str, Any], file_path: Path):
    boundary = f"----SubGen{uuid.uuid4().hex}"
    chunks = []
    for name, value in fields.items():
        if value is None:
            continue
        values = value if isinstance(value, (list, tuple)) else [value]
        for field_value in values:
            chunks.extend([
                f"--{boundary}\r\n".encode("ascii"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                str(field_value).encode("utf-8"),
                b"\r\n",
            ])
    chunks.extend([
        f"--{boundary}\r\n".encode("ascii"),
        (
            f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode("utf-8"),
        file_path.read_bytes(),
        b"\r\n",
        f"--{boundary}--\r\n".encode("ascii"),
    ])
    return boundary, b"".join(chunks)


def post_multipart_json(
    url: str,
    fields: Mapping[str, Any],
    file_path: Path,
    headers: Mapping[str, str],
    timeout: int = 600,
) -> Dict[str, Any]:
    boundary, body = _multipart_body(fields, file_path)
    max_retries = 6
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}", **headers},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            if (exc.code == 429 or 500 <= exc.code < 600) and attempt < max_retries:
                delay = 2.0 * (2 ** attempt) + random.uniform(0.1, 1.0)
                if exc.code == 429:
                    delay = provider_retry_delay(error_body, delay)
                time.sleep(delay)
                continue
            raise RuntimeError(f"Audio transcription failed: HTTP {exc.code}: {error_body}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < max_retries:
                time.sleep(2.0 * (2 ** attempt) + random.uniform(0.1, 1.0))
                continue
            raise RuntimeError(f"Audio transcription failed: {exc}") from exc


def call_xai_transcription(
    config: Mapping[str, Any],
    audio_path: str,
    model: str = "speech-to-text",
    language: Optional[str] = None,
    prompt: str = "",
    allow_empty: bool = False,
) -> TranscriptionArtifact:
    _, api_key = require_provider_key(config, "xai")
    fields: Dict[str, Any] = {"format": "true"}
    if language:
        fields["language"] = language
    if prompt:
        fields["keyterm"] = prompt
    response = post_multipart_json(
        "https://api.x.ai/v1/stt",
        fields,
        Path(audio_path),
        {"Authorization": f"Bearer {api_key}"},
    )
    return artifact_from_response(
        "xai",
        model,
        response,
        "native_word",
        allow_empty=allow_empty,
    )


def call_openai_audio_transcription(
    config: Mapping[str, Any],
    audio_path: str,
    model: str,
    language: Optional[str] = None,
    prompt: str = "",
    allow_empty: bool = False,
) -> TranscriptionArtifact:
    api_env_key = config.get("openai_api_env_key", "OPENAI_API_KEY")
    api_key = os.environ.get(api_env_key)
    if not api_key:
        raise RuntimeError(f"OpenAI API key not found in environment variable: {api_env_key}")

    fields: Dict[str, Any] = {"model": model}
    if language:
        fields["language"] = language
    if prompt and model != "gpt-4o-transcribe-diarize":
        fields["prompt"] = prompt
    if model == "whisper-1":
        fields.update({
            "response_format": "verbose_json",
            "timestamp_granularities[]": ["word", "segment"],
        })
        timing_kind = "native_word"
    elif model == "gpt-4o-transcribe-diarize":
        fields.update({"response_format": "diarized_json", "chunking_strategy": "auto"})
        timing_kind = "native_segment"
    else:
        fields["response_format"] = "json"
        timing_kind = "none"

    response = post_multipart_json(
        "https://api.openai.com/v1/audio/transcriptions",
        fields,
        Path(audio_path),
        {"Authorization": f"Bearer {api_key}"},
    )
    return artifact_from_response(
        "openai",
        model,
        response,
        timing_kind,
        allow_empty=allow_empty,
    )


def call_openai_compatible_audio_transcription(
    config: Mapping[str, Any],
    provider_id: str,
    audio_path: str,
    model: str,
    timing_kind: str = "none",
    language: Optional[str] = None,
    prompt: str = "",
    allow_empty: bool = False,
) -> TranscriptionArtifact:
    provider, api_key = require_provider_key(config, provider_id)
    base_url = str(provider.get("base_url") or "").rstrip("/")
    if not base_url:
        raise RuntimeError(f"{provider['name']} needs a base_url for audio transcription.")
    fields: Dict[str, Any] = {"model": model}
    if language:
        fields["language"] = language
    if prompt:
        fields["prompt"] = prompt
    if timing_kind in {"native_word", "native_segment"}:
        fields["response_format"] = "verbose_json"
        granularities = []
        if timing_kind == "native_word":
            granularities.append("word")
        granularities.append("segment")
        fields["timestamp_granularities[]"] = granularities
    else:
        fields["response_format"] = "json"
    response = post_multipart_json(
        f"{base_url}/audio/transcriptions",
        fields,
        Path(audio_path),
        {"Authorization": f"Bearer {api_key}"},
    )
    return artifact_from_response(
        provider_id,
        model,
        response,
        timing_kind,
        allow_empty=allow_empty,
    )


def call_cohere_transcription(
    config: Mapping[str, Any],
    audio_path: str,
    model: str,
    language: Optional[str],
    allow_empty: bool = False,
) -> TranscriptionArtifact:
    if not language:
        raise RuntimeError(
            "Cohere Transcribe requires an explicit ISO-639-1 source language. "
            "Choose the source language instead of Auto Detect."
        )
    _, api_key = require_provider_key(config, "cohere")
    response = post_multipart_json(
        "https://api.cohere.com/v2/audio/transcriptions",
        {"model": model, "language": language, "temperature": 0},
        Path(audio_path),
        {"Authorization": f"Bearer {api_key}"},
    )
    return artifact_from_response(
        "cohere",
        model,
        response,
        "none",
        allow_empty=allow_empty,
    )


def _google_timestamp_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "string"},
                        "end": {"type": "string"},
                        "text": {"type": "string"},
                        "speaker": {"type": "string"},
                        "language": {"type": "string"},
                    },
                    "required": ["start", "end", "text", "language"],
                },
            },
            "language": {"type": "string"},
        },
        "required": ["segments", "language"],
    }


def call_google_timestamped_transcription(
    config: Mapping[str, Any],
    audio_path: str,
    model: str,
    language: Optional[str] = None,
    prompt: str = "",
    allow_empty: bool = False,
) -> TranscriptionArtifact:
    _, api_key = require_provider_key(config, "google")
    audio_path_obj = Path(audio_path)
    if audio_path_obj.stat().st_size > 18 * 1024 * 1024:
        raise RuntimeError("Google inline audio request exceeds SubGen's 18 MB safety limit.")
    language_hint = language or "auto-detect every spoken language"
    instruction = (
        "Transcribe the complete audio from the first spoken word through the final spoken word. "
        f"Source language instruction: {language_hint}. Preserve every spoken language, dialect, "
        "filler, hesitation, false start, and repetition exactly in sequence. For every repeated "
        "word or phrase, output one textual occurrence for each distinct audible occurrence: no "
        "more and no fewer, regardless of how many times it is spoken. Never continue a repeated "
        "phrase after its last audible occurrence, and never remove genuine repetitions. Never "
        "translate, summarize, polish, deduplicate, or invent speech during silence. Return chronological "
        "segments with accurate start and end timestamps in MM:SS.mmm form. A long pause must "
        "remain a gap between segments; no segment may stretch across silence. Give temporally "
        "distinguishable repetitions separate segments and distinct audible intervals. Set each segment's "
        "language to its ISO 639 language code. Set the top-level language to one ISO 639 code when "
        "the audio is monolingual, or exactly 'mixed' when it contains multiple spoken languages."
    )
    if prompt:
        instruction += f" Vocabulary and context hints: {prompt}"
    payload = {
        "contents": [{
            "parts": [
                {"text": instruction},
                {
                    "inline_data": {
                        "mime_type": "audio/mp3",
                        "data": base64.b64encode(audio_path_obj.read_bytes()).decode("ascii"),
                    }
                },
            ]
        }],
        "generationConfig": gemini_transcription_generation_config(
            temperature=0.0,
            responseMimeType="application/json",
            responseJsonSchema=_google_timestamp_schema(),
        ),
    }
    identity = gemini_transcription_request_identity(
        payload,
        model=model,
        audio_sha256=hashlib.sha256(audio_path_obj.read_bytes()).hexdigest(),
        prompt_version=CURRENT_PRODUCTION_PROMPT_VERSION,
        generation_scope="direct_timestamped_full_audio",
    )
    response = post_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        payload,
        {"x-goog-api-key": api_key},
        timeout=600,
    )
    raw_text = response["candidates"][0]["content"]["parts"][0]["text"]
    parsed = json.loads(raw_text)
    parsed["usageMetadata"] = gemini_transcription_response_metadata(
        response, identity
    )
    return artifact_from_response(
        "google",
        model,
        parsed,
        "prompted_segment",
        allow_empty=allow_empty,
    )
