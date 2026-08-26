"""Versioned Gemini transcription prompts, schemas, parsing, and benchmarks."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter


CURRENT_PRODUCTION_PROMPT_VERSION = "production_transcriber_v2_general_repetition"
PROFESSIONAL_PROMPT_VERSION = "professional_transcriber_v1"
PROFESSIONAL_STRUCTURED_PROMPT_VERSION = "professional_transcriber_structured_v1"
GEMINI_TRANSCRIPTION_THINKING_CONFIG_VERSION = "gemini_2_5_zero_thinking_v1"
GEMINI_TRANSCRIPTION_THINKING_CONFIG = {"thinkingBudget": 0}


PRODUCTION_TRANSCRIBER_SYSTEM_INSTRUCTION = (
    "Transcribe this audio accurately and completely from the first spoken word "
    "to the final spoken word. Source language: {language_hint}. "
    "If the audio contains multiple spoken languages, transcribe each spoken passage "
    "in the language actually spoken. Do not translate speech during transcription. "
    "Do not force all speech into the declared source language. "
    "Preserve dialect, names, numbers, idioms, filler words, hesitations, false starts, "
    "repeated words, repeated phrases, and natural wording. For repetitions or chants, "
    "output one textual occurrence for each audible occurrence: no more and no fewer. "
    "Never infer additional repetitions from rhythm, music, or a repeated pattern. "
    "Do not summarize, polish, deduplicate, normalize away dialect, or remove repetitions. "
    "If there is silence or a long pause, do not invent filler text; continue the transcript "
    "only when speech resumes. Return only the transcript text, without explanations."
)


PROFESSIONAL_TRANSCRIBER_SYSTEM_INSTRUCTION = (
    "You are a professional multilingual verbatim audio transcriber. "
    "Process the supplied audio chronologically from its beginning to its end. "

    "Transcribe only speech that is actually audible. Do not summarize, translate, "
    "paraphrase, correct, normalize, improve, or infer missing words. Preserve the "
    "original language, dialect, names, numbers, negation, quotations, hesitations, "
    "disfluencies, and code-switching. "

    "Return one separate segment for every acoustically distinct utterance or speaker "
    "change. When a word or phrase is audibly repeated, represent each distinct audible "
    "occurrence exactly once and assign it to its own audible interval whenever the "
    "occurrences can be temporally distinguished. "

    "Do not infer additional repetitions from rhythm, chanting, music, echo, "
    "reverberation, background accompaniment, or the preceding generated text. "
    "Do not continue a repeated phrase after its final audible occurrence. "
    "Do not omit speech that occurs after a repeated phrase or difficult region. "

    "Provide absolute start_seconds and end_seconds measured as decimal seconds from "
    "the beginning of the supplied audio. Every segment must satisfy "
    "start_seconds < end_seconds. Timestamps must progress chronologically, remain "
    "within the supplied audio duration, and refer to audible intervals. Do not reuse "
    "one audio interval to justify multiple transcript segments. "

    "Continue until the final audible speech has been processed, then stop when the "
    "supplied audio ends. Do not invent speech during silence, music, noise, or after "
    "the end of the audio. "

    "When speech is genuinely unintelligible, use [unclear] rather than inventing a "
    "confident word. Return only the requested transcript structure. Do not include "
    "commentary, analysis, explanations, confidence claims, introductions, or Markdown."
)


def professional_transcription_schema():
    return {
        "type": "object",
        "properties": {
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start_seconds": {"type": "number"},
                        "end_seconds": {"type": "number"},
                        "text": {"type": "string"},
                    },
                    "required": ["start_seconds", "end_seconds", "text"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["segments"],
        "additionalProperties": False,
    }


def build_transcription_instruction(mode, language=None, context_hint=""):
    language_hint = language or "auto-detect"
    if mode == CURRENT_PRODUCTION_PROMPT_VERSION:
        instruction = PRODUCTION_TRANSCRIBER_SYSTEM_INSTRUCTION.format(language_hint=language_hint)
    elif mode in {PROFESSIONAL_PROMPT_VERSION, PROFESSIONAL_STRUCTURED_PROMPT_VERSION}:
        instruction = PROFESSIONAL_TRANSCRIBER_SYSTEM_INSTRUCTION
        instruction += f" Supplied language hint: {language_hint}; this is a hint only, not a required output language."
        if mode == PROFESSIONAL_PROMPT_VERSION:
            instruction += " Return plain transcript text in chronological order."
        else:
            instruction += " Return the structured segments object required by the response schema."
    else:
        raise ValueError(f"Unknown transcription prompt mode: {mode}")
    if context_hint:
        instruction += f" Vocabulary/context hints: {context_hint}"
    return instruction


def gemini_transcription_generation_config(**values):
    """Return a transcription GenerationConfig that cannot omit zero-thinking."""
    return {
        **values,
        "thinkingConfig": dict(GEMINI_TRANSCRIPTION_THINKING_CONFIG),
    }


def _canonical_json_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def gemini_transcription_request_identity(
    payload,
    *,
    model,
    audio_sha256,
    prompt_version,
    generation_scope="full_audio",
    source_offset_seconds=0.0,
):
    """Identify the exact serialized payload without persisting inline audio."""
    return {
        "schema": "subgen_gemini_transcription_request_v1",
        "request_sha256": hashlib.sha256(_canonical_json_bytes(payload)).hexdigest(),
        "model": str(model),
        "audio_sha256": str(audio_sha256),
        "prompt_version": str(prompt_version),
        "generation_scope": str(generation_scope),
        "source_offset_seconds": float(source_offset_seconds),
        "thinking_config_version": GEMINI_TRANSCRIPTION_THINKING_CONFIG_VERSION,
        "thinking_config": dict(GEMINI_TRANSCRIPTION_THINKING_CONFIG),
        "credential_material_persisted": False,
    }


def gemini_transcription_response_metadata(response, identity):
    """Bind raw response/usage to its request and detect thinking regressions."""
    usage = dict((response or {}).get("usageMetadata") or {})
    thoughts = usage.get("thoughtsTokenCount")
    try:
        thoughts = int(thoughts) if thoughts is not None else None
    except (TypeError, ValueError):
        thoughts = None
    verification_basis = "explicit_thoughts_token_count"
    if thoughts is None:
        try:
            total = int(usage["totalTokenCount"])
            prompt = int(usage.get("promptTokenCount") or 0)
            candidates = int(usage.get("candidatesTokenCount") or 0)
            tool = int(usage.get("toolUsePromptTokenCount") or 0)
        except (KeyError, TypeError, ValueError):
            total = prompt = candidates = tool = None
        if total is not None and total == prompt + candidates + tool:
            # Gemini omits thoughtsTokenCount when zero. Exact conservation in
            # usageMetadata is the provider's machine-checkable equivalent.
            thoughts = 0
            verification_basis = "usage_total_equals_non_thinking_components"
        else:
            verification_basis = "provider_did_not_report_verifiable_thinking_usage"
    usage.update({
        **dict(identity or {}),
        "response_sha256": hashlib.sha256(
            _canonical_json_bytes(response or {})
        ).hexdigest(),
        "model_version": (response or {}).get("modelVersion"),
        "thinking_tokens": thoughts,
        "zero_thinking_usage_verified": thoughts == 0 if thoughts is not None else None,
        "zero_thinking_verification_basis": verification_basis,
        "hidden_thinking_exhaustion_detected": bool(thoughts and thoughts > 0),
    })
    return usage


def gemini_transcription_request_metadata(mode, *, generation_scope="full_audio", source_offset_seconds=0.0):
    if mode not in {
        CURRENT_PRODUCTION_PROMPT_VERSION,
        PROFESSIONAL_PROMPT_VERSION,
        PROFESSIONAL_STRUCTURED_PROMPT_VERSION,
    }:
        raise ValueError(f"Unknown transcription prompt mode: {mode}")
    return {
        "prompt_version": mode,
        "generation_scope": generation_scope,
        "source_offset_seconds": float(source_offset_seconds),
        "timestamp_semantics": "gemini_proposal_advisory" if mode == PROFESSIONAL_STRUCTURED_PROMPT_VERSION else None,
        "thinking_config_version": GEMINI_TRANSCRIPTION_THINKING_CONFIG_VERSION,
        "thinking_config": dict(GEMINI_TRANSCRIPTION_THINKING_CONFIG),
    }


def parse_structured_transcription(value):
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL)
        value = json.loads(text)
    if not isinstance(value, dict) or not isinstance(value.get("segments"), list):
        raise ValueError("Structured Gemini transcription must contain a segments array")
    result = []
    for index, raw in enumerate(value["segments"], 1):
        if not isinstance(raw, dict) or set(raw) != {"start_seconds", "end_seconds", "text"}:
            raise ValueError(f"Structured Gemini segment {index} does not match the strict schema")
        if isinstance(raw["start_seconds"], bool) or not isinstance(raw["start_seconds"], (int, float)):
            raise ValueError(f"Structured Gemini segment {index} start_seconds is not numeric")
        if isinstance(raw["end_seconds"], bool) or not isinstance(raw["end_seconds"], (int, float)):
            raise ValueError(f"Structured Gemini segment {index} end_seconds is not numeric")
        if not isinstance(raw["text"], str):
            raise ValueError(f"Structured Gemini segment {index} text is not a string")
        result.append({
            "id": f"gemini-{index}",
            "text": raw["text"],
            "gemini_proposed_start": float(raw["start_seconds"]),
            "gemini_proposed_end": float(raw["end_seconds"]),
            "canonical_source": "gemini",
            "prompt_version": PROFESSIONAL_STRUCTURED_PROMPT_VERSION,
            "independently_verified": False,
            "manually_edited": False,
        })
    return result


def structured_segments_text(segments):
    return " ".join(str(segment.get("text") or "").strip() for segment in segments if str(segment.get("text") or "").strip())


def _tokens(value):
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return value.split()


def _edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, left_token in enumerate(left, 1):
        current = [i]
        for j, right_token in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (left_token != right_token),
            ))
        previous = current
    return previous[-1]


def benchmark_transcript(reference, candidate, *, expected_repetitions=None, expected_final_text=None):
    reference_tokens = _tokens(reference)
    candidate_tokens = _tokens(candidate)
    distance = _edit_distance(reference_tokens, candidate_tokens)
    word_error_rate = distance / max(1, len(reference_tokens))
    reference_counts = Counter(reference_tokens)
    candidate_counts = Counter(candidate_tokens)
    excess = sum(max(0, count - reference_counts[token]) for token, count in candidate_counts.items())
    omitted = sum(max(0, count - candidate_counts[token]) for token, count in reference_counts.items())
    repetition_results = []
    for item in expected_repetitions or []:
        phrase_tokens = _tokens(item["text"])
        occurrences = 0
        if phrase_tokens:
            for index in range(0, len(candidate_tokens) - len(phrase_tokens) + 1):
                if candidate_tokens[index:index + len(phrase_tokens)] == phrase_tokens:
                    occurrences += 1
        repetition_results.append({
            "reference_text": item["text"],
            "expected": int(item["count"]),
            "observed": occurrences,
            "correct": occurrences == int(item["count"]),
        })
    final_tokens = _tokens(expected_final_text)
    end_complete = not final_tokens or any(
        candidate_tokens[index:index + len(final_tokens)] == final_tokens
        for index in range(max(0, len(candidate_tokens) - len(final_tokens) * 3), len(candidate_tokens) - len(final_tokens) + 1)
    )
    return {
        "reference_words": len(reference_tokens),
        "candidate_words": len(candidate_tokens),
        "word_error_rate": round(word_error_rate, 6),
        "lexical_accuracy": round(max(0.0, 1.0 - word_error_rate), 6),
        "omitted_reference_words": omitted,
        "unsupported_candidate_words": excess,
        "completeness": round(max(0.0, 1.0 - omitted / max(1, len(reference_tokens))), 6),
        "hallucination_rate": round(excess / max(1, len(candidate_tokens)), 6),
        "repetition_results": repetition_results,
        "repetition_correct": all(item["correct"] for item in repetition_results),
        "end_of_audio_complete": end_complete,
    }


def select_professional_prompt_role(benchmark_cases, *, material_regression=0.02):
    """Return a fail-closed prompt role from human-referenced benchmark cases."""
    required_tags = {
        "ordinary_speech", "multiple_languages", "names", "numbers", "negation", "pauses",
        "music_under_speech", "genuine_repetition", "no_repetition", "speech_after_repetition",
        "long_recording", "ar_179", "middle_loop",
    }
    observed_tags = set()
    complete_cases = []
    for case in benchmark_cases or []:
        if not case.get("human_verified_reference"):
            continue
        observed_tags.update(case.get("tags") or [])
        if all(mode in case.get("results", {}) for mode in (
            CURRENT_PRODUCTION_PROMPT_VERSION,
            PROFESSIONAL_PROMPT_VERSION,
            PROFESSIONAL_STRUCTURED_PROMPT_VERSION,
        )):
            complete_cases.append(case)
    missing_tags = sorted(required_tags - observed_tags)
    if not complete_cases or missing_tags:
        return {
            "selected_default": CURRENT_PRODUCTION_PROMPT_VERSION,
            "professional_role": "recovery_diagnostic_only",
            "eligible_for_default_change": False,
            "reason": "representative_human_verified_benchmark_incomplete",
            "missing_tags": missing_tags,
            "evaluated_cases": len(complete_cases),
        }
    averages = {}
    for mode in (CURRENT_PRODUCTION_PROMPT_VERSION, PROFESSIONAL_PROMPT_VERSION, PROFESSIONAL_STRUCTURED_PROMPT_VERSION):
        scores = [case["results"][mode] for case in complete_cases]
        averages[mode] = {
            "lexical_accuracy": sum(score["lexical_accuracy"] for score in scores) / len(scores),
            "completeness": sum(score["completeness"] for score in scores) / len(scores),
            "hallucination_rate": sum(score["hallucination_rate"] for score in scores) / len(scores),
            "end_complete_rate": sum(bool(score["end_of_audio_complete"]) for score in scores) / len(scores),
        }
    baseline = averages[CURRENT_PRODUCTION_PROMPT_VERSION]
    candidates = [PROFESSIONAL_PROMPT_VERSION, PROFESSIONAL_STRUCTURED_PROMPT_VERSION]
    non_regressing = [mode for mode in candidates if (
        averages[mode]["lexical_accuracy"] >= baseline["lexical_accuracy"] - material_regression
        and averages[mode]["completeness"] >= baseline["completeness"] - material_regression
        and averages[mode]["hallucination_rate"] <= baseline["hallucination_rate"] + material_regression
    )]
    selected = max(non_regressing, key=lambda mode: (
        averages[mode]["lexical_accuracy"], averages[mode]["completeness"], -averages[mode]["hallucination_rate"]
    )) if non_regressing else CURRENT_PRODUCTION_PROMPT_VERSION
    return {
        "selected_default": selected,
        "professional_role": "default" if selected != CURRENT_PRODUCTION_PROMPT_VERSION else "recovery_diagnostic_only",
        "eligible_for_default_change": bool(non_regressing),
        "reason": "human_verified_benchmark_no_material_regression" if non_regressing else "professional_modes_materially_regressed",
        "missing_tags": [],
        "evaluated_cases": len(complete_cases),
        "averages": averages,
    }
