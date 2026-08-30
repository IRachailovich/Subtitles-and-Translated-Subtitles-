import base64
import hashlib
import json
import math
import os
import urllib.error
import urllib.request
from pathlib import Path
import time
import random
import re

from subgen_transcription import (
    CURRENT_PRODUCTION_PROMPT_VERSION,
    PROFESSIONAL_STRUCTURED_PROMPT_VERSION,
    build_transcription_instruction,
    gemini_transcription_generation_config,
    gemini_transcription_request_identity,
    gemini_transcription_request_metadata,
    gemini_transcription_response_metadata,
    parse_structured_transcription,
    professional_transcription_schema,
    structured_segments_text,
)


def _write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _persist_gemini_request_manifest(output_dir, identity, payload):
    if not output_dir:
        return
    _write_json_atomic(Path(output_dir) / "request_manifest.json", {
        **identity,
        "generation_config": dict((payload or {}).get("generationConfig") or {}),
        "inline_audio_persisted": False,
    })


def _persist_gemini_raw_response(output_dir, response):
    if output_dir:
        _write_json_atomic(Path(output_dir) / "raw_response.json", response)


BUILTIN_PROVIDERS = {
    "openai": {
        "name": "OpenAI",
        "env_key": "OPENAI_API_KEY",
        "transcription": True,
        "translation": True,
        "transcription_model": "whisper-1",
        "translation_model": "gpt-4o",
    },
    "google": {
        "name": "Google Gemini",
        "env_key": "GOOGLE_API_KEY",
        "transcription": True,
        "translation": True,
        "transcription_model": "gemini-2.5-flash",
        "translation_model": "gemini-2.5-pro",
    },
    "deepseek": {
        "name": "DeepSeek",
        "env_key": "DEEPSEEK_API_KEY",
        "transcription": False,
        "translation": True,
        "translation_model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
    },
    "anthropic": {
        "name": "Anthropic Claude",
        "env_key": "ANTHROPIC_API_KEY",
        "transcription": False,
        "translation": True,
        "translation_model": "claude-3-5-sonnet-latest",
        "base_url": "https://api.anthropic.com",
    },
    "xai": {
        "name": "xAI",
        "env_key": "XAI_API_KEY",
        "transcription": True,
        "translation": True,
        "transcription_model": "speech-to-text",
        "translation_model": "grok-4",
        "base_url": "https://api.x.ai/v1",
        "api_style": "openai_compatible",
    },
    "cohere": {
        "name": "Cohere",
        "env_key": "COHERE_API_KEY",
        "transcription": True,
        "translation": True,
        "transcription_model": "cohere-transcribe-03-2026",
        "translation_model": "command-a-plus-05-2026",
        "base_url": "https://api.cohere.com/v2",
        "api_style": "cohere",
    },
}


def normalize_provider_id(value):
    return "".join(char if char.isalnum() else "_" for char in value.lower()).strip("_")


def get_provider_registry(config=None):
    registry = {key: value.copy() for key, value in BUILTIN_PROVIDERS.items()}
    for provider_id, provider in (config or {}).get("custom_providers", {}).items():
        merged = {
            "name": provider.get("name", provider_id),
            "env_key": provider.get("env_key", f"SUBGEN_{provider_id.upper()}_API_KEY"),
            "transcription": bool(provider.get("transcription", False)),
            "translation": bool(provider.get("translation", True)),
            "transcription_model": provider.get("transcription_model", ""),
            "translation_model": provider.get("translation_model", ""),
            "base_url": provider.get("base_url", ""),
            "api_style": provider.get("api_style", "openai_compatible"),
        }
        registry[provider_id] = merged
    return registry


def configured_providers(config=None, capability=None):
    registry = get_provider_registry(config)
    providers = {}
    for provider_id, provider in registry.items():
        if capability and not provider.get(capability):
            continue
        if os.environ.get(provider["env_key"]):
            providers[provider_id] = provider
    return providers


def get_provider(config, provider_id, capability=None):
    provider = get_provider_registry(config).get(provider_id)
    if not provider:
        raise RuntimeError(f"Unknown provider: {provider_id}")
    if capability and not provider.get(capability):
        raise RuntimeError(f"{provider['name']} does not support {capability} in SubGen.")
    return provider


def require_provider_key(config, provider_id):
    provider = get_provider(config, provider_id)
    api_key = os.environ.get(provider["env_key"])
    if not api_key:
        raise RuntimeError(
            f"{provider['env_key']} is not set. Add the {provider['name']} API key during SubGen setup."
        )
    return provider, api_key


def translation_schema():
    return {
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


def provider_retry_delay(error_body, fallback_delay):
    try:
        parsed = json.loads(error_body)
    except Exception:
        parsed = {}
    for detail in (parsed.get("error") or {}).get("details", []):
        retry_delay = detail.get("retryDelay")
        if isinstance(retry_delay, str):
            match = re.match(r"^(\d+(?:\.\d+)?)s$", retry_delay.strip())
            if match:
                return max(fallback_delay, float(match.group(1)) + random.uniform(0.25, 1.25))
    return fallback_delay


def provider_quota_is_non_retryable(error_body):
    """Daily/project quota exhaustion cannot recover inside an HTTP retry loop."""
    try:
        parsed = json.loads(error_body)
    except Exception:
        return False
    error = parsed.get("error") or {}
    if str(error.get("status") or "").upper() != "RESOURCE_EXHAUSTED":
        return False
    for detail in error.get("details") or []:
        for violation in detail.get("violations") or []:
            quota_id = str(violation.get("quotaId") or "").casefold()
            metric = str(violation.get("quotaMetric") or "").casefold()
            if "perday" in quota_id or "per_day" in quota_id or "requests_per_day" in metric:
                return True
    return "current quota" in str(error.get("message") or "").casefold() and "per day" in str(error)


def post_json(url, payload, headers, timeout=240):
    max_retries = 6
    base_delay = 2.0  # seconds
    
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            if e.code == 429 and provider_quota_is_non_retryable(error_body):
                raise RuntimeError(
                    f"Provider request failed with non-retryable daily quota: HTTP {e.code}: {error_body}"
                ) from e
            # Retry on rate limit (429) or server errors (5xx)
            if (e.code == 429 or 500 <= e.code < 600) and attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0.1, 1.0)
                if e.code == 429:
                    delay = provider_retry_delay(error_body, delay)
                print(f"Request failed with HTTP {e.code}. Retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(delay)
                continue
            raise RuntimeError(f"Provider request failed: HTTP {e.code}: {error_body}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            # Retry on network timeout or connection reset/failure
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0.1, 1.0)
                print(f"Request failed: {e}. Retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(delay)
                continue
            raise RuntimeError(f"Provider request failed: {e}") from e


def parse_json_text(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].lstrip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Provider returned invalid JSON: {text}") from e


def call_openai_compatible_translation(provider, api_key, prompt, instructions, model):
    base_url = provider.get("base_url", "").rstrip("/")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": instructions + "\nReturn valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
    }
    response = post_json(
        f"{base_url}/chat/completions",
        payload,
        {"Authorization": f"Bearer {api_key}"},
    )
    parsed = parse_json_text(response["choices"][0]["message"]["content"])
    parsed["_usage"] = response.get("usage")
    return parsed


def call_google_translation(api_key, prompt, instructions, model):
    payload = {
        "system_instruction": {"parts": [{"text": instructions}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
            "responseJsonSchema": translation_schema(),
        },
    }
    response = post_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        payload,
        {"x-goog-api-key": api_key},
    )
    text = response["candidates"][0]["content"]["parts"][0]["text"]
    parsed = parse_json_text(text)
    parsed["_usage"] = response.get("usageMetadata")
    return parsed


def call_anthropic_translation(provider, api_key, prompt, instructions, model):
    response = post_json(
        f"{provider.get('base_url', 'https://api.anthropic.com').rstrip('/')}/v1/messages",
        {
            "model": model,
            "max_tokens": 8192,
            "temperature": 0.1,
            "system": instructions + "\nReturn valid JSON only using the requested translations structure.",
            "messages": [{"role": "user", "content": prompt}],
        },
        {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    text = "".join(item.get("text", "") for item in response.get("content", []) if item.get("type") == "text")
    parsed = parse_json_text(text)
    parsed["_usage"] = response.get("usage")
    return parsed


def call_cohere_translation(provider, api_key, prompt, instructions, model):
    response = post_json(
        f"{provider.get('base_url', 'https://api.cohere.com/v2').rstrip('/')}/chat",
        {
            "model": model,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": instructions},
                {
                    "role": "user",
                    "content": prompt + "\nGenerate a JSON object using the requested translations structure.",
                },
            ],
            "response_format": {
                "type": "json_object",
                "schema": translation_schema(),
            },
        },
        {"Authorization": f"Bearer {api_key}"},
    )
    content = (response.get("message") or {}).get("content") or []
    text = "".join(item.get("text", "") for item in content if item.get("type") == "text")
    parsed = parse_json_text(text)
    parsed["_usage"] = response.get("usage")
    return parsed


def call_provider_translation(config, provider_id, prompt, instructions, model=None):
    provider, api_key = require_provider_key(config, provider_id)
    if not provider.get("translation"):
        raise RuntimeError(f"{provider['name']} does not support translation in SubGen.")
    model = model or provider.get("translation_model")

    if provider_id == "google":
        return call_google_translation(api_key, prompt, instructions, model)
    if provider_id == "anthropic":
        return call_anthropic_translation(provider, api_key, prompt, instructions, model)
    if provider_id == "cohere":
        return call_cohere_translation(provider, api_key, prompt, instructions, model)
    if provider.get("api_style") == "openai_compatible" or provider_id == "deepseek":
        return call_openai_compatible_translation(provider, api_key, prompt, instructions, model)

    raise RuntimeError(f"Translation adapter is not implemented for {provider['name']}.")


def google_transcription_output_limit(config, duration_seconds=None):
    explicit_limit = config.get("google_transcription_max_output_tokens")
    if explicit_limit not in {None, ""}:
        return max(1, int(explicit_limit)), "explicit"

    minimum = max(1, int(config.get("google_transcription_min_output_tokens", 512)))
    ceiling = max(minimum, int(config.get("google_transcription_output_token_ceiling", 65536)))
    tokens_per_second = max(
        1.0,
        float(config.get("google_transcription_output_tokens_per_second", 32.0)),
    )
    if duration_seconds is None or float(duration_seconds) <= 0:
        return min(ceiling, 16384), "duration_unavailable_fallback"
    estimated = math.ceil(float(duration_seconds) * tokens_per_second + 256)
    return min(ceiling, max(minimum, estimated)), "audio_duration"


def call_google_transcription(
    config,
    audio_path,
    model=None,
    language=None,
    prompt="",
    duration_seconds=None,
    temperature=None,
    prompt_version=CURRENT_PRODUCTION_PROMPT_VERSION,
    request_timeout_seconds=600,
    diagnostic_output_dir=None,
    allow_empty=False,
):
    provider, api_key = require_provider_key(config, "google")
    audio_path = Path(audio_path)
    if audio_path.stat().st_size > 18 * 1024 * 1024:
        raise RuntimeError(
            "Google inline audio request exceeds SubGen's 18 MB safety limit. "
            "Lower api_audio_bitrate or use OpenAI transcription for this file."
        )

    model = model or provider["transcription_model"]
    language_hint = language or "auto-detect"
    instruction = build_transcription_instruction(
        prompt_version,
        language=language_hint,
        context_hint=prompt,
    )

    output_limit, output_limit_strategy = google_transcription_output_limit(
        config,
        duration_seconds=duration_seconds,
    )
    selected_temperature = (
        float(temperature)
        if temperature is not None
        else float(config.get("google_transcription_temperature", 0.0))
    )
    payload = {
        "contents": [{
            "parts": [
                {"text": instruction},
                {
                    "inline_data": {
                        "mime_type": "audio/mp3",
                        "data": base64.b64encode(audio_path.read_bytes()).decode("ascii"),
                    }
                },
            ]
        }],
        "generationConfig": gemini_transcription_generation_config(
            temperature=selected_temperature,
            maxOutputTokens=output_limit,
        ),
    }
    identity = gemini_transcription_request_identity(
        payload,
        model=model,
        audio_sha256=hashlib.sha256(audio_path.read_bytes()).hexdigest(),
        prompt_version=prompt_version,
    )
    _persist_gemini_request_manifest(diagnostic_output_dir, identity, payload)
    response = post_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        payload,
        {"x-goog-api-key": api_key},
        timeout=float(request_timeout_seconds),
    )
    _persist_gemini_raw_response(diagnostic_output_dir, response)
    candidates = response.get("candidates") or []
    usage = gemini_transcription_response_metadata(response, identity)
    usage["max_output_tokens"] = payload["generationConfig"]["maxOutputTokens"]
    usage["output_token_limit_strategy"] = output_limit_strategy
    usage["temperature"] = selected_temperature
    usage.update(gemini_transcription_request_metadata(prompt_version))
    if not candidates:
        if allow_empty:
            usage["finish_reason"] = None
            return "", usage
        raise RuntimeError("Google transcription response did not contain a candidate.")
    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
    if not text:
        if allow_empty:
            usage["finish_reason"] = candidate.get("finishReason")
            return "", usage
        raise RuntimeError("Google transcription response did not contain transcript text.")
    usage["finish_reason"] = candidate.get("finishReason")
    return text, usage


def call_google_structured_transcription(
    config,
    audio_path,
    model=None,
    language=None,
    prompt="",
    duration_seconds=None,
    temperature=None,
    *,
    generation_scope="full_audio",
    source_offset_seconds=0.0,
    request_timeout_seconds=600,
    diagnostic_output_dir=None,
):
    """Request structured Gemini cues while retaining timestamps as proposals only."""
    provider, api_key = require_provider_key(config, "google")
    audio_path = Path(audio_path)
    if audio_path.stat().st_size > 18 * 1024 * 1024:
        raise RuntimeError(
            "Google inline audio request exceeds SubGen's 18 MB safety limit. "
            "Lower api_audio_bitrate or use a bounded recovery suffix."
        )
    model = model or provider["transcription_model"]
    output_limit, output_limit_strategy = google_transcription_output_limit(
        config, duration_seconds=duration_seconds
    )
    selected_temperature = (
        float(temperature)
        if temperature is not None
        else float(config.get("google_transcription_temperature", 0.0))
    )
    instruction = build_transcription_instruction(
        PROFESSIONAL_STRUCTURED_PROMPT_VERSION,
        language=language or "auto-detect",
        context_hint=prompt,
    )
    payload = {
        "contents": [{
            "parts": [
                {"text": instruction},
                {"inline_data": {
                    "mime_type": "audio/mp3",
                    "data": base64.b64encode(audio_path.read_bytes()).decode("ascii"),
                }},
            ]
        }],
        "generationConfig": gemini_transcription_generation_config(
            temperature=selected_temperature,
            maxOutputTokens=output_limit,
            responseMimeType="application/json",
            responseJsonSchema=professional_transcription_schema(),
            audioTranscriptionConfig={"wordTimestamp": True},
        ),
    }
    identity = gemini_transcription_request_identity(
        payload,
        model=model,
        audio_sha256=hashlib.sha256(audio_path.read_bytes()).hexdigest(),
        prompt_version=PROFESSIONAL_STRUCTURED_PROMPT_VERSION,
        generation_scope=generation_scope,
        source_offset_seconds=source_offset_seconds,
    )
    _persist_gemini_request_manifest(diagnostic_output_dir, identity, payload)
    response = post_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        payload,
        {"x-goog-api-key": api_key},
        timeout=float(request_timeout_seconds),
    )
    _persist_gemini_raw_response(diagnostic_output_dir, response)
    candidates = response.get("candidates") or []
    if not candidates:
        raise RuntimeError("Google structured transcription response did not contain a candidate.")
    candidate = candidates[0]
    raw_text = "".join(
        part.get("text", "")
        for part in ((candidate.get("content") or {}).get("parts") or [])
        if isinstance(part, dict)
    ).strip()
    segments = parse_structured_transcription(raw_text)
    for segment in segments:
        segment.update({
            "generation_scope": generation_scope,
            "source_offset_seconds": float(source_offset_seconds),
        })
    usage = gemini_transcription_response_metadata(response, identity)
    usage.update({
        "finish_reason": candidate.get("finishReason"),
        "max_output_tokens": output_limit,
        "output_token_limit_strategy": output_limit_strategy,
        "temperature": selected_temperature,
        **gemini_transcription_request_metadata(
            PROFESSIONAL_STRUCTURED_PROMPT_VERSION,
            generation_scope=generation_scope,
            source_offset_seconds=source_offset_seconds,
        ),
    })
    return {
        "text": structured_segments_text(segments),
        "segments": segments,
        "usage": usage,
        "prompt_version": PROFESSIONAL_STRUCTURED_PROMPT_VERSION,
        "model": model,
    }
