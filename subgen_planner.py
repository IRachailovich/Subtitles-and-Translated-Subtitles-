"""Capability-based pipeline planning for SubGen.

The planner is deliberately separate from provider execution. A model may support a
feature in its public API while SubGen's adapter does not yet expose that feature.
Only adapter capabilities are eligible for automatic pipeline selection.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


PLAN_VERSION = "2026-07-13-capability-planner-v1"

TIMING_NONE = "none"
TIMING_PROMPTED_SEGMENT = "prompted_segment"
TIMING_NATIVE_SEGMENT = "native_segment"
TIMING_NATIVE_WORD = "native_word"

ROUTE_LOCAL_WORD_TIMESTAMPS = "local_word_timestamps"
ROUTE_DIRECT_TIMESTAMPED_TRANSCRIPT = "direct_timestamped_transcript"
ROUTE_CANONICAL_TEXT_ANCHOR_ALIGNMENT = "canonical_text_anchor_alignment"


@dataclass(frozen=True)
class ModelCapabilities:
    provider: str
    model: str
    display_name: str
    audio_transcription: bool = False
    transcription_kind: str = "none"
    model_timing_kind: str = TIMING_NONE
    adapter_timing_kind: str = TIMING_NONE
    translation: bool = False
    language_detection: bool = False
    mixed_language: bool = False
    diarization: bool = False
    structured_output: bool = False
    requires_source_language: bool = False
    adapter: str = ""

    @property
    def has_usable_timestamps(self) -> bool:
        return self.adapter_timing_kind != TIMING_NONE

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PipelineStage:
    id: str
    label: str
    provider: Optional[str] = None
    model: Optional[str] = None
    engine: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class PipelinePlan:
    route: str
    title: str
    summary: str
    transcription: ModelCapabilities
    stages: Tuple[PipelineStage, ...]
    translation_provider: str
    translation_model: Optional[str]
    timing_anchor_provider: Optional[str]
    timing_anchor_model: Optional[str]
    timing_mode: Optional[str]
    uses_separate_timing_anchors: bool
    uses_whisperx: bool
    plan_version: str = PLAN_VERSION
    source_plan_id: str = ""
    plan_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "plan_version": self.plan_version,
            "source_plan_id": self.source_plan_id,
            "plan_id": self.plan_id,
            "route": self.route,
            "title": self.title,
            "summary": self.summary,
            "transcription": self.transcription.to_dict(),
            "translation_provider": self.translation_provider,
            "translation_model": self.translation_model,
            "timing_anchor_provider": self.timing_anchor_provider,
            "timing_anchor_model": self.timing_anchor_model,
            "timing_mode": self.timing_mode,
            "uses_separate_timing_anchors": self.uses_separate_timing_anchors,
            "uses_whisperx": self.uses_whisperx,
            "stages": [stage.to_dict() for stage in self.stages],
        }
        return payload


def _model(
    provider: str,
    model: str,
    display_name: str,
    **capabilities: Any,
) -> ModelCapabilities:
    return ModelCapabilities(provider, model, display_name, **capabilities)


# This catalog describes both public model capability and current adapter support.
# Public capability alone never activates a route.
MODEL_CATALOG: Dict[Tuple[str, str], ModelCapabilities] = {
    ("google", "gemini-2.5-flash"): _model(
        "google",
        "gemini-2.5-flash",
        "Google Gemini 2.5 Flash",
        audio_transcription=True,
        transcription_kind="semantic_llm",
        model_timing_kind=TIMING_PROMPTED_SEGMENT,
        adapter_timing_kind=TIMING_NONE,
        translation=True,
        language_detection=True,
        mixed_language=True,
        structured_output=True,
        adapter="google_gemini_text",
    ),
    ("google", "gemini-2.5-pro"): _model(
        "google",
        "gemini-2.5-pro",
        "Google Gemini 2.5 Pro",
        audio_transcription=True,
        transcription_kind="semantic_llm",
        model_timing_kind=TIMING_PROMPTED_SEGMENT,
        adapter_timing_kind=TIMING_NONE,
        translation=True,
        language_detection=True,
        mixed_language=True,
        structured_output=True,
        adapter="google_gemini_text",
    ),
    ("google", "gemini-3.5-flash"): _model(
        "google",
        "gemini-3.5-flash",
        "Google Gemini 3.5 Flash",
        audio_transcription=True,
        transcription_kind="semantic_llm",
        model_timing_kind=TIMING_PROMPTED_SEGMENT,
        adapter_timing_kind=TIMING_PROMPTED_SEGMENT,
        translation=True,
        language_detection=True,
        mixed_language=True,
        diarization=True,
        structured_output=True,
        adapter="google_gemini_timestamped",
    ),
    ("openai", "whisper-1"): _model(
        "openai",
        "whisper-1",
        "OpenAI Whisper-1",
        audio_transcription=True,
        transcription_kind="dedicated_asr",
        model_timing_kind=TIMING_NATIVE_WORD,
        adapter_timing_kind=TIMING_NATIVE_WORD,
        language_detection=True,
        mixed_language=True,
        adapter="openai_audio_transcription",
    ),
    ("openai", "gpt-4o-transcribe"): _model(
        "openai",
        "gpt-4o-transcribe",
        "OpenAI GPT-4o Transcribe",
        audio_transcription=True,
        transcription_kind="semantic_asr",
        language_detection=True,
        mixed_language=True,
        adapter="openai_audio_transcription",
    ),
    ("openai", "gpt-4o-mini-transcribe"): _model(
        "openai",
        "gpt-4o-mini-transcribe",
        "OpenAI GPT-4o Mini Transcribe",
        audio_transcription=True,
        transcription_kind="semantic_asr",
        language_detection=True,
        mixed_language=True,
        adapter="openai_audio_transcription",
    ),
    ("openai", "gpt-4o-transcribe-diarize"): _model(
        "openai",
        "gpt-4o-transcribe-diarize",
        "OpenAI GPT-4o Transcribe Diarize",
        audio_transcription=True,
        transcription_kind="semantic_asr",
        model_timing_kind=TIMING_NATIVE_SEGMENT,
        adapter_timing_kind=TIMING_NATIVE_SEGMENT,
        language_detection=True,
        mixed_language=True,
        diarization=True,
        adapter="openai_audio_transcription",
    ),
    ("xai", "speech-to-text"): _model(
        "xai",
        "speech-to-text",
        "xAI Speech to Text",
        audio_transcription=True,
        transcription_kind="dedicated_asr",
        model_timing_kind=TIMING_NATIVE_WORD,
        adapter_timing_kind=TIMING_NATIVE_WORD,
        language_detection=True,
        mixed_language=True,
        diarization=True,
        adapter="xai_speech_to_text",
    ),
    ("cohere", "cohere-transcribe-03-2026"): _model(
        "cohere",
        "cohere-transcribe-03-2026",
        "Cohere Transcribe",
        audio_transcription=True,
        transcription_kind="dedicated_asr",
        language_detection=False,
        mixed_language=False,
        requires_source_language=True,
        adapter="cohere_audio_transcription",
    ),
    ("cohere", "cohere-transcribe-arabic-07-2026"): _model(
        "cohere",
        "cohere-transcribe-arabic-07-2026",
        "Cohere Transcribe Arabic",
        audio_transcription=True,
        transcription_kind="dedicated_asr",
        language_detection=False,
        mixed_language=False,
        requires_source_language=True,
        adapter="cohere_audio_transcription",
    ),
    ("openai", "gpt-4o"): _model(
        "openai",
        "gpt-4o",
        "OpenAI GPT-4o",
        translation=True,
        structured_output=True,
    ),
    ("deepseek", "deepseek-chat"): _model(
        "deepseek",
        "deepseek-chat",
        "DeepSeek Chat",
        translation=True,
        structured_output=True,
    ),
    ("anthropic", "claude-3-5-sonnet-latest"): _model(
        "anthropic",
        "claude-3-5-sonnet-latest",
        "Anthropic Claude Sonnet",
        translation=True,
        structured_output=True,
    ),
    ("xai", "grok-4"): _model(
        "xai",
        "grok-4",
        "xAI Grok 4",
        translation=True,
        structured_output=True,
    ),
    ("cohere", "command-a-plus-05-2026"): _model(
        "cohere",
        "command-a-plus-05-2026",
        "Cohere Command A Plus",
        translation=True,
        structured_output=True,
    ),
}


MODEL_GUIDE_METADATA: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("google", "gemini-2.5-flash"): {
        "provider_display": "Google",
        "model_family": "Semantic multimodal LLM",
        "timing_description": "No timestamps in SubGen canonical-text mode",
        "guide_timing_kind": TIMING_NONE,
        "docs_url": "https://ai.google.dev/gemini-api/docs/audio",
        "api_url": "https://aistudio.google.com/apikey",
    },
    ("google", "gemini-2.5-pro"): {
        "provider_display": "Google",
        "model_family": "Semantic multimodal LLM",
        "timing_description": "No timestamps in SubGen canonical-text mode",
        "guide_timing_kind": TIMING_NONE,
        "docs_url": "https://ai.google.dev/gemini-api/docs/audio",
        "api_url": "https://aistudio.google.com/apikey",
    },
    ("google", "gemini-3.5-flash"): {
        "provider_display": "Google",
        "model_family": "Semantic multimodal LLM",
        "timing_description": "LLM-generated segment timestamps; independently verified",
        "docs_url": "https://ai.google.dev/gemini-api/docs/audio",
        "api_url": "https://aistudio.google.com/apikey",
    },
    ("openai", "whisper-1"): {
        "provider_display": "OpenAI",
        "model_family": "Dedicated ASR",
        "timing_description": "Native word and segment timestamps",
        "docs_url": "https://developers.openai.com/api/docs/guides/speech-to-text",
        "api_url": "https://platform.openai.com/api-keys",
    },
    ("openai", "gpt-4o-transcribe"): {
        "provider_display": "OpenAI",
        "model_family": "GPT-powered speech-to-text",
        "timing_description": "Text only; separate timing required",
        "docs_url": "https://developers.openai.com/api/docs/models/gpt-4o-transcribe",
        "api_url": "https://platform.openai.com/api-keys",
    },
    ("openai", "gpt-4o-mini-transcribe"): {
        "provider_display": "OpenAI",
        "model_family": "GPT-powered speech-to-text",
        "timing_description": "Text only; separate timing required",
        "docs_url": "https://developers.openai.com/api/docs/models/gpt-4o-mini-transcribe",
        "api_url": "https://platform.openai.com/api-keys",
    },
    ("openai", "gpt-4o-transcribe-diarize"): {
        "provider_display": "OpenAI",
        "model_family": "GPT-powered diarized ASR",
        "timing_description": "Native speaker-labelled segment timestamps",
        "docs_url": "https://developers.openai.com/api/docs/models/gpt-4o-transcribe-diarize",
        "api_url": "https://platform.openai.com/api-keys",
    },
    ("openai", "gpt-4o"): {
        "provider_display": "OpenAI",
        "model_family": "Semantic translation LLM",
        "timing_description": "Translation only in SubGen",
        "docs_url": "https://developers.openai.com/api/docs/models/gpt-4o",
        "api_url": "https://platform.openai.com/api-keys",
    },
    ("xai", "speech-to-text"): {
        "provider_display": "xAI",
        "model_family": "Dedicated speech-to-text",
        "timing_description": "Native word timestamps; optional speaker labels",
        "docs_url": "https://docs.x.ai/developers/model-capabilities/audio/speech-to-text",
        "api_url": "https://console.x.ai/",
    },
    ("xai", "grok-4"): {
        "provider_display": "xAI",
        "model_family": "Semantic translation LLM",
        "timing_description": "Translation only in SubGen",
        "docs_url": "https://docs.x.ai/developers/models",
        "api_url": "https://console.x.ai/",
    },
    ("cohere", "cohere-transcribe-03-2026"): {
        "provider_display": "Cohere",
        "model_family": "Dedicated ASR",
        "timing_description": "Text only; separate timing required",
        "docs_url": "https://docs.cohere.com/docs/transcribe",
        "api_url": "https://dashboard.cohere.com/api-keys",
    },
    ("cohere", "cohere-transcribe-arabic-07-2026"): {
        "provider_display": "Cohere",
        "model_family": "Arabic-focused dedicated ASR",
        "timing_description": "Text only; separate timing required",
        "docs_url": "https://docs.cohere.com/docs/transcribe",
        "api_url": "https://dashboard.cohere.com/api-keys",
    },
    ("cohere", "command-a-plus-05-2026"): {
        "provider_display": "Cohere",
        "model_family": "Semantic translation LLM",
        "timing_description": "Translation only in SubGen",
        "docs_url": "https://docs.cohere.com/docs/models",
        "api_url": "https://dashboard.cohere.com/api-keys",
    },
    ("deepseek", "deepseek-chat"): {
        "provider_display": "DeepSeek",
        "model_family": "Semantic translation LLM",
        "timing_description": "Translation only in SubGen",
        "docs_url": "https://api-docs.deepseek.com/",
        "api_url": "https://platform.deepseek.com/api_keys",
    },
    ("anthropic", "claude-3-5-sonnet-latest"): {
        "provider_display": "Anthropic",
        "model_family": "Semantic translation LLM",
        "timing_description": "Translation only in SubGen",
        "docs_url": "https://docs.anthropic.com/en/docs/about-claude/models/overview",
        "api_url": "https://console.anthropic.com/settings/keys",
    },
}


GUIDE_ONLY_MODELS: Tuple[Dict[str, Any], ...] = (
    {
        "provider": "google_cloud",
        "model": "chirp_3",
        "display_name": "Google Cloud Chirp 3",
        "provider_display": "Google Cloud",
        "model_family": "Generative multilingual ASR",
        "audio_transcription": True,
        "transcription_kind": "generative_asr",
        "model_timing_kind": TIMING_NATIVE_WORD,
        "adapter_timing_kind": TIMING_NONE,
        "translation": False,
        "language_detection": True,
        "mixed_language": True,
        "diarization": True,
        "structured_output": True,
        "requires_source_language": False,
        "adapter": "",
        "guide_timing_kind": TIMING_NATIVE_WORD,
        "timing_description": "Native word timestamps; SubGen adapter not integrated yet",
        "docs_url": "https://docs.cloud.google.com/speech-to-text/docs/models/chirp-3",
        "api_url": "https://console.cloud.google.com/apis/library/speech.googleapis.com",
        "subgen_supported": False,
        "semantic_ai": False,
    },
)


PROVIDER_DEFAULTS: Dict[str, ModelCapabilities] = {
    "google": _model(
        "google", "", "Google audio model", audio_transcription=True,
        transcription_kind="semantic_llm", language_detection=True,
        mixed_language=True, adapter="google_gemini_text",
    ),
    "openai": _model(
        "openai", "", "OpenAI audio model", audio_transcription=True,
        transcription_kind="semantic_asr", language_detection=True,
        mixed_language=True, adapter="openai_audio_transcription",
    ),
    "cohere": _model(
        "cohere", "", "Cohere audio model", audio_transcription=True,
        transcription_kind="dedicated_asr", requires_source_language=True,
        adapter="cohere_audio_transcription",
    ),
    "xai": _model(
        "xai", "", "xAI audio model", audio_transcription=True,
        transcription_kind="dedicated_asr", adapter="xai_speech_to_text",
    ),
}

TRANSLATION_MODEL_DEFAULTS = {
    "openai": "gpt-4o",
    "google": "gemini-2.5-pro",
    "deepseek": "deepseek-chat",
    "anthropic": "claude-3-5-sonnet-latest",
    "xai": "grok-4",
    "cohere": "command-a-plus-05-2026",
}


def _custom_capabilities(config: Mapping[str, Any], provider: str, model: str) -> Optional[ModelCapabilities]:
    declared = config.get("model_capabilities", {}).get(provider, {}).get(model)
    if not declared:
        return None
    allowed = set(ModelCapabilities.__dataclass_fields__) - {"provider", "model"}
    unknown = sorted(set(declared) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown capability fields for {provider}/{model}: {', '.join(unknown)}"
        )
    return ModelCapabilities(
        provider=provider,
        model=model,
        display_name=declared.get("display_name") or f"{provider} {model}",
        **{key: value for key, value in declared.items() if key != "display_name"},
    )


def resolve_model_capabilities(
    config: Mapping[str, Any],
    provider: str,
    model: Optional[str],
) -> ModelCapabilities:
    model = model or ""
    custom = _custom_capabilities(config, provider, model)
    if custom:
        capabilities = custom
    elif (provider, model) in MODEL_CATALOG:
        capabilities = MODEL_CATALOG[(provider, model)]
    elif provider in PROVIDER_DEFAULTS:
        default = PROVIDER_DEFAULTS[provider]
        capabilities = ModelCapabilities(
            **{**default.to_dict(), "model": model, "display_name": f"{default.display_name}: {model or 'default'}"}
        )
    else:
        raise ValueError(
            f"No transcription capabilities are declared for {provider}/{model}. "
            "Declare model_capabilities explicitly before using it in automatic mode."
        )

    if not capabilities.audio_transcription:
        raise ValueError(f"{provider}/{model} is not an audio transcription model.")
    if capabilities.adapter_timing_kind not in {
        TIMING_NONE,
        TIMING_PROMPTED_SEGMENT,
        TIMING_NATIVE_SEGMENT,
        TIMING_NATIVE_WORD,
    }:
        raise ValueError(
            f"Unsupported adapter_timing_kind for {provider}/{model}: "
            f"{capabilities.adapter_timing_kind}"
        )
    return capabilities


def _selected_model(config: Mapping[str, Any], provider: str, capability: str) -> Optional[str]:
    explicit_key = f"{capability}_model"
    selected = (
        config.get(explicit_key)
        or config.get("provider_models", {}).get(provider, {}).get(capability)
        or None
    )
    if selected:
        return selected
    if capability == "translation":
        return TRANSLATION_MODEL_DEFAULTS.get(provider)
    return None


def _translation_stage(provider: str, model: Optional[str]) -> PipelineStage:
    if provider == "local":
        return PipelineStage("semantic_translation", "Translate with local MarianMT", provider="local")
    return PipelineStage(
        "semantic_translation",
        "Translate semantically without changing source timestamps",
        provider=provider,
        model=model,
    )


def _finalize_plan(plan: PipelinePlan) -> PipelinePlan:
    source_payload = {
        "plan_version": plan.plan_version,
        "route": plan.route,
        "transcription": plan.transcription.to_dict(),
        "timing_anchor_provider": plan.timing_anchor_provider,
        "timing_anchor_model": plan.timing_anchor_model,
        "timing_mode": plan.timing_mode,
        "uses_separate_timing_anchors": plan.uses_separate_timing_anchors,
        "uses_whisperx": plan.uses_whisperx,
        "stages": [
            stage.to_dict()
            for stage in plan.stages
            if stage.id not in {
                "semantic_translation",
                "translation_semantic_gate",
                "review_and_burn",
            }
        ],
    }
    source_canonical = json.dumps(
        source_payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    source_plan_id = hashlib.sha256(source_canonical.encode("utf-8")).hexdigest()[:16]
    plan = replace(plan, source_plan_id=source_plan_id)
    payload = plan.to_dict()
    payload["plan_id"] = ""
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    plan_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return replace(plan, plan_id=plan_id)


def build_pipeline_plan(config: Mapping[str, Any], target_language: Optional[str] = None) -> PipelinePlan:
    """Build a deterministic plan without executing any model.

    Existing configuration keys remain authoritative. Automatic planning only decides
    whether timestamps from the transcription adapter are sufficient to skip the
    separate timing-anchor and WhisperX stages.
    """

    transcription_provider = config.get("transcription_provider") or "local"
    translation_provider = config.get("translation_provider") or "local"
    translation_model = _selected_model(config, translation_provider, "translation")
    translate = bool(target_language)

    if transcription_provider == "local":
        local_model = config.get("model_size") or "small"
        transcription = ModelCapabilities(
            provider="local",
            model=local_model,
            display_name=f"Local Whisper {local_model}",
            audio_transcription=True,
            transcription_kind="local_asr",
            model_timing_kind=TIMING_NATIVE_WORD,
            adapter_timing_kind=TIMING_NATIVE_WORD,
            language_detection=True,
            mixed_language=True,
            adapter="local_whisper",
        )
        stages = [
            PipelineStage("prepare_audio", "Extract compact audio with FFmpeg", engine="ffmpeg"),
            PipelineStage(
                "canonical_transcription_with_timestamps",
                "Transcribe locally with word timestamps",
                provider="local",
                model=local_model,
            ),
            PipelineStage("source_integrity_gate", "Verify source subtitle integrity and timing"),
        ]
        if translate:
            stages.extend([
                _translation_stage(translation_provider, translation_model),
                PipelineStage("translation_semantic_gate", "Verify semantic translation equivalence"),
            ])
        stages.append(PipelineStage("review_and_burn", "Review, edit, approve, and burn subtitles"))
        return _finalize_plan(PipelinePlan(
            route=ROUTE_LOCAL_WORD_TIMESTAMPS,
            title="Local timestamped transcription",
            summary="Local Whisper supplies both source text and word timestamps.",
            transcription=transcription,
            stages=tuple(stages),
            translation_provider=translation_provider,
            translation_model=translation_model,
            timing_anchor_provider=None,
            timing_anchor_model=None,
            timing_mode=None,
            uses_separate_timing_anchors=False,
            uses_whisperx=False,
        ))

    transcription_model = _selected_model(config, transcription_provider, "transcription")
    transcription = resolve_model_capabilities(config, transcription_provider, transcription_model)
    if transcription.requires_source_language and not config.get("source_language"):
        raise ValueError(
            f"{transcription.display_name} requires an explicit source language; "
            "Auto Detect is not supported by this model."
        )

    stages = [PipelineStage("prepare_audio", "Prepare compact API audio", engine="ffmpeg")]
    if transcription.has_usable_timestamps:
        stages.extend([
            PipelineStage(
                "canonical_transcription_with_timestamps",
                "Transcribe once with canonical text and timestamps",
                provider=transcription.provider,
                model=transcription.model,
            ),
            PipelineStage(
                "source_integrity_gate",
                "Independently verify text, speech coverage, silence boundaries, and cue timing",
                engine="whisperx_audio_verifier",
            ),
        ])
        if translate:
            stages.extend([
                _translation_stage(translation_provider, translation_model),
                PipelineStage("translation_semantic_gate", "Verify semantic translation equivalence"),
            ])
        stages.append(PipelineStage("review_and_burn", "Review, edit, approve, and burn subtitles"))
        return _finalize_plan(PipelinePlan(
            route=ROUTE_DIRECT_TIMESTAMPED_TRANSCRIPT,
            title="Direct timestamped transcription",
            summary=(
                f"{transcription.display_name} supplies canonical text and "
                f"{transcription.adapter_timing_kind.replace('_', ' ')} timestamps; "
                "OpenAI Whisper-1 and WhisperX are not used to generate cue timing. "
                "SubGen still applies an independent audio-grounded verification gate."
            ),
            transcription=transcription,
            stages=tuple(stages),
            translation_provider=translation_provider,
            translation_model=translation_model,
            timing_anchor_provider=None,
            timing_anchor_model=None,
            timing_mode=None,
            uses_separate_timing_anchors=False,
            uses_whisperx=False,
        ))

    timing_anchor_provider = config.get("timing_anchor_provider") or "openai"
    timing_anchor_model = (
        config.get("timing_anchor_model")
        or ("whisper-1" if timing_anchor_provider == "openai" else config.get("model_size") or "small")
    )
    timing_mode = config.get("api_transcript_timing_mode") or "precise"
    if timing_anchor_provider not in {"openai", "local"}:
        raise ValueError(
            "The current canonical-text fusion adapter supports OpenAI Whisper-1 or "
            "local Whisper timing anchors only."
        )
    if timing_mode not in {"precise", "fuzzy", "forced"}:
        raise ValueError(f"Unsupported timing mode for automatic planning: {timing_mode}")

    anchor_label = (
        "Obtain word timing anchors with OpenAI Whisper-1"
        if timing_anchor_provider == "openai"
        else f"Obtain word timing anchors with local Whisper {timing_anchor_model}"
    )
    stages.extend([
        PipelineStage(
            "canonical_transcription",
            "Transcribe canonical source text once",
            provider=transcription.provider,
            model=transcription.model,
        ),
        PipelineStage(
            "timing_anchor_transcription",
            anchor_label,
            provider=timing_anchor_provider,
            model=timing_anchor_model,
        ),
    ])
    uses_whisperx = (
        timing_mode in {"precise", "forced"}
        and not bool(config.get("longform_enabled", True))
    )
    if uses_whisperx:
        stages.append(PipelineStage(
            "forced_alignment",
            "Force-align canonical text to audio with WhisperX",
            engine="whisperx",
        ))
    else:
        stages.append(PipelineStage(
            "anchor_reconciliation",
            "Map canonical text monotonically onto native timing anchors",
            engine="subgen_anchor_reconciliation",
        ))
    stages.append(PipelineStage(
        "source_integrity_gate",
        "Verify transcript text, speech coverage, silence boundaries, and cue timing",
    ))
    if translate:
        stages.extend([
            _translation_stage(translation_provider, translation_model),
            PipelineStage("translation_semantic_gate", "Verify semantic translation equivalence"),
        ])
    stages.append(PipelineStage("review_and_burn", "Review, edit, approve, and burn subtitles"))

    return _finalize_plan(PipelinePlan(
        route=ROUTE_CANONICAL_TEXT_ANCHOR_ALIGNMENT,
        title="Canonical text with separate audio alignment",
        summary=(
            f"{transcription.display_name} supplies canonical text; "
            f"{anchor_label.lower()}; "
            + ("WhisperX performs forced alignment." if uses_whisperx else "SubGen reconciles the timing anchors.")
        ),
        transcription=transcription,
        stages=tuple(stages),
        translation_provider=translation_provider,
        translation_model=translation_model,
        timing_anchor_provider=timing_anchor_provider,
        timing_anchor_model=timing_anchor_model,
        timing_mode=timing_mode,
        uses_separate_timing_anchors=True,
        uses_whisperx=uses_whisperx,
    ))


def public_model_catalog(config: Optional[Mapping[str, Any]] = None) -> Iterable[Dict[str, Any]]:
    catalog = {key: value for key, value in MODEL_CATALOG.items()}
    for provider, models in (config or {}).get("model_capabilities", {}).items():
        for model in models:
            capabilities = _custom_capabilities(config or {}, provider, model)
            if capabilities:
                catalog[(provider, model)] = capabilities
    public_models = []
    for key in sorted(catalog):
        payload = catalog[key].to_dict()
        payload.update(MODEL_GUIDE_METADATA.get(key, {}))
        payload.setdefault("provider_display", payload["provider"])
        payload.setdefault("model_family", "Custom model")
        payload.setdefault("timing_description", "Capabilities declared by the user")
        payload.setdefault("guide_timing_kind", payload["adapter_timing_kind"])
        payload["semantic_ai"] = (
            payload["transcription_kind"] in {"semantic_llm", "semantic_asr"}
            or (payload["translation"] and not payload["audio_transcription"])
        )
        payload["subgen_supported"] = True
        public_models.append(payload)
    public_models.extend(dict(model) for model in GUIDE_ONLY_MODELS)
    return public_models
