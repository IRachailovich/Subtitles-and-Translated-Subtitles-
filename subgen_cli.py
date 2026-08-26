import argparse
import json
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from subgen_paths import CONFIG_PATH, ENV_PATH, SOURCE_DIR
from subgen_version import __version__

from subgen_pipeline import (
    CONFIG,
    format_usd,
    get_supported_languages,
    load_dotenv,
    main as run_pipeline,
    normalize_subtitle_mode,
    retranslate_review_selected_cues,
)
from subgen_providers import (
    BUILTIN_PROVIDERS,
    configured_providers,
    get_provider,
    get_provider_registry,
    normalize_provider_id,
)
from subgen_db import get_review_manifest, list_review_manifests, save_review_manifest
from subgen_review import (
    approve_review,
    assert_burn_allowed,
    load_review,
    render_srt_bytes,
    save_review,
)


APP_DIR = SOURCE_DIR

PROVIDER_BILLING_URLS = {
    "openai": "https://platform.openai.com/usage",
    "google": "https://console.cloud.google.com/billing",
    "deepseek": "https://platform.deepseek.com/usage",
    "anthropic": "https://console.anthropic.com/settings/billing",
    "xai": "https://console.x.ai/",
    "cohere": "https://dashboard.cohere.com/",
}

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".wmv", ".flv"}
BATCH_STATE_FILENAME = ".subgen_batch_state.json"
BATCH_STOP_FILENAME = ".subgen_stop_after_current"


LANGUAGE_STYLE_DEFAULTS = {
    "default": {
        "font_name": "Arial",
        "font_size": 28,
        "primary_color": "#FFFFFF",
        "outline_color": "#000000",
        "back_color": "#000000",
        "outline_width": 1,
        "shadow": 1,
        "border_style": 3,
        "alignment": 2,
        "margin_v": 30,
        "margin_l": 20,
        "margin_r": 20,
    },
    "ar": {
        "font_name": "Amiri",
        "font_size": 44,
        "primary_color": "#FFFFFF",
        "outline_color": "#E361F7",
        "back_color": "#000000",
        "outline_width": 1,
        "shadow": 1,
        "border_style": 4,
        "alignment": 2,
        "margin_v": 38,
        "margin_l": 24,
        "margin_r": 24,
    },
    "fa": {
        "font_name": "Amiri",
        "font_size": 44,
        "primary_color": "#FFFFFF",
        "outline_color": "#E361F7",
        "back_color": "#000000",
        "outline_width": 1,
        "shadow": 1,
        "border_style": 4,
        "alignment": 2,
        "margin_v": 38,
        "margin_l": 24,
        "margin_r": 24,
    },
    "he": {
        "font_name": "Arial",
        "font_size": 34,
        "primary_color": "#FFFFFF",
        "outline_color": "#000000",
        "back_color": "#000000",
        "outline_width": 2,
        "shadow": 1,
        "border_style": 3,
        "alignment": 2,
        "margin_v": 34,
        "margin_l": 22,
        "margin_r": 22,
    },
    "en": {
        "font_name": "Arial",
        "font_size": 28,
        "primary_color": "#FFFFFF",
        "outline_color": "#000000",
        "back_color": "#000000",
        "outline_width": 1,
        "shadow": 1,
        "border_style": 3,
        "alignment": 2,
        "margin_v": 30,
        "margin_l": 20,
        "margin_r": 20,
    },
}


def default_config():
    config = CONFIG.copy()
    config.update({
        "setup_complete": False,
        "preferred_transcription_provider": "google",
        "preferred_translation_provider": "openai",
        "preferred_openai_profile": "default",
        "openai_profiles": {
            "default": {
                "label": "default",
                "env_key": "OPENAI_API_KEY",
            }
        },
        "provider_models": {
            "openai": {
                "transcription": "whisper-1",
                "translation": "gpt-4o",
            },
            "google": {
                "transcription": "gemini-2.5-flash",
                "translation": "gemini-2.5-pro",
            },
            "xai": {
                "transcription": "speech-to-text",
                "translation": "grok-4",
            },
            "cohere": {
                "transcription": "cohere-transcribe-03-2026",
                "translation": "command-a-plus-05-2026",
            },
        },
        "pipeline_selection_mode": "automatic",
        "custom_providers": {},
        "styles": LANGUAGE_STYLE_DEFAULTS,
        "last_output_dir": "",
    })
    return config


def migrate_config_values(config):
    if "subtitle_mode" not in config:
        config["subtitle_mode"] = normalize_subtitle_mode(
            legacy_tiktok_style=config.get("tiktok_style", False)
        )
    # Ensure timing mode uses precise (WhisperX) or maps correctly
    timing_mapping = {
        "best": "precise",
        "api_fuzzy": "fuzzy",
        "anchor_text": "local_whisper",
    }
    current_mode = config.get("api_transcript_timing_mode")
    if current_mode in timing_mapping or current_mode in {"proportional", "forced", None}:
        config["api_transcript_timing_mode"] = timing_mapping.get(current_mode, current_mode or "precise")

    # Migrate providers to API by default ONLY if they are not already set
    if config.get("transcription_provider") is None:
        config["transcription_provider"] = "openai"
    if config.get("transcription_backend") is None:
        config["transcription_backend"] = "openai"
    if config.get("translation_provider") is None:
        config["translation_provider"] = "openai"
    if config.get("translation_backend") is None:
        config["translation_backend"] = "openai"

    # Preserve explicit transcription model choices. Pipeline planning depends on
    # the selected model's real capabilities, so migration must not substitute a
    # different audio model behind the user's back.
    if config.get("llm_model") == "gpt-5.2":
        config["llm_model"] = "gpt-4o"
    if config.get("qa_model") == "gpt-5.2":
        config["qa_model"] = "gpt-4o-mini"
    if config.get("visual_style_model") == "gpt-5.2":
        config["visual_style_model"] = "gpt-4o-mini"

    # Migrate provider_models if they exist
    provider_models = config.setdefault("provider_models", {})
    openai_models = provider_models.setdefault("openai", {})
    if not openai_models.get("transcription"):
        openai_models["transcription"] = "whisper-1"
    if openai_models.get("translation") == "gpt-5.2" or not openai_models.get("translation"):
        openai_models["translation"] = "gpt-4o"

    # Clean up pricing config if it has the old keys
    pricing = config.get("api_pricing", {})
    if "openai" in pricing:
        if "gpt-4o-transcribe" in pricing["openai"] or "gpt-5.4" in pricing["openai"]:
            from subgen_pipeline import CONFIG as pipeline_config
            config["api_pricing"]["openai"] = pipeline_config["api_pricing"]["openai"]

    return config


def load_config():
    if not CONFIG_PATH.exists():
        return default_config()

    try:
        loaded = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return default_config()

    if "subtitle_mode" not in loaded:
        loaded["subtitle_mode"] = normalize_subtitle_mode(
            legacy_tiktok_style=loaded.get("tiktok_style", False)
        )
    merged = default_config()
    merged.update(loaded)
    loaded_styles = loaded.get("styles", {})
    merged_styles = {}
    for language_code, default_style in LANGUAGE_STYLE_DEFAULTS.items():
        merged_styles[language_code] = {
            **default_style,
            **loaded_styles.get(language_code, {}),
        }
    for language_code, loaded_style in loaded_styles.items():
        if language_code not in merged_styles:
            merged_styles[language_code] = {
                **LANGUAGE_STYLE_DEFAULTS.get("default", {}),
                **loaded_style,
            }
    merged["styles"] = merged_styles
    ensure_openai_profiles(merged)
    migrate_config_values(merged)
    return merged


def save_config(config):
    CONFIG_PATH.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def read_env_values():
    values = {}
    if ENV_PATH.exists():
        for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()
    try:
        import keyring
        credential_keys = {
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "DEEPSEEK_API_KEY",
            "ANTHROPIC_API_KEY",
        }
        for profiles in (load_config().get("api_profiles", {}) or {}).values():
            credential_keys.update(
                profile.get("env_key")
                for profile in (profiles or {}).values()
                if profile.get("env_key")
            )
        for key in credential_keys:
            stored = keyring.get_password("SubGen", key)
            if stored:
                values[key] = stored
    except Exception:
        pass
    return values


def write_env_values(values):
    try:
        import keyring
        for key, value in values.items():
            keyring.set_password("SubGen", key, value)
        ENV_PATH.write_text(
            "# API keys are stored in the operating-system credential locker.\n",
            encoding="utf-8",
        )
        return
    except Exception:
        pass
    lines = [
        "# SubGen provider API keys.",
        "# This file is ignored by git. Do not share it.",
    ]
    lines.extend(f"{key}={value}" for key, value in sorted(values.items()))
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_env_value(key, value):
    values = read_env_values()
    values[key] = value
    write_env_values(values)
    os.environ[key] = value


def prompt_api_key_update(provider_name, env_key):
    current_key = os.environ.get(env_key, "")
    if current_key:
        value = input(f"{provider_name} API key ({env_key}) [keep existing]: ").strip()
        return value if value else current_key

    api_key = prompt(f"{provider_name} API key ({env_key})")
    while not api_key:
        print("API key cannot be empty.")
        api_key = prompt(f"{provider_name} API key ({env_key})")
    return api_key


def ensure_env_file():
    if not ENV_PATH.exists():
        write_env_values({})


def prompt(message, default=None):
    suffix = f" [{default}]" if default not in (None, "") else ""
    value = input(f"{message}{suffix}: ").strip()
    return value if value else default


def clean_path_value(value):
    return str(value or "").strip().strip('"').strip("'")


def normalize_openai_profile_id(value):
    return normalize_provider_id(value or "default") or "default"


def openai_profile_env_key(profile_id):
    profile_id = normalize_openai_profile_id(profile_id)
    if profile_id == "default":
        return "OPENAI_API_KEY"
    return f"OPENAI_API_KEY_{profile_id.upper()}"


def ensure_openai_profiles(config):
    profiles = config.setdefault("openai_profiles", {})
    if "default" not in profiles:
        profiles["default"] = {"label": "default", "env_key": "OPENAI_API_KEY"}
    for profile_id, profile in list(profiles.items()):
        normalized_id = normalize_openai_profile_id(profile_id)
        if normalized_id != profile_id:
            profiles[normalized_id] = profile
            del profiles[profile_id]
            profile_id = normalized_id
        if not isinstance(profile, dict):
            profile = {"label": str(profile_id), "env_key": openai_profile_env_key(profile_id)}
            profiles[profile_id] = profile
        profile.setdefault("label", profile_id)
        profile.setdefault("env_key", openai_profile_env_key(profile_id))
    config.setdefault("preferred_openai_profile", "default")
    if config["preferred_openai_profile"] not in profiles:
        config["preferred_openai_profile"] = "default"
    return profiles


def configured_openai_profiles(config):
    profiles = ensure_openai_profiles(config)
    return {
        profile_id: profile
        for profile_id, profile in profiles.items()
        if os.environ.get(profile.get("env_key", ""))
    }


def has_configured_openai_profile(config):
    return bool(configured_openai_profiles(config))


def choose_openai_profile(config, default_profile=None, include_unconfigured=False):
    profiles = ensure_openai_profiles(config)
    configured = configured_openai_profiles(config)
    options = []
    for profile_id, profile in profiles.items():
        env_key = profile.get("env_key")
        is_configured = profile_id in configured
        if is_configured or include_unconfigured:
            suffix = "" if is_configured else " - API key not configured"
            options.append((f"{profile.get('label', profile_id)} ({env_key}){suffix}", profile_id))

    if not options:
        raise RuntimeError("No OpenAI API key profiles are configured. Run SubGen config and add an OpenAI profile.")

    default_profile = default_profile or config.get("preferred_openai_profile", "default")
    default_index = next(
        (index for index, (_, profile_id) in enumerate(options) if profile_id == default_profile),
        0,
    )
    return choose_from_list("OpenAI API key profile", options, default_index=default_index)


def get_openai_profile(config, profile_id):
    profiles = ensure_openai_profiles(config)
    profile_id = normalize_openai_profile_id(profile_id)
    if profile_id not in profiles:
        raise RuntimeError(f"Unknown OpenAI profile: {profile_id}")
    return profile_id, profiles[profile_id]


def parse_glossary_argument(value):
    entries = []
    for raw_entry in str(value or "").split(";"):
        if "=" not in raw_entry:
            continue
        source, target = raw_entry.split("=", 1)
        source = source.strip()
        target = target.strip()
        if source and target:
            entries.append({"source": source, "target": target, "forbidden": []})
    return entries


def prompt_yes_no(message, default=False):
    default_text = "Y/n" if default else "y/N"
    value = input(f"{message} [{default_text}]: ").strip().lower()
    if not value:
        return default
    return value in {"y", "yes"}


def choose_from_list(title, options, default_index=0):
    print(f"\n{title}")
    for index, (label, _) in enumerate(options, 1):
        marker = " (default)" if index - 1 == default_index else ""
        print(f"{index}. {label}{marker}")

    while True:
        raw = input("Choose number: ").strip()
        if not raw:
            return options[default_index][1]
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][1]
        print("Invalid choice.")


def choose_language(allow_none=True):
    languages = get_supported_languages()
    options = []
    if allow_none:
        options.append(("None - keep original language", None))
    options.extend([f"{code} - {name}", code] for code, name in languages.items())
    return choose_from_list("Target language", options, default_index=0)


def get_style(config, language_code):
    styles = config.get("styles", {})
    return {**styles.get("default", {}), **styles.get(language_code or "default", {})}


def print_style(style):
    print("\nSubtitle style")
    for key in [
        "font_name",
        "font_size",
        "primary_color",
        "outline_color",
        "back_color",
        "outline_width",
        "shadow",
        "border_style",
        "alignment",
        "margin_v",
        "margin_l",
        "margin_r",
    ]:
        print(f"{key}: {style.get(key)}")


def prompt_int(message, default):
    while True:
        val = prompt(message, str(default))
        try:
            return int(val)
        except ValueError:
            print("Invalid integer. Please enter a valid number.")


def edit_style(style):
    edited = style.copy()
    print("\nPress Enter to keep the current value.")
    edited["font_name"] = prompt("Font name", edited.get("font_name"))
    edited["font_size"] = prompt_int("Font size", edited.get("font_size"))
    edited["primary_color"] = prompt("Text color hex", edited.get("primary_color"))
    edited["outline_color"] = prompt("Outline color hex", edited.get("outline_color"))
    edited["back_color"] = prompt("Background color hex", edited.get("back_color"))
    edited["outline_width"] = prompt_int("Outline width", edited.get("outline_width"))
    edited["shadow"] = prompt_int("Shadow", edited.get("shadow"))
    edited["border_style"] = prompt_int("Border style", edited.get("border_style"))
    edited["alignment"] = prompt_int("ASS alignment", edited.get("alignment", 2))
    edited["margin_v"] = prompt_int("Vertical margin", edited.get("margin_v", 30))
    edited["margin_l"] = prompt_int("Left margin", edited.get("margin_l", 20))
    edited["margin_r"] = prompt_int("Right margin", edited.get("margin_r", 20))
    return edited


def provider_model(config, provider_id, capability):
    provider = get_provider(config, provider_id, capability)
    return (
        config.get("provider_models", {}).get(provider_id, {}).get(capability)
        or provider.get(f"{capability}_model")
        or ""
    )


def resolve_qa_enabled(config, args):
    cli_value = getattr(args, "qa_enabled", None)
    if cli_value is not None:
        return bool(cli_value)
    return bool(config.get("qa_enabled", CONFIG["qa_enabled"]))


def resolve_qa_model(config, args):
    return getattr(args, "qa_model", None) or config.get("qa_model", CONFIG["qa_model"])


def resolve_qa_policy(config, args):
    return getattr(args, "qa_policy", None) or config.get("qa_policy", CONFIG["qa_policy"])


def resolve_api_transcript_timing_mode(config, args, transcription_provider):
    explicit_value = getattr(args, "api_transcript_timing_mode", None)
    if explicit_value:
        if transcription_provider != "local" and explicit_value == "local_whisper":
            print(
                "Warning: local_whisper uses local Whisper text and is disabled for API transcription. "
                "Using precise instead."
            )
            return "precise"
        return explicit_value
    configured_value = config.get("api_transcript_timing_mode", CONFIG["api_transcript_timing_mode"])
    if transcription_provider != "local" and configured_value in {"local_whisper", "proportional"}:
        print(
            f"Warning: stored api_transcript_timing_mode={configured_value} uses weaker source text. "
            "Using precise for this API run."
        )
        return "precise"
    return configured_value


def resolve_visual_style_enabled(config, args):
    cli_value = getattr(args, "visual_style_enabled", None)
    if cli_value is not None:
        return bool(cli_value)
    return bool(config.get("visual_style_enabled", CONFIG["visual_style_enabled"]))


def resolve_tiktok_style_enabled(config, args):
    cli_value = getattr(args, "tiktok_style", None)
    if cli_value is not None:
        return bool(cli_value)
    return bool(config.get("tiktok_style", CONFIG.get("tiktok_style", False)))


def resolve_subtitle_mode_enabled(config, args):
    explicit_mode = getattr(args, "subtitle_mode", None)
    if explicit_mode:
        return normalize_subtitle_mode(explicit_mode)
    legacy_value = getattr(args, "tiktok_style", None)
    if legacy_value is not None:
        return "tiktok" if legacy_value else "normal"
    return normalize_subtitle_mode(
        config.get("subtitle_mode"),
        legacy_tiktok_style=config.get("tiktok_style", False),
    )


def resolve_visual_style_model(config, args):
    return (
        getattr(args, "visual_style_model", None)
        or config.get("visual_style_model", CONFIG["visual_style_model"])
    )


def configure_openai_provider(config):
    provider = BUILTIN_PROVIDERS["openai"]
    profiles = ensure_openai_profiles(config)
    print("\nConfigure OpenAI")
    print("OpenAI profiles")
    for profile_id, profile in profiles.items():
        status = "configured" if os.environ.get(profile["env_key"]) else "not configured"
        preferred = " default" if profile_id == config.get("preferred_openai_profile") else ""
        print(f"- {profile_id}: {profile['env_key']} | {status}{preferred}")

    options = [
        (f"Update {profile_id} ({profile['env_key']})", profile_id)
        for profile_id, profile in profiles.items()
    ]
    options.append(("Add new OpenAI profile", "__new__"))
    choice = choose_from_list("OpenAI profile to configure", options, default_index=0)

    if choice == "__new__":
        profile_label = prompt("New OpenAI profile name")
        while not profile_label:
            profile_label = prompt("New OpenAI profile name")
        profile_id = normalize_openai_profile_id(profile_label)
        env_key = openai_profile_env_key(profile_id)
        profiles[profile_id] = {
            "label": profile_label,
            "env_key": env_key,
        }
    else:
        profile_id = choice
        profile = profiles[profile_id]
        profile_label = profile.get("label", profile_id)
        env_key = profile["env_key"]

    api_key = prompt_api_key_update(provider["name"], env_key)
    update_env_value(env_key, api_key)
    if prompt_yes_no(f"Use '{profile_id}' as the default OpenAI profile?", default=False):
        config["preferred_openai_profile"] = profile_id

    models = config.setdefault("provider_models", {}).setdefault("openai", {})
    if provider.get("transcription"):
        models["transcription"] = prompt(
            "Transcription model",
            models.get("transcription") or provider.get("transcription_model"),
        )
    if provider.get("translation"):
        models["translation"] = prompt(
            "Translation model",
            models.get("translation") or provider.get("translation_model"),
        )


def configure_builtin_provider(config, provider_id):
    if provider_id == "openai":
        configure_openai_provider(config)
        return

    provider = BUILTIN_PROVIDERS[provider_id]
    print(f"\nConfigure {provider['name']}")
    env_key = provider["env_key"]
    api_key = prompt_api_key_update(provider["name"], env_key)
    update_env_value(env_key, api_key)

    models = config.setdefault("provider_models", {}).setdefault(provider_id, {})
    if provider.get("transcription"):
        models["transcription"] = prompt(
            "Transcription model",
            models.get("transcription") or provider.get("transcription_model"),
        )
    if provider.get("translation"):
        models["translation"] = prompt(
            "Translation model",
            models.get("translation") or provider.get("translation_model"),
        )


def configure_custom_provider(config):
    print("\nCustom OpenAI-compatible translation provider")
    name = prompt("Provider name")
    while not name:
        name = prompt("Provider name")
    provider_id = normalize_provider_id(name)
    base_url = prompt("Base URL, without /chat/completions")
    model = prompt("Translation model name")
    env_key = f"SUBGEN_{provider_id.upper()}_API_KEY"
    api_key = prompt(f"API key ({env_key})")
    while not api_key:
        api_key = prompt(f"API key ({env_key})")

    config.setdefault("custom_providers", {})[provider_id] = {
        "name": name,
        "env_key": env_key,
        "base_url": base_url.rstrip("/"),
        "api_style": "openai_compatible",
        "transcription": False,
        "translation": True,
        "translation_model": model,
    }
    config.setdefault("provider_models", {}).setdefault(provider_id, {})["translation"] = model
    update_env_value(env_key, api_key)


def provider_choices(config, capability, include_unconfigured=False):
    options = [("Local/offline", "local")]
    registry = get_provider_registry(config)
    for provider_id, provider in registry.items():
        if not provider.get(capability):
            continue
        configured = (
            has_configured_openai_profile(config)
            if provider_id == "openai"
            else bool(os.environ.get(provider["env_key"]))
        )
        if configured or include_unconfigured:
            suffix = "" if configured else " - API key not configured"
            options.append((f"{provider['name']}{suffix}", provider_id))
    return options


def choose_provider(config, capability, default_provider="local", include_unconfigured=False):
    options = provider_choices(config, capability, include_unconfigured=include_unconfigured)
    default_index = next(
        (index for index, (_, provider_id) in enumerate(options) if provider_id == default_provider),
        0,
    )
    return choose_from_list(f"{capability.title()} provider", options, default_index=default_index)


def configure_provider_menu(config):
    while True:
        options = [
            (provider["name"], provider_id)
            for provider_id, provider in BUILTIN_PROVIDERS.items()
        ]
        options.extend([
            ("Custom OpenAI-compatible translation provider", "custom"),
            ("Finish provider setup", "done"),
        ])
        choice = choose_from_list("Add or update provider", options, default_index=len(options) - 1)
        if choice == "done":
            break
        if choice == "custom":
            configure_custom_provider(config)
        else:
            configure_builtin_provider(config, choice)


def choose_default_providers(config):
    config["preferred_transcription_provider"] = choose_provider(
        config,
        "transcription",
        config.get("preferred_transcription_provider", "local"),
    )
    config["preferred_translation_provider"] = choose_provider(
        config,
        "translation",
        config.get("preferred_translation_provider", "local"),
    )


def run_setup(args=None):
    load_dotenv(ENV_PATH)
    config = load_config()
    print("SubGen setup")
    ensure_env_file()

    if prompt_yes_no("Add or update API providers?", default=True):
        configure_provider_menu(config)

    choose_default_providers(config)
    config["setup_complete"] = True
    save_config(config)
    print(f"Setup saved: {CONFIG_PATH}")
    print(f"API key file: {ENV_PATH}")
    return config


def configure_subtitle_style(config):
    config = load_config()
    languages = get_supported_languages()
    options = [("default - fallback style", "default")]
    options.extend([f"{code} - {name}", code] for code, name in languages.items())
    language_code = choose_from_list("Configure subtitle style", options, default_index=0)
    current = get_style(config, language_code)
    print_style(current)

    if prompt_yes_no("Edit this style?", default=True):
        config.setdefault("styles", {})[language_code] = edit_style(current)
        save_config(config)
        print(f"Saved global style for {language_code}.")

    return config


def run_config(args=None):
    load_dotenv(ENV_PATH)
    config = load_config()
    choice = choose_from_list(
        "Configuration",
        [
            ("Subtitle styles by language", "styles"),
            ("API providers and keys", "providers"),
            ("Default transcription/translation providers", "defaults"),
        ],
        default_index=0,
    )
    if choice == "styles":
        return configure_subtitle_style(config)
    if choice == "providers":
        configure_provider_menu(config)
    else:
        choose_default_providers(config)
    save_config(config)
    return config


def interactive_run(args=None):
    if args is None:
        import argparse
        args = argparse.Namespace()

    load_dotenv(ENV_PATH)
    config = load_config()

    if not config.get("setup_complete"):
        config = run_setup(args)

    no_prompts = getattr(args, "no_prompts", False)

    if no_prompts:
        transcription_provider = getattr(args, "transcription_provider", None) or config.get("preferred_transcription_provider", "local")
        translation_provider = getattr(args, "translation_provider", None) or config.get("preferred_translation_provider", "local")
    else:
        transcription_provider = getattr(args, "transcription_provider", None) or choose_provider(
            config,
            "transcription",
            config.get("preferred_transcription_provider", "local"),
        )
        translation_provider = getattr(args, "translation_provider", None) or choose_provider(
            config,
            "translation",
            config.get("preferred_translation_provider", "local"),
        )

    qa_enabled = resolve_qa_enabled(config, args)
    qa_provider = config.get("qa_provider", CONFIG["qa_provider"])
    visual_style_enabled = resolve_visual_style_enabled(config, args)
    visual_style_provider = config.get("visual_style_provider", CONFIG["visual_style_provider"])
    openai_profile = None
    openai_api_env_key = "OPENAI_API_KEY"
    if (
        "openai" in {transcription_provider, translation_provider}
        or (qa_enabled and qa_provider == "openai")
        or (visual_style_enabled and visual_style_provider == "openai")
    ):
        requested_profile = getattr(args, "openai_profile", None)
        if requested_profile:
            requested_profile = normalize_openai_profile_id(requested_profile)
            profiles = ensure_openai_profiles(config)
            profiles.setdefault(requested_profile, {
                "label": requested_profile,
                "env_key": openai_profile_env_key(requested_profile),
            })
            openai_profile = requested_profile
        elif no_prompts:
            openai_profile = config.get("preferred_openai_profile", "default")
        else:
            openai_profile = choose_openai_profile(
                config,
                default_profile=config.get("preferred_openai_profile", "default"),
            )
        openai_profile, profile = get_openai_profile(config, openai_profile)
        openai_api_env_key = profile["env_key"]
        if not os.environ.get(openai_api_env_key):
            raise RuntimeError(
                f"OpenAI profile '{openai_profile}' uses {openai_api_env_key}, but that key is not set. "
                "Run SubGen config and add/update the OpenAI provider profile."
            )
        config["preferred_openai_profile"] = openai_profile

    if no_prompts:
        video_path_value = clean_path_value(getattr(args, "video", None))
        if not video_path_value:
            raise RuntimeError("Video path must be provided via --video when --no-prompts is used.")
        video_path = Path(video_path_value).expanduser()
    else:
        video_path_value = clean_path_value(getattr(args, "video", None) or prompt("Video path"))
        video_path = Path(video_path_value).expanduser()
        while not video_path.exists():
            print("Video file not found.")
            video_path = Path(clean_path_value(prompt("Video path"))).expanduser()

    output_dir_default = getattr(args, "output_dir", None) or config.get("last_output_dir") or str(video_path.parent)
    if no_prompts:
        output_dir = Path(clean_path_value(output_dir_default)).expanduser()
    else:
        output_dir = Path(
            clean_path_value(output_dir_default if getattr(args, "output_dir", None) else prompt("Output folder", output_dir_default))
        ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    config["last_output_dir"] = str(output_dir)

    target_language_arg = getattr(args, "target_language", None)
    if target_language_arg == "none":
        target_language = None
    elif target_language_arg:
        target_language = target_language_arg
    elif no_prompts:
        target_language = None
    else:
        target_language = choose_language(allow_none=True)
    style_language = target_language or "default"
    style = get_style(config, style_language)
    print_style(style)
    if no_prompts:
        style_config = style
    elif prompt_yes_no("Keep this style for this video?", default=True):
        style_config = style
    else:
        style_config = edit_style(style)
        if prompt_yes_no(f"Save these changes globally for {style_language}?", default=False):
            config.setdefault("styles", {})[style_language] = style_config

    source_language = getattr(args, "source_language", None)
    source_dialect = getattr(args, "source_dialect", None) or config.get("source_dialect", "auto")
    if transcription_provider != "local" and not no_prompts:
        source_language = prompt("Source language code for transcription, or blank for auto", config.get("source_language") or "")
        source_language = source_language or None
        source_dialect = prompt("Source dialect hint", config.get("source_dialect", "auto"))

    subtitle_mode = resolve_subtitle_mode_enabled(config, args)
    if (
        not getattr(args, "no_prompts", False)
        and getattr(args, "subtitle_mode", None) is None
        and getattr(args, "tiktok_style", None) is None
    ):
        options = ["auto", "normal", "tiktok"]
        subtitle_mode = choose_from_list(
            "Subtitle format",
            options,
            default_index=options.index(subtitle_mode),
        )
    args.subtitle_mode = subtitle_mode
    args.tiktok_style = subtitle_mode == "tiktok"

    save_config(config)

    pipeline_config = config.copy()
    glossary = list(config.get("translation_glossary", []))
    glossary.extend(parse_glossary_argument(getattr(args, "glossary", None)))
    pipeline_config.update({
        "transcription_provider": transcription_provider,
        "translation_provider": translation_provider,
        "transcription_model": (
            getattr(args, "transcription_model", None)
            or provider_model(config, transcription_provider, "transcription")
            if transcription_provider != "local"
            else config.get("model_size", "small")
        ),
        "translation_model": (
            getattr(args, "translation_model", None)
            or provider_model(config, translation_provider, "translation")
            if translation_provider != "local"
            else None
        ),
        "model_size": getattr(args, "model_size", None) or config.get("model_size", "small"),
        "api_transcript_timing_mode": resolve_api_transcript_timing_mode(config, args, transcription_provider),
        "source_language": source_language,
        "source_dialect": source_dialect,
        "target_dialect": config.get("target_dialect", "natural English"),
        "translation_glossary": glossary,
        "openai_profile": openai_profile,
        "openai_api_env_key": openai_api_env_key,
        "qa_enabled": qa_enabled,
        "qa_provider": qa_provider,
        "qa_model": resolve_qa_model(config, args),
        "qa_policy": resolve_qa_policy(config, args),
        "visual_style_enabled": visual_style_enabled,
        "visual_style_provider": visual_style_provider,
        "visual_style_model": resolve_visual_style_model(config, args),
        "subtitle_mode": resolve_subtitle_mode_enabled(config, args),
        "tiktok_style": resolve_subtitle_mode_enabled(config, args) == "tiktok",
    })

    run_pipeline(
        str(video_path),
        target_language=target_language,
        style_config=style_config,
        pipeline_config=pipeline_config,
        keep_files=False,
        force=getattr(args, "force", False),
        no_burn=getattr(args, "no_burn", False),
        output_dir=str(output_dir),
    )


def print_languages(args=None):
    print("Supported target languages")
    print("none - keep original language")
    for code, name in get_supported_languages().items():
        print(f"{code} - {name}")


def print_providers(args=None):
    load_dotenv(ENV_PATH)
    config = load_config()
    print("SubGen providers")
    for provider_id, provider in get_provider_registry(config).items():
        configured = (
            "configured"
            if (has_configured_openai_profile(config) if provider_id == "openai" else os.environ.get(provider["env_key"]))
            else "not configured"
        )
        capabilities = []
        if provider.get("transcription"):
            capabilities.append("transcription")
        if provider.get("translation"):
            capabilities.append("translation")
        print(f"{provider_id}: {provider['name']} | {', '.join(capabilities)} | {configured}")
        if provider_id == "openai":
            for profile_id, profile in ensure_openai_profiles(config).items():
                profile_status = "configured" if os.environ.get(profile["env_key"]) else "not configured"
                print(f"  - profile {profile_id}: {profile['env_key']} | {profile_status}")
    print("local: Local/offline | transcription, translation | always available")


def manifest_paths(root, recursive=False):
    root = Path(root).expanduser().resolve()
    if root.is_file() and root.name.endswith(".manifest.json"):
        return [root]
    pattern = "**/*.manifest.json" if recursive else "*.manifest.json"
    return sorted(root.glob(pattern)) if root.exists() else []


def load_manifest_cost(path):
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    report = manifest.get("api_cost_report")
    if not report:
        return None
    return {
        "path": path,
        "input": manifest.get("input", ""),
        "known_cost_usd": float(report.get("known_cost_usd", 0) or 0),
        "unknown_cost_event_count": int(report.get("unknown_cost_event_count", 0) or 0),
        "totals": report.get("totals", {}),
    }


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def safe_folder_name(name):
    cleaned = "".join(char if char not in '<>:"/\\|?*' and ord(char) >= 32 else "_" for char in name)
    cleaned = cleaned.strip(" .")
    return cleaned or "video"


def is_video_path(path):
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def safe_extract_zip(zip_path, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            member_path = (destination / member.filename).resolve()
            if destination not in member_path.parents and member_path != destination:
                raise RuntimeError(f"Unsafe zip member path: {member.filename}")
        archive.extractall(destination)
    return destination


def read_video_list_file(path):
    entries = []
    base_dir = Path(path).resolve().parent
    for raw_line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = clean_path_value(raw_line)
        if not line or line.startswith("#"):
            continue
        candidate = Path(line).expanduser()
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        entries.append(candidate.resolve())
    return entries


def discover_batch_videos(input_path, output_root):
    input_path = Path(clean_path_value(input_path)).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not input_path.exists():
        raise RuntimeError(f"Batch input not found: {input_path}")

    source_root = input_path
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        source_root = output_root / "_batch_sources" / safe_folder_name(input_path.stem)
        if not source_root.exists():
            print(f"Extracting zip to {source_root}")
            safe_extract_zip(input_path, source_root)
        else:
            print(f"Reusing extracted zip folder: {source_root}")

    if source_root.is_dir():
        videos = sorted(path.resolve() for path in source_root.rglob("*") if path.is_file() and is_video_path(path))
    elif source_root.is_file() and is_video_path(source_root):
        videos = [source_root]
    elif source_root.is_file() and source_root.suffix.lower() in {".txt", ".list"}:
        videos = [path for path in read_video_list_file(source_root) if is_video_path(path)]
    else:
        raise RuntimeError("Batch input must be a video file, folder, zip file, or .txt/.list file of video paths.")

    existing = []
    missing = []
    for video in videos:
        if Path(video).exists():
            existing.append(Path(video).resolve())
        else:
            missing.append(str(video))
    if missing:
        print("Warning: some listed videos were not found and will be skipped:")
        for item in missing:
            print(f"- {item}")
    if not existing:
        raise RuntimeError(f"No video files found in {input_path}")
    return existing


def output_dirs_for_videos(videos, output_root):
    used_names = {}
    rows = []
    for video in videos:
        base_name = safe_folder_name(Path(video).stem)
        count = used_names.get(base_name, 0) + 1
        used_names[base_name] = count
        folder_name = base_name if count == 1 else f"{base_name}_{count}"
        rows.append({
            "source": str(Path(video).resolve()),
            "output_dir": str((Path(output_root) / folder_name).resolve()),
            "status": "pending",
        })
    return rows


def batch_state_path(output_root):
    return Path(output_root).expanduser().resolve() / BATCH_STATE_FILENAME


def batch_stop_path(output_root):
    return Path(output_root).expanduser().resolve() / BATCH_STOP_FILENAME


def load_batch_state(output_root):
    path = batch_state_path(output_root)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_batch_state(output_root, state):
    path = batch_state_path(output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def build_batch_pipeline_config(config, args, transcription_provider, translation_provider, openai_profile, openai_api_env_key, source_language, source_dialect):
    glossary = list(config.get("translation_glossary", []))
    glossary.extend(parse_glossary_argument(getattr(args, "glossary", None)))
    qa_enabled = resolve_qa_enabled(config, args)
    qa_provider = config.get("qa_provider", CONFIG["qa_provider"])
    visual_style_enabled = resolve_visual_style_enabled(config, args)
    visual_style_provider = config.get("visual_style_provider", CONFIG["visual_style_provider"])
    return {
        **config,
        "transcription_provider": transcription_provider,
        "translation_provider": translation_provider,
        "transcription_model": (
            getattr(args, "transcription_model", None)
            or provider_model(config, transcription_provider, "transcription")
            if transcription_provider != "local"
            else config.get("model_size", "small")
        ),
        "translation_model": (
            getattr(args, "translation_model", None)
            or provider_model(config, translation_provider, "translation")
            if translation_provider != "local"
            else None
        ),
        "model_size": getattr(args, "model_size", None) or config.get("model_size", "small"),
        "api_transcript_timing_mode": resolve_api_transcript_timing_mode(config, args, transcription_provider),
        "api_transcription_chunking": (
            getattr(args, "api_transcription_chunking", None)
            if getattr(args, "api_transcription_chunking", None) is not None
            else config.get("api_transcription_chunking", CONFIG["api_transcription_chunking"])
        ),
        "api_transcription_chunk_seconds": (
            getattr(args, "api_transcription_chunk_seconds", None)
            or config.get("api_transcription_chunk_seconds", CONFIG["api_transcription_chunk_seconds"])
        ),
        "api_transcription_chunk_overlap_seconds": (
            getattr(args, "api_transcription_chunk_overlap_seconds", None)
            or config.get("api_transcription_chunk_overlap_seconds", CONFIG["api_transcription_chunk_overlap_seconds"])
        ),
        "source_language": source_language,
        "source_dialect": source_dialect,
        "target_dialect": config.get("target_dialect", "natural English"),
        "translation_glossary": glossary,
        "openai_profile": openai_profile,
        "openai_api_env_key": openai_api_env_key,
        "qa_enabled": qa_enabled,
        "qa_provider": qa_provider,
        "qa_model": resolve_qa_model(config, args),
        "qa_policy": resolve_qa_policy(config, args),
        "visual_style_enabled": visual_style_enabled,
        "visual_style_provider": visual_style_provider,
        "visual_style_model": resolve_visual_style_model(config, args),
        "subtitle_mode": resolve_subtitle_mode_enabled(config, args),
        "tiktok_style": resolve_subtitle_mode_enabled(config, args) == "tiktok",
    }


def resolve_openai_profile_for_run(
    config,
    args,
    transcription_provider,
    translation_provider,
    qa_enabled=False,
    qa_provider="openai",
    visual_style_enabled=False,
    visual_style_provider="openai",
):
    openai_profile = None
    openai_api_env_key = "OPENAI_API_KEY"
    if (
        "openai" not in {transcription_provider, translation_provider}
        and not (qa_enabled and qa_provider == "openai")
        and not (visual_style_enabled and visual_style_provider == "openai")
    ):
        return openai_profile, openai_api_env_key

    requested_profile = getattr(args, "openai_profile", None)
    if requested_profile:
        requested_profile = normalize_openai_profile_id(requested_profile)
        profiles = ensure_openai_profiles(config)
        profiles.setdefault(requested_profile, {
            "label": requested_profile,
            "env_key": openai_profile_env_key(requested_profile),
        })
        openai_profile = requested_profile
    else:
        openai_profile = choose_openai_profile(
            config,
            default_profile=config.get("preferred_openai_profile", "default"),
        )
    openai_profile, profile = get_openai_profile(config, openai_profile)
    openai_api_env_key = profile["env_key"]
    if not os.environ.get(openai_api_env_key):
        raise RuntimeError(
            f"OpenAI profile '{openai_profile}' uses {openai_api_env_key}, but that key is not set. "
            "Run SubGen config and add/update the OpenAI provider profile."
        )
    config["preferred_openai_profile"] = openai_profile
    return openai_profile, openai_api_env_key


def run_costs(args=None):
    config = load_config()
    default_root = getattr(args, "output_dir", None) or config.get("last_output_dir") or "."
    rows = [
        item
        for item in (load_manifest_cost(path) for path in manifest_paths(default_root, getattr(args, "recursive", False)))
        if item
    ]
    if not rows:
        print(f"No API cost reports found in {Path(default_root).expanduser().resolve()}")
        return []

    print("SubGen local API cost reports")
    for row in rows:
        totals = row["totals"]
        label = Path(row["input"]).name if row["input"] else row["path"].name
        print(
            f"- {label}: {format_usd(row['known_cost_usd'])}, "
            f"tokens total {totals.get('total_tokens', 0)}"
        )
        if row["unknown_cost_event_count"]:
            print(f"  unknown-cost events: {row['unknown_cost_event_count']}")

    known_total = sum(row["known_cost_usd"] for row in rows)
    unknown_total = sum(row["unknown_cost_event_count"] for row in rows)
    print(f"Known local total: {format_usd(known_total)}")
    if unknown_total:
        print(f"Unknown-cost events total: {unknown_total}")
    return rows


def run_balance(args=None):
    print("Provider balance / usage")
    print("Exact account balance is provider-side billing data; SubGen does not store it locally.")
    print("Open the provider dashboard for the authoritative current balance/usage:")
    for provider_id, url in PROVIDER_BILLING_URLS.items():
        print(f"- {provider_id}: {url}")

    output_dir = getattr(args, "output_dir", None)
    if output_dir:
        print("")
        run_costs(args)


def resolve_batch_target_language(args):
    target_language_arg = getattr(args, "target_language", None)
    if target_language_arg == "none":
        return None
    if target_language_arg:
        return target_language_arg
    return choose_language(allow_none=True)


def run_batch(args=None):
    if args is None:
        import argparse
        args = argparse.Namespace()
    load_dotenv(ENV_PATH)
    config = load_config()
    if not config.get("setup_complete"):
        config = run_setup(args)

    output_root = Path(clean_path_value(getattr(args, "output_dir", None) or prompt("Batch output folder"))).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config["last_output_dir"] = str(output_root)
    existing_state = load_batch_state(output_root)
    resuming = bool(existing_state and not getattr(args, "restart", False))

    input_value = clean_path_value(
        getattr(args, "input", None)
        or (existing_state or {}).get("input")
        or prompt("Video file/folder/zip/list path")
    )

    target_language = (
        resolve_batch_target_language(args)
        if not resuming or getattr(args, "target_language", None)
        else existing_state.get("target_language")
    )
    if target_language and target_language not in get_supported_languages():
        raise RuntimeError(f"Unsupported target language '{target_language}'. Run SubGen languages for valid codes.")
    style_language = target_language or "default"
    style_config = get_style(config, style_language)
    print_style(style_config)

    transcription_provider = (
        getattr(args, "transcription_provider", None)
        or (existing_state or {}).get("transcription_provider")
        or choose_provider(config, "transcription", config.get("preferred_transcription_provider", "local"))
    )
    translation_provider = (
        getattr(args, "translation_provider", None)
        or (existing_state or {}).get("translation_provider")
        or choose_provider(config, "translation", config.get("preferred_translation_provider", "local"))
    )
    if resuming and getattr(args, "visual_style_enabled", None) is None and "visual_style_enabled" in existing_state:
        args.visual_style_enabled = existing_state.get("visual_style_enabled")
    if resuming and getattr(args, "subtitle_mode", None) is None:
        if existing_state.get("subtitle_mode"):
            args.subtitle_mode = existing_state.get("subtitle_mode")
        elif getattr(args, "tiktok_style", None) is None and "tiktok_style" in existing_state:
            args.tiktok_style = existing_state.get("tiktok_style")
    if resuming and getattr(args, "visual_style_model", None) is None and existing_state.get("visual_style_model"):
        args.visual_style_model = existing_state.get("visual_style_model")
    qa_enabled = resolve_qa_enabled(config, args)
    qa_provider = config.get("qa_provider", CONFIG["qa_provider"])
    visual_style_enabled = resolve_visual_style_enabled(config, args)
    visual_style_provider = config.get("visual_style_provider", CONFIG["visual_style_provider"])
    if resuming and getattr(args, "openai_profile", None) is None and existing_state.get("openai_profile"):
        args.openai_profile = existing_state.get("openai_profile")
    openai_profile, openai_api_env_key = resolve_openai_profile_for_run(
        config,
        args,
        transcription_provider,
        translation_provider,
        qa_enabled=qa_enabled,
        qa_provider=qa_provider,
        visual_style_enabled=visual_style_enabled,
        visual_style_provider=visual_style_provider,
    )

    source_language = getattr(args, "source_language", None) or (existing_state or {}).get("source_language")
    source_dialect = (
        getattr(args, "source_dialect", None)
        or (existing_state or {}).get("source_dialect")
        or config.get("source_dialect", "auto")
    )
    if transcription_provider != "local" and not getattr(args, "no_prompts", False):
        source_language = prompt("Source language code for transcription, or blank for auto", source_language or config.get("source_language") or "")
        source_language = source_language or None
        source_dialect = prompt("Source dialect hint", source_dialect)

    subtitle_mode = resolve_subtitle_mode_enabled(config, args)
    if (
        not getattr(args, "no_prompts", False)
        and getattr(args, "subtitle_mode", None) is None
        and getattr(args, "tiktok_style", None) is None
    ):
        options = ["auto", "normal", "tiktok"]
        subtitle_mode = choose_from_list(
            "Subtitle format",
            options,
            default_index=options.index(subtitle_mode),
        )
    args.subtitle_mode = subtitle_mode
    args.tiktok_style = subtitle_mode == "tiktok"

    save_config(config)

    stop_path = batch_stop_path(output_root)
    if stop_path.exists():
        stop_path.unlink()

    if resuming:
        state = existing_state
        print(f"Resuming batch from {batch_state_path(output_root)}")
    else:
        videos = discover_batch_videos(input_value, output_root)
        state = {
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "input": str(Path(input_value).expanduser().resolve()),
            "output_root": str(output_root),
            "target_language": target_language,
            "transcription_provider": transcription_provider,
            "translation_provider": translation_provider,
            "openai_profile": openai_profile,
            "source_language": source_language,
            "source_dialect": source_dialect,
            "visual_style_enabled": visual_style_enabled,
            "visual_style_model": resolve_visual_style_model(config, args),
            "subtitle_mode": resolve_subtitle_mode_enabled(config, args),
            "tiktok_style": resolve_subtitle_mode_enabled(config, args) == "tiktok",
            "videos": output_dirs_for_videos(videos, output_root),
        }
        save_batch_state(output_root, state)

    videos = state.get("videos", [])
    total = len(videos)
    resumable_done = {"ready_for_review", "needs_attention", "approved", "completed"}
    completed = sum(1 for item in videos if item.get("status") in resumable_done)
    print(f"Batch videos: {total}. Already completed: {completed}.")

    pipeline_config = build_batch_pipeline_config(
        config,
        args,
        transcription_provider,
        translation_provider,
        openai_profile,
        openai_api_env_key,
        source_language,
        source_dialect,
    )

    for index, item in enumerate(videos, 1):
        if item.get("status") in resumable_done:
            continue
        if stop_path.exists():
            print("Graceful stop requested before starting next video.")
            break

        video_path = Path(item["source"])
        video_output_dir = Path(item["output_dir"])
        video_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nBatch {index}/{total}: {video_path.name}")
        print(f"Output folder: {video_output_dir}")

        item["status"] = "running"
        item["started_at"] = utc_now()
        item.pop("error", None)
        state["updated_at"] = utc_now()
        save_batch_state(output_root, state)

        try:
            run_pipeline(
                str(video_path),
                target_language=target_language,
                style_config=style_config,
                pipeline_config=pipeline_config,
                keep_files=getattr(args, "keep_files", False),
                force=getattr(args, "force", False),
                no_burn=True,
                output_dir=str(video_output_dir),
            )
        except Exception as exc:
            item["status"] = "failed"
            item["error"] = str(exc)
            item["finished_at"] = utc_now()
            state["updated_at"] = utc_now()
            save_batch_state(output_root, state)
            print(f"Batch item failed: {exc}")
            if not getattr(args, "continue_on_error", False):
                break
            continue

        review_paths = sorted(video_output_dir.glob("*.review.json"))
        if not review_paths:
            raise RuntimeError("Batch preparation completed without a durable review manifest.")
        prepared_review = load_review(review_paths[-1])
        item["review_path"] = str(review_paths[-1])
        item["review_id"] = prepared_review.get("review_id")
        item["review_state"] = prepared_review.get("state")
        item["status"] = (
            "needs_attention" if prepared_review.get("state") == "NEEDS_ATTENTION" else "ready_for_review"
        )
        item["finished_at"] = utc_now()
        state["updated_at"] = utc_now()
        save_batch_state(output_root, state)

        if stop_path.exists():
            print("Graceful stop requested. Current video completed; stopping batch now.")
            break

    completed = sum(1 for item in videos if item.get("status") in resumable_done)
    failed = sum(1 for item in videos if item.get("status") == "failed")
    pending = total - completed - failed
    print(f"\nBatch summary: completed {completed}/{total}, failed {failed}, pending/running {pending}.")
    print(f"State file: {batch_state_path(output_root)}")


def run_batch_stop(args=None):
    output_root = Path(clean_path_value(getattr(args, "output_dir", None) or prompt("Batch output folder"))).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    path = batch_stop_path(output_root)
    path.write_text(f"stop requested at {utc_now()}\n", encoding="utf-8")
    print(f"Stop requested. Batch will finish the current video before stopping: {path}")


def run_batch_status(args=None):
    output_root = Path(clean_path_value(getattr(args, "output_dir", None) or prompt("Batch output folder"))).expanduser().resolve()
    state = load_batch_state(output_root)
    if not state:
        print(f"No batch state found in {output_root}")
        return
    videos = state.get("videos", [])
    total = len(videos)
    review_ready = {"ready_for_review", "needs_attention", "approved", "completed"}
    completed = sum(1 for item in videos if item.get("status") in review_ready)
    failed = sum(1 for item in videos if item.get("status") == "failed")
    running = sum(1 for item in videos if item.get("status") == "running")
    pending = total - completed - failed - running
    print(f"Batch status: completed {completed}/{total}, running {running}, failed {failed}, pending {pending}")
    if batch_stop_path(output_root).exists():
        print("Stop requested: yes")
    for index, item in enumerate(videos, 1):
        status = item.get("status", "pending")
        print(f"{index}. {status}: {Path(item.get('source', '')).name}")


def resolve_review_for_cli(args):
    if getattr(args, "manifest", None):
        return load_review(args.manifest), Path(args.manifest).resolve()
    review = get_review_manifest(args.video_id, getattr(args, "target_language", None))
    if not review:
        raise RuntimeError("Review not found. Supply --manifest or a known --video-id.")
    source_path = ((review.get("source_location") or {}).get("path"))
    parent = Path(source_path).resolve().parent if source_path else Path.cwd()
    return review, parent / f"{review['video_id']}.review.json"


def run_review_list(args=None):
    reviews = list_review_manifests(states=getattr(args, "state", None))
    for review in reviews:
        blocking = sum(
            1 for issue in review.get("issues") or []
            if issue.get("blocking") and issue.get("status") == "unresolved"
        )
        print(
            f"{review.get('video_id')}\t{review.get('state')}\t"
            f"{review.get('target_language') or 'source'}\tblocking={blocking}\t"
            f"{review.get('source_location')}"
        )


def run_review_inspect(args):
    review, _ = resolve_review_for_cli(args)
    print(json.dumps(review, ensure_ascii=False, indent=2))


def run_review_approve(args):
    review, manifest_path = resolve_review_for_cli(args)
    approve_review(
        review,
        actor=getattr(args, "actor", None) or "cli_user",
        accept_warnings=bool(getattr(args, "accept_warnings", False)),
    )
    save_review(manifest_path, review)
    save_review_manifest(review)
    print(f"Approved draft hash: {review['approval']['approved_draft_hash']}")
    print(f"Review manifest: {manifest_path}")


def run_review_retranslate(args):
    review, manifest_path = resolve_review_for_cli(args)
    result = retranslate_review_selected_cues(
        review,
        args.cue_id,
        load_config(),
        device=getattr(args, "device", None) or "cpu",
        actor=getattr(args, "actor", None) or "cli_user",
    )
    save_review(manifest_path, review)
    save_review_manifest(review)
    print(
        f"Retranslated {result['translated_cue_count']} selected cue(s): "
        f"{', '.join(result['selected_source_cue_ids'])}"
    )
    print(f"Review manifest: {manifest_path}")


def run_review_burn(args):
    review, manifest_path = resolve_review_for_cli(args)
    source_value = args.video or ((review.get("source_location") or {}).get("path") or "")
    source_path = Path(source_value).resolve()
    if not source_path.is_file():
        raise RuntimeError("The prepared source video is unavailable; use --video with the exact original media.")
    gate = assert_burn_allowed(review, source_path)
    output_root = Path(args.output_dir).resolve() if args.output_dir else manifest_path.parent
    output_root.mkdir(parents=True, exist_ok=True)
    suffix = review.get("target_language") or review.get("source_language") or "source"
    approved_srt = output_root / f"{source_path.stem}.approved.{suffix}.srt"
    approved_srt.write_bytes(gate["subtitle_bytes"])
    save_review(manifest_path, review)
    output = run_pipeline(
        str(source_path),
        srt_path_arg=str(approved_srt),
        target_language=review.get("target_language") or review.get("source_language"),
        pipeline_config=load_config(),
        force=bool(args.force),
        output_dir=str(output_root),
        approved_review_path=str(manifest_path),
    )
    print(f"Burned approved draft: {output}")


def build_parser():
    parser = argparse.ArgumentParser(prog="SubGen", description="Interactive subtitle transcription and translation CLI.")
    parser.add_argument("--version", action="version", version=f"SubGen {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("setup", help="Run first-time setup again.")
    subparsers.add_parser("config", help="Edit styles, providers, or default providers.")
    subparsers.add_parser("languages", help="List supported target languages.")
    subparsers.add_parser("providers", help="List providers, capabilities, and configuration status.")
    costs_parser = subparsers.add_parser("costs", help="Show local per-video API usage and cost reports.")
    costs_parser.add_argument("--output-dir", help="Folder containing SubGen manifest files. Default: last output folder.")
    costs_parser.add_argument("--recursive", action="store_true", help="Scan output folder recursively.")

    balance_parser = subparsers.add_parser("balance", help="Show provider billing links and optional local SubGen totals.")
    balance_parser.add_argument("--output-dir", help="Also total local SubGen manifest costs from this folder.")
    balance_parser.add_argument("--recursive", action="store_true", help="Scan output folder recursively.")

    review_parser = subparsers.add_parser("review", help="List, inspect, approve, or burn durable review drafts.")
    review_actions = review_parser.add_subparsers(dest="review_action", required=True)
    review_list_parser = review_actions.add_parser("list", help="List prepared reviews and blocking issue counts.")
    review_list_parser.add_argument("--state", action="append", help="Filter by persistent review state.")
    for action in ("inspect", "approve", "retranslate", "burn"):
        action_parser = review_actions.add_parser(action, help=f"{action.title()} one prepared review.")
        selector = action_parser.add_mutually_exclusive_group(required=True)
        selector.add_argument("--manifest", help="Path to the durable .review.json file.")
        selector.add_argument("--video-id", help="Prepared video ID in the local review database.")
        action_parser.add_argument("--target-language", help="Target language when selecting by video ID.")
        if action == "approve":
            action_parser.add_argument("--accept-warnings", action="store_true", help="Explicitly accept unresolved warnings.")
            action_parser.add_argument("--actor", help="Approval identity recorded in history.")
        if action == "retranslate":
            action_parser.add_argument(
                "--cue-id",
                action="append",
                required=True,
                help="Source cue ID to retranslate; repeat for multiple selected cues.",
            )
            action_parser.add_argument("--actor", help="Identity recorded in translation history.")
            action_parser.add_argument("--device", help="Translation device override.")
        if action == "burn":
            action_parser.add_argument("--video", help="Exact original source media when its recorded local path is unavailable.")
            action_parser.add_argument("--output-dir", help="Destination for approved SRT and rendered video.")
            action_parser.add_argument("--force", action="store_true", help="Re-render even if an output exists.")

    batch_parser = subparsers.add_parser("batch", help="Process a folder, zip, list, or single video automatically.")
    batch_parser.add_argument("--input", "-i", help="Video file, folder, zip file, or .txt/.list file of video paths.")
    batch_parser.add_argument("--output-dir", "-o", help="Root output folder. SubGen creates one subfolder per video.")
    batch_parser.add_argument("--target-language", "-t", help="Target language code, or 'none' to keep original language.")
    batch_parser.add_argument("--transcription-provider", help="Provider ID for transcription.")
    batch_parser.add_argument("--translation-provider", help="Provider ID for translation.")
    batch_parser.add_argument("--openai-profile", help="OpenAI API key profile for OpenAI provider usage.")
    batch_parser.add_argument("--transcription-model", help="Override transcription model for this batch.")
    batch_parser.add_argument("--translation-model", help="Override translation model for this batch.")
    batch_parser.add_argument("--source-language", help="Optional source language code for API transcription.")
    batch_parser.add_argument("--source-dialect", help="Optional source dialect hint.")
    batch_parser.add_argument("--model-size", help="Local Whisper model size for local timing anchors.")
    batch_parser.add_argument("--glossary", help="Per-run glossary entries: source=target;source2=target2")
    batch_parser.add_argument("--api-transcript-timing-mode", choices=["precise", "forced", "fuzzy", "local_whisper", "proportional"])
    batch_chunk_group = batch_parser.add_mutually_exclusive_group()
    batch_chunk_group.add_argument("--api-transcription-chunking", dest="api_transcription_chunking", action="store_const", const="auto", default=None, help="Chunk long API transcription audio automatically.")
    batch_chunk_group.add_argument("--no-api-transcription-chunking", dest="api_transcription_chunking", action="store_const", const="off", help="Send API transcription audio as one request when possible.")
    batch_parser.add_argument("--api-transcription-chunk-seconds", type=float, help="Seconds per OpenAI transcription chunk. Default: config value.")
    batch_parser.add_argument("--api-transcription-chunk-overlap-seconds", type=float, help="Overlap seconds between OpenAI transcription chunks. Default: config value.")
    batch_qa_group = batch_parser.add_mutually_exclusive_group()
    batch_qa_group.add_argument("--qa", dest="qa_enabled", action="store_true", default=None, help="Enable LLM source subtitle QA before translation.")
    batch_qa_group.add_argument("--no-qa", dest="qa_enabled", action="store_false", help="Disable LLM source subtitle QA for this batch.")
    batch_parser.add_argument("--qa-model", help="OpenAI model for source subtitle QA. Default: config qa_model.")
    batch_parser.add_argument("--qa-policy", choices=["stop", "warn"], help="Stop on failed QA or only warn. Default: stop.")
    batch_visual_group = batch_parser.add_mutually_exclusive_group()
    batch_visual_group.add_argument("--visual-style", dest="visual_style_enabled", action="store_true", default=None, help="Use vision LLM frame review to adapt subtitle styling.")
    batch_visual_group.add_argument("--no-visual-style", dest="visual_style_enabled", action="store_false", help="Disable vision LLM subtitle style adaptation.")
    batch_parser.add_argument("--subtitle-mode", choices=["auto", "normal", "tiktok"], help="Subtitle format. Auto resolves each video independently.")
    batch_tiktok_group = batch_parser.add_mutually_exclusive_group()
    batch_tiktok_group.add_argument("--tiktok-style", dest="tiktok_style", action="store_true", default=None, help="Format subtitles in rapid-fire TikTok style (short segments, 1 line).")
    batch_tiktok_group.add_argument("--no-tiktok-style", dest="tiktok_style", action="store_false", help="Disable rapid-fire TikTok style.")
    batch_parser.add_argument("--visual-style-model", help="OpenAI vision-capable model for subtitle style adaptation.")
    batch_parser.add_argument("--restart", action="store_true", help="Create a new batch state instead of resuming.")
    batch_parser.add_argument("--continue-on-error", action="store_true", help="Continue with the next video after a failure.")
    batch_parser.add_argument("--no-burn", action="store_true", help="Generate SRT only; do not burn subtitles into video.")
    batch_parser.add_argument("--keep-files", action="store_true", help="Keep intermediate audio files.")
    batch_parser.add_argument("--force", action="store_true", help="Regenerate outputs even if reusable files exist.")
    batch_parser.add_argument("--no-prompts", action="store_true", help="Avoid optional prompts when flags are supplied.")

    batch_stop_parser = subparsers.add_parser("batch-stop", help="Ask a running batch to stop after the current video.")
    batch_stop_parser.add_argument("--output-dir", "-o", help="Batch root output folder.")

    batch_status_parser = subparsers.add_parser("batch-status", help="Show progress for a batch output folder.")
    batch_status_parser.add_argument("--output-dir", "-o", help="Batch root output folder.")

    run_parser = subparsers.add_parser("run", help="Run the interactive subtitle generator.")
    run_parser.add_argument("--setup", action="store_true", help="Run setup before starting.")
    run_parser.add_argument("--video", help="Video path. If omitted, SubGen asks interactively.")
    run_parser.add_argument("--output-dir", help="Output folder. If omitted, SubGen asks interactively.")
    run_parser.add_argument("--target-language", "-t", help="Target language code, or 'none' to keep original language.")
    run_parser.add_argument("--transcription-provider", help="Provider ID for transcription, for example local, openai, or google.")
    run_parser.add_argument("--translation-provider", help="Provider ID for translation, for example local, openai, google, deepseek, or anthropic.")
    run_parser.add_argument("--openai-profile", help="OpenAI API key profile to use when transcription or translation provider is openai.")
    run_parser.add_argument("--transcription-model", help="Override the selected transcription provider's model for this run.")
    run_parser.add_argument("--translation-model", help="Override the selected translation provider's model for this run.")
    run_parser.add_argument("--source-language", help="Optional source language code for API transcription.")
    run_parser.add_argument("--source-dialect", help="Optional source dialect hint.")
    run_parser.add_argument("--model-size", help="Local Whisper model size for local transcription/timing anchors. Example: small, medium, large-v3.")
    run_parser.add_argument("--glossary", help="Per-run glossary entries: source=target;source2=target2")
    run_parser.add_argument(
        "--api-transcript-timing-mode",
        choices=["precise", "forced", "fuzzy", "local_whisper", "proportional"],
        help="How API transcript text is matched to subtitle timings. Default: precise.",
    )
    run_chunk_group = run_parser.add_mutually_exclusive_group()
    run_chunk_group.add_argument("--api-transcription-chunking", dest="api_transcription_chunking", action="store_const", const="auto", default=None, help="Chunk long API transcription audio automatically.")
    run_chunk_group.add_argument("--no-api-transcription-chunking", dest="api_transcription_chunking", action="store_const", const="off", help="Send API transcription audio as one request when possible.")
    run_parser.add_argument("--api-transcription-chunk-seconds", type=float, help="Seconds per OpenAI transcription chunk. Default: config value.")
    run_parser.add_argument("--api-transcription-chunk-overlap-seconds", type=float, help="Overlap seconds between OpenAI transcription chunks. Default: config value.")
    run_qa_group = run_parser.add_mutually_exclusive_group()
    run_qa_group.add_argument("--qa", dest="qa_enabled", action="store_true", default=None, help="Enable LLM source subtitle QA before translation.")
    run_qa_group.add_argument("--no-qa", dest="qa_enabled", action="store_false", help="Disable LLM source subtitle QA for this run.")
    run_parser.add_argument("--qa-model", help="OpenAI model for source subtitle QA. Default: config qa_model.")
    run_parser.add_argument("--qa-policy", choices=["stop", "warn"], help="Stop on failed QA or only warn. Default: stop.")
    run_visual_group = run_parser.add_mutually_exclusive_group()
    run_visual_group.add_argument("--visual-style", dest="visual_style_enabled", action="store_true", default=None, help="Use vision LLM frame review to adapt subtitle styling.")
    run_visual_group.add_argument("--no-visual-style", dest="visual_style_enabled", action="store_false", help="Disable vision LLM subtitle style adaptation.")
    run_parser.add_argument("--subtitle-mode", choices=["auto", "normal", "tiktok"], help="Subtitle format. Auto resolves this video from its orientation.")
    run_tiktok_group = run_parser.add_mutually_exclusive_group()
    run_tiktok_group.add_argument("--tiktok-style", dest="tiktok_style", action="store_true", default=None, help="Format subtitles in rapid-fire TikTok style (short segments, 1 line).")
    run_tiktok_group.add_argument("--no-tiktok-style", dest="tiktok_style", action="store_false", help="Disable rapid-fire TikTok style.")
    run_parser.add_argument("--visual-style-model", help="OpenAI vision-capable model for subtitle style adaptation.")
    run_parser.add_argument("--force", action="store_true", help="Regenerate outputs for this run.")
    run_parser.add_argument("--no-burn", action="store_true", help="Generate SRT only; do not burn subtitles into video.")
    run_parser.add_argument("--no-prompts", action="store_true", help="Avoid optional prompts when flags are supplied.")

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "setup":
        run_setup(args)
    elif args.command == "config":
        if not load_config().get("setup_complete"):
            run_setup(args)
        run_config(args)
    elif args.command == "languages":
        print_languages(args)
    elif args.command == "providers":
        print_providers(args)
    elif args.command == "costs":
        run_costs(args)
    elif args.command == "balance":
        run_balance(args)
    elif args.command == "review":
        if args.review_action == "list":
            run_review_list(args)
        elif args.review_action == "inspect":
            run_review_inspect(args)
        elif args.review_action == "approve":
            run_review_approve(args)
        elif args.review_action == "retranslate":
            run_review_retranslate(args)
        elif args.review_action == "burn":
            run_review_burn(args)
    elif args.command == "batch":
        run_batch(args)
    elif args.command == "batch-stop":
        run_batch_stop(args)
    elif args.command == "batch-status":
        run_batch_status(args)
    else:
        if getattr(args, "setup", False):
            run_setup(args)
        interactive_run(args)


if __name__ == "__main__":
    main()
