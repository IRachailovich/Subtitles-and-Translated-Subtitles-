import os
import sys
import json
import urllib.parse
import threading
import http.server
import socketserver
import uuid
import shutil
import argparse
import io
import time
from copy import deepcopy
from pathlib import Path


def _configure_utf8_stream(stream):
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except (OSError, ValueError, io.UnsupportedOperation):
            pass


def _write_stream_safely(stream, message):
    """Write diagnostic output without allowing a legacy console encoding to abort a job."""
    try:
        stream.write(message)
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "ascii"
        escaped = message.encode(encoding, errors="backslashreplace").decode(encoding)
        try:
            stream.write(escaped)
        except (OSError, UnicodeError, ValueError):
            return
    except (OSError, ValueError):
        return
    try:
        stream.flush()
    except (OSError, ValueError):
        pass


_configure_utf8_stream(sys.stdout)
_configure_utf8_stream(sys.stderr)

# Add parent directory to path so we can import subgen modules
PARENT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PARENT_DIR))

from subgen_paths import CONFIG_PATH, ENV_PATH, MOBILE_TOKEN_PATH, UPLOADS_DIR
from subgen_version import __version__
from subgen_drive import (
    DriveAuthorization,
    DriveBatchStore,
    GoogleDriveClient,
    cleanup_work_directory,
    configure_drive_client,
    default_output_folder_name,
    disconnect_drive,
    drive_auth_status,
    drive_folder_url,
    extract_drive_folder_id,
    safe_file_name as safe_drive_file_name,
    safe_folder_name as safe_drive_name,
)
from subgen_editor import build_waveform_peaks, validate_editor_segments
from subgen_planner import build_pipeline_plan, public_model_catalog
from subgen_mobile import (
    ACCESS_COOKIE_NAME,
    build_mobile_urls,
    detect_client_browser,
    detect_client_platform,
    is_loopback_address,
    load_or_create_access_token,
    mobile_access_diagnostics,
    request_token,
    request_windows_mobile_access_repair,
    rotate_access_token,
    token_matches,
)

# Import database module
try:
    import subgen_db
    from subgen_db import (
        init_db,
        create_job,
        update_job_status,
        calculate_video_hash,
        get_subtitle_draft,
        save_subtitle_draft,
        get_review_manifest,
        save_review_manifest,
    )
except ImportError as e:
    print(f"Warning: Could not import subgen_db: {e}")

try:
    import subgen_pipeline
    from subgen_pipeline import (
        CONFIG,
        get_supported_languages,
        main as run_pipeline,
        normalize_subtitle_mode,
        retranslate_review_selected_cues,
        resolve_cache_action,
    )
    import subgen_providers

    PIPELINE_DEFAULT_CONFIG = deepcopy(CONFIG)

    def merged_runtime_config(user_config):
        """Overlay persisted user choices without losing newer source defaults."""
        merged = deepcopy(PIPELINE_DEFAULT_CONFIG)
        user_config = dict(user_config or {})
        if "subtitle_mode" not in user_config:
            user_config["subtitle_mode"] = normalize_subtitle_mode(
                legacy_tiktok_style=user_config.get("tiktok_style", False)
            )
        merged.update(user_config)
        return merged

    # Load CONFIG from subgen_config.json if it exists to get user profiles
    config_path = CONFIG_PATH
    if config_path.exists():
        try:
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                CONFIG.clear()
                CONFIG.update(merged_runtime_config(user_config))
                print(f"[CONFIG] Successfully loaded user configuration from {config_path}")
        except Exception as e:
            print(f"[CONFIG] Error loading user configuration: {e}")

    # Resolve active OpenAI API profile from config
    preferred_profile = CONFIG.get("preferred_openai_profile", "default")
    profiles = CONFIG.get("openai_profiles", {})
    profile = profiles.get(preferred_profile, {})
    CONFIG["openai_api_env_key"] = profile.get("env_key", "OPENAI_API_KEY")
except ImportError as e:
    print(f"Warning: Could not import subgen modules: {e}")
    # Fallback mock functions if imports fail
    run_pipeline = None
    get_supported_languages = lambda: {"en": "English", "es": "Spanish"}
    resolve_cache_action = lambda action=None, force=False: (
        action or ("retime" if force else "reuse_all"),
        {"force_burn": bool(force)},
    )
    normalize_subtitle_mode = lambda value=None, legacy_tiktok_style=False: (
        value if value in {"auto", "normal", "tiktok"}
        else ("tiktok" if legacy_tiktok_style else "normal")
    )
    CONFIG = {}

from subgen_review import (
    approve_review,
    assert_burn_allowed,
    complete_burn,
    confirm_translation_current,
    format_srt_time,
    independent_speech_coverage,
    load_review,
    make_issue as make_review_issue,
    new_review,
    normalize_cues,
    replace_draft,
    review_revision_hash,
    resolve_issue,
    save_review,
    set_ready_for_review,
    sha256_file as sha256_full_file,
)

# Helpers to read and write .env values (with fallback)
try:
    from subgen_cli import read_env_values, update_env_value
except ImportError:
    def read_env_values():
        values = {}
        env_path = ENV_PATH
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    values[k.strip()] = v.strip()
        return values

    def update_env_value(key, value):
        values = read_env_values()
        values[key] = value
        env_path = ENV_PATH
        lines = [
            "# SubGen provider API keys.",
            "# This file is ignored by git. Do not share it.",
        ]
        lines.extend(f"{k}={v}" for k, v in sorted(values.items()))
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.environ[key] = value

def init_api_profiles():
    profiles = CONFIG.setdefault("api_profiles", {})
    preferred = CONFIG.setdefault("preferred_profiles", {})
    registry = subgen_providers.get_provider_registry(CONFIG)
    for provider_id, provider in registry.items():
        if provider_id == "openai" and CONFIG.get("openai_profiles"):
            profiles.setdefault(provider_id, CONFIG["openai_profiles"])
        provider_profiles = profiles.setdefault(provider_id, {})
        provider_profiles.setdefault(
            "default",
            {
                "label": "default",
                "env_key": provider.get("env_key") or f"{provider_id.upper()}_API_KEY",
            },
        )
        preferred.setdefault(
            provider_id,
            CONFIG.get("preferred_openai_profile", "default")
            if provider_id == "openai"
            else "default",
        )

def resolve_and_inject_keys(pipeline_config):
    init_api_profiles()
    pref_profiles = CONFIG.get("preferred_profiles", {})
    api_profs = CONFIG.get("api_profiles", {})
    env_vals = read_env_values()
    
    registry = subgen_providers.get_provider_registry(CONFIG)
    for prov_id, provider in registry.items():
        pref = pref_profiles.get(prov_id, "default")
        prof_info = api_profs.get(prov_id, {}).get(pref, {})
        env_key = prof_info.get("env_key")
        
        if env_key:
            # Retrieve the key value from .env or os.environ
            val = env_vals.get(env_key) or os.environ.get(env_key)
            if val:
                # Set the standard environment variable that the pipeline expects
                standard_key = provider.get("env_key") or f"{prov_id.upper()}_API_KEY"
                os.environ[standard_key] = val
                # Also set in pipeline_config for OpenAI backward compatibility
                if prov_id == "openai":
                    pipeline_config["openai_api_env_key"] = standard_key

# Call on startup
init_api_profiles()

# Global Server State
server_state = {
    "status": "idle", # idle, processing, waiting_for_review, burning, completed, error
    "stage": None,     # transcription, alignment, translation, burning
    "progress": 0,
    "status_label": "Ready",
    "new_logs": [],
    "logs": [],
    "video_path": None,
    "video_hash": None,
    "source_lang": None,
    "target_lang": None,
    "segments": [],
    "review": None,
    "issues": [],
    "approval": None,
    "output_video": None,
    "final_srt": None,
    "cache_action": "reuse_all",
    "error_message": None
}

state_lock = threading.Lock()
session_cleanup_lock = threading.Lock()
pending_session_cleanup = set()

MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
mobile_access = {
    "enabled": False,
    "port": 8080,
    "token": None,
    "diagnostics": None,
    "diagnostics_at": 0,
    "last_device": None,
}
drive_authorization = DriveAuthorization()
drive_batch_store = DriveBatchStore()
drive_batch_lock = threading.RLock()
drive_batch_thread = None
drive_batch_starting = False


def public_drive_batch_state(state):
    if not state:
        return None
    result = dict(state)
    result.pop("configuration", None)
    items = []
    for item in state.get("items", []):
        public_item = dict(item)
        public_item.pop("local_input", None)
        public_item.pop("local_output", None)
        items.append(public_item)
    result["items"] = items
    result["total"] = len(items)
    result["completed"] = sum(
        1 for item in items
        if item.get("status") in {"ready_for_review", "needs_attention", "approved", "completed"}
    )
    result["failed"] = sum(1 for item in items if item.get("status") == "failed")
    return result


def drive_batch_is_running():
    with drive_batch_lock:
        return bool(drive_batch_starting or (drive_batch_thread and drive_batch_thread.is_alive()))


def build_drive_batch_pipeline_config(configuration):
    pipeline_config = CONFIG.copy()
    allowed = (
        "transcription_provider",
        "transcription_model",
        "translation_provider",
        "translation_model",
        "timing_anchor_provider",
        "timing_anchor_model",
        "api_transcript_timing_mode",
        "model_size",
        "source_language",
        "subtitle_mode",
        "tiktok_style",
    )
    for key in allowed:
        if configuration.get(key) is not None:
            pipeline_config[key] = configuration[key]
    pipeline_config["subtitle_mode"] = normalize_subtitle_mode(
        configuration.get("subtitle_mode"),
        legacy_tiktok_style=configuration.get("tiktok_style", False),
    )
    pipeline_config["tiktok_style"] = pipeline_config["subtitle_mode"] == "tiktok"
    resolve_and_inject_keys(pipeline_config)
    return pipeline_config


def drive_batch_artifacts(output_dir):
    output_dir = Path(output_dir)
    return sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file()
        and not path.name.startswith(".")
        and not path.name.endswith((".part", ".tmp"))
    )


def run_drive_batch_thread(batch_id):
    global drive_batch_thread, drive_batch_starting
    state = drive_batch_store.load(batch_id)
    if not state:
        return
    try:
        client = GoogleDriveClient()
        state["status"] = "running"
        state["error"] = None
        drive_batch_store.save(state)

        destination_parent_id = state.get("destination_folder_id") or state["source_folder_id"]
        if not state.get("output_folder_id"):
            root = client.create_folder(
                default_output_folder_name(state["source_folder_name"], state.get("target_language")),
                destination_parent_id,
                app_properties={"subgenOutputRoot": "true", "subgenBatchId": state["id"]},
            )
            state["output_folder_id"] = root["id"]
            state["output_folder_url"] = root.get("webViewLink") or drive_folder_url(root["id"])
            drive_batch_store.save(state)

        pipeline_config = build_drive_batch_pipeline_config(state.get("configuration") or {})
        style_config = (state.get("configuration") or {}).get("style_config") or None
        target_language = state.get("target_language")
        work_root = drive_batch_store.work_dir(batch_id)

        for index, item in enumerate(state.get("items", [])):
            latest = drive_batch_store.load(batch_id) or state
            if latest.get("stop_requested"):
                state = latest
                state["status"] = "stopped"
                state["current_index"] = None
                drive_batch_store.save(state)
                add_log("Google Drive batch stopped before the next video.", "system")
                return
            state = latest
            item = state["items"][index]
            if item.get("status") in {"ready_for_review", "needs_attention", "approved", "burning", "completed"}:
                continue

            state["current_index"] = index
            item.update({"status": "running", "stage": "downloading", "progress": 0, "error": None})
            drive_batch_store.save(state)
            add_log(f"Drive batch {index + 1}/{len(state['items'])}: downloading {item['source_name']}", "system")

            item_work = work_root / f"{index + 1:04d}_{safe_drive_name(item['source_id'])}"
            input_dir = item_work / "input"
            output_dir = item_work / "output"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            input_path = input_dir / safe_drive_file_name(item["source_name"])

            try:
                client.download_file(
                    {
                        "id": item["source_id"],
                        "name": item["source_name"],
                        "capabilities": {"canDownload": item.get("can_download", True)},
                    },
                    input_path,
                    progress=lambda value, s=state, row=item: (
                        row.update({"progress": round(value * 20)}), drive_batch_store.save(s)
                    ),
                )
                item.update({"stage": "processing", "progress": 22})
                drive_batch_store.save(state)
                add_log(f"Drive batch: processing {item['source_name']} through the selected SubGen pipeline.", "pipeline")
                source_location = {
                    "kind": "google_drive",
                    "drive_id": item["source_id"],
                    "name": item["source_name"],
                    "relative_path": item.get("relative_path") or item["source_name"],
                    "size": item.get("size"),
                    "modified_time": item.get("modified_time"),
                    "source_folder_id": state.get("source_folder_id"),
                    "source_folder_name": state.get("source_folder_name"),
                    "source_folder_url": state.get("source_folder_url"),
                }
                run_pipeline(
                    str(input_path),
                    target_language=target_language,
                    style_config=style_config,
                    pipeline_config=pipeline_config,
                    no_burn=True,
                    output_dir=str(output_dir),
                    source_location=source_location,
                )

                artifacts = drive_batch_artifacts(output_dir)
                if not artifacts:
                    raise RuntimeError("The pipeline completed without exportable artifacts.")
                review_paths = [path for path in artifacts if path.name.endswith(".review.json")]
                if not review_paths:
                    raise RuntimeError("Drive preparation completed without durable review metadata.")
                prepared_review = load_review(review_paths[-1])
                prepared_review["source_location"] = source_location
                save_review(review_paths[-1], prepared_review)
                save_review_manifest(prepared_review)
                item.update({
                    "source_hash": prepared_review.get("source_hash"),
                    "review_id": prepared_review.get("review_id"),
                    "review_video_id": prepared_review.get("video_id"),
                    "review_state": prepared_review.get("state"),
                    "review_file_name": review_paths[-1].name,
                })
                item.update({"stage": "uploading", "progress": 82})
                drive_batch_store.save(state)
                if not item.get("output_folder_id"):
                    output_folder = client.ensure_folder(item["output_folder_name"], state["output_folder_id"])
                    item["output_folder_id"] = output_folder["id"]
                    item["output_folder_url"] = output_folder.get("webViewLink") or drive_folder_url(output_folder["id"])
                    drive_batch_store.save(state)
                uploaded = []
                for artifact_index, artifact in enumerate(artifacts, 1):
                    remote = client.upload_file(artifact, item["output_folder_id"])
                    uploaded.append({"id": remote["id"], "name": remote["name"], "url": remote.get("webViewLink")})
                    item["progress"] = 82 + round(18 * artifact_index / len(artifacts))
                    drive_batch_store.save(state)
                item.update({
                    "status": "needs_attention" if prepared_review.get("state") == "NEEDS_ATTENTION" else "ready_for_review",
                    "stage": "review",
                    "progress": 100,
                    "artifacts": uploaded,
                    "finished_at": time.time(),
                })
                drive_batch_store.save(state)
                add_log(f"Drive batch: prepared {item['source_name']} for review without burning.", "success")
            except Exception as exc:
                item.update({
                    "status": "failed",
                    "stage": "failed",
                    "error": str(exc),
                    "finished_at": time.time(),
                })
                drive_batch_store.save(state)
                add_log(f"Drive batch item failed ({item['source_name']}): {exc}", "error")
            finally:
                if item.get("status") == "failed":
                    add_log(
                        "Drive batch preserved failed-item checkpoints for resume: "
                        f"{item_work}",
                        "system",
                    )
                else:
                    cleanup_work_directory(item_work)

        state = drive_batch_store.load(batch_id) or state
        state["current_index"] = None
        failures = [item for item in state.get("items", []) if item.get("status") == "failed"]
        state["status"] = "ready_for_review_with_errors" if failures else "ready_for_review"
        drive_batch_store.save(state)
        add_log(
            f"Google Drive batch finished: {len(state['items']) - len(failures)} completed, {len(failures)} failed.",
            "success" if not failures else "warning",
        )
    except Exception as exc:
        state = drive_batch_store.load(batch_id) or state
        state["status"] = "error"
        state["error"] = str(exc)
        state["current_index"] = None
        drive_batch_store.save(state)
        add_log(f"Google Drive batch error: {exc}", "error")
    finally:
        with drive_batch_lock:
            drive_batch_thread = None
            drive_batch_starting = False


def run_drive_burn_thread(batch_id, item_index):
    state = drive_batch_store.load(batch_id)
    if not state or item_index < 0 or item_index >= len(state.get("items", [])):
        return
    item = state["items"][item_index]
    burn_work = drive_batch_store.work_dir(batch_id) / f"burn_{item_index + 1:04d}_{safe_drive_name(item['source_id'])}"
    try:
        review = get_review_manifest(item.get("review_video_id"), state.get("target_language") or "source")
        if not review:
            raise RuntimeError("Durable prepared review is unavailable for this Drive item.")
        item.update({"status": "burning", "stage": "redownloading", "progress": 0, "error": None})
        drive_batch_store.save(state)
        input_dir = burn_work / "input"
        output_dir = burn_work / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        source_path = input_dir / safe_drive_file_name(item["source_name"])
        client = GoogleDriveClient()
        client.download_file(
            {"id": item["source_id"], "name": item["source_name"], "capabilities": {"canDownload": item.get("can_download", True)}},
            source_path,
            progress=lambda value: (item.update({"progress": round(value * 20)}), drive_batch_store.save(state)),
        )
        gate = assert_burn_allowed(review, source_path)
        if gate["source_hash"] != item.get("source_hash"):
            raise RuntimeError("Re-downloaded Drive source hash differs from the prepared source hash.")
        approved_srt = output_dir / f"{source_path.stem}.approved.{state.get('target_language') or 'source'}.srt"
        approved_srt.write_bytes(gate["subtitle_bytes"])
        review_path = output_dir / f"{source_path.stem}.review.json"
        save_review(review_path, review)
        item.update({"stage": "burning", "progress": 25})
        drive_batch_store.save(state)
        output_video = Path(run_pipeline(
            str(source_path),
            srt_path_arg=str(approved_srt),
            target_language=state.get("target_language"),
            style_config=(state.get("configuration") or {}).get("style_config") or None,
            pipeline_config=build_drive_batch_pipeline_config(state.get("configuration") or {}),
            force=True,
            no_burn=False,
            output_dir=str(output_dir),
            approved_review_path=str(review_path),
        ))
        if not output_video.is_file():
            raise RuntimeError("Drive burn completed without an output video.")
        item.update({"stage": "uploading", "progress": 85})
        drive_batch_store.save(state)
        uploads = []
        for artifact in (approved_srt, review_path, output_video):
            remote = client.upload_file(artifact, item["output_folder_id"])
            uploads.append({
                "id": remote["id"], "name": remote["name"], "url": remote.get("webViewLink"),
                "sha256": sha256_full_file(artifact),
            })
        completed_review = load_review(review_path)
        item.update({
            "status": "completed",
            "stage": "completed",
            "progress": 100,
            "review_state": completed_review.get("state"),
            "final_artifacts": uploads,
            "finished_at": time.time(),
        })
        drive_batch_store.save(state)
        add_log(f"Drive batch: burned exact approved draft for {item['source_name']}.", "success")
    except Exception as exc:
        state = drive_batch_store.load(batch_id) or state
        item = state["items"][item_index]
        item.update({"status": "burn_failed", "stage": "burn_failed", "error": str(exc), "finished_at": time.time()})
        drive_batch_store.save(state)
        add_log(f"Drive approved burn failed ({item['source_name']}): {exc}", "error")
    finally:
        cleanup_work_directory(burn_work)


def sandbox_job_dir(video_path, job_id=None):
    """Return the owned upload job directory, never an arbitrary filesystem path."""
    if not video_path:
        return None
    try:
        candidate = Path(video_path).resolve()
        jobs_root = (UPLOADS_DIR / "jobs").resolve()
        relative = candidate.relative_to(jobs_root)
    except (OSError, TypeError, ValueError):
        return None
    if len(relative.parts) < 2:
        return None
    owned_job_id = relative.parts[0]
    if job_id and str(job_id) != owned_job_id:
        return None
    return jobs_root / owned_job_id


def delete_sandbox_video_session(video_path, job_id=None):
    job_dir = sandbox_job_dir(video_path, job_id=job_id)
    if job_dir is None:
        return {"status": "not_sandboxed", "deleted": False}
    shutil.rmtree(job_dir, ignore_errors=True)
    deleted = not job_dir.exists()
    if deleted:
        add_log(f"Deleted completed video session storage: {job_dir.name}", "system")
    return {"status": "deleted" if deleted else "delete_failed", "deleted": deleted}


def release_video_session(video_path, job_id=None):
    """Delete an ended upload session now, or immediately after active work finishes."""
    if not job_id:
        return {"status": "not_sandboxed", "deleted": False}
    job_dir = sandbox_job_dir(video_path, job_id=job_id)
    if job_dir is None:
        return {"status": "not_sandboxed", "deleted": False}
    owned_job_id = job_dir.name
    with state_lock:
        active = (
            server_state.get("job_id") == owned_job_id
            and server_state.get("status") in {"processing", "burning"}
        )
    if active:
        with session_cleanup_lock:
            pending_session_cleanup.add(owned_job_id)
        return {"status": "deferred", "deleted": False}
    return delete_sandbox_video_session(video_path, job_id=owned_job_id)


def finish_deferred_video_session_release(job_id, video_path):
    with session_cleanup_lock:
        if job_id not in pending_session_cleanup:
            return
        pending_session_cleanup.discard(job_id)
    delete_sandbox_video_session(video_path, job_id=job_id)


def cleanup_all_video_sessions():
    """Remove volatile uploads without touching API keys, SQLite, or model caches."""
    jobs_dir = UPLOADS_DIR / "jobs"
    shutil.rmtree(jobs_dir, ignore_errors=True)
    with session_cleanup_lock:
        pending_session_cleanup.clear()

def add_log(text, log_type="info", to_terminal=True):
    with state_lock:
        log_entry = {"text": text, "type": log_type}
        server_state["new_logs"].append(log_entry)
        server_state["logs"].append(log_entry)
    
    if to_terminal:
        # Write directly to the original stdout to avoid infinite recursion
        if '_original_stdout' in globals():
            _write_stream_safely(_original_stdout, f"[{log_type.upper()}] {text}\n")
        else:
            _write_stream_safely(sys.stdout, f"[{log_type.upper()}] {text}\n")

# Intercept sys.stdout to capture logs
class StdoutRedirector:
    def __init__(self, original_stdout, log_type="info"):
        self.original_stdout = original_stdout
        self.log_type = log_type

    def write(self, message):
        _write_stream_safely(self.original_stdout, message)
        stripped = message.strip()
        if stripped:
            # Parse progress and stage from pipeline prints
            parse_pipeline_log(stripped)
            add_log(stripped, self.log_type, to_terminal=False)

    def flush(self):
        try:
            self.original_stdout.flush()
        except (OSError, ValueError):
            pass

def parse_pipeline_log(message):
    msg_lower = message.lower()
    with state_lock:
        if "step 1:" in msg_lower or "extracting audio" in msg_lower or "preparing compact audio" in msg_lower:
            server_state["stage"] = "transcription"
            server_state["progress"] = 10
            server_state["status_label"] = "Extracting audio track..."
        elif "step 2:" in msg_lower or "transcribing" in msg_lower:
            server_state["stage"] = "transcription"
            server_state["progress"] = 25
            server_state["status_label"] = "Transcribing audio..."
        elif "step 2b:" in msg_lower:
            server_state["stage"] = "transcription"
            server_state["progress"] = 30
            server_state["status_label"] = "Running API transcription..."
        elif "step 2c:" in msg_lower or "reviewing source" in msg_lower:
            server_state["stage"] = "transcription"
            server_state["progress"] = 40
            server_state["status_label"] = "Running subtitle QA..."
        elif "aligning" in msg_lower or "forced_align" in msg_lower:
            server_state["stage"] = "alignment"
            server_state["progress"] = 50
            server_state["status_label"] = "Aligning word timings..."
        elif "step 3:" in msg_lower or "translating" in msg_lower:
            server_state["stage"] = "translation"
            server_state["progress"] = 70
            server_state["status_label"] = "Translating segments..."
        elif "step 4:" in msg_lower or "writing translated" in msg_lower:
            server_state["stage"] = "translation"
            server_state["progress"] = 85
            server_state["status_label"] = "Saving translated subtitles..."
        elif "step 5:" in msg_lower or "burning" in msg_lower:
            server_state["stage"] = "burning"
            server_state["progress"] = 92
            server_state["status_label"] = "Burning subtitles into video..."

# Redirect stdout globally
_original_stdout = sys.stdout
_original_stderr = sys.stderr
sys.stdout = StdoutRedirector(sys.stdout, "info")
sys.stderr = StdoutRedirector(sys.stderr, "error")

def parse_srt_file(filepath):
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if not content:
            return []
        
        segments = []
        blocks = content.replace('\r\n', '\n').split('\n\n')
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            try:
                lines = block.split('\n')
                if len(lines) >= 2:
                    index = int(lines[0].strip())
                    times = lines[1].split('-->')
                    if len(times) != 2:
                        continue
                    start = times[0].strip()
                    end = times[1].strip()
                    text = '\n'.join(lines[2:]) if len(lines) > 2 else ""
                    segments.append({
                        "index": index,
                        "start": start,
                        "end": end,
                        "text": text,
                        "translation": text
                    })
            except Exception as block_err:
                if '_original_stdout' in globals():
                    _write_stream_safely(
                        _original_stdout,
                        f"[WARNING] Skipping malformed SRT block: {block[:50]}... Error: {block_err}\n",
                    )
        return segments
    except Exception as e:
        add_log(f"Error parsing SRT file {filepath}: {e}", "error")
        return []

def write_srt_file(segments, filepath):
    try:
        content = "".join(
            f"{seg['index']}\n{seg['start']} --> {seg['end']}\n{seg['translation']}\n\n"
            for seg in segments
        )
        path = Path(filepath)
        if path.exists() and path.read_text(encoding='utf-8') == content:
            add_log(f"Subtitles unchanged; keeping existing SRT file: {filepath}", "info")
            return True
        path.write_text(content, encoding='utf-8')
        add_log(f"Successfully saved edited subtitles to {filepath}", "success")
        return True
    except Exception as e:
        add_log(f"Error writing SRT file {filepath}: {e}", "error")
        return False


def review_to_editor_segments(review):
    source = normalize_cues((review or {}).get("source_draft") or [])
    translated = normalize_cues((review or {}).get("translation_draft") or [])
    by_index = {cue.get("index"): cue for cue in translated}
    result = []
    for index, cue in enumerate(source, 1):
        target = by_index.get(cue.get("index"))
        result.append({
            "id": cue.get("id"),
            "index": index,
            "start": format_srt_time(cue.get("start")),
            "end": format_srt_time(cue.get("end")),
            "text": cue.get("text", ""),
            "translation": (target or cue).get("text", ""),
            "provenance": {
                key: cue.get(key)
                for key in (
                    "canonical_source", "previous_source", "prompt_version", "generation_scope",
                    "source_offset_seconds", "independently_verified", "manually_edited",
                )
                if cue.get(key) is not None
            },
        })
    return result


def update_review_from_editor(review, segments, *, actor="human", translation_confirmed=False):
    normalized = normalize_cues(segments)
    source = []
    translated = []
    has_translation = bool(review.get("target_language") and review.get("target_language") != review.get("source_language"))
    for index, item in enumerate(normalized, 1):
        source.append({
            "id": str(item.get("id") or f"source-{index}"),
            "index": index,
            "start": item["start"],
            "end": item["end"],
            "text": item.get("text", ""),
        })
        if has_translation:
            translated.append({
                "id": f"translation-{item.get('id') or index}",
                "index": index,
                "start": item["start"],
                "end": item["end"],
                "text": item.get("translation", ""),
            })

    def comparable(cues):
        return [
            (round(float(cue.get("start", 0)), 6), round(float(cue.get("end", 0)), 6), str(cue.get("text") or ""))
            for cue in normalize_cues(cues)
        ]

    source_changed = comparable(source) != comparable(review.get("source_draft") or [])
    timing_changed = [item[:2] for item in comparable(source)] != [
        item[:2] for item in comparable(review.get("source_draft") or [])
    ]
    translation_changed = has_translation and comparable(translated) != comparable(review.get("translation_draft") or [])
    if source_changed:
        replace_draft(review, "source", source, actor=actor, operation="editor_source_update")
    if translation_changed:
        replace_draft(review, "translation", translated, actor=actor, operation="editor_translation_update")
    if has_translation and translation_confirmed:
        confirm_translation_current(review, actor=actor)
    if timing_changed and (review.get("coverage_report") or {}).get("speech_intervals"):
        previous = review["coverage_report"]
        review["issues"] = [
            issue for issue in review.get("issues") or []
            if issue.get("code") not in {"uncovered_speech", "low_speech_coverage"}
        ]
        tolerances = previous.get("tolerances") or {}
        refreshed = independent_speech_coverage(
            review.get("video_id"),
            previous.get("speech_intervals"),
            [(cue["start"], cue["end"]) for cue in source],
            media_duration=previous.get("media_duration_seconds", 0),
            **{
                key: tolerances[key]
                for key in (
                    "vad_boundary_uncertainty", "alignment_padding", "ignore_shorter_than",
                    "critical_gap_seconds", "warning_gap_seconds", "min_coverage_ratio",
                )
                if key in tolerances
            },
        )
        review["coverage_report"] = refreshed
        review["issues"].extend(refreshed.get("issues") or [])
    if source_changed or translation_changed:
        review["state"] = "IN_REVIEW"
    return review


def persist_active_review(review, video_path=None):
    save_review_manifest(review)
    candidate = None
    with state_lock:
        final_srt = server_state.get("final_srt")
    if final_srt:
        srt = Path(final_srt).resolve()
        candidate = srt.parent / f"{Path(video_path).stem if video_path else srt.stem}.review.json"
        save_review(candidate, review)
    with state_lock:
        server_state["review"] = review
        server_state["issues"] = review.get("issues") or []
        server_state["approval"] = review.get("approval")
        server_state["segments"] = review_to_editor_segments(review)
    return candidate


# Background Worker Threads
def literal_video_artifacts(directory, base):
    """Return artifacts whose names literally belong to a video stem."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    artifacts = []
    for candidate in directory.iterdir():
        if not candidate.is_file() or not candidate.name.startswith(base):
            continue
        remainder = candidate.name[len(base):]
        if remainder.startswith((".", "_")):
            artifacts.append(candidate)
    return artifacts


def run_pipeline_thread(
    job_id,
    video_path,
    source_lang,
    target_lang,
    trans_provider,
    transcription_model,
    model_size,
    trans_provider_api,
    translation_model,
    timing_anchor_provider,
    timing_mode,
    subtitle_mode,
    cache_action="reuse_all",
):
    global server_state
    try:
        with state_lock:
            server_state["job_id"] = job_id
            server_state["status"] = "processing"
            server_state["stage"] = "transcription"
            server_state["progress"] = 5
            server_state["status_label"] = "Starting pipeline..."
            server_state["video_path"] = video_path
            server_state["source_lang"] = source_lang
            server_state["target_lang"] = target_lang
            server_state["cache_action"] = cache_action

        video_path_obj = Path(video_path).resolve()
        base = video_path_obj.stem
        output_dir = video_path_obj.parent

        # Setup configuration
        pipeline_config = CONFIG.copy()
        pipeline_config["transcription_provider"] = trans_provider
        if transcription_model:
            pipeline_config["transcription_model"] = transcription_model
        pipeline_config["source_language"] = source_lang
        if trans_provider == "local":
            pipeline_config["model_size"] = model_size
        pipeline_config["translation_provider"] = trans_provider_api
        if translation_model:
            pipeline_config["translation_model"] = translation_model
        pipeline_config["timing_anchor_provider"] = timing_anchor_provider
        pipeline_config["api_transcript_timing_mode"] = timing_mode
        subtitle_mode = normalize_subtitle_mode(
            subtitle_mode,
            legacy_tiktok_style=subtitle_mode is True,
        )
        pipeline_config["subtitle_mode"] = subtitle_mode
        pipeline_config["tiktok_style"] = subtitle_mode == "tiktok"

        # Inject active API keys
        resolve_and_inject_keys(pipeline_config)

        # Run pipeline with no_burn=True so we can edit before burning
        add_log(f"Running SubGen pipeline on {video_path_obj.name} (No-Burn Mode)...", "system")
        
        pipeline_output = None
        if run_pipeline:
            source_kind = (
                "local_upload"
                if UPLOADS_DIR.resolve() in video_path_obj.parents
                else "local"
            )
            pipeline_output = run_pipeline(
                str(video_path_obj),
                target_language=target_lang,
                pipeline_config=pipeline_config,
                no_burn=True,
                cache_action=cache_action,
                source_location={
                    "kind": source_kind,
                    "path": str(video_path_obj),
                    "name": video_path_obj.name,
                    "job_id": job_id,
                },
            )
        else:
            raise RuntimeError(
                "The shared SubGen pipeline could not be imported. Refusing to create simulated review output."
            )

        # Locate the generated SRT files
        # Source SRT
        source_srt_path = None
        target_srt_path = None

        if pipeline_output:
            returned_path = Path(pipeline_output).resolve()
            if returned_path.is_file() and returned_path.suffix.lower() == ".srt":
                if target_lang:
                    target_srt_path = returned_path
                else:
                    source_srt_path = returned_path
        
        # Look for SRT files in output directory
        # The pipeline saves files as {base}.{lang}.srt or {base}.srt
        for f in literal_video_artifacts(output_dir, base):
            if f.suffix.lower() != ".srt":
                continue
            if target_lang and f.name.endswith(f".{target_lang}.srt") and target_srt_path is None:
                target_srt_path = f
            elif (
                target_lang
                and source_srt_path is None
                and not f.name.endswith(f".{target_lang}.srt")
            ):
                source_srt_path = f
            elif not target_lang and source_srt_path is None:
                source_srt_path = f

        if not target_srt_path and source_srt_path:
            target_srt_path = source_srt_path
        if not source_srt_path and target_srt_path:
            source_srt_path = target_srt_path

        if not target_srt_path or not target_srt_path.is_file():
            raise RuntimeError(
                "The subtitle pipeline completed without producing a readable final SRT file."
            )

        # Parse segments
        source_segs = parse_srt_file(source_srt_path) if source_srt_path else []
        target_segs = parse_srt_file(target_srt_path) if target_srt_path else []
        
        merged_segments = []
        for s in source_segs:
            translation = s["text"]
            for t in target_segs:
                if t["index"] == s["index"]:
                    translation = t["text"]
                    break
            merged_segments.append({
                "index": s["index"],
                "start": s["start"],
                "end": s["end"],
                "text": s["text"],
                "translation": translation
            })

        if not merged_segments and target_segs:
            merged_segments = target_segs

        with state_lock:
            active_video_hash = server_state.get("video_hash")
        review = get_review_manifest(active_video_hash, target_lang or "source") if active_video_hash else None
        if not review:
            full_source_hash = sha256_full_file(video_path_obj)
            source_draft = [
                {
                    "id": f"source-{segment['index']}",
                    "index": segment["index"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment.get("text", ""),
                    "canonical_source": "gemini" if trans_provider == "google" else trans_provider,
                    "manually_edited": False,
                }
                for segment in merged_segments
            ]
            translation_draft = []
            if target_lang and target_lang != source_lang:
                translation_draft = [
                    {
                        "id": f"translation-{segment['index']}",
                        "index": segment["index"],
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment.get("translation", ""),
                        "canonical_source": "automatic_translation",
                        "manually_edited": False,
                    }
                    for segment in merged_segments
                ]
            review = new_review(
                active_video_hash or full_source_hash,
                full_source_hash,
                source_language=source_lang,
                target_language=target_lang,
                provider=trans_provider,
                model=transcription_model,
                prompt_version=pipeline_config.get("google_transcription_prompt_version"),
                source_draft=source_draft,
                translation_draft=translation_draft,
                source_location={"kind": "local", "path": str(video_path_obj)},
            )
            set_ready_for_review(review)
        merged_segments = review_to_editor_segments(review)

        with state_lock:
            server_state["segments"] = merged_segments
            server_state["review"] = review
            server_state["issues"] = review.get("issues") or []
            server_state["approval"] = review.get("approval")
            server_state["final_srt"] = str(target_srt_path)
            server_state["status"] = "waiting_for_review"
            server_state["stage"] = "review"
            server_state["progress"] = 100
            server_state["status_label"] = "Waiting for subtitle review"
            
        update_job_status(job_id, "waiting_for_review", progress=100, stage="review", status_label="Waiting for subtitle review")
        persist_active_review(review, video_path=video_path_obj)
        copy_results_to_output_dir(video_path_obj.name, job_dir=video_path_obj.parent, include_video=False)
        add_log("Pipeline completed. Subtitles loaded into editor.", "success")

    except Exception as e:
        with state_lock:
            server_state["status"] = "error"
            server_state["error_message"] = str(e)
            server_state["status_label"] = "Error occurred"
        update_job_status(job_id, "failed", error_message=str(e), status_label="Error occurred")
        add_log(f"Pipeline Thread Error: {e}", "error")
        import traceback
        traceback.print_exc()
    finally:
        finish_deferred_video_session_release(job_id, video_path)

def run_burning_thread(job_id, video_path, target_lang, review_path, style_config, force_burn=False):
    global server_state
    try:
        with state_lock:
            server_state["status"] = "burning"
            server_state["stage"] = "burning"
            server_state["progress"] = 5
            server_state["status_label"] = "Saving edited subtitles..."

        video_path_obj = Path(video_path).resolve()
        base = video_path_obj.stem
        output_dir = video_path_obj.parent
        
        review = load_review(review_path)
        burn_gate = assert_burn_allowed(review, video_path_obj)

        # Write the exact approved bytes; never reserialize client-supplied text here.
        with state_lock:
            final_srt_value = server_state.get("final_srt")
        if not final_srt_value:
            raise RuntimeError("No reviewed subtitle file is available for burning.")
        srt_path = Path(final_srt_value).resolve()
        if not srt_path.is_file():
            raise RuntimeError(f"Reviewed subtitle file not found: {srt_path}")
        srt_path.write_bytes(burn_gate["subtitle_bytes"])
        if sha256_full_file(srt_path) != burn_gate["approved_draft_hash"]:
            raise RuntimeError(f"Could not reproduce the approved subtitle bytes: {srt_path}")

        with state_lock:
            server_state["progress"] = 20
            server_state["status_label"] = "Burning subtitles with FFmpeg..."

        # Call pipeline with srt_path_arg and no_burn=False
        add_log(f"Burning subtitles into {video_path_obj.name}...", "system")
        
        pipeline_output = None
        if run_pipeline:
            # We pass the style config directly
            pipeline_output = run_pipeline(
                str(video_path_obj),
                srt_path_arg=str(srt_path),
                target_language=target_lang,
                style_config=style_config,
                no_burn=False,
                force=force_burn,
                approved_review_path=str(review_path),
            )
        else:
            # Mock burning
            import time
            time.sleep(3)

        # Locate final video
        # The pipeline saves as {base}_subtitled_{lang}.mp4 or similar
        lang_suffix = f"_{target_lang}" if target_lang else ""
        output_video_path = None
        if pipeline_output:
            returned_path = Path(pipeline_output).resolve()
            if returned_path.is_file() and returned_path.suffix.lower() == ".mp4":
                output_video_path = returned_path
        if output_video_path is None:
            output_video_path = output_dir / f"{base}_subtitled{lang_suffix}.mp4"
        
        if not output_video_path.exists():
            # Fallback check
            candidates = [
                artifact
                for artifact in literal_video_artifacts(output_dir, base)
                if artifact.suffix.lower() == ".mp4"
                and "_subtitled" in artifact.stem[len(base):]
            ]
            if candidates:
                output_video_path = max(candidates, key=lambda path: path.stat().st_mtime)
                
        if not output_video_path.exists():
            raise RuntimeError("Subtitle burning completed without producing an output video.")

        exported_paths = copy_results_to_output_dir(
            video_path_obj.name,
            job_dir=video_path_obj.parent,
            include_video=True,
        )
        final_video_dest = Path(
            exported_paths.get(str(output_video_path.resolve()), str(output_video_path.resolve()))
        )
        final_srt_dest = Path(
            exported_paths.get(str(srt_path.resolve()), str(srt_path.resolve()))
        )

        output_dir_str = CONFIG.get("last_output_dir")
        if output_dir_str:
            expected_video_dest = Path(output_dir_str).resolve() / output_video_path.name
            expected_srt_dest = Path(output_dir_str).resolve() / srt_path.name
            if not expected_video_dest.is_file():
                raise RuntimeError(
                    f"Video was rendered but could not be exported to {expected_video_dest}"
                )
            if not expected_srt_dest.is_file():
                raise RuntimeError(
                    f"Subtitles were saved but could not be exported to {expected_srt_dest}"
                )
            final_video_dest = expected_video_dest
            final_srt_dest = expected_srt_dest

        with state_lock:
            server_state["output_video"] = str(final_video_dest)
            server_state["final_srt"] = str(final_srt_dest)
            server_state["status"] = "completed"
            server_state["progress"] = 100
            server_state["status_label"] = "Completed"
            completed_review = get_review_manifest(server_state.get("video_hash"), target_lang or "source")
            server_state["review"] = completed_review
            server_state["issues"] = (completed_review or {}).get("issues") or []
            server_state["approval"] = (completed_review or {}).get("approval")

        update_job_status(job_id, "completed", progress=100, stage="burning", status_label="Completed")
        add_log(f"Burning complete! Output video: {final_video_dest}", "success")

    except Exception as e:
        with state_lock:
            server_state["status"] = "error"
            server_state["error_message"] = str(e)
            server_state["status_label"] = "Error during burning"
        update_job_status(job_id, "failed", error_message=str(e), status_label="Error during burning")
        add_log(f"Burning Thread Error: {e}", "error")
    finally:
        finish_deferred_video_session_release(job_id, video_path)

def copy_results_to_output_dir(filename, job_dir=None, include_video=True):
    exported_paths = {}
    try:
        base = Path(filename).stem
        output_dir_str = CONFIG.get("last_output_dir")
        if not output_dir_str:
            return exported_paths
            
        output_dir = Path(output_dir_str)
        output_dir.mkdir(parents=True, exist_ok=True)
            
        source_dir = Path(job_dir) if job_dir else UPLOADS_DIR
        artifacts = literal_video_artifacts(source_dir, base)
        
        # Copy manifest/txt/srt
        for f in artifacts:
            if not (
                f.name.endswith(".manifest.json")
                or f.name.endswith("api_transcript.txt")
                or f.suffix.lower() == ".srt"
            ):
                continue
            dest = output_dir / f.name
            if f.resolve() != dest.resolve():
                shutil.copy2(f, dest)
                print(f"[EXPORT] Copied {f.name} to output directory.")
            exported_paths[str(f.resolve())] = str(dest.resolve())
            
        # Copy MP4 files
        if include_video:
            for mp4_file in artifacts:
                if (
                    mp4_file.suffix.lower() != ".mp4"
                    or "_subtitled" not in mp4_file.stem[len(base):]
                ):
                    continue
                dest = output_dir / mp4_file.name
                if mp4_file.resolve() != dest.resolve():
                    shutil.copy2(mp4_file, dest)
                    print(f"[EXPORT] Copied video {mp4_file.name} to output directory.")
                exported_paths[str(mp4_file.resolve())] = str(dest.resolve())
    except Exception as e:
        print(f"[EXPORT] Error exporting results: {e}")
    return exported_paths


# check_and_copy_cache removed as caching is now managed in SQLite database

# Custom HTTP Request Handler
class SubGenRequestHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress request logging to prevent terminal flooding from status polling
        pass

    def translate_path(self, path):
        # Serve files from the 'web' folder for static requests
        root = Path(__file__).resolve().parent
        
        # If the path is an API call, don't modify it
        if path.startswith('/api'):
            return path
            
        # Strip query parameters
        path_clean = urllib.parse.unquote(path.split('?')[0])
        
        # Serve index.html by default
        if path_clean == '/':
            return str(root / 'index.html')
            
        candidate = (root / path_clean.lstrip('/')).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            return str(root / "__not_found__")
        return str(candidate)

    def client_is_local(self):
        return is_loopback_address(self.client_address[0])

    def request_is_authorized(self):
        if self.client_is_local():
            return True
        if not mobile_access["enabled"]:
            return False
        return token_matches(request_token(self.headers), mobile_access["token"])

    def send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def discard_small_request_body(self, max_bytes=64 * 1024):
        """Drain a small rejected body so Windows can deliver the HTTP response."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            return False
        if content_length <= 0 or content_length > int(max_bytes):
            return False
        remaining = content_length
        while remaining:
            chunk = self.rfile.read(min(16 * 1024, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
        return remaining == 0

    def reject_unauthorized(self, api_request=False):
        if api_request:
            self.send_json({"error": "Pair this device from the SubGen desktop app."}, 401)
            return
        body = (
            "<!doctype html><html><meta name='viewport' content='width=device-width'>"
            "<title>SubGen pairing required</title><style>body{font-family:system-ui;background:#070a13;"
            "color:#f8fafc;display:grid;place-items:center;min-height:100vh;margin:0}main{max-width:28rem;"
            "padding:2rem}p{color:#94a3b8;line-height:1.6}</style><main><h1>Pairing required</h1>"
            "<p>Open Mobile Access in the SubGen desktop app and scan its QR code.</p></main></html>"
        ).encode("utf-8")
        self.send_response(401)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_pairing_page(self):
        body = b"""<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="referrer" content="no-referrer"><title>Pair with SubGen</title><style>
body{font-family:system-ui;background:#070a13;color:#f8fafc;display:grid;place-items:center;min-height:100vh;margin:0}
main{width:min(28rem,calc(100% - 2rem));text-align:center}p{color:#94a3b8;line-height:1.6}.state{color:#22d3ee}
</style></head><body><main><h1>Connecting to SubGen</h1><p class="state" id="state">Verifying this device...</p></main>
<script>(async()=>{const state=document.getElementById('state');const token=location.hash.slice(1);history.replaceState(null,'','/pair');
if(!token){state.textContent='This pairing link is incomplete.';return}try{const response=await fetch('/api/mobile/pair',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token})});
if(!response.ok)throw new Error();state.textContent='Paired. Opening SubGen...';location.replace('/')}catch(error){state.textContent='Pairing failed. Refresh the QR code in the desktop app.'}})();</script>
</body></html>"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)

    def mobile_urls(self):
        return build_mobile_urls(mobile_access["port"], mobile_access["token"])

    def mobile_status(self, force=False):
        now = time.monotonic()
        if force or not mobile_access.get("diagnostics") or now - mobile_access.get("diagnostics_at", 0) > 2:
            urls = self.mobile_urls() if mobile_access["enabled"] else []
            mobile_access["diagnostics"] = mobile_access_diagnostics(
                urls,
                executable_path=sys.executable,
                port=mobile_access["port"],
            )
            mobile_access["diagnostics_at"] = now
        return mobile_access["diagnostics"]

    def request_has_valid_origin(self):
        if self.client_is_local():
            return True
        origin = self.headers.get("Origin")
        if not origin:
            return True
        parsed = urllib.parse.urlparse(origin)
        return parsed.scheme in {"http", "https"} and parsed.netloc == self.headers.get("Host")

    def remote_path_is_allowed(self, path):
        if self.client_is_local():
            return True
        try:
            candidate = Path(path).resolve()
        except (OSError, TypeError, ValueError):
            return False
        try:
            candidate.relative_to(UPLOADS_DIR.resolve())
            return True
        except ValueError:
            pass
        with state_lock:
            allowed_values = [
                server_state.get("video_path"),
                server_state.get("output_video"),
                server_state.get("final_srt"),
            ]
        for allowed in allowed_values:
            if not allowed:
                continue
            try:
                if candidate == Path(allowed).resolve():
                    return True
            except (OSError, TypeError, ValueError):
                continue
        return False

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)

        if path == '/pair' and mobile_access["enabled"]:
            self.send_pairing_page()
            return

        # API: Desktop shell readiness probe.
        if path == '/api/health':
            payload = json.dumps({"status": "ok", "version": __version__}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            return

        if not self.request_is_authorized():
            self.reject_unauthorized(api_request=path.startswith('/api/'))
            return

        if path == '/api/session':
            user_agent = self.headers.get("User-Agent", "")
            self.send_json({
                "is_local": self.client_is_local(),
                "lan_enabled": bool(mobile_access["enabled"]),
                "paired": True,
                "client_platform": detect_client_platform(user_agent),
                "client_browser": detect_client_browser(user_agent),
            })
            return

        if path == '/api/drive/status':
            status = drive_authorization.status()
            if status.get("connected"):
                try:
                    GoogleDriveClient()
                except Exception as exc:
                    status.update({"connected": False, "status": "error", "error": str(exc)})
            status["batch_running"] = drive_batch_is_running()
            status["latest_batch"] = public_drive_batch_state(drive_batch_store.latest())
            status["can_configure"] = self.client_is_local()
            self.send_json(status)
            return

        if path == '/api/drive/batch/status':
            batch_id = query.get('id', [None])[0]
            state = drive_batch_store.load(batch_id) if batch_id else drive_batch_store.latest()
            self.send_json({
                "batch": public_drive_batch_state(state),
                "running": drive_batch_is_running(),
            })
            return

        if path == '/api/drive/batch/item':
            batch_id = query.get('id', [None])[0]
            try:
                item_index = int(query.get('index', [0])[0])
            except (TypeError, ValueError):
                self.send_json({"error": "Invalid Drive item index."}, 400)
                return
            state = drive_batch_store.load(batch_id) if batch_id else None
            if not state or item_index < 0 or item_index >= len(state.get("items", [])):
                self.send_json({"error": "Drive batch item not found."}, 404)
                return
            item = state["items"][item_index]
            review = get_review_manifest(item.get("review_video_id"), state.get("target_language") or "source")
            self.send_json({
                "item": public_drive_batch_state({"items": [item]}).get("items", [item])[0],
                "review": review,
                "segments": review_to_editor_segments(review) if review else [],
            })
            return

        if path == '/api/mobile/access':
            if not self.client_is_local():
                self.send_json({"error": "Mobile access settings are available on this computer only."}, 403)
                return
            urls = self.mobile_urls() if mobile_access["enabled"] else []
            self.send_json({
                "enabled": bool(mobile_access["enabled"]),
                "urls": urls,
                "token_suffix": mobile_access["token"][-6:] if mobile_access["token"] else None,
                "diagnostics": self.mobile_status(),
                "last_device": mobile_access.get("last_device"),
            })
            return

        if path == '/api/mobile/qr':
            if not self.client_is_local():
                self.send_json({"error": "QR access is available on this computer only."}, 403)
                return
            urls = self.mobile_urls()
            if not urls:
                self.send_json({"error": "No private network address was detected."}, 503)
                return
            try:
                index = min(max(int(query.get('index', [0])[0]), 0), len(urls) - 1)
                import segno
                output = io.BytesIO()
                segno.make(urls[index]["pairing_url"], error="m").save(
                    output, kind="svg", scale=5, border=2, dark="#070a13", light="#ffffff"
                )
                body = output.getvalue()
                self.send_response(200)
                self.send_header("Content-Type", "image/svg+xml")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                self.send_json({"error": f"Could not create pairing QR: {exc}"}, 500)
            return

        # API: Status
        if path == '/api/process/status':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            with state_lock:
                # Return state and clear new logs
                response_data = server_state.copy()
                server_state["new_logs"] = []
                
            self.wfile.write(json.dumps(response_data).encode('utf-8'))
            return

        # API: Config
        elif path == '/api/config':
            # Reload from disk to reflect changes immediately
            config_path = CONFIG_PATH
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        user_config = json.load(f)
                        CONFIG.clear()
                        CONFIG.update(merged_runtime_config(user_config))
                except Exception as e:
                    print(f"Error reloading config: {e}")

            response_config = dict(CONFIG)
            if not self.client_is_local():
                for private_key in (
                    "last_output_dir",
                    "openai_api_env_key",
                    "openai_profiles",
                    "api_profiles",
                    "preferred_profiles",
                ):
                    response_config.pop(private_key, None)
            self.send_json(response_config)
            return

        elif path == '/api/pipeline/catalog':
            registry = subgen_providers.get_provider_registry(CONFIG)
            providers = {
                provider_id: {
                    key: provider.get(key)
                    for key in (
                        "name",
                        "transcription",
                        "translation",
                        "transcription_model",
                        "translation_model",
                    )
                }
                for provider_id, provider in registry.items()
            }
            self.send_json({
                "providers": providers,
                "models": list(public_model_catalog(CONFIG)),
                "selected_models": CONFIG.get("provider_models", {}),
            })
            return

        # API: Config Keys
        elif path == '/api/config/keys':
            if not self.client_is_local():
                self.send_json({"api_profiles": {}})
                return
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            init_api_profiles()
            env_vals = read_env_values()
            
            res_profiles = {}
            api_profs = CONFIG.get("api_profiles", {})
            pref_profiles = CONFIG.get("preferred_profiles", {})
            
            for prov_id, profiles_dict in api_profs.items():
                res_profiles[prov_id] = []
                preferred = pref_profiles.get(prov_id, "default")
                for pid, pinfo in profiles_dict.items():
                    env_key = pinfo.get("env_key", f"{prov_id.upper()}_API_KEY")
                    has_key = bool(env_vals.get(env_key) or os.environ.get(env_key))
                    res_profiles[prov_id].append({
                        "id": pid,
                        "label": pinfo.get("label", pid),
                        "env_key": env_key,
                        "configured": has_key,
                        "active": pid == preferred
                    })
                    
            self.wfile.write(json.dumps({"api_profiles": res_profiles}).encode('utf-8'))
            return

        # API: Languages
        elif path == '/api/languages':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            langs = get_supported_languages()
            self.wfile.write(json.dumps(langs).encode('utf-8'))
            return

        # API: Check Local Path
        elif path == '/api/check-local-path':
            if not self.client_is_local():
                self.send_json({"error": "Local Windows paths are available on the desktop only."}, 403)
                return
            path_str = query.get('path', [None])[0]
            if not path_str:
                self.send_error(400, "Missing 'path' parameter")
                return
                
            local_path = Path(path_str)
            if not local_path.exists() or not local_path.is_file():
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"exists": False}).encode('utf-8'))
                return

            # File exists!
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                "exists": True,
                "filename": local_path.name,
                "filepath": str(local_path.resolve()),
                "size_bytes": local_path.stat().st_size
            }).encode('utf-8'))
            return

        # API: Serve Video File (With Range Support for seeking)
        elif path == '/api/video':
            file_path_str = query.get('path', [None])[0]
            if not file_path_str:
                self.send_error(400, "Missing 'path' parameter")
                return
                
            file_path = Path(file_path_str)
            if not self.remote_path_is_allowed(file_path):
                self.send_json({"error": "This file is not part of the active SubGen job."}, 403)
                return
            if not file_path.exists():
                self.send_error(404, "Video file not found")
                return

            file_size = file_path.stat().st_size
            range_header = self.headers.get('Range')

            if range_header:
                self.serve_file_range(file_path, file_size, range_header)
            else:
                self.send_response(200)
                self.send_header('Content-Type', 'video/mp4')
                self.send_header('Content-Length', str(file_size))
                self.send_header('Accept-Ranges', 'bytes')
                self.end_headers()
                with open(file_path, 'rb') as f:
                    self.wfile.write(f.read())
            return

        # API: Serve SRT File
        elif path == '/api/srt':
            file_path_str = query.get('path', [None])[0]
            if not file_path_str:
                self.send_error(400, "Missing 'path' parameter")
                return
            file_path = Path(file_path_str)
            if not self.remote_path_is_allowed(file_path):
                self.send_json({"error": "This file is not part of the active SubGen job."}, 403)
                return
            if not file_path.exists():
                self.send_error(404, "SRT file not found")
                return

            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.send_header('Content-Disposition', f'attachment; filename="{file_path.name}"')
            self.end_headers()
            with open(file_path, 'rb') as f:
                self.wfile.write(f.read())
            return

        # API: Check Video Cache
        elif path == '/api/video/check-cache':
            file_path_str = query.get('path', [None])[0]
            if not file_path_str:
                self.send_error(400, "Missing 'path' parameter")
                return
                
            file_path = Path(file_path_str)
            if not self.remote_path_is_allowed(file_path):
                self.send_json({"error": "This file is not part of the active SubGen job."}, 403)
                return
            if not file_path.exists():
                self.send_error(404, "Video file not found")
                return
                
            try:
                from subgen_db import calculate_video_hash, get_cached_transcription
                video_hash = calculate_video_hash(file_path)
                cached = get_cached_transcription(video_hash)
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "exists": cached is not None,
                    "hash": video_hash
                }).encode('utf-8'))
            except Exception as e:
                self.send_error(500, f"Error checking cache: {e}")
            return

        # API: Load an autosaved subtitle draft.
        elif path == '/api/editor/draft':
            video_hash = query.get('video_hash', [None])[0]
            target_language = query.get('target_language', ['source'])[0]
            if not video_hash:
                self.send_error(400, "Missing 'video_hash' parameter")
                return
            review = get_review_manifest(video_hash, target_language)
            draft = (
                {
                    "segments": review_to_editor_segments(review),
                    "approved": bool(review.get("approval")),
                    "review": review,
                }
                if review else get_subtitle_draft(video_hash, target_language)
            )
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"draft": draft}, ensure_ascii=False).encode('utf-8'))
            return

        # API: Audio waveform used by the subtitle timing editor.
        elif path == '/api/video/waveform':
            file_path_str = query.get('path', [None])[0]
            if not file_path_str:
                self.send_error(400, "Missing 'path' parameter")
                return
            file_path = Path(file_path_str)
            if not self.remote_path_is_allowed(file_path):
                self.send_json({"error": "This file is not part of the active SubGen job."}, 403)
                return
            if not file_path.exists() or not file_path.is_file():
                self.send_error(404, "Video file not found")
                return
            try:
                peaks = build_waveform_peaks(file_path, query.get('bins', [900])[0])
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"peaks": peaks}).encode('utf-8'))
            except Exception as exc:
                self.send_error(500, f"Waveform generation failed: {exc}")
            return

        # Serve static files normally
        super().do_GET()

    def serve_file_range(self, filepath, file_size, range_header):
        try:
            # Parse Range: e.g. "bytes=0-1000" or "bytes=2000-"
            range_str = range_header.strip().split('=')[1]
            parts = range_str.split('-')
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if parts[1] else file_size - 1
            
            if start >= file_size:
                self.send_response(416) # Range Not Satisfiable
                self.send_header('Content-Range', f'bytes */{file_size}')
                self.end_headers()
                return
                
            if end >= file_size:
                end = file_size - 1
                
            chunk_size = end - start + 1
            
            self.send_response(206) # Partial Content
            self.send_header('Content-Type', 'video/mp4')
            self.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
            self.send_header('Content-Length', str(chunk_size))
            self.send_header('Accept-Ranges', 'bytes')
            self.end_headers()
            
            with open(filepath, 'rb') as f:
                f.seek(start)
                remaining = chunk_size
                while remaining > 0:
                    chunk = f.read(min(remaining, 64 * 1024))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # Browsers routinely cancel obsolete media range requests while seeking.
            return
        except Exception as e:
            print(f"Error serving range for {filepath}: {e}")

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path == '/api/mobile/pair':
            if not mobile_access["enabled"]:
                self.send_json({"error": "Mobile access is disabled."}, 404)
                return
            try:
                content_length = min(int(self.headers.get('Content-Length', 0)), 4096)
                data = json.loads(self.rfile.read(content_length).decode('utf-8'))
            except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
                self.send_json({"error": "Invalid pairing request."}, 400)
                return
            if not token_matches(data.get("token"), mobile_access["token"]):
                self.send_json({"error": "Invalid pairing code."}, 403)
                return
            user_agent = self.headers.get("User-Agent", "")
            device = {
                "platform": detect_client_platform(user_agent),
                "browser": detect_client_browser(user_agent),
                "paired_at": int(time.time()),
            }
            mobile_access["last_device"] = device
            body = json.dumps({"status": "paired", "device": device}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header(
                "Set-Cookie",
                f"{ACCESS_COOKIE_NAME}={mobile_access['token']}; Path=/; HttpOnly; SameSite=Strict",
            )
            self.end_headers()
            self.wfile.write(body)
            return

        if not self.request_is_authorized():
            self.reject_unauthorized(api_request=True)
            return
        if not self.request_has_valid_origin():
            self.send_json({"error": "Request origin does not match this SubGen server."}, 403)
            return

        if path == '/api/mobile/rotate':
            if not self.client_is_local():
                self.send_json({"error": "Only the desktop app can rotate mobile access."}, 403)
                return
            mobile_access["token"] = rotate_access_token(MOBILE_TOKEN_PATH)
            mobile_access["last_device"] = None
            self.send_json({"status": "rotated"})
            return

        if path == '/api/mobile/repair':
            if not self.client_is_local():
                self.send_json({"error": "Only the desktop app can repair mobile access."}, 403)
                return
            urls = self.mobile_urls() if mobile_access["enabled"] else []
            interfaces = sorted({entry.get("interface") for entry in urls if entry.get("interface")})
            try:
                request_windows_mobile_access_repair(
                    mobile_access["port"],
                    interfaces,
                    executable_path=sys.executable,
                )
                mobile_access["diagnostics"] = None
                mobile_access["diagnostics_at"] = 0
                self.send_json({"status": "awaiting_administrator"}, 202)
            except Exception as exc:
                self.send_json({"error": f"Could not start Windows network repair: {exc}"}, 500)
            return

        if path == '/api/drive/configure':
            if not self.client_is_local():
                self.send_json({"error": "Google OAuth setup is available on this computer only."}, 403)
                return
            try:
                content_length = min(int(self.headers.get('Content-Length', 0)), 64 * 1024)
                data = json.loads(self.rfile.read(content_length).decode('utf-8'))
                configured_path = configure_drive_client(data.get("client_json_path"))
                self.send_json({"status": "configured", "filename": configured_path.name})
            except Exception as exc:
                self.send_json({"error": str(exc)}, 400)
            return

        if path == '/api/drive/connect':
            if not self.client_is_local():
                self.send_json({"error": "Connect Google Drive from the desktop app."}, 403)
                return
            try:
                self.send_json(drive_authorization.start(), 202)
            except Exception as exc:
                self.send_json({"error": str(exc)}, 400)
            return

        if path == '/api/drive/disconnect':
            if not self.client_is_local():
                self.send_json({"error": "Disconnect Google Drive from the desktop app."}, 403)
                return
            disconnect_drive()
            self.send_json({"status": "disconnected"})
            return

        if path == '/api/drive/folder/inspect':
            try:
                content_length = min(int(self.headers.get('Content-Length', 0)), 64 * 1024)
                data = json.loads(self.rfile.read(content_length).decode('utf-8'))
                folder_id = extract_drive_folder_id(data.get("folder_url"))
                scan = GoogleDriveClient().scan_videos(folder_id)
                self.send_json({
                    "folder": {
                        "id": scan.folder_id,
                        "name": scan.folder_name,
                        "video_count": len(scan.videos),
                        "total_bytes": sum(video.get("size", 0) for video in scan.videos),
                        "videos": [
                            {
                                "id": video["id"],
                                "name": video["name"],
                                "relative_path": video["relative_path"],
                                "size": video.get("size", 0),
                                "can_download": (video.get("capabilities") or {}).get("canDownload", True),
                            }
                            for video in scan.videos
                        ],
                    }
                })
            except Exception as exc:
                self.send_json({"error": str(exc)}, 400)
            return

        if path == '/api/drive/batch/start':
            global drive_batch_thread, drive_batch_starting
            try:
                content_length = min(int(self.headers.get('Content-Length', 0)), 1024 * 1024)
                data = json.loads(self.rfile.read(content_length).decode('utf-8'))
                with drive_batch_lock:
                    if drive_batch_is_running():
                        raise RuntimeError("A Google Drive batch is already running.")
                    drive_batch_starting = True
                with state_lock:
                    if server_state.get("status") in {"processing", "burning"}:
                        raise RuntimeError("Finish the current single-video job before starting a batch.")
                source_url = data.get("source_folder_url")
                source_id = extract_drive_folder_id(source_url)
                client = GoogleDriveClient()
                scan = client.scan_videos(source_id)
                if not scan.videos:
                    raise RuntimeError("No supported video files were found in this Google Drive folder.")
                destination_url = str(data.get("destination_folder_url") or "").strip()
                destination_id = extract_drive_folder_id(destination_url) if destination_url else source_id
                client.get_folder(destination_id)
                state = drive_batch_store.create(
                    {
                        "source_folder_url": source_url,
                        "destination_folder_url": destination_url or None,
                        "destination_folder_id": destination_id,
                        "target_language": data.get("target_language") or None,
                        "configuration": data.get("configuration") or {},
                    },
                    scan,
                )
                with drive_batch_lock:
                    drive_batch_thread = threading.Thread(
                        target=run_drive_batch_thread,
                        args=(state["id"],),
                        name=f"subgen-drive-batch-{state['id'][:8]}",
                        daemon=True,
                    )
                    drive_batch_starting = False
                    drive_batch_thread.start()
                self.send_json({"batch": public_drive_batch_state(state)}, 202)
            except Exception as exc:
                with drive_batch_lock:
                    drive_batch_starting = False
                self.send_json({"error": str(exc)}, 400)
            return

        if path == '/api/drive/batch/stop':
            try:
                content_length = min(int(self.headers.get('Content-Length', 0)), 64 * 1024)
                data = json.loads(self.rfile.read(content_length).decode('utf-8'))
                batch_id = data.get("id") or (drive_batch_store.latest() or {}).get("id")
                if not batch_id:
                    raise RuntimeError("No Google Drive batch is available.")
                state = drive_batch_store.request_stop(batch_id)
                self.send_json({"batch": public_drive_batch_state(state)}, 202)
            except Exception as exc:
                self.send_json({"error": str(exc)}, 400)
            return

        if path == '/api/drive/batch/resume':
            try:
                content_length = min(int(self.headers.get('Content-Length', 0)), 64 * 1024)
                data = json.loads(self.rfile.read(content_length).decode('utf-8'))
                with drive_batch_lock:
                    if drive_batch_is_running():
                        raise RuntimeError("A Google Drive batch is already running.")
                    drive_batch_starting = True
                batch_id = data.get("id") or (drive_batch_store.latest() or {}).get("id")
                state = drive_batch_store.load(batch_id) if batch_id else None
                if not state:
                    raise RuntimeError("Google Drive batch was not found.")
                for item in state.get("items", []):
                    if item.get("status") in {"failed", "pending", "running"}:
                        item.update({"status": "pending", "stage": "queued", "progress": 0, "error": None})
                state.update({"status": "pending", "stop_requested": False, "current_index": None, "error": None})
                drive_batch_store.save(state, preserve_stop_request=False)
                with drive_batch_lock:
                    drive_batch_thread = threading.Thread(
                        target=run_drive_batch_thread,
                        args=(state["id"],),
                        name=f"subgen-drive-batch-{state['id'][:8]}",
                        daemon=True,
                    )
                    drive_batch_starting = False
                    drive_batch_thread.start()
                self.send_json({"batch": public_drive_batch_state(state)}, 202)
            except Exception as exc:
                with drive_batch_lock:
                    drive_batch_starting = False
                self.send_json({"error": str(exc)}, 400)
            return

        if path in {
            '/api/drive/batch/item/save', '/api/drive/batch/item/approve',
            '/api/drive/batch/item/burn', '/api/drive/batch/item/issue/resolve',
            '/api/drive/batch/item/retranslate',
        }:
            content_length = min(int(self.headers.get('Content-Length', 0)), 8 * 1024 * 1024)
            data = json.loads(self.rfile.read(content_length).decode('utf-8'))
            state = drive_batch_store.load(data.get("batch_id"))
            try:
                item_index = int(data.get("item_index"))
                item = state["items"][item_index]
            except (TypeError, ValueError, IndexError, KeyError):
                self.send_json({"error": "Drive batch item not found."}, 404)
                return
            review = get_review_manifest(item.get("review_video_id"), state.get("target_language") or "source")
            if not review:
                self.send_json({"error": "Prepared Drive review not found."}, 404)
                return
            if path.endswith('/issue/resolve'):
                try:
                    resolve_issue(
                        review, data.get("issue_id"), data.get("status"), actor="drive_editor",
                        reason=data.get("reason"),
                    )
                except (KeyError, ValueError) as exc:
                    self.send_json({"error": str(exc)}, 400)
                    return
                save_review_manifest(review)
                item.update({"review_state": review.get("state"), "status": "needs_attention"})
                drive_batch_store.save(state)
                self.send_json({"review": review})
                return
            if path.endswith('/retranslate'):
                try:
                    result = retranslate_review_selected_cues(
                        review,
                        data.get("cue_ids") or [],
                        CONFIG,
                        device=CONFIG.get("device", "cpu"),
                        actor="drive_editor",
                    )
                    save_review_manifest(review)
                    item.update({
                        "review_state": review.get("state"),
                        "status": "needs_attention",
                    })
                    drive_batch_store.save(state)
                except (KeyError, ValueError, RuntimeError) as exc:
                    self.send_json({"error": str(exc)}, 400)
                    return
                self.send_json({
                    "review": review,
                    "segments": review_to_editor_segments(review),
                    "retranslation": {
                        key: result[key]
                        for key in (
                            "selected_source_cue_ids",
                            "translated_cue_count",
                            "provider",
                            "model",
                        )
                    },
                })
                return
            if path.endswith('/save'):
                update_review_from_editor(
                    review, data.get("segments") or [], actor="drive_editor",
                    translation_confirmed=bool(data.get("translation_confirmed", False)),
                )
                save_review_manifest(review)
                item.update({
                    "review_state": review.get("state"),
                    "status": "needs_attention" if review.get("state") == "NEEDS_ATTENTION" else "ready_for_review",
                })
                drive_batch_store.save(state)
                self.send_json({"review": review})
                return
            if path.endswith('/approve'):
                try:
                    update_review_from_editor(
                        review, data.get("segments") or review_to_editor_segments(review), actor="drive_editor",
                        translation_confirmed=bool(data.get("translation_confirmed", False)),
                    )
                    approve_review(
                        review, actor="drive_editor",
                        accept_warnings=bool(data.get("accept_warnings", False)),
                    )
                    save_review_manifest(review)
                    item.update({"review_state": "APPROVED", "status": "approved"})
                    drive_batch_store.save(state)
                except ValueError as exc:
                    self.send_json({"error": str(exc), "issues": review.get("issues") or []}, 409)
                    return
                self.send_json({"review": review, "approved_draft_hash": review["approval"]["approved_draft_hash"]})
                return
            if review.get("state") != "APPROVED" or not review.get("approval"):
                self.send_json({"error": "Drive burn requires an explicitly approved draft."}, 409)
                return
            item.update({"status": "burning", "stage": "queued_for_redownload", "progress": 0})
            drive_batch_store.save(state)
            thread = threading.Thread(
                target=run_drive_burn_thread,
                args=(state["id"], item_index),
                name=f"subgen-drive-burn-{item_index}",
                daemon=True,
            )
            thread.start()
            self.send_json({"status": "burning", "batch": public_drive_batch_state(state)}, 202)
            return

        if path == '/api/pipeline/preview':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8')) if body else {}
            preview_config = dict(CONFIG)
            for key in (
                "transcription_provider",
                "transcription_model",
                "timing_anchor_provider",
                "timing_anchor_model",
                "api_transcript_timing_mode",
                "translation_provider",
                "translation_model",
                "model_size",
                "source_language",
            ):
                if data.get(key) is not None:
                    preview_config[key] = data.get(key)
            try:
                plan = build_pipeline_plan(
                    preview_config,
                    target_language=data.get("target_language"),
                )
            except (ValueError, RuntimeError) as exc:
                self.send_json({"error": str(exc)})
                return
            self.send_json({"plan": plan.to_dict()})
            return

        # API: Validate or autosave edited subtitle segments.
        if path in ['/api/editor/validate', '/api/editor/draft']:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            segments = data.get('segments') or []
            report = validate_editor_segments(segments)
            if path == '/api/editor/draft':
                video_hash = data.get('video_hash')
                target_language = data.get('target_language') or 'source'
                if not video_hash:
                    self.send_error(400, "Missing video_hash")
                    return
                review = get_review_manifest(video_hash, target_language)
                if not review:
                    self.send_json({"error": "No prepared review exists for this video."}, 409)
                    return
                update_review_from_editor(
                    review,
                    segments,
                    actor="local_editor",
                    translation_confirmed=bool(data.get("translation_confirmed", False)),
                )
                persist_active_review(review, video_path=server_state.get("video_path"))
                save_subtitle_draft(video_hash, target_language, segments, approved=False)
                report["review"] = {
                    "state": review.get("state"),
                    "approval": review.get("approval"),
                    "field_state": review.get("field_state"),
                    "issues": review.get("issues") or [],
                }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(report, ensure_ascii=False).encode('utf-8'))
            return

        if path == '/api/editor/issue/resolve':
            content_length = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(content_length).decode('utf-8'))
            video_hash = data.get("video_hash")
            target_language = data.get("target_language") or "source"
            review = get_review_manifest(video_hash, target_language) if video_hash else None
            if not review:
                self.send_json({"error": "Prepared review not found."}, 404)
                return
            try:
                resolve_issue(
                    review,
                    data.get("issue_id"),
                    data.get("status"),
                    actor="local_editor",
                    reason=data.get("reason"),
                )
                persist_active_review(review, video_path=server_state.get("video_path"))
            except (KeyError, ValueError) as exc:
                self.send_json({"error": str(exc)}, 400)
                return
            self.send_json({"review": review})
            return

        if path == '/api/editor/retranslate':
            content_length = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(content_length).decode('utf-8'))
            video_hash = data.get("video_hash") or server_state.get("video_hash")
            target_language = (
                data.get("target_language")
                or server_state.get("target_lang")
                or "source"
            )
            review = (
                get_review_manifest(video_hash, target_language)
                if video_hash else None
            )
            if not review:
                self.send_json({"error": "Prepared review not found."}, 404)
                return
            try:
                result = retranslate_review_selected_cues(
                    review,
                    data.get("cue_ids") or [],
                    CONFIG,
                    device=CONFIG.get("device", "cpu"),
                    actor="local_or_lan_editor",
                )
                persist_active_review(
                    review, video_path=server_state.get("video_path")
                )
            except (KeyError, ValueError, RuntimeError) as exc:
                self.send_json({"error": str(exc)}, 400)
                return
            self.send_json({
                "review": review,
                "segments": review_to_editor_segments(review),
                "retranslation": {
                    key: result[key]
                    for key in (
                        "selected_source_cue_ids",
                        "translated_cue_count",
                        "provider",
                        "model",
                    )
                },
            })
            return

        if path == '/api/process/approve':
            content_length = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(content_length).decode('utf-8'))
            video_hash = data.get("video_hash") or server_state.get("video_hash")
            target_language = data.get("target_language") or server_state.get("target_lang") or "source"
            review = get_review_manifest(video_hash, target_language) if video_hash else None
            if not review:
                self.send_json({"error": "Prepared review not found."}, 404)
                return
            segments = data.get("segments") or review_to_editor_segments(review)
            validation = validate_editor_segments(segments)
            if not validation.get("accept"):
                self.send_json(validation, 409)
                return
            try:
                update_review_from_editor(
                    review,
                    segments,
                    actor="local_editor",
                    translation_confirmed=bool(data.get("translation_confirmed", False)),
                )
                approve_review(
                    review,
                    actor="local_editor",
                    accept_warnings=bool(data.get("accept_warnings", False)),
                )
                review_path = persist_active_review(review, video_path=server_state.get("video_path"))
            except ValueError as exc:
                self.send_json({"error": str(exc), "issues": review.get("issues") or []}, 409)
                return
            self.send_json({
                "status": "approved",
                "approved_draft_hash": review["approval"]["approved_draft_hash"],
                "review_path": str(review_path) if review_path else None,
                "review": review,
            })
            return

        # API Config POST endpoints
        if path in ['/api/config/save-key', '/api/config/set-active-profile', '/api/config/update-output-dir']:
            if not self.client_is_local():
                self.send_json({"error": "Provider and output settings can be changed on the desktop only."}, 403)
                return
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            if path == '/api/config/save-key':
                provider_id = data.get('provider')
                name = data.get('name', '').strip().lower()
                api_key = data.get('api_key')
                
                if not provider_id or not name or not api_key:
                    self.send_error(400, "Missing provider, name, or api_key")
                    return
                    
                # Generate env key
                if name == "default":
                    env_key = f"{provider_id.upper()}_API_KEY"
                else:
                    env_key = f"{provider_id.upper()}_API_KEY_{name.upper()}"
                    
                update_env_value(env_key, api_key)
                
                # Update CONFIG
                init_api_profiles()
                if provider_id not in CONFIG["api_profiles"]:
                    CONFIG["api_profiles"][provider_id] = {}
                CONFIG["api_profiles"][provider_id][name] = {
                    "label": name,
                    "env_key": env_key
                }
                
                # Maintain backwards compatibility for OpenAI
                if provider_id == "openai":
                    if "openai_profiles" not in CONFIG:
                        CONFIG["openai_profiles"] = {}
                    CONFIG["openai_profiles"][name] = {
                        "label": name,
                        "env_key": env_key
                    }
                
                # Save config file
                config_path = CONFIG_PATH
                try:
                    with open(config_path, 'w', encoding='utf-8') as f:
                        json.dump(CONFIG, f, indent=2)
                except Exception as e:
                    print(f"Error saving config: {e}")
                    
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success"}).encode('utf-8'))
                return

            elif path == '/api/config/set-active-profile':
                provider_id = data.get('provider')
                profile = data.get('profile')
                if not provider_id or not profile:
                    self.send_error(400, "Missing provider or profile name")
                    return
                    
                init_api_profiles()
                CONFIG["preferred_profiles"][provider_id] = profile
                
                # Maintain backwards compatibility for OpenAI
                if provider_id == "openai":
                    CONFIG["preferred_openai_profile"] = profile
                    profiles_dict = CONFIG.get("openai_profiles", {})
                    profile_info = profiles_dict.get(profile, {})
                    CONFIG["openai_api_env_key"] = profile_info.get("env_key", "OPENAI_API_KEY")
                
                # Save config file
                config_path = CONFIG_PATH
                try:
                    with open(config_path, 'w', encoding='utf-8') as f:
                        json.dump(CONFIG, f, indent=2)
                except Exception as e:
                    print(f"Error saving config: {e}")
                    
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success"}).encode('utf-8'))
                return

            elif path == '/api/config/update-output-dir':
                output_dir = data.get('output_dir', '').strip()
                if not output_dir:
                    self.send_error(400, "Missing output_dir")
                    return
                    
                CONFIG["last_output_dir"] = output_dir
                
                # Save config file
                config_path = CONFIG_PATH
                try:
                    with open(config_path, 'w', encoding='utf-8') as f:
                        json.dump(CONFIG, f, indent=2)
                except Exception as e:
                    print(f"Error saving config: {e}")
                    
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success"}).encode('utf-8'))
                return

        # API: End the current uploaded-video session.
        if path == '/api/session/release-video':
            try:
                content_length = min(int(self.headers.get('Content-Length', 0)), 64 * 1024)
                body = self.rfile.read(content_length)
                data = json.loads(body.decode('utf-8')) if body else {}
            except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
                self.send_json({"error": "Invalid video session release request."}, 400)
                return
            result = release_video_session(data.get('video_path'), data.get('job_id'))
            self.send_json(result, 202 if result["status"] == "deferred" else 200)
            return

        # API: Upload Video
        if path == '/api/upload':
            if self.headers.get('Content-Type', '').split(';', 1)[0] != 'application/octet-stream':
                self.discard_small_request_body()
                self.send_json({"error": "Upload must use application/octet-stream."}, 415)
                return
            try:
                content_length = int(self.headers.get('Content-Length', 0))
            except ValueError:
                content_length = 0
            if content_length <= 0:
                self.send_json({"error": "The uploaded file is empty."}, 400)
                return
            if content_length > MAX_UPLOAD_BYTES:
                self.send_json({"error": "Video exceeds the 2 GB upload limit."}, 413)
                return
            encoded_filename = self.headers.get('X-SubGen-Filename', 'video.mp4')
            filename = urllib.parse.unquote(encoded_filename)
            job_id = str(uuid.uuid4())
            uploads_dir = UPLOADS_DIR / "jobs" / job_id
            uploads_dir.mkdir(parents=True, exist_ok=True)
            safe_filename = "".join([c for c in filename if c.isalpha() or c.isdigit() or c in "._-"]).strip()
            if not safe_filename:
                safe_filename = "video.mp4"
            filepath = uploads_dir / safe_filename
            partial_path = filepath.with_suffix(filepath.suffix + ".part")
            remaining = content_length
            try:
                with open(partial_path, 'wb') as output:
                    while remaining:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise ConnectionError("Upload ended before the declared file size was received.")
                        output.write(chunk)
                        remaining -= len(chunk)
                partial_path.replace(filepath)
            except Exception as exc:
                partial_path.unlink(missing_ok=True)
                shutil.rmtree(uploads_dir, ignore_errors=True)
                self.send_json({"error": f"Upload failed: {exc}"}, 400)
                return

            self.send_json({
                "filename": safe_filename,
                "filepath": str(filepath.resolve()),
                "job_id": job_id
            })
            return

        # API: Start Process
        elif path == '/api/process/start':
            if drive_batch_is_running():
                self.send_json({"error": "Stop or finish the active Google Drive batch first."}, 409)
                return
            content_length = int(self.headers.get('Content-Length'))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            video_path = data.get('video_path')
            if video_path and not self.remote_path_is_allowed(video_path):
                self.send_json({"error": "This file is not part of the active SubGen job."}, 403)
                return
            source_lang = data.get('source_language')
            target_lang = data.get('target_language')
            trans_provider = data.get('transcription_provider', 'google')
            transcription_model = data.get('transcription_model')
            model_size = data.get('model_size', 'small')
            trans_provider_api = data.get('translation_provider', 'local')
            translation_model = data.get('translation_model')
            subtitle_mode = normalize_subtitle_mode(
                data.get('subtitle_mode'),
                legacy_tiktok_style=data.get('tiktok_style', False),
            )
            output_dir = data.get('last_output_dir')
            if not self.client_is_local():
                output_dir = None
            api_transcript_timing_mode = data.get('api_transcript_timing_mode', 'precise')
            timing_anchor_provider = data.get('timing_anchor_provider', 'openai')

            # Save selections to CONFIG and persist
            CONFIG["transcription_provider"] = trans_provider
            if transcription_model:
                CONFIG["transcription_model"] = transcription_model
                CONFIG.setdefault("provider_models", {}).setdefault(trans_provider, {})[
                    "transcription"
                ] = transcription_model
            CONFIG["translation_provider"] = trans_provider_api
            if translation_model:
                CONFIG["translation_model"] = translation_model
                CONFIG.setdefault("provider_models", {}).setdefault(trans_provider_api, {})[
                    "translation"
                ] = translation_model
            CONFIG["model_size"] = model_size
            CONFIG["subtitle_mode"] = subtitle_mode
            CONFIG["tiktok_style"] = subtitle_mode == "tiktok"
            CONFIG["api_transcript_timing_mode"] = api_transcript_timing_mode
            CONFIG["timing_anchor_provider"] = timing_anchor_provider
            if output_dir:
                output_dir = output_dir.strip()
                CONFIG["last_output_dir"] = output_dir

            # Save config file
            config_path = CONFIG_PATH
            try:
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(CONFIG, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Error saving config: {e}")

            # Generate a job_id if we don't have one (e.g., if local path was used)
            job_id = data.get('job_id') or str(uuid.uuid4())
            
            # Compute hash of the video
            video_hash = None
            if video_path:
                try:
                    video_hash = calculate_video_hash(video_path)
                except Exception:
                    pass
            with state_lock:
                server_state["video_hash"] = video_hash
            try:
                cache_action, _ = resolve_cache_action(
                    data.get('cache_action'),
                    force=bool(data.get('force', False)),
                )
            except ValueError as e:
                self.send_error(400, str(e))
                return
            
            create_job(job_id, video_path, video_hash, "processing")

            # Mark ownership active before starting the worker so a simultaneous
            # page-close release cannot remove its source during thread startup.
            with state_lock:
                server_state["job_id"] = job_id
                server_state["video_path"] = video_path
                server_state["status"] = "processing"

            # Start thread
            t = threading.Thread(
                target=run_pipeline_thread,
                args=(
                    job_id,
                    video_path,
                    source_lang,
                    target_lang,
                    trans_provider,
                    transcription_model,
                    model_size,
                    trans_provider_api,
                    translation_model,
                    timing_anchor_provider,
                    api_transcript_timing_mode,
                    subtitle_mode,
                    cache_action,
                )
            )
            t.daemon = True
            t.start()

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "processing"}).encode('utf-8'))
            return

        # API: Burn Subtitles
        elif path == '/api/process/burn':
            content_length = int(self.headers.get('Content-Length'))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            video_path = data.get('video_path')
            target_lang = data.get('target_language')
            segments = data.get('segments')
            style_config = data.get('style_config')
            if video_path and not self.remote_path_is_allowed(video_path):
                self.send_json({"error": "This file is not part of the active SubGen job."}, 403)
                return
            validation = validate_editor_segments(segments)
            if not validation['accept']:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(validation, ensure_ascii=False).encode('utf-8'))
                return
            _, cache_policy = resolve_cache_action(server_state.get("cache_action", "reuse_all"))
            force_burn = bool(data.get("force_burn", False)) or cache_policy["force_burn"]

            video_hash = server_state.get("video_hash")
            if not video_hash and video_path:
                video_hash = calculate_video_hash(video_path)
            review = get_review_manifest(video_hash, target_lang or 'source') if video_hash else None
            if not review:
                self.send_json({"error": "Burn requires a prepared review."}, 409)
                return
            candidate = json.loads(json.dumps(review))
            update_review_from_editor(candidate, segments, actor="burn_request_comparison")
            if (
                candidate.get("draft_hash") != review.get("draft_hash")
                or review_revision_hash(candidate) != review_revision_hash(review)
            ):
                self.send_json({"error": "Editor content changed after approval. Save and approve the new version."}, 409)
                return
            try:
                assert_burn_allowed(review, video_path)
                review_path = persist_active_review(review, video_path=video_path)
            except ValueError as exc:
                self.send_json({"error": str(exc), "issues": review.get("issues") or []}, 409)
                return
            if not review_path:
                self.send_json({"error": "Approved review manifest could not be persisted."}, 500)
                return

            # Retrieve or generate job_id
            job_id = server_state.get("job_id") or str(uuid.uuid4())
            update_job_status(job_id, "processing", stage="burning", status_label="Burning subtitles...")
            with state_lock:
                server_state["job_id"] = job_id
                server_state["video_path"] = video_path
                server_state["status"] = "burning"

            # Start thread
            t = threading.Thread(
                target=run_burning_thread,
                args=(job_id, video_path, target_lang, str(review_path), style_config, force_burn)
            )
            t.daemon = True
            t.start()

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "burning"}).encode('utf-8'))
            return

        self.send_error(404, "Not Found")

class SubGenThreadingServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def run_server(port=8080, host="127.0.0.1", open_browser=True, lan_access=False):
    # Initialize the SQLite database on startup
    try:
        init_db()
        print("[DB] SQLite database initialized successfully.")
        
        # Clean up any leftover sandboxed job files from previous server sessions
        uploads_jobs_dir = UPLOADS_DIR / "jobs"
        if uploads_jobs_dir.exists():
            cleanup_all_video_sessions()
            uploads_jobs_dir.mkdir(parents=True, exist_ok=True)
            print("[DB] Cleaned up temporary sandboxed job directories from previous sessions.")
    except Exception as e:
        print(f"[DB] Error initializing database or cleaning up jobs: {e}")

    mobile_access["enabled"] = bool(lan_access)
    mobile_access["port"] = int(port)
    mobile_access["token"] = load_or_create_access_token(MOBILE_TOKEN_PATH) if lan_access else None

    handler = SubGenRequestHandler
    with SubGenThreadingServer((host, port), handler) as httpd:
        print(f"\n==================================================")
        print(f"  SubGen Premium Studio UI is running at:")
        print(f"  http://localhost:{port}")
        if lan_access:
            urls = build_mobile_urls(port, mobile_access["token"])
            if urls:
                print(f"  Mobile access: {urls[0]['base_url']} (pair from the desktop QR)")
            else:
                print("  Mobile access enabled, but no private IPv4 address was detected.")
        print(f"==================================================\n")
        
        # Automatically open browser
        if open_browser:
            import webbrowser
            import threading
            threading.Timer(0.5, lambda: webbrowser.open(f"http://localhost:{port}")).start()
        
        try:
            httpd.serve_forever()
        finally:
            cleanup_all_video_sessions()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SubGen local application server")
    parser.add_argument("legacy_port", nargs="?", type=int)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--desktop", action="store_true")
    parser.add_argument("--lan", action="store_true")
    args = parser.parse_args()
    run_server(
        port=args.port or args.legacy_port or 8080,
        host=args.host,
        open_browser=not args.desktop,
        lan_access=args.lan,
    )
