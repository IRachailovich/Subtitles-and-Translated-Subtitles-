import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import zipfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import or_, select

from .config import load_settings
from .db import Artifact, Job, JobStatus, ProviderCredential, Video, create_database
from .security import CredentialCipher
from .srt import parse_srt, render_srt
from .storage import create_storage
from subgen_review import (
    assert_burn_allowed,
    combined_review_cues,
    load_review,
    save_review,
)


PROVIDER_ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "xai": "XAI_API_KEY",
    "cohere": "COHERE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


def utcnow():
    return datetime.now(timezone.utc)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_duration(path):
    completed = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError("FFprobe could not read the uploaded video")
    return float(completed.stdout.strip())


def content_type(path):
    suffix = Path(path).suffix.lower()
    return {
        ".mp4": "video/mp4",
        ".srt": "application/x-subrip; charset=utf-8",
        ".txt": "text/plain; charset=utf-8",
        ".json": "application/json",
    }.get(suffix, "application/octet-stream")


def claim_job(session_factory, worker_id, lease_seconds):
    now = utcnow()
    with session_factory.begin() as session:
        statement = (
            select(Job)
            .where(
                Job.status.in_([JobStatus.QUEUED.value, JobStatus.BURN_QUEUED.value]),
                or_(Job.lease_until.is_(None), Job.lease_until < now),
            )
            .order_by(Job.created_at)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        job = session.scalar(statement)
        if not job:
            return None
        job.lease_owner = worker_id
        job.lease_until = now + timedelta(seconds=lease_seconds)
        job.status = JobStatus.BURNING.value if job.status == JobStatus.BURN_QUEUED.value else JobStatus.PROCESSING.value
        job.stage = "burning" if job.status == JobStatus.BURNING.value else "preparing"
        job.progress = 2
        session.flush()
        return job.id


@contextmanager
def decrypted_provider_environment(session_factory, cipher, user_id, pipeline_config):
    original = {}
    injected = []
    with session_factory() as session:
        credentials = list(session.scalars(select(ProviderCredential).where(ProviderCredential.user_id == user_id)))
    preferred_profiles = pipeline_config.get("credential_profiles") or {}
    for credential in credentials:
        preferred = preferred_profiles.get(credential.provider, "default")
        if credential.profile != preferred:
            continue
        env_key = PROVIDER_ENV_KEYS.get(credential.provider)
        if not env_key:
            custom = (pipeline_config.get("custom_providers") or {}).get(credential.provider) or {}
            env_key = custom.get("env_key")
        if not env_key:
            continue
        original[env_key] = os.environ.get(env_key)
        os.environ[env_key] = cipher.decrypt(
            user_id, credential.provider, credential.profile, credential.ciphertext
        )
        injected.append(env_key)
    try:
        yield
    finally:
        for env_key in injected:
            if original[env_key] is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = original[env_key]


def refresh_job_lease(session_factory, job_id, worker_id, lease_seconds):
    with session_factory.begin() as session:
        job = session.get(Job, job_id)
        if (
            not job
            or job.lease_owner != worker_id
            or job.status == JobStatus.CANCELLED.value
        ):
            return False
        job.lease_until = utcnow() + timedelta(seconds=lease_seconds)
        return True


def job_is_cancelled(session_factory, job_id):
    with session_factory() as session:
        job = session.get(Job, job_id)
        return not job or job.status == JobStatus.CANCELLED.value


def terminate_pipeline_process(process):
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def restore_longform_checkpoint(storage, object_key, output_root, scratch_root):
    if not storage.exists(object_key):
        return False
    archive_path = Path(scratch_root) / "longform-checkpoint.zip"
    storage.download(object_key, archive_path)
    output_root = Path(output_root).resolve()
    with zipfile.ZipFile(archive_path, "r") as archive:
        for member in archive.infolist():
            destination = (output_root / member.filename).resolve()
            if output_root not in destination.parents and destination != output_root:
                raise RuntimeError("Cloud long-form checkpoint contains an unsafe path.")
        archive.extractall(output_root)
    return True


def upload_longform_checkpoint(
    storage,
    object_key,
    output_root,
    scratch_root,
    previous_signature=None,
):
    output_root = Path(output_root)
    files = sorted({
        *output_root.glob("**/longform-manifest.json"),
        *output_root.glob("**/chunk-*.result.json"),
        *output_root.glob("**/chunk-*.failure.json"),
        *output_root.glob("**/chunk-*.initial-rejected.json"),
        *output_root.glob("**/coverage-recovery-*.result.json"),
        *output_root.glob("**/coverage-recovery-*.failure.json"),
    })
    files = [path for path in files if path.is_file()]
    if not files:
        return previous_signature
    signature = hashlib.sha256(json.dumps([
        (
            path.relative_to(output_root).as_posix(),
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in files
    ], separators=(",", ":")).encode("utf-8")).hexdigest()
    if signature == previous_signature:
        return signature
    archive_path = Path(scratch_root) / "longform-checkpoint.zip"
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for path in files:
            archive.write(path, path.relative_to(output_root).as_posix())
    storage.upload_file(archive_path, object_key, "application/zip")
    return signature


def run_pipeline_subprocess(
    request,
    work_root,
    env,
    *,
    heartbeat=None,
    cancelled=None,
    heartbeat_seconds=30,
    timeout_seconds=43200,
):
    request_path = work_root / "pipeline-request.json"
    result_path = work_root / "pipeline-result.json"
    request_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
    command = [sys.executable, "-m", "subgen_cloud.pipeline_runner", str(request_path), str(result_path)]
    log_path = work_root / "pipeline.log"
    process_kwargs = {}
    if os.name == "nt":
        process_kwargs["creationflags"] = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
    else:
        process_kwargs["start_new_session"] = True
    started = time.monotonic()
    last_heartbeat = 0.0
    with log_path.open("w", encoding="utf-8", errors="replace") as log_stream:
        process = subprocess.Popen(
            command,
            env=env,
            text=True,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            **process_kwargs,
        )
        while process.poll() is None:
            now = time.monotonic()
            if timeout_seconds and now - started > float(timeout_seconds):
                terminate_pipeline_process(process)
                raise RuntimeError(
                    "Subtitle pipeline exceeded its configured hard deadline "
                    f"of {int(timeout_seconds)} seconds."
                )
            if cancelled and cancelled():
                terminate_pipeline_process(process)
                raise RuntimeError("Subtitle pipeline was cancelled.")
            if heartbeat and now - last_heartbeat >= float(heartbeat_seconds):
                if not heartbeat():
                    terminate_pipeline_process(process)
                    raise RuntimeError(
                        "Subtitle pipeline lost its worker lease or was cancelled."
                    )
                last_heartbeat = now
            time.sleep(1)
        returncode = process.returncode
    if returncode != 0:
        tail = "\n".join(log_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()[-40:])
        raise RuntimeError(f"Subtitle pipeline failed.\n{tail}")
    if not result_path.exists():
        raise RuntimeError("Subtitle pipeline exited without a result file.")
    return json.loads(result_path.read_text(encoding="utf-8"))["output_path"]


def add_artifact(session, storage, job, path, kind, language=None, parent_id=None, provenance=None):
    path = Path(path)
    digest = sha256_file(path)
    key = f"users/{job.user_id}/videos/{job.video_id}/jobs/{job.id}/{kind}/{path.name}"
    storage.upload_file(path, key, content_type(path))
    artifact = Artifact(
        user_id=job.user_id,
        video_id=job.video_id,
        job_id=job.id,
        kind=kind,
        language=language,
        object_key=key,
        sha256=digest,
        parent_artifact_id=parent_id,
        provenance=provenance or {},
    )
    session.add(artifact)
    session.flush()
    return artifact


def find_source_and_target_srts(root, target_language):
    paths = list(Path(root).glob("*.srt"))
    target = next((path for path in paths if target_language and path.name.endswith(f".{target_language}.srt")), None)
    source = next((path for path in paths if path != target), None)
    return source or target, target or source


def process_transcription_job(
    job_id,
    settings,
    session_factory,
    storage,
    cipher,
    worker_id,
):
    with session_factory() as session:
        job = session.get(Job, job_id)
        video = session.get(Video, job.video_id)
        source_key = video.object_key
        filename = video.original_name
        user_id = job.user_id
        pipeline_config = dict(job.pipeline_config or {})
        style_config = dict(job.style_config or {})
        target_language = job.target_language
        cache_action = job.cache_action

    with tempfile.TemporaryDirectory(prefix=f"subgen-{job_id[:8]}-") as temporary:
        root = Path(temporary)
        input_path = root / "input" / filename
        output_root = root / "output"
        data_root = root / "data"
        output_root.mkdir(parents=True)
        data_root.mkdir(parents=True)
        checkpoint_key = (
            f"users/{user_id}/videos/{job.video_id}/jobs/{job_id}/"
            "checkpoint/longform-checkpoint.zip"
        )
        restore_longform_checkpoint(
            storage,
            checkpoint_key,
            output_root,
            root,
        )
        storage.download(source_key, input_path)
        cache_key = f"users/{user_id}/videos/{job.video_id}/cache/subgen.db"
        if storage.exists(cache_key):
            storage.download(cache_key, data_root / "subgen.db")
        digest = sha256_file(input_path)
        duration_seconds = probe_duration(input_path)
        if duration_seconds > settings.max_video_minutes * 60:
            raise RuntimeError(
                f"Video is {duration_seconds / 60:.1f} minutes; this deployment limit is "
                f"{settings.max_video_minutes} minutes to control compute cost."
            )
        if not settings.enable_gpu_worker:
            pipeline_config["device"] = "cpu"
        with session_factory.begin() as session:
            video = session.get(Video, job.video_id)
            if not video.sha256:
                video.sha256 = digest
            video.duration_seconds = round(duration_seconds)
            job = session.get(Job, job_id)
            job.stage = "transcription"
            job.progress = 10

        env = os.environ.copy()
        env["SUBGEN_DATA_DIR"] = str(data_root)
        request = {
            "video_path": str(input_path),
            "output_dir": str(output_root),
            "target_language": target_language,
            "pipeline_config": pipeline_config,
            "style_config": style_config,
            "cache_action": cache_action,
            "no_burn": True,
            "source_location": {
                "kind": "cloud_object",
                "object_key": source_key,
                "name": filename,
                "video_id": job.video_id,
            },
        }
        with decrypted_provider_environment(session_factory, cipher, user_id, pipeline_config):
            for env_key in PROVIDER_ENV_KEYS.values():
                if os.environ.get(env_key):
                    env[env_key] = os.environ[env_key]
            checkpoint_state = {"signature": None}

            def heartbeat():
                active = refresh_job_lease(
                    session_factory,
                    job_id,
                    worker_id,
                    settings.worker_lease_seconds,
                )
                if active:
                    checkpoint_state["signature"] = upload_longform_checkpoint(
                        storage,
                        checkpoint_key,
                        output_root,
                        root,
                        checkpoint_state["signature"],
                    )
                return active

            run_pipeline_subprocess(
                request,
                root,
                env,
                heartbeat=heartbeat,
                cancelled=lambda: job_is_cancelled(session_factory, job_id),
                heartbeat_seconds=settings.worker_heartbeat_seconds,
                timeout_seconds=settings.worker_pipeline_timeout_seconds,
            )
        cache_db = data_root / "subgen.db"
        if cache_db.exists():
            storage.upload_file(cache_db, cache_key, "application/vnd.sqlite3")

        source_srt, target_srt = find_source_and_target_srts(output_root, target_language)
        if not target_srt or not target_srt.exists():
            raise RuntimeError("Pipeline completed without a subtitle file")
        review_segments = parse_srt(target_srt.read_text(encoding="utf-8"))
        if not review_segments:
            raise RuntimeError("Pipeline produced an empty or invalid subtitle file")
        review_paths = sorted(output_root.glob("*.review.json"))
        if not review_paths:
            raise RuntimeError("Pipeline completed without a durable review manifest")
        review_path = review_paths[-1]
        review = load_review(review_path)
        review["video_id"] = job_id
        review["source_hash"] = digest
        review["source_location"] = {"kind": "cloud_object", "object_key": source_key, "name": filename}
        save_review(review_path, review)
        review_segments = combined_review_cues(review)
        provenance = {
            "video_sha256": digest,
            "pipeline_plan": pipeline_config.get("_pipeline_plan"),
            "cache_action": cache_action,
        }
        with session_factory.begin() as session:
            job = session.get(Job, job_id)
            source_artifact = add_artifact(
                session, storage, job, source_srt, "source_srt", None, provenance=provenance
            )
            if target_srt.resolve() != source_srt.resolve():
                add_artifact(
                    session,
                    storage,
                    job,
                    target_srt,
                    "target_srt",
                    target_language,
                    parent_id=source_artifact.id,
                    provenance=provenance,
                )
            log_path = root / "pipeline.log"
            add_artifact(session, storage, job, log_path, "pipeline_log", provenance=provenance)
            add_artifact(session, storage, job, review_path, "review_manifest", provenance=provenance)
            job.review_segments = review_segments
            job.review_manifest = review
            job.status = (
                JobStatus.NEEDS_ATTENTION.value
                if review.get("state") == "NEEDS_ATTENTION"
                else JobStatus.WAITING_FOR_REVIEW.value
            )
            job.stage = "review"
            job.progress = 100
            job.lease_owner = None
            job.lease_until = None


def process_burn_job(
    job_id,
    settings,
    session_factory,
    storage,
    cipher,
    worker_id,
):
    with session_factory() as session:
        job = session.get(Job, job_id)
        video = session.get(Video, job.video_id)
        source_key = video.object_key
        filename = video.original_name
        review = dict(job.review_manifest or {})
        pipeline_config = dict(job.pipeline_config or {})
        style_config = dict(job.style_config or {})
        target_language = job.target_language
        user_id = job.user_id
    if not review:
        raise RuntimeError("Cannot burn a job without approved subtitle cues")

    with tempfile.TemporaryDirectory(prefix=f"subgen-burn-{job_id[:8]}-") as temporary:
        root = Path(temporary)
        input_path = root / "input" / filename
        output_root = root / "output"
        data_root = root / "data"
        output_root.mkdir(parents=True)
        data_root.mkdir(parents=True)
        storage.download(source_key, input_path)
        burn_gate = assert_burn_allowed(review, input_path)
        srt_path = output_root / f"reviewed.{target_language or 'source'}.srt"
        srt_path.write_bytes(burn_gate["subtitle_bytes"])
        review_path = output_root / "approved.review.json"
        save_review(review_path, review)
        env = os.environ.copy()
        env["SUBGEN_DATA_DIR"] = str(data_root)
        request = {
            "video_path": str(input_path),
            "srt_path": str(srt_path),
            "output_dir": str(output_root),
            "target_language": target_language,
            "pipeline_config": pipeline_config,
            "style_config": style_config,
            "cache_action": "reburn",
            "force": True,
            "no_burn": False,
            "approved_review_path": str(review_path),
        }
        with decrypted_provider_environment(session_factory, cipher, user_id, pipeline_config):
            output_path = Path(run_pipeline_subprocess(
                request,
                root,
                env,
                heartbeat=lambda: refresh_job_lease(
                    session_factory,
                    job_id,
                    worker_id,
                    settings.worker_lease_seconds,
                ),
                cancelled=lambda: job_is_cancelled(session_factory, job_id),
                heartbeat_seconds=settings.worker_heartbeat_seconds,
                timeout_seconds=settings.worker_pipeline_timeout_seconds,
            ))
        if not output_path.exists() or output_path.suffix.lower() != ".mp4":
            raise RuntimeError("Burn stage completed without an MP4 output")
        with session_factory.begin() as session:
            job = session.get(Job, job_id)
            reviewed = add_artifact(session, storage, job, srt_path, "reviewed_srt", target_language)
            add_artifact(
                session, storage, job, output_path, "output_video", target_language, parent_id=reviewed.id
            )
            add_artifact(session, storage, job, root / "pipeline.log", "burn_log")
            completed_review = load_review(review_path)
            add_artifact(session, storage, job, review_path, "completed_review_manifest")
            job.review_manifest = completed_review
            job.status = JobStatus.COMPLETED.value
            job.stage = "completed"
            job.progress = 100
            job.lease_owner = None
            job.lease_until = None


def fail_job(session_factory, job_id, error):
    with session_factory.begin() as session:
        job = session.get(Job, job_id)
        if job and job.status != JobStatus.CANCELLED.value:
            burn_failure = job.status in {JobStatus.BURNING.value, JobStatus.BURN_QUEUED.value}
            job.status = JobStatus.BURN_FAILED.value if burn_failure else JobStatus.FAILED.value
            job.stage = "burn_failed" if burn_failure else "failed"
            job.error_message = str(error)[-8000:]
            job.lease_owner = None
            job.lease_until = None


def run_worker(once=False):
    settings = load_settings()
    settings.validate()
    if not settings.processing_enabled:
        raise RuntimeError("Cloud processing is disabled by the deployment safety switch")
    if not settings.enable_pipeline_worker:
        raise RuntimeError("Set SUBGEN_ENABLE_PIPELINE_WORKER=true to run cloud jobs")
    _, session_factory = create_database(settings.database_url)
    storage = create_storage(settings)
    cipher = CredentialCipher(settings.credential_master_key, allow_development_key=settings.dev_auth_enabled)
    worker_id = f"{socket.gethostname()}:{os.getpid()}"
    while True:
        job_id = claim_job(session_factory, worker_id, settings.worker_lease_seconds)
        if not job_id:
            if once:
                return
            time.sleep(settings.worker_poll_seconds)
            continue
        try:
            with session_factory() as session:
                status_value = session.get(Job, job_id).status
            if status_value == JobStatus.BURNING.value:
                process_burn_job(
                    job_id,
                    settings,
                    session_factory,
                    storage,
                    cipher,
                    worker_id,
                )
            else:
                process_transcription_job(
                    job_id,
                    settings,
                    session_factory,
                    storage,
                    cipher,
                    worker_id,
                )
        except Exception as exc:
            fail_job(session_factory, job_id, exc)
        if once:
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    arguments = parser.parse_args()
    run_worker(once=arguments.once)
