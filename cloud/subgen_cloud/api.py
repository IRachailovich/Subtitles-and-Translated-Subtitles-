import json
import math
import mimetypes
import os
import re
import subprocess
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select

from .config import load_settings
from .dispatch import dispatch_worker
from .db import (
    ACTIVE_JOB_STATUSES,
    Artifact,
    Job,
    JobStatus,
    ProviderCredential,
    Upload,
    UserSettings,
    Video,
    create_database,
    new_id,
)
from .schemas import (
    ApproveRequest,
    CredentialUpdate,
    JobCreate,
    PartRequest,
    ReviewUpdate,
    RetranslateRequest,
    SettingsUpdate,
    UploadComplete,
    UploadInitiate,
    IssueResolution,
)
from .security import CredentialCipher, TokenVerifier, install_auth_dependencies
from .storage import LocalObjectStorage, create_storage
from subgen_review import (
    approve_review,
    combined_review_cues,
    new_review,
    resolve_issue,
    set_ready_for_review,
    update_review_from_combined,
)
from . import __version__


SAFE_CONFIG_KEY = re.compile(r"^[a-zA-Z0-9_.-]{1,120}$")
SECRET_WORDS = ("api_key", "apikey", "secret", "password", "credential", "access_token", "refresh_token")


def durable_review_for_job(job):
    review = dict(job.review_manifest or {})
    if review:
        return review
    if not job.review_segments:
        return {}
    video = job.video
    review = new_review(
        job.id,
        (video.sha256 if video else None) or "legacy-source-hash-unavailable",
        source_language=None,
        target_language=None,
        provider=(job.pipeline_config or {}).get("transcription_provider"),
        model=(job.pipeline_config or {}).get("transcription_model"),
        prompt_version=(job.pipeline_config or {}).get("google_transcription_prompt_version"),
        source_draft=job.review_segments,
        source_location={"kind": "cloud_object", "object_key": video.object_key if video else None},
    )
    set_ready_for_review(review)
    return review


def reject_embedded_secrets(value, path="config"):
    if isinstance(value, dict):
        for key, child in value.items():
            if not SAFE_CONFIG_KEY.match(str(key)):
                raise HTTPException(status_code=400, detail=f"Invalid configuration key at {path}")
            normalized = str(key).lower()
            if any(word in normalized for word in SECRET_WORDS):
                raise HTTPException(status_code=400, detail="API keys and secrets must use the encrypted Credentials section")
            reject_embedded_secrets(child, f"{path}.{key}")
    elif isinstance(value, list):
        for child in value:
            reject_embedded_secrets(child, path)


def user_object_key(user_id, video_id, filename):
    clean_name = "".join(char for char in Path(filename).name if char.isalnum() or char in "._- ").strip()
    clean_name = clean_name or "video"
    return f"users/{user_id}/videos/{video_id}/source/{clean_name}"


def serialize_job(job):
    return {
        "id": job.id,
        "video_id": job.video_id,
        "target_language": job.target_language,
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
        "error_message": job.error_message,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }


def create_app(settings=None, session_factory=None, storage=None):
    settings = settings or load_settings()
    if session_factory is None:
        _, session_factory = create_database(settings.database_url)
    storage = storage or create_storage(settings)
    verifier = TokenVerifier(settings)
    cipher = CredentialCipher(settings.credential_master_key, allow_development_key=settings.dev_auth_enabled)

    @asynccontextmanager
    async def lifespan(_app):
        settings.validate()
        yield

    app = FastAPI(title="SubGen Cloud API", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-SubGen-Dev-User"],
        expose_headers=["ETag"],
    )
    current_user = install_auth_dependencies(app, session_factory, verifier)
    app.state.settings = settings
    app.state.session_factory = session_factory
    app.state.storage = storage
    app.state.cipher = cipher

    def dispatch_queued_job(job_id):
        try:
            dispatch_worker(settings)
        except RuntimeError as exc:
            with session_factory.begin() as session:
                job = session.get(Job, job_id)
                if job and job.status in {JobStatus.QUEUED.value, JobStatus.BURN_QUEUED.value}:
                    job.status = JobStatus.FAILED.value
                    job.stage = "dispatch_failed"
                    job.error_message = str(exc)
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "storage": settings.storage_backend,
            "pipeline_worker": settings.enable_pipeline_worker,
            "gpu_worker": settings.enable_gpu_worker,
            "processing_enabled": settings.processing_enabled,
            "worker_dispatch_mode": settings.worker_dispatch_mode,
        }

    @app.get("/v1/public-config")
    def public_config():
        return {
            "auth_url": settings.auth_public_url,
            "auth_public_key": settings.auth_public_key,
            "development_auth": settings.dev_auth_enabled,
            "max_upload_bytes": settings.max_upload_bytes,
            "part_size": settings.upload_part_bytes,
            "retention_days": settings.artifact_retention_days,
            "gpu_enabled": settings.enable_gpu_worker,
            "processing_enabled": settings.processing_enabled,
            "max_video_minutes": settings.max_video_minutes,
            "max_jobs_per_user_day": settings.max_jobs_per_user_day,
            "max_jobs_per_user_month": settings.max_jobs_per_user_month,
        }

    @app.get("/v1/me")
    def me(user=Depends(current_user)):
        return user

    @app.get("/v1/settings")
    def get_settings(user=Depends(current_user)):
        with session_factory() as session:
            row = session.get(UserSettings, user["id"])
            return {"revision": row.revision if row else 0, "config": row.config if row else {}}

    @app.put("/v1/settings")
    def update_settings(payload: SettingsUpdate, user=Depends(current_user)):
        reject_embedded_secrets(payload.config)
        with session_factory.begin() as session:
            row = session.get(UserSettings, user["id"])
            if row and payload.revision is not None and row.revision != payload.revision:
                raise HTTPException(status_code=409, detail="Settings changed on another device; reload before saving")
            if row:
                row.config = payload.config
                row.revision += 1
            else:
                row = UserSettings(user_id=user["id"], config=payload.config, revision=1)
                session.add(row)
            session.flush()
            return {"revision": row.revision, "config": row.config}

    @app.get("/v1/credentials")
    def list_credentials(user=Depends(current_user)):
        with session_factory() as session:
            rows = session.scalars(select(ProviderCredential).where(ProviderCredential.user_id == user["id"]))
            return [{
                "provider": row.provider,
                "profile": row.profile,
                "configured": True,
                "updated_at": row.updated_at,
            } for row in rows]

    @app.put("/v1/credentials")
    def save_credential(payload: CredentialUpdate, user=Depends(current_user)):
        encrypted = cipher.encrypt(user["id"], payload.provider, payload.profile, payload.api_key)
        with session_factory.begin() as session:
            row = session.scalar(select(ProviderCredential).where(
                ProviderCredential.user_id == user["id"],
                ProviderCredential.provider == payload.provider,
                ProviderCredential.profile == payload.profile,
            ))
            if row:
                row.ciphertext = encrypted
            else:
                session.add(ProviderCredential(
                    user_id=user["id"],
                    provider=payload.provider,
                    profile=payload.profile,
                    ciphertext=encrypted,
                ))
        return {"provider": payload.provider, "profile": payload.profile, "configured": True}

    @app.delete("/v1/credentials/{provider}/{profile}", status_code=204)
    def delete_credential(provider: str, profile: str, user=Depends(current_user)):
        with session_factory.begin() as session:
            row = session.scalar(select(ProviderCredential).where(
                ProviderCredential.user_id == user["id"],
                ProviderCredential.provider == provider,
                ProviderCredential.profile == profile,
            ))
            if row:
                session.delete(row)
        return Response(status_code=204)

    @app.post("/v1/uploads/initiate")
    def initiate_upload(payload: UploadInitiate, user=Depends(current_user)):
        if payload.size_bytes > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="Video exceeds this deployment's upload limit")
        with session_factory.begin() as session:
            if payload.sha256:
                existing = session.scalar(select(Video).where(
                    Video.user_id == user["id"], Video.sha256 == payload.sha256.lower(), Video.upload_complete.is_(True)
                ))
                if existing and storage.exists(existing.object_key):
                    return {"video_id": existing.id, "reused": True, "complete": True}
                if existing:
                    existing.upload_complete = False
            video = Video(
                user_id=user["id"],
                sha256=payload.sha256.lower() if payload.sha256 else None,
                original_name=payload.filename,
                size_bytes=payload.size_bytes,
                object_key="pending",
            )
            session.add(video)
            session.flush()
            video.object_key = user_object_key(user["id"], video.id, payload.filename)
            upload_id = new_id()
            storage_upload_id = storage.initiate_upload(video.object_key, upload_id)
            upload = Upload(
                id=upload_id,
                user_id=user["id"],
                video_id=video.id,
                storage_upload_id=storage_upload_id,
                part_size=settings.upload_part_bytes,
            )
            session.add(upload)
            return {
                "upload_id": upload.id,
                "video_id": video.id,
                "part_size": upload.part_size,
                "part_count": math.ceil(payload.size_bytes / upload.part_size),
                "reused": False,
            }

    @app.post("/v1/uploads/{upload_id}/part-url")
    def get_part_url(upload_id: str, payload: PartRequest, user=Depends(current_user)):
        with session_factory() as session:
            upload = session.scalar(select(Upload).where(Upload.id == upload_id, Upload.user_id == user["id"]))
            if not upload or upload.completed:
                raise HTTPException(status_code=404, detail="Active upload not found")
            video = session.get(Video, upload.video_id)
            return {"url": storage.part_url(upload.storage_upload_id, payload.part_number, video.object_key)}

    @app.put("/v1/uploads/{upload_id}/parts/{part_number}")
    async def local_upload_part(upload_id: str, part_number: int, request: Request, user=Depends(current_user)):
        if not isinstance(storage, LocalObjectStorage):
            raise HTTPException(status_code=404, detail="Direct object-storage URL required")
        with session_factory() as session:
            upload = session.scalar(select(Upload).where(Upload.id == upload_id, Upload.user_id == user["id"]))
            if not upload or upload.completed:
                raise HTTPException(status_code=404, detail="Active upload not found")
            body = await request.body()
            if len(body) > upload.part_size:
                raise HTTPException(status_code=413, detail="Upload part is too large")
            etag = storage.write_part(upload.storage_upload_id, part_number, body)
            return Response(status_code=200, headers={"ETag": etag})

    @app.post("/v1/uploads/{upload_id}/complete")
    def complete_upload(upload_id: str, payload: UploadComplete, user=Depends(current_user)):
        with session_factory.begin() as session:
            upload = session.scalar(select(Upload).where(Upload.id == upload_id, Upload.user_id == user["id"]))
            if not upload or upload.completed:
                raise HTTPException(status_code=404, detail="Active upload not found")
            video = session.get(Video, upload.video_id)
            parts = [part.model_dump() for part in payload.parts]
            expected_count = math.ceil(video.size_bytes / upload.part_size)
            part_numbers = sorted(part["part_number"] for part in parts)
            if part_numbers != list(range(1, expected_count + 1)):
                raise HTTPException(status_code=400, detail="Upload parts are incomplete or duplicated")
            storage.complete_upload(video.object_key, upload.storage_upload_id, parts)
            if storage.object_size(video.object_key) != video.size_bytes:
                storage.delete(video.object_key)
                raise HTTPException(status_code=400, detail="Completed upload size does not match the selected video")
            upload.completed = True
            video.upload_complete = True
            return {"video_id": video.id, "complete": True}

    @app.post("/v1/jobs", status_code=201)
    def create_job(payload: JobCreate, user=Depends(current_user)):
        reject_embedded_secrets(payload.pipeline_config, "pipeline_config")
        result = None
        queued_job_id = None
        with session_factory.begin() as session:
            video = session.scalar(select(Video).where(Video.id == payload.video_id, Video.user_id == user["id"]))
            if not video or not video.upload_complete:
                raise HTTPException(status_code=404, detail="Uploaded video not found")
            if payload.cache_action == "reuse_all":
                candidates = session.scalars(select(Job).where(
                    Job.user_id == user["id"],
                    Job.video_id == video.id,
                    Job.target_language == payload.target_language,
                    Job.status == JobStatus.COMPLETED.value,
                ).order_by(Job.created_at.desc()).limit(20))
                for candidate in candidates:
                    if candidate.pipeline_config != payload.pipeline_config or candidate.style_config != payload.style_config:
                        continue
                    output = session.scalar(select(Artifact).where(
                        Artifact.job_id == candidate.id,
                        Artifact.user_id == user["id"],
                        Artifact.kind == "output_video",
                    ))
                    if output and storage.exists(output.object_key):
                        result = serialize_job(candidate)
                        result["reused"] = True
                        return result
            if not settings.processing_enabled:
                raise HTTPException(
                    status_code=503,
                    detail="Cloud processing is safely disabled by the deployment owner; cached completed outputs remain available.",
                )
            active = session.scalar(select(func.count()).select_from(Job).where(
                Job.user_id == user["id"], Job.status.in_(ACTIVE_JOB_STATUSES)
            ))
            if active >= settings.max_active_jobs_per_user:
                raise HTTPException(status_code=429, detail="Finish or cancel the current job before starting another")
            global_active = session.scalar(select(func.count()).select_from(Job).where(
                Job.status.in_(ACTIVE_JOB_STATUSES)
            ))
            if global_active >= settings.max_active_jobs_global:
                raise HTTPException(status_code=429, detail="The zero-cost worker is busy; try again after the active job finishes")
            cutoff = datetime.now(timezone.utc) - timedelta(days=1)
            user_daily = session.scalar(select(func.count()).select_from(Job).where(
                Job.user_id == user["id"], Job.created_at >= cutoff
            ))
            if user_daily >= settings.max_jobs_per_user_day:
                raise HTTPException(status_code=429, detail="This account reached the deployment's 24-hour processing limit")
            global_daily = session.scalar(select(func.count()).select_from(Job).where(Job.created_at >= cutoff))
            if global_daily >= settings.max_jobs_global_day:
                raise HTTPException(status_code=429, detail="This deployment reached its 24-hour processing safety limit")
            month_cutoff = datetime.now(timezone.utc) - timedelta(days=30)
            user_monthly = session.scalar(select(func.count()).select_from(Job).where(
                Job.user_id == user["id"], Job.created_at >= month_cutoff
            ))
            if user_monthly >= settings.max_jobs_per_user_month:
                raise HTTPException(status_code=429, detail="This account reached the deployment's 30-day processing limit")
            global_monthly = session.scalar(select(func.count()).select_from(Job).where(
                Job.created_at >= month_cutoff
            ))
            if global_monthly >= settings.max_jobs_global_month:
                raise HTTPException(status_code=429, detail="This deployment reached its 30-day processing safety limit")
            job = Job(
                user_id=user["id"],
                video_id=video.id,
                target_language=payload.target_language,
                pipeline_config=payload.pipeline_config,
                style_config=payload.style_config,
                cache_action=payload.cache_action,
            )
            session.add(job)
            session.flush()
            queued_job_id = job.id
            result = serialize_job(job)
        dispatch_queued_job(queued_job_id)
        return result

    @app.get("/v1/videos")
    def list_videos(user=Depends(current_user)):
        with session_factory.begin() as session:
            videos = list(session.scalars(select(Video).where(
                Video.user_id == user["id"], Video.upload_complete.is_(True)
            ).order_by(Video.created_at.desc()).limit(100)))
            available = []
            for video in videos:
                if storage.exists(video.object_key):
                    available.append(video)
                else:
                    video.upload_complete = False
            return [{
                "id": video.id,
                "name": video.original_name,
                "size_bytes": video.size_bytes,
                "sha256": video.sha256,
                "created_at": video.created_at,
            } for video in available]

    @app.get("/v1/jobs")
    def list_jobs(user=Depends(current_user)):
        with session_factory() as session:
            jobs = session.scalars(select(Job).where(Job.user_id == user["id"]).order_by(Job.created_at.desc()).limit(100))
            return [serialize_job(job) for job in jobs]

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str, user=Depends(current_user)):
        with session_factory() as session:
            job = session.scalar(select(Job).where(Job.id == job_id, Job.user_id == user["id"]))
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")
            result = serialize_job(job)
            result["artifacts"] = [{
                "id": artifact.id,
                "kind": artifact.kind,
                "language": artifact.language,
            } for artifact in session.scalars(select(Artifact).where(Artifact.job_id == job.id, Artifact.user_id == user["id"]))]
            return result

    @app.delete("/v1/jobs/{job_id}", status_code=204)
    def cancel_job(job_id: str, user=Depends(current_user)):
        with session_factory.begin() as session:
            job = session.scalar(select(Job).where(Job.id == job_id, Job.user_id == user["id"]))
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")
            if job.status in {JobStatus.COMPLETED.value, JobStatus.FAILED.value}:
                raise HTTPException(status_code=409, detail="Finished jobs cannot be cancelled")
            job.status = JobStatus.CANCELLED.value
            job.stage = "cancelled"
        return Response(status_code=204)

    @app.get("/v1/jobs/{job_id}/review")
    def get_review(job_id: str, user=Depends(current_user)):
        with session_factory() as session:
            job = session.scalar(select(Job).where(Job.id == job_id, Job.user_id == user["id"]))
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")
            had_durable_review = bool(job.review_manifest)
            review = durable_review_for_job(job)
            return {
                "status": job.status,
                "segments": combined_review_cues(review) if had_durable_review else (job.review_segments or []),
                "review": review,
                "issues": review.get("issues") or [],
                "approval": review.get("approval"),
            }

    @app.put("/v1/jobs/{job_id}/review")
    def save_review(job_id: str, payload: ReviewUpdate, user=Depends(current_user)):
        with session_factory.begin() as session:
            job = session.scalar(select(Job).where(Job.id == job_id, Job.user_id == user["id"]))
            editable_states = {
                JobStatus.WAITING_FOR_REVIEW.value, JobStatus.NEEDS_ATTENTION.value,
                JobStatus.IN_REVIEW.value, JobStatus.STALE_AFTER_EDIT.value,
                JobStatus.APPROVED.value,
            }
            if not job or job.status not in editable_states:
                raise HTTPException(status_code=409, detail="Job is not waiting for review")
            review = durable_review_for_job(job)
            if not review:
                raise HTTPException(status_code=409, detail="Durable review manifest is unavailable")
            update_review_from_combined(
                review, payload.segments, actor=f"cloud_user:{user['id']}",
                translation_confirmed=payload.translation_confirmed,
            )
            job.review_manifest = review
            job.review_segments = combined_review_cues(review)
            job.approved_draft_hash = None
            job.approved_revision_hash = None
            job.approved_at = None
            job.status = JobStatus.STALE_AFTER_EDIT.value
            job.stage = "review"
        return {"saved": True, "segments": len(payload.segments), "review": review}

    @app.post("/v1/jobs/{job_id}/review/retranslate")
    def retranslate_review(
        job_id: str,
        payload: RetranslateRequest,
        user=Depends(current_user),
    ):
        with session_factory() as session:
            job = session.scalar(select(Job).where(
                Job.id == job_id, Job.user_id == user["id"]
            ))
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")
            review = durable_review_for_job(job)
            initial_revision = review.get("revision_hash")
            pipeline_config = dict(job.pipeline_config or {})
            provider = pipeline_config.get("translation_provider", "local")
            credential = None
            if provider != "local":
                profile = (
                    (pipeline_config.get("credential_profiles") or {}).get(provider)
                    or "default"
                )
                credential = session.scalar(select(ProviderCredential).where(
                    ProviderCredential.user_id == user["id"],
                    ProviderCredential.provider == provider,
                    ProviderCredential.profile == profile,
                ))
                if not credential:
                    raise HTTPException(
                        status_code=409,
                        detail=f"No authorized {provider} credential profile is available",
                    )
        environment = os.environ.copy()
        if credential:
            from subgen_providers import get_provider_registry

            provider_definition = get_provider_registry(pipeline_config).get(provider) or {}
            env_key = provider_definition.get("env_key")
            if not env_key:
                raise HTTPException(
                    status_code=409,
                    detail=f"Provider {provider} has no credential environment mapping",
                )
            environment[env_key] = cipher.decrypt(
                user["id"],
                credential.provider,
                credential.profile,
                credential.ciphertext,
            )
        request = {
            "review": review,
            "cue_ids": payload.cue_ids,
            "pipeline_config": pipeline_config,
            "actor": f"cloud_user:{user['id']}",
        }
        completed = subprocess.run(
            [sys.executable, "-m", "cloud.subgen_cloud.retranslate_runner"],
            input=json.dumps(request, ensure_ascii=False),
            text=True,
            capture_output=True,
            env=environment,
        )
        marker = "SUBGEN_RETRANSLATION_RESULT="
        output_line = next(
            (
                line[len(marker):]
                for line in reversed(completed.stdout.splitlines())
                if line.startswith(marker)
            ),
            None,
        )
        if completed.returncode or not output_line:
            raise HTTPException(
                status_code=502,
                detail=(
                    "Selected-cue translation worker failed: "
                    + (completed.stderr.strip() or "no result returned")
                )[:2000],
            )
        result = json.loads(output_line)
        updated_review = result["review"]
        with session_factory.begin() as session:
            job = session.scalar(select(Job).where(
                Job.id == job_id, Job.user_id == user["id"]
            ))
            current = durable_review_for_job(job)
            if current.get("revision_hash") != initial_revision:
                raise HTTPException(
                    status_code=409,
                    detail="Review changed while selected cues were being translated",
                )
            job.review_manifest = updated_review
            job.review_segments = combined_review_cues(updated_review)
            job.approved_draft_hash = None
            job.approved_revision_hash = None
            job.approved_at = None
            job.status = JobStatus.STALE_AFTER_EDIT.value
            job.stage = "review"
        return {
            "review": updated_review,
            "segments": combined_review_cues(updated_review),
            "retranslation": result["retranslation"],
        }

    @app.post("/v1/jobs/{job_id}/issues/{issue_id}/resolve")
    def resolve_review_issue(job_id: str, issue_id: str, payload: IssueResolution, user=Depends(current_user)):
        with session_factory.begin() as session:
            job = session.scalar(select(Job).where(Job.id == job_id, Job.user_id == user["id"]))
            if not job or job.status not in {
                JobStatus.WAITING_FOR_REVIEW.value, JobStatus.NEEDS_ATTENTION.value,
                JobStatus.IN_REVIEW.value, JobStatus.STALE_AFTER_EDIT.value,
            }:
                raise HTTPException(status_code=409, detail="Job is not in review")
            review = durable_review_for_job(job)
            try:
                resolve_issue(
                    review, issue_id, payload.status,
                    actor=f"cloud_user:{user['id']}", reason=payload.reason,
                )
            except (KeyError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            job.review_manifest = review
            job.status = JobStatus.IN_REVIEW.value
            job.approved_draft_hash = None
            job.approved_revision_hash = None
            job.approved_at = None
        return {"review": review}

    @app.post("/v1/jobs/{job_id}/approve")
    def approve_job(job_id: str, payload: ApproveRequest, user=Depends(current_user)):
        if not settings.processing_enabled:
            raise HTTPException(status_code=503, detail="Cloud processing is safely disabled by the deployment owner")
        with session_factory.begin() as session:
            job = session.scalar(select(Job).where(Job.id == job_id, Job.user_id == user["id"]))
            if not job or job.status not in {
                JobStatus.WAITING_FOR_REVIEW.value, JobStatus.NEEDS_ATTENTION.value,
                JobStatus.IN_REVIEW.value, JobStatus.STALE_AFTER_EDIT.value,
            }:
                raise HTTPException(status_code=409, detail="Job is not waiting for review")
            review = durable_review_for_job(job)
            if not review:
                raise HTTPException(status_code=409, detail="No reviewed subtitles are available")
            try:
                if payload.translation_confirmed:
                    update_review_from_combined(
                        review, combined_review_cues(review), actor=f"cloud_user:{user['id']}",
                        translation_confirmed=True,
                    )
                approve_review(
                    review, actor=f"cloud_user:{user['id']}",
                    accept_warnings=payload.accept_warnings,
                )
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            if payload.style_config is not None:
                job.style_config = payload.style_config
            job.review_manifest = review
            job.review_segments = combined_review_cues(review)
            job.approved_draft_hash = review["approval"]["approved_draft_hash"]
            job.approved_revision_hash = review["approval"]["approved_revision_hash"]
            job.approved_at = datetime.now(timezone.utc)
            job.status = JobStatus.APPROVED.value
            job.stage = "approved"
            result = serialize_job(job)
        return result

    @app.post("/v1/jobs/{job_id}/burn")
    def burn_approved_job(job_id: str, user=Depends(current_user)):
        """Queue a distinct burn operation for the exact approved review revision."""
        if not settings.processing_enabled:
            raise HTTPException(status_code=503, detail="Cloud processing is safely disabled by the deployment owner")
        with session_factory.begin() as session:
            job = session.scalar(select(Job).where(Job.id == job_id, Job.user_id == user["id"]))
            if not job or job.status not in {JobStatus.APPROVED.value, JobStatus.BURN_FAILED.value}:
                raise HTTPException(status_code=409, detail="Job is not approved for burn")
            review = durable_review_for_job(job)
            approval = review.get("approval") or {}
            if not approval.get("approved_draft_hash"):
                raise HTTPException(status_code=409, detail="Approved subtitle hash is unavailable")
            if approval.get("approved_draft_hash") != job.approved_draft_hash:
                raise HTTPException(status_code=409, detail="Stored approval hash does not match the review manifest")
            job.status = JobStatus.BURN_QUEUED.value
            job.stage = "burn_queued"
            job.progress = 0
            job.error_message = None
            result = serialize_job(job)
        dispatch_queued_job(job_id)
        return result

    @app.get("/v1/artifacts/{artifact_id}/download")
    def download_artifact(artifact_id: str, user=Depends(current_user)):
        with session_factory() as session:
            artifact = session.scalar(select(Artifact).where(
                Artifact.id == artifact_id, Artifact.user_id == user["id"]
            ))
            if not artifact:
                raise HTTPException(status_code=404, detail="Artifact not found")
            url = (
                f"{settings.public_base_url.rstrip('/')}/v1/artifacts/{artifact.id}/content"
                if isinstance(storage, LocalObjectStorage)
                else storage.signed_download_url(artifact.object_key)
            )
            return {"url": url, "expires_in": 900}

    @app.get("/v1/artifacts/{artifact_id}/content")
    def local_artifact_content(artifact_id: str, user=Depends(current_user)):
        if not isinstance(storage, LocalObjectStorage):
            raise HTTPException(status_code=404, detail="Object not found")
        with session_factory() as session:
            artifact = session.scalar(select(Artifact).where(
                Artifact.id == artifact_id, Artifact.user_id == user["id"]
            ))
            if not artifact:
                raise HTTPException(status_code=404, detail="Artifact not found")
            path = storage.open_object(artifact.object_key)
            if not path.exists():
                raise HTTPException(status_code=404, detail="Object not found")
            return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0], filename=path.name)

    @app.get("/v1/local-objects/{object_key:path}")
    def local_object(object_key: str, user=Depends(current_user)):
        if not isinstance(storage, LocalObjectStorage) or not object_key.startswith(f"users/{user['id']}/"):
            raise HTTPException(status_code=404, detail="Object not found")
        path = storage.open_object(object_key)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Object not found")
        return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0])

    static_root = Path(__file__).resolve().parents[2] / "cloud_web"
    if static_root.exists():
        assets_root = Path(__file__).resolve().parents[2] / "web" / "assets"
        if assets_root.exists():
            app.mount("/assets", StaticFiles(directory=assets_root), name="assets")
        app.mount("/", StaticFiles(directory=static_root, html=True), name="cloud-web")
    return app
