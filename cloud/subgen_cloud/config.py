import base64
import os
from dataclasses import dataclass, field
from pathlib import Path


def _boolean(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _integer(name, default):
    value = os.environ.get(name)
    return int(value) if value else default


def _csv(name, default=""):
    return tuple(item.strip() for item in os.environ.get(name, default).split(",") if item.strip())


@dataclass(frozen=True)
class CloudSettings:
    database_url: str = field(default_factory=lambda: os.environ.get(
        "SUBGEN_CLOUD_DATABASE_URL", "sqlite:///./subgen-cloud.db"
    ))
    auth_issuer: str = field(default_factory=lambda: os.environ.get("SUBGEN_AUTH_ISSUER", ""))
    auth_audience: str = field(default_factory=lambda: os.environ.get("SUBGEN_AUTH_AUDIENCE", "authenticated"))
    auth_jwks_url: str = field(default_factory=lambda: os.environ.get("SUBGEN_AUTH_JWKS_URL", ""))
    auth_public_url: str = field(default_factory=lambda: os.environ.get("SUBGEN_AUTH_PUBLIC_URL", ""))
    auth_public_key: str = field(default_factory=lambda: os.environ.get("SUBGEN_AUTH_PUBLIC_KEY", ""))
    dev_auth_enabled: bool = field(default_factory=lambda: _boolean("SUBGEN_DEV_AUTH", False))
    credential_master_key: str = field(default_factory=lambda: os.environ.get("SUBGEN_CREDENTIAL_MASTER_KEY", ""))
    storage_backend: str = field(default_factory=lambda: os.environ.get("SUBGEN_STORAGE_BACKEND", "local"))
    storage_endpoint: str = field(default_factory=lambda: os.environ.get("SUBGEN_STORAGE_ENDPOINT", ""))
    storage_region: str = field(default_factory=lambda: os.environ.get("SUBGEN_STORAGE_REGION", "auto"))
    storage_bucket: str = field(default_factory=lambda: os.environ.get("SUBGEN_STORAGE_BUCKET", "subgen"))
    storage_access_key: str = field(default_factory=lambda: os.environ.get("SUBGEN_STORAGE_ACCESS_KEY", ""))
    storage_secret_key: str = field(default_factory=lambda: os.environ.get("SUBGEN_STORAGE_SECRET_KEY", ""))
    local_storage_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("SUBGEN_CLOUD_STORAGE_DIR", "./cloud-data")
    ).resolve())
    public_base_url: str = field(default_factory=lambda: os.environ.get("SUBGEN_PUBLIC_BASE_URL", "http://localhost:8000"))
    cors_origins: tuple[str, ...] = field(default_factory=lambda: _csv(
        "SUBGEN_CORS_ORIGINS", "http://localhost:8000,http://localhost:8080"
    ))
    max_upload_bytes: int = field(default_factory=lambda: _integer("SUBGEN_MAX_UPLOAD_BYTES", 8 * 1024**3))
    max_active_jobs_per_user: int = field(default_factory=lambda: _integer("SUBGEN_MAX_ACTIVE_JOBS", 1))
    max_active_jobs_global: int = field(default_factory=lambda: _integer("SUBGEN_MAX_ACTIVE_JOBS_GLOBAL", 1))
    max_jobs_per_user_day: int = field(default_factory=lambda: _integer("SUBGEN_MAX_JOBS_PER_USER_DAY", 1))
    max_jobs_global_day: int = field(default_factory=lambda: _integer("SUBGEN_MAX_JOBS_GLOBAL_DAY", 1))
    max_jobs_per_user_month: int = field(default_factory=lambda: _integer("SUBGEN_MAX_JOBS_PER_USER_MONTH", 4))
    max_jobs_global_month: int = field(default_factory=lambda: _integer("SUBGEN_MAX_JOBS_GLOBAL_MONTH", 6))
    max_video_minutes: int = field(default_factory=lambda: _integer("SUBGEN_MAX_VIDEO_MINUTES", 180))
    upload_part_bytes: int = field(default_factory=lambda: _integer("SUBGEN_UPLOAD_PART_BYTES", 8 * 1024**2))
    artifact_retention_days: int = field(default_factory=lambda: _integer("SUBGEN_RETENTION_DAYS", 2))
    worker_poll_seconds: int = field(default_factory=lambda: _integer("SUBGEN_WORKER_POLL_SECONDS", 5))
    worker_lease_seconds: int = field(default_factory=lambda: _integer("SUBGEN_WORKER_LEASE_SECONDS", 7200))
    worker_heartbeat_seconds: int = field(default_factory=lambda: _integer("SUBGEN_WORKER_HEARTBEAT_SECONDS", 30))
    worker_pipeline_timeout_seconds: int = field(default_factory=lambda: _integer("SUBGEN_WORKER_PIPELINE_TIMEOUT_SECONDS", 43200))
    enable_pipeline_worker: bool = field(default_factory=lambda: _boolean("SUBGEN_ENABLE_PIPELINE_WORKER", False))
    enable_gpu_worker: bool = field(default_factory=lambda: _boolean("SUBGEN_ENABLE_GPU_WORKER", False))
    processing_enabled: bool = field(default_factory=lambda: _boolean("SUBGEN_PROCESSING_ENABLED", False))
    worker_dispatch_mode: str = field(default_factory=lambda: os.environ.get("SUBGEN_WORKER_DISPATCH_MODE", "disabled"))
    gcp_project_id: str = field(default_factory=lambda: os.environ.get("SUBGEN_GCP_PROJECT_ID", ""))
    gcp_region: str = field(default_factory=lambda: os.environ.get("SUBGEN_GCP_REGION", ""))
    gcp_worker_job: str = field(default_factory=lambda: os.environ.get("SUBGEN_GCP_WORKER_JOB", ""))

    def validate(self):
        if self.worker_dispatch_mode not in {"disabled", "polling", "gcp_cloud_run_job"}:
            raise RuntimeError("SUBGEN_WORKER_DISPATCH_MODE must be disabled, polling, or gcp_cloud_run_job.")
        if self.processing_enabled and self.worker_dispatch_mode == "disabled":
            raise RuntimeError("Cloud processing cannot be enabled without a worker dispatch mode.")
        if self.worker_dispatch_mode == "gcp_cloud_run_job" and not all((
            self.gcp_project_id, self.gcp_region, self.gcp_worker_job
        )):
            raise RuntimeError("Google Cloud Run job dispatch requires project, region, and worker job settings.")
        limits = (
            self.max_active_jobs_per_user,
            self.max_active_jobs_global,
            self.max_jobs_per_user_day,
            self.max_jobs_global_day,
            self.max_jobs_per_user_month,
            self.max_jobs_global_month,
            self.max_video_minutes,
            self.worker_heartbeat_seconds,
            self.worker_pipeline_timeout_seconds,
        )
        if any(value < 1 for value in limits):
            raise RuntimeError("Cloud job and video safety limits must be positive integers.")
        if not self.dev_auth_enabled and not (self.auth_issuer and self.auth_jwks_url):
            raise RuntimeError("Configure OIDC authentication or explicitly enable SUBGEN_DEV_AUTH for local development.")
        if self.storage_backend == "s3" and not all((
            self.storage_endpoint,
            self.storage_bucket,
            self.storage_access_key,
            self.storage_secret_key,
        )):
            raise RuntimeError("S3/R2 storage is selected but its endpoint, bucket, or credentials are missing.")
        if self.credential_master_key:
            try:
                key = base64.urlsafe_b64decode(self.credential_master_key.encode("ascii"))
            except Exception as exc:
                raise RuntimeError("SUBGEN_CREDENTIAL_MASTER_KEY must be URL-safe base64.") from exc
            if len(key) != 32:
                raise RuntimeError("SUBGEN_CREDENTIAL_MASTER_KEY must decode to exactly 32 bytes.")
        elif not self.dev_auth_enabled:
            raise RuntimeError("SUBGEN_CREDENTIAL_MASTER_KEY is required outside local development.")


def load_settings():
    settings = CloudSettings()
    settings.local_storage_dir.mkdir(parents=True, exist_ok=True)
    return settings
