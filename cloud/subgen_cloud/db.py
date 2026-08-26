import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    inspect,
    select,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


def utcnow():
    return datetime.now(timezone.utc)


def new_id():
    return str(uuid.uuid4())


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    WAITING_FOR_REVIEW = "waiting_for_review"
    NEEDS_ATTENTION = "needs_attention"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    STALE_AFTER_EDIT = "stale_after_edit"
    BURN_QUEUED = "burn_queued"
    BURNING = "burning"
    COMPLETED = "completed"
    FAILED = "failed"
    BURN_FAILED = "burn_failed"
    CANCELLED = "cancelled"


ACTIVE_JOB_STATUSES = {
    JobStatus.QUEUED.value,
    JobStatus.PROCESSING.value,
    JobStatus.WAITING_FOR_REVIEW.value,
    JobStatus.NEEDS_ATTENTION.value,
    JobStatus.IN_REVIEW.value,
    JobStatus.APPROVED.value,
    JobStatus.STALE_AFTER_EDIT.value,
    JobStatus.BURN_QUEUED.value,
    JobStatus.BURNING.value,
}


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "cloud_users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    auth_subject: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    email: Mapped[str | None] = mapped_column(String(320))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class UserSettings(Base):
    __tablename__ = "cloud_user_settings"

    user_id: Mapped[str] = mapped_column(ForeignKey("cloud_users.id", ondelete="CASCADE"), primary_key=True)
    revision: Mapped[int] = mapped_column(Integer, default=1)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class ProviderCredential(Base):
    __tablename__ = "cloud_provider_credentials"
    __table_args__ = (UniqueConstraint("user_id", "provider", "profile", name="uq_cloud_provider_profile"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("cloud_users.id", ondelete="CASCADE"), index=True)
    provider: Mapped[str] = mapped_column(String(80), nullable=False)
    profile: Mapped[str] = mapped_column(String(80), default="default")
    ciphertext: Mapped[str] = mapped_column(Text, nullable=False)
    key_version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class Video(Base):
    __tablename__ = "cloud_videos"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("cloud_users.id", ondelete="CASCADE"), index=True)
    sha256: Mapped[str | None] = mapped_column(String(64))
    original_name: Mapped[str] = mapped_column(String(512), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    duration_seconds: Mapped[int | None] = mapped_column(Integer)
    object_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    upload_complete: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Upload(Base):
    __tablename__ = "cloud_uploads"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("cloud_users.id", ondelete="CASCADE"), index=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("cloud_videos.id", ondelete="CASCADE"), index=True)
    storage_upload_id: Mapped[str] = mapped_column(String(512), nullable=False)
    part_size: Mapped[int] = mapped_column(Integer, nullable=False)
    completed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Job(Base):
    __tablename__ = "cloud_jobs"
    __table_args__ = (
        Index("ix_cloud_jobs_claim", "status", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("cloud_users.id", ondelete="CASCADE"), index=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("cloud_videos.id", ondelete="CASCADE"), index=True)
    target_language: Mapped[str | None] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(40), default=JobStatus.QUEUED.value, index=True)
    stage: Mapped[str] = mapped_column(String(80), default="queued")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    pipeline_config: Mapped[dict] = mapped_column(JSON, default=dict)
    style_config: Mapped[dict] = mapped_column(JSON, default=dict)
    cache_action: Mapped[str] = mapped_column(String(40), default="reuse_all")
    review_segments: Mapped[list] = mapped_column(JSON, default=list)
    review_manifest: Mapped[dict] = mapped_column(JSON, default=dict)
    approved_draft_hash: Mapped[str | None] = mapped_column(String(64))
    approved_revision_hash: Mapped[str | None] = mapped_column(String(64))
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[str | None] = mapped_column(Text)
    lease_owner: Mapped[str | None] = mapped_column(String(128))
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    video: Mapped[Video] = relationship()


class Artifact(Base):
    __tablename__ = "cloud_artifacts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("cloud_users.id", ondelete="CASCADE"), index=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("cloud_videos.id", ondelete="CASCADE"), index=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("cloud_jobs.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(80), nullable=False)
    language: Mapped[str | None] = mapped_column(String(32))
    object_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    parent_artifact_id: Mapped[str | None] = mapped_column(ForeignKey("cloud_artifacts.id"))
    provenance: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


def create_database(database_url):
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    engine = create_engine(database_url, pool_pre_ping=True, connect_args=connect_args)
    Base.metadata.create_all(engine)
    # Additive runtime migration for existing SQLite/PostgreSQL deployments.
    # New installs receive these columns through metadata.create_all above.
    existing = {column["name"] for column in inspect(engine).get_columns("cloud_jobs")}
    additions = {
        "review_manifest": "JSON",
        "approved_draft_hash": "VARCHAR(64)",
        "approved_revision_hash": "VARCHAR(64)",
        "approved_at": "TIMESTAMP",
    }
    with engine.begin() as connection:
        for column, sql_type in additions.items():
            if column not in existing:
                connection.execute(text(f"ALTER TABLE cloud_jobs ADD COLUMN {column} {sql_type}"))
    return engine, sessionmaker(bind=engine, expire_on_commit=False)


def get_or_create_user(session, subject, email=None):
    user = session.scalar(select(User).where(User.auth_subject == subject))
    if user:
        if email and user.email != email:
            user.email = email
        return user
    user = User(auth_subject=subject, email=email)
    session.add(user)
    session.flush()
    return user
