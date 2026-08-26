from typing import Any

from pydantic import BaseModel, Field, field_validator


class SettingsUpdate(BaseModel):
    config: dict[str, Any]
    revision: int | None = None


class CredentialUpdate(BaseModel):
    provider: str = Field(min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9_-]+$")
    profile: str = Field(default="default", min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9_-]+$")
    api_key: str = Field(min_length=8, max_length=8192)


class UploadInitiate(BaseModel):
    filename: str = Field(min_length=1, max_length=512)
    size_bytes: int = Field(gt=0)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")

    @field_validator("filename")
    @classmethod
    def safe_filename(cls, value):
        if value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError("Use a filename without path separators")
        return value


class PartRequest(BaseModel):
    part_number: int = Field(ge=1, le=10000)


class CompletedPart(BaseModel):
    part_number: int = Field(ge=1, le=10000)
    etag: str = Field(min_length=1, max_length=256)


class UploadComplete(BaseModel):
    parts: list[CompletedPart] = Field(min_length=1, max_length=10000)


class JobCreate(BaseModel):
    video_id: str
    target_language: str | None = Field(default=None, max_length=32)
    pipeline_config: dict[str, Any] = Field(default_factory=dict)
    style_config: dict[str, Any] = Field(default_factory=dict)
    cache_action: str = Field(default="reuse_all", pattern=r"^(reuse_all|reburn|retime|regenerate_all)$")


class ReviewUpdate(BaseModel):
    segments: list[dict[str, Any]] = Field(min_length=1, max_length=10000)
    translation_confirmed: bool = False


class RetranslateRequest(BaseModel):
    cue_ids: list[str] = Field(min_length=1, max_length=1000)


class ApproveRequest(BaseModel):
    style_config: dict[str, Any] | None = None
    accept_warnings: bool = False
    translation_confirmed: bool = False


class IssueResolution(BaseModel):
    status: str = Field(pattern=r"^(corrected|accepted|dismissed_with_reason)$")
    reason: str | None = Field(default=None, max_length=2000)
