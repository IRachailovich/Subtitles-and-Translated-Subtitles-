import os
import hashlib
import sqlite3
import json
import contextlib
from pathlib import Path

from subgen_paths import DB_PATH


def normalize_source_location(source_location):
    """Return a stable, JSON-safe description of a media source."""
    if not source_location:
        return None
    if not isinstance(source_location, dict):
        raise TypeError("source_location must be a mapping")

    location = {
        str(key): str(value) if isinstance(value, Path) else value
        for key, value in source_location.items()
        if value is not None
    }
    kind = str(location.get("kind") or "").strip().lower()
    if not kind:
        raise ValueError("source_location requires a kind")
    location["kind"] = kind

    if kind == "google_drive":
        drive_id = (
            location.get("drive_id")
            or location.get("file_id")
            or location.get("source_id")
        )
        if not drive_id:
            raise ValueError("Google Drive provenance requires a Drive file ID")
        location["drive_id"] = str(drive_id)
        location.setdefault(
            "url",
            f"https://drive.google.com/file/d/{location['drive_id']}/view",
        )
    elif kind == "cloud_object":
        if not location.get("object_key"):
            raise ValueError("Cloud-object provenance requires an object key")
        location["object_key"] = str(location["object_key"])
    elif kind in {"local", "local_upload"}:
        if not location.get("path"):
            raise ValueError("Local provenance requires a path")
        location["path"] = str(Path(location["path"]).expanduser().resolve())

    # Prove that every value can be persisted before the production run starts.
    json.dumps(location, ensure_ascii=False, sort_keys=True)
    return location


def source_location_identity(source_location):
    """Return a stable identity within a source system, not a display name."""
    location = normalize_source_location(source_location)
    if not location:
        return None
    kind = location["kind"]
    if kind == "google_drive":
        return location["drive_id"]
    if kind == "cloud_object":
        return location["object_key"]
    if kind in {"local", "local_upload"}:
        return location["path"]
    for key in ("source_id", "id", "uri", "url", "path"):
        if location.get(key):
            return str(location[key])
    return hashlib.sha256(
        json.dumps(location, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

def calculate_video_hash(file_path):
    """Calculate SHA-256 hash of the first 8MB and last 8MB of a file for fast identification."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Video file not found: {file_path}")
    
    file_size = file_path.stat().st_size
    hasher = hashlib.sha256()
    
    chunk_size = 8 * 1024 * 1024  # 8MB
    
    with open(file_path, "rb") as f:
        if file_size <= chunk_size:
            hasher.update(f.read())
        else:
            # Read first 8MB
            hasher.update(f.read(chunk_size))
            # Read last 8MB
            f.seek(-chunk_size, os.SEEK_END)
            hasher.update(f.read(chunk_size))
            
    # Include file size to prevent collision on identical headers
    hasher.update(str(file_size).encode("utf-8"))
    return hasher.hexdigest()

def get_db_connection():
    """Create and return a SQLite database connection with row factory enabled."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

@contextlib.contextmanager
def db_session():
    """Context manager that ensures the database connection is committed and closed."""
    conn = get_db_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    """Initialize the SQLite database schema if tables do not exist."""
    with db_session() as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        
        # 1. Video cache metadata table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS video_cache (
            video_hash TEXT PRIMARY KEY,
            file_size INTEGER NOT NULL,
            duration REAL,
            detected_language TEXT,
            content_sha256 TEXT,
            effective_style_config TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        video_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(video_cache)").fetchall()
        }
        if "content_sha256" not in video_columns:
            conn.execute("ALTER TABLE video_cache ADD COLUMN content_sha256 TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_video_cache_content_sha256 "
            "ON video_cache(content_sha256)"
        )
        
        # 2. Transcription cache table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS transcriptions (
            video_hash TEXT PRIMARY KEY,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            transcript_text TEXT NOT NULL,
            segments_json TEXT NOT NULL,
            alignment_info TEXT,
            prompt_version TEXT,
            request_config_version TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (video_hash) REFERENCES video_cache(video_hash) ON DELETE CASCADE
        );
        """)
        transcription_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(transcriptions)").fetchall()
        }
        if "prompt_version" not in transcription_columns:
            conn.execute("ALTER TABLE transcriptions ADD COLUMN prompt_version TEXT")
        if "request_config_version" not in transcription_columns:
            conn.execute("ALTER TABLE transcriptions ADD COLUMN request_config_version TEXT")
        
        # 3. Translation cache table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS translations (
            video_hash TEXT,
            target_language TEXT,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            source_signature TEXT,
            segments_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (video_hash, target_language),
            FOREIGN KEY (video_hash) REFERENCES video_cache(video_hash) ON DELETE CASCADE
        );
        """)
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(translations)").fetchall()
        }
        if "source_signature" not in columns:
            conn.execute("ALTER TABLE translations ADD COLUMN source_signature TEXT")
        
        # 4. Background jobs status table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            video_path TEXT NOT NULL,
            video_hash TEXT,
            status TEXT NOT NULL,
            progress INTEGER DEFAULT 0,
            stage TEXT,
            status_label TEXT,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # 5. User-edited subtitle drafts, saved before the burn stage.
        conn.execute("""
        CREATE TABLE IF NOT EXISTS subtitle_drafts (
            video_hash TEXT NOT NULL,
            target_language TEXT NOT NULL,
            segments_json TEXT NOT NULL,
            approved INTEGER NOT NULL DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (video_hash, target_language),
            FOREIGN KEY (video_hash) REFERENCES video_cache(video_hash) ON DELETE CASCADE
        );
        """)

        # 6. Durable, versioned review workflow manifests shared by source/web/frozen.
        conn.execute("""
        CREATE TABLE IF NOT EXISTS review_manifests (
            video_hash TEXT NOT NULL,
            target_language TEXT NOT NULL,
            review_id TEXT NOT NULL,
            state TEXT NOT NULL,
            draft_hash TEXT,
            approved_draft_hash TEXT,
            manifest_json TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (video_hash, target_language)
        );
        """)

        # 7. Durable source provenance. A media hash can have several legitimate
        # locations, and the source record must exist even if transcription fails.
        conn.execute("""
        CREATE TABLE IF NOT EXISTS media_sources (
            video_hash TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            source_id TEXT NOT NULL,
            source_name TEXT,
            source_path TEXT,
            source_url TEXT,
            parent_id TEXT,
            metadata_json TEXT NOT NULL,
            first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (video_hash, source_kind, source_id),
            FOREIGN KEY (video_hash) REFERENCES video_cache(video_hash) ON DELETE CASCADE
        );
        """)


def register_media_source(
    video_hash,
    file_size,
    source_location,
    *,
    content_sha256=None,
    duration=None,
):
    """Persist hash-bound media provenance before provider work begins."""
    location = normalize_source_location(source_location)
    if not video_hash or not location:
        return location
    source_id = source_location_identity(location)
    source_path = (
        location.get("relative_path")
        or location.get("path")
        or location.get("object_key")
    )
    source_name = location.get("name")
    if not source_name and source_path:
        source_name = Path(str(source_path)).name
    parent_id = (
        location.get("source_folder_id")
        or location.get("parent_id")
        or location.get("folder_id")
    )

    init_db()
    with db_session() as conn:
        conn.execute("""
            INSERT INTO video_cache (
                video_hash, file_size, duration, content_sha256
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(video_hash) DO UPDATE SET
                file_size = excluded.file_size,
                duration = COALESCE(excluded.duration, video_cache.duration),
                content_sha256 = COALESCE(
                    excluded.content_sha256,
                    video_cache.content_sha256
                )
        """, (video_hash, int(file_size), duration, content_sha256))
        conn.execute("""
            INSERT INTO media_sources (
                video_hash, source_kind, source_id, source_name, source_path,
                source_url, parent_id, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_hash, source_kind, source_id) DO UPDATE SET
                source_name = excluded.source_name,
                source_path = excluded.source_path,
                source_url = excluded.source_url,
                parent_id = excluded.parent_id,
                metadata_json = excluded.metadata_json,
                last_seen = CURRENT_TIMESTAMP
        """, (
            video_hash,
            location["kind"],
            source_id,
            source_name,
            source_path,
            location.get("url"),
            parent_id,
            json.dumps(location, ensure_ascii=False, sort_keys=True),
        ))
    return location


def get_media_sources(video_hash):
    """Return every known source location for the same hash-bound media."""
    init_db()
    with db_session() as conn:
        rows = conn.execute("""
            SELECT source_kind, source_id, source_name, source_path, source_url,
                   parent_id, metadata_json, first_seen, last_seen
            FROM media_sources
            WHERE video_hash = ?
            ORDER BY first_seen, source_kind, source_id
        """, (video_hash,)).fetchall()
    result = []
    for row in rows:
        location = json.loads(row["metadata_json"])
        location.update({
            "source_id": row["source_id"],
            "first_seen": row["first_seen"],
            "last_seen": row["last_seen"],
        })
        result.append(location)
    return result

def get_cached_transcription(video_hash):
    """Retrieve transcription from cache if it exists."""
    with db_session() as conn:
        row = conn.execute("""
            SELECT t.transcript_text, t.segments_json, t.alignment_info, t.provider, t.model,
                   t.prompt_version, t.request_config_version, v.detected_language,
                   v.duration, v.content_sha256
            FROM transcriptions t
            JOIN video_cache v ON t.video_hash = v.video_hash
            WHERE t.video_hash = ?
        """, (video_hash,)).fetchone()
        
        if row:
            return {
                "transcript_text": row["transcript_text"],
                "segments": json.loads(row["segments_json"]),
                "alignment_info": json.loads(row["alignment_info"]) if row["alignment_info"] else None,
                "transcription_provider": row["provider"],
                "transcription_model": row["model"],
                "prompt_version": row["prompt_version"],
                "request_config_version": row["request_config_version"],
                "detected_language": row["detected_language"],
                "duration": row["duration"],
                "content_sha256": row["content_sha256"],
                "source_locations": get_media_sources(video_hash),
            }
        return None


def get_subtitle_draft(video_hash, target_language):
    with db_session() as conn:
        row = conn.execute("""
            SELECT segments_json, approved, updated_at
            FROM subtitle_drafts
            WHERE video_hash = ? AND target_language = ?
        """, (video_hash, target_language or "source")).fetchone()
        if not row:
            return None
        return {
            "segments": json.loads(row["segments_json"]),
            "approved": bool(row["approved"]),
            "updated_at": row["updated_at"],
        }


def save_subtitle_draft(video_hash, target_language, segments, approved=False):
    with db_session() as conn:
        conn.execute("""
            INSERT INTO subtitle_drafts (
                video_hash, target_language, segments_json, approved, updated_at
            ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(video_hash, target_language) DO UPDATE SET
                segments_json = excluded.segments_json,
                approved = excluded.approved,
                updated_at = CURRENT_TIMESTAMP
        """, (
            video_hash,
            target_language or "source",
            json.dumps(segments, ensure_ascii=False),
            1 if approved else 0,
        ))

def save_transcription(
    video_hash, file_size, duration, detected_language, provider, model,
    transcript_text, segments, alignment_info, prompt_version=None,
    request_config_version=None,
):
    """Upsert transcription results in the database cache."""
    with db_session() as conn:
        # 1. Upsert video_cache metadata
        conn.execute("""
            INSERT INTO video_cache (video_hash, file_size, duration, detected_language)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(video_hash) DO UPDATE SET
                file_size = excluded.file_size,
                duration = excluded.duration,
                detected_language = excluded.detected_language
        """, (video_hash, file_size, duration, detected_language))
        
        # 2. Upsert transcriptions
        conn.execute("""
            INSERT INTO transcriptions (
                video_hash, provider, model, transcript_text, segments_json, alignment_info,
                prompt_version, request_config_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_hash) DO UPDATE SET
                provider = excluded.provider,
                model = excluded.model,
                transcript_text = excluded.transcript_text,
                segments_json = excluded.segments_json,
                alignment_info = excluded.alignment_info,
                prompt_version = excluded.prompt_version,
                request_config_version = excluded.request_config_version
        """, (
            video_hash, 
            provider, 
            model, 
            transcript_text, 
            json.dumps(segments, ensure_ascii=False),
            json.dumps(alignment_info, ensure_ascii=False) if alignment_info else None,
            prompt_version,
            request_config_version,
        ))


def get_review_manifest(video_hash, target_language=None):
    init_db()
    with db_session() as conn:
        row = conn.execute("""
            SELECT manifest_json, updated_at
            FROM review_manifests
            WHERE video_hash = ? AND target_language = ?
        """, (video_hash, target_language or "source")).fetchone()
        if not row:
            return None
        manifest = json.loads(row["manifest_json"])
        manifest["persisted_at"] = row["updated_at"]
        return manifest


def save_review_manifest(review):
    init_db()
    video_hash = str(review.get("video_id") or review.get("source_hash") or "")
    if not video_hash:
        raise ValueError("Review manifest requires a source hash or video ID")
    target_language = review.get("target_language") or "source"
    approval = review.get("approval") or {}
    with db_session() as conn:
        conn.execute("""
            INSERT INTO review_manifests (
                video_hash, target_language, review_id, state, draft_hash,
                approved_draft_hash, manifest_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(video_hash, target_language) DO UPDATE SET
                review_id = excluded.review_id,
                state = excluded.state,
                draft_hash = excluded.draft_hash,
                approved_draft_hash = excluded.approved_draft_hash,
                manifest_json = excluded.manifest_json,
                updated_at = CURRENT_TIMESTAMP
        """, (
            video_hash,
            target_language,
            review.get("review_id"),
            review.get("state"),
            review.get("draft_hash"),
            approval.get("approved_draft_hash"),
            json.dumps(review, ensure_ascii=False),
        ))


def list_review_manifests(states=None):
    init_db()
    values = [str(value) for value in (states or [])]
    query = "SELECT manifest_json, updated_at FROM review_manifests"
    parameters = []
    if values:
        query += f" WHERE state IN ({','.join('?' for _ in values)})"
        parameters.extend(values)
    query += " ORDER BY updated_at DESC"
    with db_session() as conn:
        rows = conn.execute(query, parameters).fetchall()
    result = []
    for row in rows:
        manifest = json.loads(row["manifest_json"])
        manifest["persisted_at"] = row["updated_at"]
        result.append(manifest)
    return result

def get_cached_translation(video_hash, target_language, source_signature=None):
    """Retrieve translation from cache if it exists."""
    with db_session() as conn:
        if source_signature is None:
            row = conn.execute("""
                SELECT segments_json, provider, model, source_signature
                FROM translations
                WHERE video_hash = ? AND target_language = ?
            """, (video_hash, target_language)).fetchone()
        else:
            row = conn.execute("""
                SELECT segments_json, provider, model, source_signature
                FROM translations
                WHERE video_hash = ? AND target_language = ? AND source_signature = ?
            """, (video_hash, target_language, source_signature)).fetchone()
        
        if row:
            return {
                "segments": json.loads(row["segments_json"]),
                "provider": row["provider"],
                "model": row["model"],
                "source_signature": row["source_signature"],
            }
        return None

def save_translation(video_hash, target_language, provider, model, segments, source_signature=None):
    """Upsert translation results in the database cache."""
    with db_session() as conn:
        # Ensure video metadata exists (fallback if transcription was external)
        conn.execute("""
            INSERT OR IGNORE INTO video_cache (video_hash, file_size)
            VALUES (?, 0)
        """, (video_hash,))
        
        # Upsert translations
        conn.execute("""
            INSERT INTO translations (video_hash, target_language, provider, model, source_signature, segments_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_hash, target_language) DO UPDATE SET
                provider = excluded.provider,
                model = excluded.model,
                source_signature = excluded.source_signature,
                segments_json = excluded.segments_json
        """, (
            video_hash,
            target_language,
            provider,
            model,
            source_signature,
            json.dumps(segments, ensure_ascii=False)
        ))

def create_job(job_id, video_path, video_hash=None, status="pending"):
    """Create a new job record in the database."""
    with db_session() as conn:
        conn.execute("""
            INSERT INTO jobs (job_id, video_path, video_hash, status, progress, stage, status_label)
            VALUES (?, ?, ?, ?, 0, 'initialized', 'Job initialized')
        """, (job_id, video_path, video_hash, status))

def update_job_status(job_id, status, progress=None, stage=None, status_label=None, error_message=None):
    """Update job progress and status in the database."""
    with db_session() as conn:
        query = "UPDATE jobs SET status = ?, updated_at = CURRENT_TIMESTAMP"
        params = [status]
        
        if progress is not None:
            query += ", progress = ?"
            params.append(progress)
        if stage is not None:
            query += ", stage = ?"
            params.append(stage)
        if status_label is not None:
            query += ", status_label = ?"
            params.append(status_label)
        if error_message is not None:
            query += ", error_message = ?"
            params.append(error_message)
            
        query += " WHERE job_id = ?"
        params.append(job_id)
        
        conn.execute(query, tuple(params))

def get_job_status(job_id):
    """Retrieve the full status of a job."""
    with db_session() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row:
            return dict(row)
        return None

def save_burned_style(video_hash, style_config):
    """Save the last successfully burned subtitle style config for a video."""
    with db_session() as conn:
        conn.execute("""
            UPDATE video_cache 
            SET effective_style_config = ?
            WHERE video_hash = ?
        """, (json.dumps(style_config, ensure_ascii=False), video_hash))

def get_burned_style(video_hash):
    """Retrieve the last successfully burned subtitle style config for a video."""
    with db_session() as conn:
        row = conn.execute("SELECT effective_style_config FROM video_cache WHERE video_hash = ?", (video_hash,)).fetchone()
        if row and row["effective_style_config"]:
            try:
                return json.loads(row["effective_style_config"])
            except Exception:
                return {}
        return {}
