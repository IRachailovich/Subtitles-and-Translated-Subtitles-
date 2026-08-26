import json
import mimetypes
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from subgen_paths import (
    DRIVE_BATCHES_DIR,
    GOOGLE_DRIVE_CLIENT_PATH,
    GOOGLE_DRIVE_TOKEN_PATH,
)


DRIVE_FOLDER_MIME = "application/vnd.google-apps.folder"
DRIVE_SHORTCUT_MIME = "application/vnd.google-apps.shortcut"
DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".wmv", ".flv"}
FOLDER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{10,}$")
FOLDER_URL_PATTERNS = (
    re.compile(r"/folders/([A-Za-z0-9_-]+)"),
    re.compile(r"[?&]id=([A-Za-z0-9_-]+)"),
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def extract_drive_folder_id(value):
    raw = str(value or "").strip()
    if FOLDER_ID_PATTERN.fullmatch(raw):
        return raw
    for pattern in FOLDER_URL_PATTERNS:
        match = pattern.search(raw)
        if match:
            return match.group(1)
    raise ValueError("Enter a Google Drive folder link or folder ID.")


def safe_folder_name(name):
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(name or "")).strip().rstrip(".")
    cleaned = cleaned or "video"
    if cleaned.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        cleaned = f"_{cleaned}"
    return cleaned[:120].rstrip(".") or "video"


def safe_file_name(name):
    original = Path(str(name or "video.mp4"))
    suffix = re.sub(r"[^A-Za-z0-9.]", "", original.suffix)[:12]
    stem = safe_folder_name(original.stem)
    return f"{stem[: max(1, 180 - len(suffix))]}{suffix}"


def is_drive_video(file_info):
    mime_type = str(file_info.get("mimeType") or "").lower()
    suffix = Path(str(file_info.get("name") or "")).suffix.lower()
    return mime_type.startswith("video/") or suffix in VIDEO_EXTENSIONS


def unique_output_names(files):
    used = {}
    result = []
    for file_info in files:
        base = safe_folder_name(Path(file_info["name"]).stem)
        count = used.get(base, 0) + 1
        used[base] = count
        result.append(base if count == 1 else f"{base}_{count}")
    return result


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def configure_drive_client(source_path):
    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Google OAuth client file was not found: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("The selected file is not valid Google OAuth client JSON.") from exc
    client = payload.get("installed")
    if not isinstance(client, dict) or not client.get("client_id") or not client.get("auth_uri"):
        raise ValueError("Use an OAuth 2.0 Desktop app client JSON downloaded from Google Cloud.")
    GOOGLE_DRIVE_CLIENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, GOOGLE_DRIVE_CLIENT_PATH)
    GOOGLE_DRIVE_TOKEN_PATH.unlink(missing_ok=True)
    return GOOGLE_DRIVE_CLIENT_PATH


def drive_auth_status():
    return {
        "client_configured": GOOGLE_DRIVE_CLIENT_PATH.is_file(),
        "connected": GOOGLE_DRIVE_TOKEN_PATH.is_file(),
    }


def disconnect_drive():
    GOOGLE_DRIVE_TOKEN_PATH.unlink(missing_ok=True)


def _google_imports():
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
    except ImportError as exc:
        raise RuntimeError(
            "Google Drive support is not installed. Reinstall SubGen with Drive dependencies."
        ) from exc
    return Request, Credentials, InstalledAppFlow, build, MediaFileUpload, MediaIoBaseDownload


def authorize_drive(open_browser=True):
    if not GOOGLE_DRIVE_CLIENT_PATH.is_file():
        raise RuntimeError("Configure a Google OAuth Desktop client JSON before connecting Drive.")
    Request, Credentials, InstalledAppFlow, _, _, _ = _google_imports()
    credentials = None
    if GOOGLE_DRIVE_TOKEN_PATH.is_file():
        try:
            credentials = Credentials.from_authorized_user_file(str(GOOGLE_DRIVE_TOKEN_PATH), [DRIVE_SCOPE])
        except (OSError, ValueError):
            GOOGLE_DRIVE_TOKEN_PATH.unlink(missing_ok=True)
    if credentials and credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
    if not credentials or not credentials.valid:
        flow = InstalledAppFlow.from_client_secrets_file(str(GOOGLE_DRIVE_CLIENT_PATH), [DRIVE_SCOPE])
        credentials = flow.run_local_server(
            host="127.0.0.1",
            port=0,
            authorization_prompt_message="Opening Google Drive authorization in your browser...",
            success_message="Google Drive is connected. You may close this browser tab.",
            open_browser=open_browser,
        )
    GOOGLE_DRIVE_TOKEN_PATH.write_text(credentials.to_json(), encoding="utf-8")
    return credentials


def build_drive_service(require_connected=True):
    Request, Credentials, _, build, _, _ = _google_imports()
    if not GOOGLE_DRIVE_TOKEN_PATH.is_file():
        if require_connected:
            raise RuntimeError("Google Drive is not connected.")
        return None
    credentials = Credentials.from_authorized_user_file(str(GOOGLE_DRIVE_TOKEN_PATH), [DRIVE_SCOPE])
    if credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
        GOOGLE_DRIVE_TOKEN_PATH.write_text(credentials.to_json(), encoding="utf-8")
    if not credentials.valid:
        raise RuntimeError("Google Drive authorization expired. Connect Drive again.")
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def _escape_query_value(value):
    return str(value).replace("\\", "\\\\").replace("'", "\\'")


@dataclass
class DriveFolderScan:
    folder_id: str
    folder_name: str
    videos: list


class GoogleDriveClient:
    def __init__(self, service=None):
        self.service = service or build_drive_service()

    def get_folder(self, folder_id):
        folder = self.service.files().get(
            fileId=folder_id,
            fields="id,name,mimeType,trashed,capabilities(canDownload)",
            supportsAllDrives=True,
        ).execute()
        if folder.get("trashed") or folder.get("mimeType") != DRIVE_FOLDER_MIME:
            raise ValueError("The Google Drive link does not identify an available folder.")
        return folder

    def _list_children(self, folder_id):
        page_token = None
        while True:
            response = self.service.files().list(
                q=f"'{_escape_query_value(folder_id)}' in parents and trashed = false",
                fields=(
                    "nextPageToken,files(id,name,mimeType,size,modifiedTime,"
                    "appProperties,capabilities(canDownload),shortcutDetails(targetId,targetMimeType))"
                ),
                pageSize=1000,
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ).execute()
            yield from response.get("files", [])
            page_token = response.get("nextPageToken")
            if not page_token:
                break

    def scan_videos(self, folder_id):
        root = self.get_folder(folder_id)
        queue = [(folder_id, "")]
        seen_folders = set()
        seen_files = set()
        videos = []
        while queue:
            current_id, relative_parent = queue.pop(0)
            if current_id in seen_folders:
                continue
            seen_folders.add(current_id)
            for item in self._list_children(current_id):
                item_id = item.get("id")
                item_mime = item.get("mimeType")
                if item_mime == DRIVE_SHORTCUT_MIME:
                    details = item.get("shortcutDetails") or {}
                    item_id = details.get("targetId")
                    item_mime = details.get("targetMimeType")
                    if not item_id:
                        continue
                relative_path = str(Path(relative_parent) / item.get("name", "unnamed"))
                if item_mime == DRIVE_FOLDER_MIME:
                    if (item.get("appProperties") or {}).get("subgenOutputRoot") == "true":
                        continue
                    queue.append((item_id, relative_path))
                    continue
                normalized = dict(item)
                normalized["id"] = item_id
                normalized["mimeType"] = item_mime
                normalized["relative_path"] = relative_path
                if is_drive_video(normalized) and item_id not in seen_files:
                    seen_files.add(item_id)
                    normalized["size"] = int(normalized.get("size") or 0)
                    videos.append(normalized)
        videos.sort(key=lambda row: (row["relative_path"].casefold(), row["id"]))
        return DriveFolderScan(folder_id=folder_id, folder_name=root["name"], videos=videos)

    def download_file(self, file_info, destination, progress=None):
        if (file_info.get("capabilities") or {}).get("canDownload") is False:
            raise PermissionError(f"Google Drive does not permit downloading {file_info['name']}.")
        _, _, _, _, _, MediaIoBaseDownload = _google_imports()
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        request = self.service.files().get_media(fileId=file_info["id"], supportsAllDrives=True)
        try:
            with partial.open("wb") as output:
                downloader = MediaIoBaseDownload(output, request, chunksize=8 * 1024 * 1024)
                done = False
                while not done:
                    status, done = downloader.next_chunk(num_retries=5)
                    if status and progress:
                        progress(float(status.progress()))
            partial.replace(destination)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
        return destination

    def create_folder(self, name, parent_id, app_properties=None):
        body = {"name": name, "mimeType": DRIVE_FOLDER_MIME, "parents": [parent_id]}
        if app_properties:
            body["appProperties"] = dict(app_properties)
        response = self.service.files().create(
            body=body,
            fields="id,name,webViewLink",
            supportsAllDrives=True,
        ).execute()
        return response

    def find_child(self, parent_id, name, mime_type=None):
        query = (
            f"'{_escape_query_value(parent_id)}' in parents and "
            f"name = '{_escape_query_value(name)}' and trashed = false"
        )
        if mime_type:
            query += f" and mimeType = '{_escape_query_value(mime_type)}'"
        response = self.service.files().list(
            q=query,
            fields="files(id,name,mimeType,webViewLink)",
            pageSize=10,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        return next(iter(response.get("files", [])), None)

    def ensure_folder(self, name, parent_id):
        return self.find_child(parent_id, name, DRIVE_FOLDER_MIME) or self.create_folder(name, parent_id)

    def upload_file(self, local_path, parent_id, progress=None):
        _, _, _, _, MediaFileUpload, _ = _google_imports()
        local_path = Path(local_path)
        mime_type = mimetypes.guess_type(local_path.name)[0] or "application/octet-stream"
        media = MediaFileUpload(str(local_path), mimetype=mime_type, resumable=True, chunksize=8 * 1024 * 1024)
        existing = self.find_child(parent_id, local_path.name)
        if existing:
            request = self.service.files().update(
                fileId=existing["id"],
                media_body=media,
                fields="id,name,webViewLink",
                supportsAllDrives=True,
            )
        else:
            request = self.service.files().create(
                body={"name": local_path.name, "parents": [parent_id]},
                media_body=media,
                fields="id,name,webViewLink",
                supportsAllDrives=True,
            )
        response = None
        while response is None:
            status, response = request.next_chunk(num_retries=5)
            if status and progress:
                progress(float(status.progress()))
        return response


class DriveAuthorization:
    def __init__(self):
        self._lock = threading.RLock()
        self._state = {**drive_auth_status(), "status": "idle", "error": None}

    def status(self):
        with self._lock:
            return {**self._state, **drive_auth_status()}

    def start(self):
        with self._lock:
            if self._state["status"] == "connecting":
                return self.status()
            self._state = {**drive_auth_status(), "status": "connecting", "error": None}
        thread = threading.Thread(target=self._run, name="subgen-drive-auth", daemon=True)
        thread.start()
        return self.status()

    def _run(self):
        try:
            authorize_drive(open_browser=True)
            update = {"status": "connected", "error": None}
        except Exception as exc:
            update = {"status": "error", "error": str(exc)}
        with self._lock:
            self._state.update(update)


class DriveBatchStore:
    def __init__(self, root=DRIVE_BATCHES_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def create(self, specification, scan):
        batch_id = str(uuid.uuid4())
        output_names = unique_output_names(scan.videos)
        state = {
            "id": batch_id,
            "status": "pending",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "source_folder_id": scan.folder_id,
            "source_folder_name": scan.folder_name,
            "source_folder_url": specification.get("source_folder_url"),
            "destination_folder_id": specification.get("destination_folder_id"),
            "destination_folder_url": specification.get("destination_folder_url"),
            "output_folder_id": None,
            "output_folder_url": None,
            "target_language": specification.get("target_language"),
            "configuration": specification.get("configuration") or {},
            "stop_requested": False,
            "current_index": None,
            "error": None,
            "items": [
                {
                    "source_id": video["id"],
                    "source_name": video["name"],
                    "relative_path": video["relative_path"],
                    "size": video.get("size", 0),
                    "can_download": (video.get("capabilities") or {}).get("canDownload", True),
                    "output_folder_name": output_name,
                    "output_folder_id": None,
                    "status": "pending",
                    "progress": 0,
                    "stage": "queued",
                    "error": None,
                }
                for video, output_name in zip(scan.videos, output_names)
            ],
        }
        self.save(state)
        return state

    def path(self, batch_id):
        return self.root / str(batch_id) / "state.json"

    def work_dir(self, batch_id):
        path = self.root / str(batch_id) / "work"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save(self, state, preserve_stop_request=True):
        with self._lock:
            path = self.path(state["id"])
            if preserve_stop_request and path.is_file() and not state.get("stop_requested"):
                try:
                    persisted = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError, json.JSONDecodeError):
                    persisted = {}
                if persisted.get("stop_requested"):
                    state["stop_requested"] = True
            state["updated_at"] = utc_now()
            atomic_write_json(path, state)

    def load(self, batch_id):
        path = self.path(batch_id)
        if not path.is_file():
            return None
        with self._lock:
            return json.loads(path.read_text(encoding="utf-8"))

    def latest(self):
        candidates = sorted(self.root.glob("*/state.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not candidates:
            return None
        return json.loads(candidates[0].read_text(encoding="utf-8"))

    def request_stop(self, batch_id):
        state = self.load(batch_id)
        if not state:
            raise FileNotFoundError("Drive batch was not found.")
        state["stop_requested"] = True
        self.save(state)
        return state


def drive_folder_url(folder_id):
    return f"https://drive.google.com/drive/folders/{folder_id}"


def default_output_folder_name(source_name, target_language):
    suffix = target_language or "source"
    return safe_folder_name(f"{source_name} - SubGen {suffix}")


def cleanup_work_directory(path, attempts=5):
    path = Path(path)
    for attempt in range(attempts):
        try:
            shutil.rmtree(path, ignore_errors=False)
            return True
        except FileNotFoundError:
            return True
        except OSError:
            if attempt + 1 == attempts:
                return False
            time.sleep(0.2 * (attempt + 1))
    return not path.exists()
