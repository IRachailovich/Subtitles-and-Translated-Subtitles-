import json
import urllib.error
import urllib.request


METADATA_TOKEN_URL = (
    "http://metadata.google.internal/computeMetadata/v1/instance/"
    "service-accounts/default/token"
)


def _metadata_access_token():
    request = urllib.request.Request(METADATA_TOKEN_URL, headers={"Metadata-Flavor": "Google"})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError) as exc:
        raise RuntimeError("Could not obtain the Cloud Run service identity token.") from exc
    token = payload.get("access_token")
    if not token:
        raise RuntimeError("Google metadata response did not contain an access token.")
    return token


def dispatch_worker(settings):
    if settings.worker_dispatch_mode == "polling":
        return
    if settings.worker_dispatch_mode != "gcp_cloud_run_job":
        raise RuntimeError("Cloud processing is not connected to a worker.")
    endpoint = (
        "https://run.googleapis.com/v2/projects/"
        f"{settings.gcp_project_id}/locations/{settings.gcp_region}/jobs/"
        f"{settings.gcp_worker_job}:run"
    )
    request = urllib.request.Request(
        endpoint,
        data=b"{}",
        method="POST",
        headers={
            "Authorization": f"Bearer {_metadata_access_token()}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            if response.status not in {200, 201}:
                raise RuntimeError(f"Cloud Run worker dispatch returned HTTP {response.status}.")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"Cloud Run worker dispatch failed: HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("Cloud Run worker dispatch could not reach the Google API.") from exc
