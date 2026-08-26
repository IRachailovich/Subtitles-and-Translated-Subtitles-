import base64
import json
import os

import functions_framework
from google.api_core.exceptions import NotFound
from google.cloud import run_v2


@functions_framework.cloud_event
def stop_subgen_worker(cloud_event):
    encoded = cloud_event.data.get("message", {}).get("data") or ""
    payload = json.loads(base64.b64decode(encoded).decode("utf-8"))
    cost = float(payload.get("costAmount") or 0)
    budget = float(payload.get("budgetAmount") or 0)
    if cost <= 0 or (budget > 0 and cost < budget):
        return "No billed-cost threshold reached"

    project = os.environ["SUBGEN_GCP_PROJECT_ID"]
    region = os.environ["SUBGEN_GCP_REGION"]
    job = os.environ.get("SUBGEN_GCP_WORKER_JOB", "subgen-worker")
    name = f"projects/{project}/locations/{region}/jobs/{job}"
    try:
        run_v2.JobsClient().delete_job(name=name).result(timeout=120)
    except NotFound:
        pass
    return f"Deleted {name} after reported cost {cost} reached budget {budget}"
