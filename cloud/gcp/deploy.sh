#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID to a dedicated Google Cloud project ID.}"
: "${SUPABASE_PROJECT_REF:?Set SUPABASE_PROJECT_REF.}"
: "${R2_ACCOUNT_ID:?Set R2_ACCOUNT_ID.}"
: "${R2_BUCKET:?Set R2_BUCKET.}"
: "${VERCEL_ORIGIN:?Set VERCEL_ORIGIN, for example https://your-app.vercel.app.}"

REGION="${REGION:-europe-west1}"
ARM_PROCESSING="${ARM_PROCESSING:-false}"
SKIP_BUILD="${SKIP_BUILD:-false}"
REPOSITORY="subgen"
API_SERVICE="subgen-api"
WORKER_JOB="subgen-worker"
API_SA="subgen-api@${PROJECT_ID}.iam.gserviceaccount.com"
WORKER_SA="subgen-worker@${PROJECT_ID}.iam.gserviceaccount.com"
IMAGE_ROOT="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}"

if [[ "${ARM_PROCESSING}" != "true" && "${ARM_PROCESSING}" != "false" ]] || \
   [[ "${SKIP_BUILD}" != "true" && "${SKIP_BUILD}" != "false" ]]; then
  echo "ARM_PROCESSING and SKIP_BUILD must be true or false." >&2
  exit 2
fi

gcloud config set project "${PROJECT_ID}"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com

if ! gcloud artifacts repositories describe "${REPOSITORY}" --location "${REGION}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${REPOSITORY}" --repository-format=docker --location "${REGION}"
fi
gcloud artifacts repositories set-cleanup-policies "${REPOSITORY}" \
  --location "${REGION}" --policy cloud/gcp/artifact-cleanup.json --no-dry-run

for account in subgen-api subgen-worker; do
  if ! gcloud iam service-accounts describe "${account}@${PROJECT_ID}.iam.gserviceaccount.com" >/dev/null 2>&1; then
    gcloud iam service-accounts create "${account}" --display-name "SubGen ${account#subgen-}"
  fi
done

put_secret() {
  local name="$1"
  local prompt="$2"
  local value
  if gcloud secrets versions describe latest --secret "${name}" >/dev/null 2>&1; then
    echo "Reusing Secret Manager value: ${name}"
    return
  fi
  if ! gcloud secrets describe "${name}" >/dev/null 2>&1; then
    gcloud secrets create "${name}" --replication-policy=automatic >/dev/null
  fi
  read -r -s -p "${prompt}: " value
  echo
  printf '%s' "${value}" | gcloud secrets versions add "${name}" --data-file=- >/dev/null
  unset value
}

echo "Secret values are read without echo and stored directly in Google Secret Manager."
put_secret subgen-database-url "Supabase pooled PostgreSQL URL"
put_secret subgen-auth-public-key "Supabase anon/public key"
put_secret subgen-r2-access-key "Cloudflare R2 access key ID"
put_secret subgen-r2-secret-key "Cloudflare R2 secret access key"

if ! gcloud secrets describe subgen-credential-master-key >/dev/null 2>&1; then
  gcloud secrets create subgen-credential-master-key --replication-policy=automatic >/dev/null
  openssl rand -base64 32 | tr '+/' '-_' | tr -d '\n' | \
    gcloud secrets versions add subgen-credential-master-key --data-file=- >/dev/null
fi

for account in "${API_SA}" "${WORKER_SA}"; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member "serviceAccount:${account}" --role roles/secretmanager.secretAccessor --quiet >/dev/null
done

if [[ "${SKIP_BUILD}" != "true" ]]; then
  gcloud builds submit --config cloud/gcp/cloudbuild.yaml \
    --substitutions "_REGION=${REGION},_TAG=latest" .
fi

COMMON_ENV="SUBGEN_AUTH_PUBLIC_URL=https://${SUPABASE_PROJECT_REF}.supabase.co,SUBGEN_AUTH_ISSUER=https://${SUPABASE_PROJECT_REF}.supabase.co/auth/v1,SUBGEN_AUTH_AUDIENCE=authenticated,SUBGEN_AUTH_JWKS_URL=https://${SUPABASE_PROJECT_REF}.supabase.co/auth/v1/.well-known/jwks.json,SUBGEN_STORAGE_BACKEND=s3,SUBGEN_STORAGE_ENDPOINT=https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com,SUBGEN_STORAGE_REGION=auto,SUBGEN_STORAGE_BUCKET=${R2_BUCKET},SUBGEN_MAX_UPLOAD_BYTES=8589934592,SUBGEN_MAX_ACTIVE_JOBS=1,SUBGEN_MAX_ACTIVE_JOBS_GLOBAL=1,SUBGEN_MAX_JOBS_PER_USER_DAY=1,SUBGEN_MAX_JOBS_GLOBAL_DAY=1,SUBGEN_MAX_JOBS_PER_USER_MONTH=4,SUBGEN_MAX_JOBS_GLOBAL_MONTH=6,SUBGEN_MAX_VIDEO_MINUTES=180,SUBGEN_RETENTION_DAYS=2,SUBGEN_ENABLE_GPU_WORKER=false,SUBGEN_PROCESSING_ENABLED=${ARM_PROCESSING}"
COMMON_SECRETS="SUBGEN_CLOUD_DATABASE_URL=subgen-database-url:latest,SUBGEN_AUTH_PUBLIC_KEY=subgen-auth-public-key:latest,SUBGEN_CREDENTIAL_MASTER_KEY=subgen-credential-master-key:latest,SUBGEN_STORAGE_ACCESS_KEY=subgen-r2-access-key:latest,SUBGEN_STORAGE_SECRET_KEY=subgen-r2-secret-key:latest"

gcloud run jobs deploy "${WORKER_JOB}" \
  --image "${IMAGE_ROOT}/worker:latest" --region "${REGION}" --service-account "${WORKER_SA}" \
  --tasks 1 --parallelism 1 --max-retries 0 --task-timeout 43200s --cpu 2 --memory 8Gi \
  --set-env-vars "${COMMON_ENV},SUBGEN_ENABLE_PIPELINE_WORKER=true,SUBGEN_WORKER_DISPATCH_MODE=gcp_cloud_run_job,SUBGEN_GCP_PROJECT_ID=${PROJECT_ID},SUBGEN_GCP_REGION=${REGION},SUBGEN_GCP_WORKER_JOB=${WORKER_JOB}" \
  --set-secrets "${COMMON_SECRETS}"

gcloud run jobs add-iam-policy-binding "${WORKER_JOB}" --region "${REGION}" \
  --member "serviceAccount:${API_SA}" --role roles/run.invoker --quiet >/dev/null

gcloud run deploy "${API_SERVICE}" \
  --image "${IMAGE_ROOT}/api:latest" --region "${REGION}" --service-account "${API_SA}" \
  --allow-unauthenticated --min 0 --max 1 --concurrency 1 --cpu 1 --memory 512Mi \
  --timeout 60s --cpu-throttling \
  --set-env-vars "${COMMON_ENV},SUBGEN_WORKER_DISPATCH_MODE=gcp_cloud_run_job,SUBGEN_GCP_PROJECT_ID=${PROJECT_ID},SUBGEN_GCP_REGION=${REGION},SUBGEN_GCP_WORKER_JOB=${WORKER_JOB},SUBGEN_CORS_ORIGINS=${VERCEL_ORIGIN}" \
  --set-secrets "${COMMON_SECRETS}"

API_URL="$(gcloud run services describe "${API_SERVICE}" --region "${REGION}" --format='value(status.url)')"
gcloud run services update "${API_SERVICE}" --region "${REGION}" \
  --update-env-vars "SUBGEN_PUBLIC_BASE_URL=${API_URL}" >/dev/null

echo
echo "API URL: ${API_URL}"
echo "Processing armed: ${ARM_PROCESSING}"
echo "Set SUBGEN_API_BASE_URL=${API_URL} in the Vercel project, then redeploy the static PWA."
