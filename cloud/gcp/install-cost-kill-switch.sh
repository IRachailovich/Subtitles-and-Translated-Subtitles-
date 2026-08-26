#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID to the dedicated SubGen Google Cloud project.}"
: "${BILLING_ACCOUNT_ID:?Set BILLING_ACCOUNT_ID without the billingAccounts/ prefix.}"

REGION="${REGION:-europe-west1}"
BUDGET_AMOUNT="${BUDGET_AMOUNT:-0.01USD}"
TOPIC="subgen-billed-cost"
FUNCTION="subgen-budget-kill-switch"
SERVICE_ACCOUNT="subgen-budget-kill@${PROJECT_ID}.iam.gserviceaccount.com"
PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"

gcloud config set project "${PROJECT_ID}"
gcloud services enable billingbudgets.googleapis.com pubsub.googleapis.com cloudfunctions.googleapis.com eventarc.googleapis.com run.googleapis.com cloudbuild.googleapis.com

if ! gcloud pubsub topics describe "${TOPIC}" >/dev/null 2>&1; then
  gcloud pubsub topics create "${TOPIC}"
fi
if ! gcloud iam service-accounts describe "${SERVICE_ACCOUNT}" >/dev/null 2>&1; then
  gcloud iam service-accounts create subgen-budget-kill --display-name "SubGen billed-cost kill switch"
fi
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member "serviceAccount:${SERVICE_ACCOUNT}" --role roles/run.admin --quiet >/dev/null

gcloud functions deploy "${FUNCTION}" --gen2 --runtime python311 --region "${REGION}" \
  --source cloud/gcp/budget-kill-switch --entry-point stop_subgen_worker \
  --trigger-topic "${TOPIC}" --service-account "${SERVICE_ACCOUNT}" \
  --memory 256Mi --timeout 180s --max-instances 1 \
  --set-env-vars "SUBGEN_GCP_PROJECT_ID=${PROJECT_ID},SUBGEN_GCP_REGION=${REGION},SUBGEN_GCP_WORKER_JOB=subgen-worker"

if ! gcloud billing budgets list --billing-account "${BILLING_ACCOUNT_ID}" \
    --filter='displayName="SubGen billed-cost stop"' --format='value(name)' | grep -q .; then
  gcloud billing budgets create \
    --billing-account "${BILLING_ACCOUNT_ID}" \
    --display-name "SubGen billed-cost stop" \
    --budget-amount "${BUDGET_AMOUNT}" \
    --calendar-period month \
    --credit-types-treatment include-all-credits \
    --filter-projects "projects/${PROJECT_NUMBER}" \
    --threshold-rule percent=1.0,basis=current-spend \
    --notifications-rule-pubsub-topic "projects/${PROJECT_ID}/topics/${TOPIC}"
fi

echo "The worker will be deleted after Google reports billed cost reaching ${BUDGET_AMOUNT}."
echo "Budget data is delayed; this is an emergency stop, not a mathematical zero-cost guarantee."
