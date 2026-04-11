#!/usr/bin/env bash
# =============================================================================
# deploy.sh — Project B (LangGraph Support Agent) Deployment Script
#
# What this script does:
#   1.  Validates prerequisites
#   2.  Checks base stack exists (deploy project-a first if not)
#   3.  Stores API keys in Secrets Manager (if not already there)
#   4.  Builds and pushes the Docker image to ECR
#   5.  Deploys the project-b service stack (ALB + ECS service)
#   6.  Waits for service to be stable
#   7.  Prints the live URL
#
# NOTE: Project B shares Aurora PostgreSQL and ECS cluster with Project A.
#       Run project-a's deploy.sh first to provision the base infrastructure
#       and ingest the corpus. Project B's deploy assumes data is already loaded.
#
# Usage:
#   export OPENAI_API_KEY="sk-..."
#   export LANGFUSE_PUBLIC_KEY="pk-lf-..."
#   export LANGFUSE_SECRET_KEY="sk-lf-..."
#   bash infra/deploy.sh
#
# Teardown:
#   bash infra/deploy.sh --teardown
#
# Redeploy (new image only, skip secret setup):
#   bash infra/deploy.sh --redeploy
# =============================================================================
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
REGION="ap-south-1"
STACK_BASE="acmera-base"
STACK_SERVICE="acmera-service-b"
ECR_REPO_NAME="acmera-project-b"
IMAGE_TAG="${IMAGE_TAG:-latest}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

step() { echo -e "\n${CYAN}${BOLD}[$(date +%H:%M:%S)]${RESET} ${BOLD}$*${RESET}"; }
ok()   { echo -e "  ${GREEN}✓${RESET} $*"; }
warn() { echo -e "  ${YELLOW}⚠${RESET}  $*"; }
die()  { echo -e "\n${RED}✗ ERROR:${RESET} $*" >&2; exit 1; }

# ── Mode flags ────────────────────────────────────────────────────────────────
TEARDOWN=false
REDEPLOY=false
for arg in "$@"; do
  case $arg in
    --teardown) TEARDOWN=true ;;
    --redeploy) REDEPLOY=true ;;
  esac
done

# ─────────────────────────────────────────────────────────────────────────────
# TEARDOWN MODE
# ─────────────────────────────────────────────────────────────────────────────
if $TEARDOWN; then
  step "Tearing down Project B service stack..."
  aws cloudformation delete-stack --stack-name "${STACK_SERVICE}" --region "${REGION}" 2>/dev/null || warn "Service stack not found"
  aws cloudformation wait stack-delete-complete --stack-name "${STACK_SERVICE}" --region "${REGION}" 2>/dev/null && ok "Service stack deleted"
  echo -e "\n${GREEN}${BOLD}Project B teardown complete.${RESET}"
  echo "  Base stack (VPC, Aurora, Redis) was NOT deleted."
  echo "  To delete it: cd ../project-a-instructor && bash infra/deploy.sh --teardown"
  exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# PRE-FLIGHT CHECKS
# ─────────────────────────────────────────────────────────────────────────────
step "Checking prerequisites..."

command -v aws    &>/dev/null || die "AWS CLI not found."
command -v docker &>/dev/null || die "Docker not found."

if ! $REDEPLOY; then
  for var in OPENAI_API_KEY LANGFUSE_PUBLIC_KEY LANGFUSE_SECRET_KEY; do
    [[ -z "${!var:-}" ]] && die "Missing required environment variable: ${var}"
  done
fi

# Verify base stack exists
if ! aws cloudformation describe-stacks --stack-name "${STACK_BASE}" --region "${REGION}" &>/dev/null; then
  die "Base stack '${STACK_BASE}' not found.\nRun project-a's deploy.sh first:\n  cd ../project-a-instructor && bash infra/deploy.sh"
fi
ok "Base stack found"

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region "${REGION}")
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO_NAME}"
ok "AWS account: ${AWS_ACCOUNT_ID}"
ok "ECR target:  ${ECR_URI}:${IMAGE_TAG}"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — STORE SECRETS (only if not already set)
# Project B needs OpenAI + LangFuse keys (no Cohere — no reranking in agent).
# deploy.sh for project-a stores these same keys, so this is a no-op if
# project-a was deployed first. We still check in case deploying standalone.
# ─────────────────────────────────────────────────────────────────────────────
if ! $REDEPLOY; then
  step "Ensuring API keys are in SSM Parameter Store..."

  store_param() {
    local name="/acmera/${1}"
    local value="${2}"
    aws ssm put-parameter \
      --name      "${name}" \
      --value     "${value}" \
      --type      SecureString \
      --overwrite \
      --region    "${REGION}" >/dev/null
    ok "Stored: ${name}"
  }

  store_param "openai-api-key"      "${OPENAI_API_KEY}"
  store_param "langfuse-public-key" "${LANGFUSE_PUBLIC_KEY}"
  store_param "langfuse-secret-key" "${LANGFUSE_SECRET_KEY}"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — BUILD AND PUSH DOCKER IMAGE
# ─────────────────────────────────────────────────────────────────────────────
step "Authenticating with ECR..."
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin \
    "${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com" >/dev/null
ok "ECR login successful"

step "Building Docker image..."
docker build \
  --platform linux/amd64 \
  -t "${ECR_REPO_NAME}:${IMAGE_TAG}" \
  "${PROJECT_ROOT}"
ok "Image built"

step "Pushing image to ECR..."
docker tag "${ECR_REPO_NAME}:${IMAGE_TAG}" "${ECR_URI}:${IMAGE_TAG}"
docker push "${ECR_URI}:${IMAGE_TAG}"
ok "Image pushed: ${ECR_URI}:${IMAGE_TAG}"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — DEPLOY SERVICE STACK
# ─────────────────────────────────────────────────────────────────────────────
step "Deploying service stack (${STACK_SERVICE})..."
aws cloudformation deploy \
  --template-file    "${SCRIPT_DIR}/service-b.yaml" \
  --stack-name       "${STACK_SERVICE}" \
  --parameter-overrides \
    "ImageUri=${ECR_URI}:${IMAGE_TAG}" \
  --capabilities     CAPABILITY_IAM \
  --region           "${REGION}" \
  --no-fail-on-empty-changeset
ok "Service stack deployed"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — WAIT FOR SERVICE STABLE
# ─────────────────────────────────────────────────────────────────────────────
step "Waiting for ECS service to be stable..."

ECS_CLUSTER=$(aws cloudformation describe-stacks \
  --stack-name "${STACK_BASE}" \
  --query      'Stacks[0].Outputs[?OutputKey==`ECSClusterName`].OutputValue' \
  --output     text --region "${REGION}")

ECS_SERVICE=$(aws cloudformation describe-stacks \
  --stack-name "${STACK_SERVICE}" \
  --query      'Stacks[0].Outputs[?OutputKey==`ECSServiceName`].OutputValue' \
  --output     text --region "${REGION}")

aws ecs wait services-stable \
  --cluster  "${ECS_CLUSTER}" \
  --services "${ECS_SERVICE}" \
  --region   "${REGION}"
ok "Service is stable and healthy"

# ─────────────────────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────────────────────
SERVICE_URL=$(aws cloudformation describe-stacks \
  --stack-name "${STACK_SERVICE}" \
  --query      'Stacks[0].Outputs[?OutputKey==`ServiceURL`].OutputValue' \
  --output     text --region "${REGION}")

echo ""
echo -e "${GREEN}${BOLD}╔════════════════════════════════════════╗${RESET}"
echo -e "${GREEN}${BOLD}║  Project B deployed successfully!       ║${RESET}"
echo -e "${GREEN}${BOLD}╠════════════════════════════════════════╣${RESET}"
echo -e "${GREEN}${BOLD}║${RESET}  URL: ${BOLD}${SERVICE_URL}${RESET}"
echo -e "${GREEN}${BOLD}╚════════════════════════════════════════╝${RESET}"
echo ""
echo "  CloudWatch logs: aws logs tail /ecs/acmera-project-b --follow --region ${REGION}"
echo "  To teardown:     bash infra/deploy.sh --teardown"
echo "  To redeploy:     bash infra/deploy.sh --redeploy"
