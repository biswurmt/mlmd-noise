#!/usr/bin/env bash
# Run the LiLAW MedMNIST replication sweep on a RunPod GPU pod.
#
# Usage: bash scripts/run_medmnist_sweep.sh [BRANCH]
#   BRANCH  git branch to clone (default: current branch)
#
# Prerequisites:
#   - runpodctl configured (RUNPOD_API_KEY set or ~/.runpod/config.toml)
#   - python3 available locally (for JSON parsing)
#   - GH_TOKEN env var set to a GitHub token with read access to this repo
#     (repo is private; obtain via: export GH_TOKEN=$(gh auth token))

set -euo pipefail

BRANCH="${1:-$(git rev-parse --abbrev-ref HEAD)}"
GH_TOKEN="${GH_TOKEN:-$(gh auth token 2>/dev/null || true)}"
REPO="https://${GH_TOKEN}@github.com/biswurmt/mlmd-noise.git"
GPU_ID="NVIDIA H100 80GB HBM3"
TEMPLATE_ID="runpod-torch-v240"
KEY_FILE="${HOME}/.ssh/id_ed25519_runpod"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/../results/medmnist"

# ── 1. SSH key ────────────────────────────────────────────────────────────────
if [[ ! -f "${KEY_FILE}" ]]; then
  echo "Generating dedicated RunPod SSH key at ${KEY_FILE} ..."
  ssh-keygen -t ed25519 -f "${KEY_FILE}" -N '' -C 'runpod'
  runpodctl ssh add-key --key-file "${KEY_FILE}.pub"
  echo "Key registered with RunPod."
else
  echo "SSH key already exists: ${KEY_FILE}"
fi

# ── 2. Create pod ─────────────────────────────────────────────────────────────
echo "Creating pod (GPU: ${GPU_ID}, template: ${TEMPLATE_ID}) ..."
POD_JSON=$(runpodctl pod create \
  --name lilaw-medmnist-sweep \
  --template-id "${TEMPLATE_ID}" \
  --gpu-id "${GPU_ID}" \
  --cloud-type COMMUNITY \
  --container-disk-in-gb 30 \
  --ssh \
  -o json 2>&1 | sed 's/\x1b\[[0-9;]*[mGKHF]//g')
POD_ID=$(echo "${POD_JSON}" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo "Pod created: ${POD_ID}"

# Ensure pod is deleted on exit
cleanup() {
  echo "Deleting pod ${POD_ID} ..."
  runpodctl pod delete "${POD_ID}" || true
  echo "Done."
}
trap cleanup EXIT

# ── 3. Wait for RUNNING ───────────────────────────────────────────────────────
echo "Waiting for pod to start (polling every 10s) ..."
for _ in $(seq 1 60); do
  STATUS=$(runpodctl pod get "${POD_ID}" -o json | \
    python3 -c "import sys,json; print(json.load(sys.stdin).get('desiredStatus',''))")
  echo "  status: ${STATUS}"
  [[ "${STATUS}" == "RUNNING" ]] && break
  sleep 10
done
[[ "${STATUS}" == "RUNNING" ]] || { echo "ERROR: pod never reached RUNNING"; exit 1; }

# Give sshd a moment to start
sleep 15

# ── 4. SSH connection details ─────────────────────────────────────────────────
SSH_INFO=$(runpodctl ssh info "${POD_ID}" -o json)
SSH_HOST=$(echo "${SSH_INFO}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['ip'])")
SSH_PORT=$(echo "${SSH_INFO}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['port'])")
echo "SSH: root@${SSH_HOST}:${SSH_PORT}"

SSH_CMD="ssh -i ${KEY_FILE} -o StrictHostKeyChecking=no -o ConnectTimeout=30 root@${SSH_HOST} -p ${SSH_PORT}"

# ── 5. Remote setup + sweep ───────────────────────────────────────────────────
echo "Running setup and sweep on pod ..."
${SSH_CMD} bash -s << REMOTE
set -euo pipefail
echo "=== Cloning repo (branch: ${BRANCH}) ==="
git clone --branch ${BRANCH} --single-branch ${REPO} /workspace/mlmd-noise

echo "=== Installing dependencies ==="
pip install -q medmnist torchvision timm scikit-learn
pip install -q -e /workspace/mlmd-noise/lilaw-poc --no-deps

echo "=== Running MedMNIST sweep ==="
cd /workspace/mlmd-noise/lilaw-poc
mkdir -p results/medmnist
python -m lilaw_poc.medmnist.experiment 2>&1 | tee /workspace/medmnist_sweep_output.txt
REMOTE

# ── 6. Download results ───────────────────────────────────────────────────────
mkdir -p "${RESULTS_DIR}"
scp -i "${KEY_FILE}" -o StrictHostKeyChecking=no -P "${SSH_PORT}" \
  "root@${SSH_HOST}:/workspace/mlmd-noise/lilaw-poc/results/medmnist/results.json" \
  "${RESULTS_DIR}/results.json"
scp -i "${KEY_FILE}" -o StrictHostKeyChecking=no -P "${SSH_PORT}" \
  "root@${SSH_HOST}:/workspace/medmnist_sweep_output.txt" \
  "${RESULTS_DIR}/sweep_output.txt"

echo ""
echo "Results saved to ${RESULTS_DIR}/"
echo "  results.json   — $(python3 -c "import json; d=json.load(open('${RESULTS_DIR}/results.json')); print(len(d), 'experiments')")"
echo "  sweep_output.txt"
