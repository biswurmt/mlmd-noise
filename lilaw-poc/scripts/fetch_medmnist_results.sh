#!/usr/bin/env bash
# Fetch MedMNIST results from a RunPod pod and delete it.
#
# Usage: bash scripts/fetch_medmnist_results.sh <POD_ID>
#
# Use this if the pod failed to push results to git, or if you want to
# grab the full sweep output log before deleting.

set -euo pipefail

POD_ID="${1:?Usage: $0 <POD_ID>}"
KEY_FILE="${HOME}/.ssh/id_ed25519_runpod"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/../results/medmnist"

# Check pod status
echo "Checking pod ${POD_ID} ..."
STATUS=$(runpodctl pod get "${POD_ID}" -o json | \
  python3 -c "import sys,json; print(json.load(sys.stdin).get('desiredStatus','UNKNOWN'))")
echo "  Status: ${STATUS}"

if [[ "${STATUS}" == "EXITED" || "${STATUS}" == "STOPPED" ]]; then
  echo "Pod is stopped. Starting it to fetch results ..."
  runpodctl pod start "${POD_ID}"
  for _ in $(seq 1 30); do
    STATUS=$(runpodctl pod get "${POD_ID}" -o json | \
      python3 -c "import sys,json; print(json.load(sys.stdin).get('desiredStatus',''))")
    [[ "${STATUS}" == "RUNNING" ]] && break
    sleep 10
  done
  sleep 15
fi

# Get SSH info
SSH_INFO=$(runpodctl ssh info "${POD_ID}" -o json)
SSH_HOST=$(echo "${SSH_INFO}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['ip'])")
SSH_PORT=$(echo "${SSH_INFO}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['port'])")

# Download results
mkdir -p "${RESULTS_DIR}"
echo "Downloading results ..."
scp -i "${KEY_FILE}" -o StrictHostKeyChecking=no -P "${SSH_PORT}" \
  "root@${SSH_HOST}:/workspace/mlmd-noise/lilaw-poc/results/medmnist/results.json" \
  "${RESULTS_DIR}/results.json" 2>/dev/null && echo "  results.json ✓" || echo "  results.json not found"
scp -i "${KEY_FILE}" -o StrictHostKeyChecking=no -P "${SSH_PORT}" \
  "root@${SSH_HOST}:/workspace/medmnist_sweep_output.txt" \
  "${RESULTS_DIR}/sweep_output.txt" 2>/dev/null && echo "  sweep_output.txt ✓" || echo "  sweep_output.txt not found"

echo ""
echo "Results saved to ${RESULTS_DIR}/"

# Ask before deleting
read -rp "Delete pod ${POD_ID}? [y/N] " confirm
if [[ "${confirm}" =~ ^[Yy]$ ]]; then
  runpodctl pod delete "${POD_ID}"
  echo "Pod deleted."
else
  echo "Pod kept. Delete manually: runpodctl pod delete ${POD_ID}"
fi
