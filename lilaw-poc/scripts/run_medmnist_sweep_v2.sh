#!/usr/bin/env bash
# Run the LiLAW MedMNIST v2 sweep on a RunPod GPU pod (fire-and-forget).
#
# Changes from v1:
#   - Datasets: bloodmnist + dermamnist
#   - Noise rates: 0%, 30%, 50%
#   - early_stopping_patience=100 (full 100-epoch LR schedule fires)
#   - Results pushed to results/medmnist-sweep-v2
#
# Usage: bash scripts/run_medmnist_sweep_v2.sh [BRANCH]
#   BRANCH  git branch to clone (default: current branch)

set -euo pipefail

BRANCH="${1:-$(git rev-parse --abbrev-ref HEAD)}"
GH_TOKEN="${GH_TOKEN:-$(gh auth token 2>/dev/null || true)}"
REPO="https://${GH_TOKEN}@github.com/biswurmt/mlmd-noise.git"
# GPU priority list: fastest first, fallback to slower if unavailable
GPU_PRIORITY=(
  "NVIDIA H100 80GB HBM3"
  "NVIDIA H100 PCIe"
  "NVIDIA H100 NVL"
  "NVIDIA A100-SXM4-80GB"
  "NVIDIA A100 80GB PCIe"
  "NVIDIA GeForce RTX 4090"
  "NVIDIA RTX A6000"
)
TEMPLATE_ID="runpod-torch-v240"
KEY_FILE="${HOME}/.ssh/id_ed25519_runpod"

# Read RUNPOD_API_KEY from config if not in env
if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
  RUNPOD_API_KEY=$(python3 -c "
import tomllib, pathlib, os
p = pathlib.Path(os.path.expanduser('~/.runpod/config.toml'))
if p.exists():
    d = tomllib.loads(p.read_text())
    print(d.get('default', {}).get('api_key', d.get('api_key', d.get('apikey', ''))))
" 2>/dev/null || true)
fi

if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
  echo "ERROR: RUNPOD_API_KEY not found in env or ~/.runpod/config.toml"
  exit 1
fi

# ── 1. SSH key ────────────────────────────────────────────────────────────────
if [[ ! -f "${KEY_FILE}" ]]; then
  echo "Generating dedicated RunPod SSH key at ${KEY_FILE} ..."
  ssh-keygen -t ed25519 -f "${KEY_FILE}" -N '' -C 'runpod'
  runpodctl ssh add-key --key-file "${KEY_FILE}.pub"
  echo "Key registered with RunPod."
else
  echo "SSH key already exists: ${KEY_FILE}"
fi

# ── 2. Create pod (try GPUs in priority order) ────────────────────────────────
POD_ID=""
GPU_ID=""
for candidate in "${GPU_PRIORITY[@]}"; do
  echo "Trying GPU: ${candidate} ..."
  POD_JSON=$(runpodctl pod create \
    --name lilaw-medmnist-sweep-v2 \
    --template-id "${TEMPLATE_ID}" \
    --gpu-id "${candidate}" \
    --cloud-type COMMUNITY \
    --container-disk-in-gb 30 \
    --ssh \
    -o json 2>&1 | sed 's/\x1b\[[0-9;]*[mGKHF]//g')
  POD_ID=$(echo "${POD_JSON}" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('id', ''))
except Exception:
    print('')
" 2>/dev/null || true)
  if [[ -n "${POD_ID}" ]]; then
    GPU_ID="${candidate}"
    echo "Pod created: ${POD_ID} (GPU: ${GPU_ID})"
    break
  fi
  echo "  unavailable, trying next ..."
done
[[ -n "${POD_ID}" ]] || { echo "ERROR: no GPU available from priority list"; exit 1; }

# NOTE: No cleanup trap — pod manages its own lifecycle.

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

# ── 4. Wait for SSH to become ready ──────────────────────────────────────────
echo "Waiting for SSH ..."
SSH_HOST=""
SSH_PORT=""
for _ in $(seq 1 30); do
  SSH_INFO=$(runpodctl ssh info "${POD_ID}" -o json 2>/dev/null || echo '{}')
  SSH_HOST=$(echo "${SSH_INFO}" | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(d.get('ip', d.get('host', d.get('publicIp', ''))))" 2>/dev/null || true)
  SSH_PORT=$(echo "${SSH_INFO}" | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(d.get('port', d.get('sshPort', '')))" 2>/dev/null || true)
  if [[ -n "${SSH_HOST}" && -n "${SSH_PORT}" ]]; then
    echo "  SSH ready: ${SSH_HOST}:${SSH_PORT}"
    break
  fi
  echo "  SSH not ready yet ..."
  sleep 10
done
[[ -n "${SSH_HOST}" && -n "${SSH_PORT}" ]] || { echo "ERROR: SSH never became ready"; exit 1; }
sleep 5
echo "SSH: root@${SSH_HOST}:${SSH_PORT}"

SSH_CMD="ssh -i ${KEY_FILE} -o StrictHostKeyChecking=no -o ConnectTimeout=30 root@${SSH_HOST} -p ${SSH_PORT}"

# ── 5. Launch detached experiment ─────────────────────────────────────────────
echo "Launching experiment (detached) ..."

${SSH_CMD} \
  XBRANCH="${BRANCH}" \
  XREPO="${REPO}" \
  XAPIKEY="${RUNPOD_API_KEY}" \
  XPODID="${POD_ID}" \
  bash -s << 'REMOTE'
cat > /workspace/run_experiment.sh << SCRIPT
#!/usr/bin/env bash
exec > >(tee /workspace/medmnist_sweep_v2_output.txt) 2>&1

# Always stop pod on exit (success or failure) to avoid burning credits
stop_pod() {
  echo "=== Stopping pod to save credits: \$(date -u) ==="
  curl -s -X POST "https://api.runpod.io/graphql?api_key=${XAPIKEY}" \
    -H "Content-Type: application/json" \
    -d '{"query":"mutation { podStop(input: {podId: \"${XPODID}\"}) { id desiredStatus }}"}'
  echo "=== Pod stop requested ==="
}
trap stop_pod EXIT

set -euo pipefail

echo "=============================="
echo "MedMNIST sweep v2 — fire-and-forget"
echo "Datasets: bloodmnist + dermamnist"
echo "Noise rates: 0%, 30%, 50%  |  3 seeds  |  100 epochs (no early stop)"
echo "Started: \$(date -u)"
echo "=============================="

echo "=== Cloning repo ==="
git clone --branch ${XBRANCH} --single-branch ${XREPO} /workspace/mlmd-noise

echo "=== Installing dependencies ==="
pip install -q medmnist torchvision timm scikit-learn
pip install -q -e /workspace/mlmd-noise/lilaw-poc --no-deps

echo "=== Running MedMNIST sweep v2 ==="
cd /workspace/mlmd-noise/lilaw-poc
mkdir -p results/medmnist_v2
python -m lilaw_poc.medmnist.experiment 2>&1

echo "=== Sweep finished: \$(date -u) ==="

echo "=== Pushing results to git ==="
cd /workspace/mlmd-noise
git config user.email "runpod-sweep@noreply.github.com"
git config user.name "RunPod Sweep"
git checkout -b results/medmnist-sweep-v2
git add -f lilaw-poc/results/medmnist_v2/results.json
git commit -m "results(medmnist-v2): bloodmnist+dermamnist, 100 epochs, no early stop

18 experiments: 2 datasets × 3 noise rates × 3 seeds.
patience=100 ensures full LR schedule fires (milestones at 50, 75)."
git push origin results/medmnist-sweep-v2

echo "=== Results pushed to results/medmnist-sweep-v2: \$(date -u) ==="
SCRIPT
chmod +x /workspace/run_experiment.sh
nohup bash /workspace/run_experiment.sh > /dev/null 2>&1 &
echo "Experiment launched in background (PID: $!)"
REMOTE

echo ""
echo "============================================"
echo "  Pod launched: ${POD_ID}"
echo "  GPU: ${GPU_ID}"
echo "  Branch: ${BRANCH}"
echo "  Results branch: results/medmnist-sweep-v2"
echo "  Estimated runtime: ~3-4 hours"
echo "============================================"
echo ""
echo "The pod will:"
echo "  1. Run 18 experiments (bloodmnist+dermamnist × 3 noise rates × 3 seeds)"
echo "  2. Push results to 'results/medmnist-sweep-v2'"
echo "  3. Stop itself (billing stops, disk retained)"
echo ""
echo "You can safely close your laptop now."
echo ""
echo "To check status:  runpodctl pod get ${POD_ID}"
echo "To check logs:    ssh -i ~/.ssh/id_ed25519_runpod -o StrictHostKeyChecking=no root@${SSH_HOST} -p ${SSH_PORT} 'tail -30 /workspace/medmnist_sweep_v2_output.txt'"
echo "To delete pod:    runpodctl pod delete ${POD_ID}"
