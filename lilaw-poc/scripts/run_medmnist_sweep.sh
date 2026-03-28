#!/usr/bin/env bash
# Run the LiLAW MedMNIST replication sweep on a RunPod GPU pod (fire-and-forget).
#
# The pod runs the experiment, pushes results to a git branch, then stops itself.
# You can close your laptop after launch — no persistent SSH session needed.
#
# Usage: bash scripts/run_medmnist_sweep.sh [BRANCH]
#   BRANCH  git branch to clone (default: current branch)
#
# After the pod finishes, fetch results with:
#   git pull origin feature/lilaw-poc
#   cat lilaw-poc/results/medmnist/results.json
#
# To manually check pod status:
#   runpodctl pod get <POD_ID>
#
# To manually fetch results + delete pod:
#   bash scripts/fetch_medmnist_results.sh <POD_ID>
#
# Prerequisites:
#   - runpodctl configured (RUNPOD_API_KEY set or ~/.runpod/config.toml)
#   - python3 available locally (for JSON parsing)
#   - GH_TOKEN env var set to a GitHub token with repo write access
#     (obtain via: export GH_TOKEN=$(gh auth token))

set -euo pipefail

BRANCH="${1:-$(git rev-parse --abbrev-ref HEAD)}"
GH_TOKEN="${GH_TOKEN:-$(gh auth token 2>/dev/null || true)}"
REPO="https://${GH_TOKEN}@github.com/biswurmt/mlmd-noise.git"
GPU_ID="NVIDIA H100 80GB HBM3"
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

# Give sshd a moment to start
sleep 15

# ── 4. SSH connection details ─────────────────────────────────────────────────
SSH_INFO=$(runpodctl ssh info "${POD_ID}" -o json)
SSH_HOST=$(echo "${SSH_INFO}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['ip'])")
SSH_PORT=$(echo "${SSH_INFO}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['port'])")
echo "SSH: root@${SSH_HOST}:${SSH_PORT}"

SSH_CMD="ssh -i ${KEY_FILE} -o StrictHostKeyChecking=no -o ConnectTimeout=30 root@${SSH_HOST} -p ${SSH_PORT}"

# ── 5. Launch detached experiment ─────────────────────────────────────────────
# Upload a self-contained script, then run it via nohup.
# The script: installs deps → runs sweep → git pushes results → stops pod.
echo "Uploading and launching experiment (detached) ..."

${SSH_CMD} bash -s << 'UPLOAD'
cat > /workspace/run_experiment.sh << 'INNER_SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
exec > >(tee /workspace/medmnist_sweep_output.txt) 2>&1

echo "=============================="
echo "MedMNIST sweep — fire-and-forget"
echo "Started: $(date -u)"
echo "=============================="

# ── Install ──
echo "=== Cloning repo ==="
git clone --branch BRANCH_PLACEHOLDER --single-branch REPO_PLACEHOLDER /workspace/mlmd-noise

echo "=== Installing dependencies ==="
pip install -q medmnist torchvision timm scikit-learn
pip install -q -e /workspace/mlmd-noise/lilaw-poc --no-deps

# ── Run experiment ──
echo "=== Running MedMNIST sweep ==="
cd /workspace/mlmd-noise/lilaw-poc
mkdir -p results/medmnist
python -m lilaw_poc.medmnist.experiment 2>&1

echo "=== Sweep finished: $(date -u) ==="

# ── Push results to git ──
echo "=== Pushing results to git ==="
cd /workspace/mlmd-noise
git config user.email "runpod-sweep@noreply.github.com"
git config user.name "RunPod Sweep"
git add lilaw-poc/results/medmnist/results.json
git commit -m "results(medmnist): add sweep results from RunPod

Automated commit from RunPod H100 sweep pod."
git push origin BRANCH_PLACEHOLDER

echo "=== Results pushed to BRANCH_PLACEHOLDER ==="

# ── Stop pod (stops billing, retains disk) ──
echo "=== Stopping pod to save credits ==="
curl -s -X POST "https://api.runpod.io/graphql?api_key=APIKEY_PLACEHOLDER" \
  -H "Content-Type: application/json" \
  -d '{"query":"mutation { podStop(input: {podId: \"PODID_PLACEHOLDER\"}) { id desiredStatus }}"}'

echo "=== Pod stop requested. Goodbye! ==="
INNER_SCRIPT
chmod +x /workspace/run_experiment.sh
UPLOAD

# Now substitute placeholders and launch
${SSH_CMD} bash -s << SUBSTITUTE
sed -i "s|BRANCH_PLACEHOLDER|${BRANCH}|g" /workspace/run_experiment.sh
sed -i "s|REPO_PLACEHOLDER|${REPO}|g" /workspace/run_experiment.sh
sed -i "s|APIKEY_PLACEHOLDER|${RUNPOD_API_KEY}|g" /workspace/run_experiment.sh
sed -i "s|PODID_PLACEHOLDER|${POD_ID}|g" /workspace/run_experiment.sh
nohup bash /workspace/run_experiment.sh > /dev/null 2>&1 &
echo "Experiment launched in background (PID: \$!)"
SUBSTITUTE

echo ""
echo "============================================"
echo "  Pod launched: ${POD_ID}"
echo "  GPU: ${GPU_ID}"
echo "  Branch: ${BRANCH}"
echo "============================================"
echo ""
echo "The pod will:"
echo "  1. Run the MedMNIST sweep (~40-55 min)"
echo "  2. Push results to '${BRANCH}'"
echo "  3. Stop itself (billing stops, disk retained)"
echo ""
echo "You can safely close your laptop now."
echo ""
echo "To check status:   runpodctl pod get ${POD_ID}"
echo "To fetch results:  git pull origin ${BRANCH}"
echo "To delete pod:     runpodctl pod delete ${POD_ID}"
