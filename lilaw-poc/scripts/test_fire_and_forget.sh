#!/usr/bin/env bash
# Smoke test for the fire-and-forget RunPod pattern.
# Runs the tiny breast_cancer experiment (569 samples, 5 epochs) on a cheap GPU.
set -euo pipefail

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
GH_TOKEN="${GH_TOKEN:-$(gh auth token 2>/dev/null || true)}"
REPO="https://${GH_TOKEN}@github.com/biswurmt/mlmd-noise.git"
GPU_ID="NVIDIA GeForce RTX 3090"
TEMPLATE_ID="runpod-torch-v240"
KEY_FILE="${HOME}/.ssh/id_ed25519_runpod"
RESULTS_BRANCH="test/fire-and-forget-smoke"

if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
  RUNPOD_API_KEY=$(python3 -c "
import tomllib, pathlib, os
p = pathlib.Path(os.path.expanduser('~/.runpod/config.toml'))
if p.exists():
    d = tomllib.loads(p.read_text())
    print(d.get('default', {}).get('api_key', d.get('api_key', '')))
" 2>/dev/null || true)
fi
[[ -n "${RUNPOD_API_KEY:-}" ]] || { echo "ERROR: no RUNPOD_API_KEY"; exit 1; }

# ── SSH key ───────────────────────────────────────────────────────────────────
if [[ ! -f "${KEY_FILE}" ]]; then
  ssh-keygen -t ed25519 -f "${KEY_FILE}" -N '' -C 'runpod'
  runpodctl ssh add-key --key-file "${KEY_FILE}.pub"
fi

# ── Create pod ────────────────────────────────────────────────────────────────
echo "Creating RTX 3090 pod for smoke test ..."
POD_JSON=$(runpodctl pod create \
  --name lilaw-smoke-test \
  --template-id "${TEMPLATE_ID}" \
  --gpu-id "${GPU_ID}" \
  --cloud-type COMMUNITY \
  --container-disk-in-gb 20 \
  --ssh \
  -o json 2>&1 | sed 's/\x1b\[[0-9;]*[mGKHF]//g')
POD_ID=$(echo "${POD_JSON}" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo "Pod: ${POD_ID}"

# ── Wait for RUNNING ─────────────────────────────────────────────────────────
echo "Waiting for pod ..."
for _ in $(seq 1 60); do
  STATUS=$(runpodctl pod get "${POD_ID}" -o json | \
    python3 -c "import sys,json; print(json.load(sys.stdin).get('desiredStatus',''))")
  echo "  ${STATUS}"
  [[ "${STATUS}" == "RUNNING" ]] && break
  sleep 10
done
[[ "${STATUS}" == "RUNNING" ]] || { echo "ERROR: pod never started"; runpodctl pod delete "${POD_ID}"; exit 1; }
sleep 15

# ── SSH info ──────────────────────────────────────────────────────────────────
SSH_INFO=$(runpodctl ssh info "${POD_ID}" -o json)
SSH_HOST=$(echo "${SSH_INFO}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['ip'])")
SSH_PORT=$(echo "${SSH_INFO}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['port'])")
SSH_CMD="ssh -i ${KEY_FILE} -o StrictHostKeyChecking=no -o ConnectTimeout=30 root@${SSH_HOST} -p ${SSH_PORT}"

# ── Upload + launch detached ──────────────────────────────────────────────────
echo "Launching smoke test (detached) ..."

${SSH_CMD} bash -s << 'UPLOAD'
cat > /workspace/run_smoke.sh << 'INNER'
#!/usr/bin/env bash
set -euo pipefail
exec > >(tee /workspace/smoke_output.txt) 2>&1

echo "=== Smoke test started: $(date -u) ==="
git clone --branch BRANCH_PH --single-branch REPO_PH /workspace/mlmd-noise
pip install -q scikit-learn pandas matplotlib
pip install -q -e /workspace/mlmd-noise/lilaw-poc --no-deps

cd /workspace/mlmd-noise
git checkout -b RESULTS_BRANCH_PH

cd lilaw-poc
mkdir -p results/smoke
python -c "
from lilaw_poc.experiment import run_sweep
run_sweep(datasets=['breast_cancer'], noise_rates=[0.1], seeds=[42], epochs=5, output_dir='results/smoke')
"

echo "=== Pushing results ==="
cd /workspace/mlmd-noise
git config user.email 'runpod-sweep@noreply.github.com'
git config user.name 'RunPod Sweep'
git add lilaw-poc/results/smoke/
git commit -m 'test: smoke test results from RunPod fire-and-forget'
git push origin RESULTS_BRANCH_PH

echo "=== Stopping pod: $(date -u) ==="
curl -s -X POST "https://api.runpod.io/graphql?api_key=APIKEY_PH" \
  -H "Content-Type: application/json" \
  -d '{"query":"mutation { podStop(input: {podId: \"PODID_PH\"}) { id desiredStatus }}"}'
echo "=== Done ==="
INNER
chmod +x /workspace/run_smoke.sh
UPLOAD

${SSH_CMD} bash -s << SUBST
sed -i "s|BRANCH_PH|${BRANCH}|g" /workspace/run_smoke.sh
sed -i "s|REPO_PH|${REPO}|g" /workspace/run_smoke.sh
sed -i "s|RESULTS_BRANCH_PH|${RESULTS_BRANCH}|g" /workspace/run_smoke.sh
sed -i "s|APIKEY_PH|${RUNPOD_API_KEY}|g" /workspace/run_smoke.sh
sed -i "s|PODID_PH|${POD_ID}|g" /workspace/run_smoke.sh
nohup bash /workspace/run_smoke.sh > /dev/null 2>&1 &
echo "Launched (PID: \$!)"
SUBST

echo ""
echo "Pod: ${POD_ID}"
echo "Results will be pushed to branch: ${RESULTS_BRANCH}"
echo "Experiment is running detached. Pod will self-stop when done."
