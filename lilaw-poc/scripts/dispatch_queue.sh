#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/dispatch_queue.sh --job-file JOBS.txt [--gpus 0,1,2,3] [--max-parallel N] [--log-dir DIR]

Behavior:
  - Reads one shell command per line from JOBS.txt.
  - Ignores blank lines and lines starting with '#'.
  - If --gpus is set, binds one running job per listed GPU via CUDA_VISIBLE_DEVICES.
  - Otherwise runs up to --max-parallel jobs concurrently without GPU binding.
EOF
}

JOB_FILE=""
GPU_LIST=""
MAX_PARALLEL=""
LOG_DIR="logs/dispatch"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-file)
      JOB_FILE="$2"
      shift 2
      ;;
    --gpus)
      GPU_LIST="$2"
      shift 2
      ;;
    --max-parallel)
      MAX_PARALLEL="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${JOB_FILE}" ]]; then
  echo "--job-file is required" >&2
  usage >&2
  exit 1
fi

if [[ ! -f "${JOB_FILE}" ]]; then
  echo "Job file not found: ${JOB_FILE}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

declare -a GPUS=()
declare -a GPU_PIDS=()
declare -a JOBS=()

while IFS= read -r line; do
  [[ "${line}" =~ ^[[:space:]]*$ ]] && continue
  [[ "${line}" =~ ^[[:space:]]*# ]] && continue
  JOBS+=("${line}")
done < "${JOB_FILE}"

if [[ "${#JOBS[@]}" -eq 0 ]]; then
  echo "No runnable jobs found in ${JOB_FILE}" >&2
  exit 1
fi

if [[ -n "${GPU_LIST}" ]]; then
  IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
  GPU_PIDS=()
  for _gpu in "${GPUS[@]}"; do
    GPU_PIDS+=("")
  done
  if [[ -z "${MAX_PARALLEL}" ]]; then
    MAX_PARALLEL="${#GPUS[@]}"
  fi
else
  if [[ -z "${MAX_PARALLEL}" ]]; then
    MAX_PARALLEL=1
  fi
fi

echo "Dispatching ${#JOBS[@]} jobs"
echo "Log directory: ${LOG_DIR}"
if [[ "${#GPUS[@]}" -gt 0 ]]; then
  echo "GPU mode: ${GPUS[*]}"
else
  echo "CPU mode: max_parallel=${MAX_PARALLEL}"
fi

running_jobs() {
  jobs -pr | wc -l | tr -d ' '
}

reap_finished_gpus() {
  if [[ "${#GPUS[@]}" -eq 0 ]]; then
    return
  fi
  local i pid
  for ((i = 0; i < ${#GPUS[@]}; i++)); do
    pid="${GPU_PIDS[$i]:-}"
    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
      GPU_PIDS[$i]=""
    fi
  done
}

launch_job() {
  local cmd="$1"
  local slot_label="$2"
  local log_file="$3"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] launching on ${slot_label}: ${cmd}" >&2
  bash -lc "${cmd}" >"${log_file}" 2>&1 &
  local pid=$!
  echo "  pid=${pid} log=${log_file}" >&2
  printf '%s' "${pid}"
}

job_index=0
for cmd in "${JOBS[@]}"; do
  job_index=$((job_index + 1))
  job_name=$(printf "job_%02d" "${job_index}")
  log_file="${LOG_DIR}/${job_name}.log"

  if [[ "${#GPUS[@]}" -eq 0 ]]; then
    while (( $(running_jobs) >= MAX_PARALLEL )); do
      sleep 1
    done
    launch_job "${cmd}" "cpu" "${log_file}" >/dev/null
    continue
  fi

  launched=0
  while true; do
    reap_finished_gpus
    for ((gpu_index = 0; gpu_index < ${#GPUS[@]}; gpu_index++)); do
      gpu="${GPUS[$gpu_index]}"
      if [[ -z "${GPU_PIDS[$gpu_index]:-}" ]]; then
        pid=$(launch_job "CUDA_VISIBLE_DEVICES=${gpu} ${cmd}" "gpu:${gpu}" "${log_file}")
        GPU_PIDS[$gpu_index]="${pid}"
        launched=1
        break
      fi
    done
    if [[ "${launched}" -eq 1 ]]; then
      break
    fi
    sleep 1
  done
done

wait
echo "All jobs completed"
