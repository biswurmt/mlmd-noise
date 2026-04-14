#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/monitor_dispatch.sh --log-dir DIR [--results-dir DIR] [--process-pattern REGEX]
                                   [--interval SECONDS] [--once]

Behavior:
  - Prints queue, process, and artifact snapshots for a dispatch run.
  - Reads job status files emitted by scripts/dispatch_queue.sh.
  - Repeats every --interval seconds unless --once is provided.
EOF
}

LOG_DIR=""
RESULTS_DIR=""
PROCESS_PATTERN="run_(medmnist|tabular)_matrix.py"
INTERVAL=60
ONCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --results-dir)
      RESULTS_DIR="$2"
      shift 2
      ;;
    --process-pattern)
      PROCESS_PATTERN="$2"
      shift 2
      ;;
    --interval)
      INTERVAL="$2"
      shift 2
      ;;
    --once)
      ONCE=1
      shift
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

if [[ -z "${LOG_DIR}" ]]; then
  echo "--log-dir is required" >&2
  usage >&2
  exit 1
fi

print_status_summary() {
  local status_file job_name state slot start_time end_time exit_code
  if ! compgen -G "${LOG_DIR}/job_*.status" >/dev/null; then
    echo "No status files under ${LOG_DIR}"
    return
  fi

  for status_file in "${LOG_DIR}"/job_*.status; do
    job_name="$(basename "${status_file}" .status)"
    state="$(awk -F= '/^state=/{value=$2} END{print value}' "${status_file}")"
    slot="$(awk -F= '/^slot=/{value=$2} END{print value}' "${status_file}")"
    start_time="$(awk -F= '/^start_time=/{value=$2} END{print value}' "${status_file}")"
    end_time="$(awk -F= '/^end_time=/{value=$2} END{print value}' "${status_file}")"
    exit_code="$(awk -F= '/^exit_code=/{value=$2} END{print value}' "${status_file}")"

    printf '%s slot=%s state=%s start=%s' \
      "${job_name}" "${slot:-unknown}" "${state:-unknown}" "${start_time:-n/a}"
    if [[ -n "${end_time}" ]]; then
      printf ' end=%s' "${end_time}"
    fi
    if [[ -n "${exit_code}" ]]; then
      printf ' exit=%s' "${exit_code}"
    fi
    printf '\n'
  done
}

snapshot() {
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="

  echo "-- gpu --"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used \
      --format=csv,noheader,nounits || true
  else
    echo "nvidia-smi not found"
  fi
  echo

  echo "-- processes --"
  pgrep -af "${PROCESS_PATTERN}" || true
  echo

  echo "-- job statuses --"
  print_status_summary
  echo

  echo "-- recent log files --"
  if [[ -d "${LOG_DIR}" ]]; then
    ls -lt "${LOG_DIR}" | head -n 20 || true
  else
    echo "Log directory not found: ${LOG_DIR}"
  fi
  echo

  if [[ -n "${RESULTS_DIR}" ]]; then
    echo "-- recent result artifacts --"
    if [[ -d "${RESULTS_DIR}" ]]; then
      find "${RESULTS_DIR}" -type f \
        \( -name 'results.json' -o -name 'run.log' -o -name 'suite_config.json' \) \
        -exec ls -lt {} + 2>/dev/null | head -n 20 || true
    else
      echo "Results directory not found: ${RESULTS_DIR}"
    fi
    echo
  fi
}

if [[ "${ONCE}" -eq 1 ]]; then
  snapshot
  exit 0
fi

while true; do
  snapshot
  sleep "${INTERVAL}"
done
