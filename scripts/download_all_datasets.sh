#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'USAGE'
Download all supported datasets.

Usage:
  bash scripts/download_all_datasets.sh [options]

Options:
  --dir PATH             Destination PhysioNet files directory.
                         Default: ./physionet.org/files
  --physionet-user USER  PhysioNet username for restricted-access datasets.
                         Required for EchoNext after accepting the DUA.
  --skip-restricted      Skip restricted-access datasets such as EchoNext.
  -h, --help             Show this help.

This script runs:
  1. scripts/download_ptb_xl.sh
  2. scripts/download_ecg_arrhythmia.sh
  3. scripts/download_challenge_2020_cpsc2018.sh
  4. scripts/download_echonext.sh
USAGE
}

COMMON_ARGS=()
RESTRICTED_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --dir" >&2
        exit 1
      fi
      COMMON_ARGS+=("--dir" "$2")
      shift 2
      ;;
    --physionet-user)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --physionet-user" >&2
        exit 1
      fi
      RESTRICTED_ARGS+=("--physionet-user" "$2")
      shift 2
      ;;
    --skip-restricted)
      RESTRICTED_ARGS+=("--skip-restricted")
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

run_download() {
  local script="$1"
  shift

  echo "Running $script"
  bash "$SCRIPT_DIR/$script" "$@"
}

run_download "download_ptb_xl.sh" "${COMMON_ARGS[@]}"
run_download "download_ecg_arrhythmia.sh" "${COMMON_ARGS[@]}"
run_download "download_challenge_2020_cpsc2018.sh" "${COMMON_ARGS[@]}"
run_download "download_echonext.sh" "${COMMON_ARGS[@]}" "${RESTRICTED_ARGS[@]}"

echo "Done. All requested datasets downloaded."
