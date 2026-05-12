#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Download all ECG benchmark datasets into the default repo layout.

Usage:
  bash scripts/download_all_public_data.sh [options]

Options:
  --dir PATH             Destination PhysioNet files directory.
                         Default: ./physionet.org/files
  --physionet-user USER  PhysioNet username. Required for restricted datasets
                         such as EchoNext after accepting the DUA.
  --skip-restricted      Skip restricted-access datasets.
  -h, --help             Show this help.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for arg in "$@"; do
  case "$arg" in
    -h|--help)
      usage
      exit 0
      ;;
  esac
done

pids=()

bash "$SCRIPT_DIR/download_ptb_xl.sh" "$@" &
pids+=($!)

bash "$SCRIPT_DIR/download_challenge_2020_cpsc2018.sh" "$@" &
pids+=($!)

bash "$SCRIPT_DIR/download_ecg_arrhythmia.sh" "$@" &
pids+=($!)

bash "$SCRIPT_DIR/download_echonext.sh" "$@" &
pids+=($!)

for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "Done. All requested data downloaded."