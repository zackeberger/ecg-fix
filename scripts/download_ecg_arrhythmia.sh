#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_physionet_download_common.sh"

usage() {
  cat <<'USAGE'
Download ECG Arrhythmia.

Usage:
  bash scripts/download_ecg_arrhythmia.sh [options]

Options:
  --dir PATH             Destination PhysioNet files directory.
                         Default: ./physionet.org/files
  -h, --help             Show this help.

Resulting layout:
  physionet.org/files/ecg-arrhythmia/1.0.0
USAGE
}

parse_common_args "$@"
download_open "https://physionet.org/files/ecg-arrhythmia/1.0.0/"

echo "Done. ECG Arrhythmia downloaded under: $PHYSIONET_DIR"
