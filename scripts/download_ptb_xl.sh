#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_physionet_download_common.sh"

usage() {
  cat <<'USAGE'
Download PTB-XL.

Usage:
  bash scripts/download_ptb_xl.sh [options]

Options:
  --dir PATH             Destination PhysioNet files directory.
                         Default: ./physionet.org/files
  -h, --help             Show this help.

Resulting layout:
  physionet.org/files/ptb-xl/1.0.3
USAGE
}

parse_common_args "$@"
download_open "https://physionet.org/files/ptb-xl/1.0.3/"

echo "Done. PTB-XL downloaded under: $PHYSIONET_DIR"
