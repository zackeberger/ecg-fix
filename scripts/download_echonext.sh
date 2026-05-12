#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_physionet_download_common.sh"

usage() {
  cat <<'USAGE'
Download EchoNext.

Usage:
  bash scripts/download_echonext.sh [options]

Options:
  --dir PATH             Destination PhysioNet files directory.
                         Default: ./physionet.org/files
  --physionet-user USER  PhysioNet username. Required after accepting the DUA.
  --skip-restricted      Skip this restricted-access dataset.
  -h, --help             Show this help.

Resulting layout:
  physionet.org/files/echonext/1.1.0
USAGE
}

parse_common_args "$@"
download_restricted "https://physionet.org/files/echonext/1.1.0/"

echo "Done. EchoNext downloaded under: $PHYSIONET_DIR"
