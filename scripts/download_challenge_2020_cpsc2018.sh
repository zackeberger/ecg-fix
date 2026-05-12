#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_physionet_download_common.sh"

usage() {
  cat <<'USAGE'
Download Challenge 2020 CPSC 2018 training data.

Usage:
  bash scripts/download_challenge_2020_cpsc2018.sh [options]

Options:
  --dir PATH             Destination PhysioNet files directory.
                         Default: ./physionet.org/files
  -h, --help             Show this help.

Resulting layout:
  physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018
USAGE
}

parse_common_args "$@"
download_open "https://physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/"

echo "Done. Challenge 2020 CPSC 2018 data downloaded under: $PHYSIONET_DIR"
