#!/usr/bin/env bash
set -euo pipefail

PHYSIONET_DIR="./physionet.org/files"
PHYSIONET_USER=""
SKIP_RESTRICTED=0

usage() {
  cat <<'EOF'
Download public ECG benchmark datasets into the default repo layout.

Usage:
  bash scripts/download_public_data.sh [options]

Options:
  --dir PATH             Destination PhysioNet files directory.
                         Default: ./physionet.org/files
  --physionet-user USER  PhysioNet username. Required for restricted datasets
                         such as EchoNext after accepting the DUA.
  --skip-restricted      Skip restricted-access datasets.
  -h, --help             Show this help.

The resulting layout matches configs/local.example.json:
  physionet.org/files/ptb-xl/1.0.3
  physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018
  physionet.org/files/ecg-arrhythmia/1.0.0
  physionet.org/files/echonext/1.1.0
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir)
      PHYSIONET_DIR="$2"
      shift 2
      ;;
    --physionet-user)
      PHYSIONET_USER="$2"
      shift 2
      ;;
    --skip-restricted)
      SKIP_RESTRICTED=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v wget >/dev/null 2>&1; then
  echo "wget is required. Install wget, then rerun this script." >&2
  exit 1
fi

mkdir -p "$PHYSIONET_DIR"

download_open() {
  local url="$1"
  echo "Downloading $url"
  wget -r -N -c -np -nH --cut-dirs=1 -P "$PHYSIONET_DIR" "$url"
}

download_restricted() {
  local url="$1"
  if [[ "$SKIP_RESTRICTED" -eq 1 ]]; then
    echo "Skipping restricted dataset: $url"
    return
  fi
  if [[ -z "$PHYSIONET_USER" ]]; then
    echo "Restricted dataset requires PhysioNet login: $url" >&2
    echo "Accept the dataset DUA, then rerun with --physionet-user USER or use --skip-restricted." >&2
    exit 1
  fi

  echo "Downloading restricted dataset $url"
  wget -r -N -c -np -nH --cut-dirs=1 -P "$PHYSIONET_DIR" --user "$PHYSIONET_USER" --ask-password "$url"
}

download_open "https://physionet.org/files/ptb-xl/1.0.3/"
download_open "https://physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/"
download_open "https://physionet.org/files/ecg-arrhythmia/1.0.0/"
download_restricted "https://physionet.org/files/echonext/1.1.0/"

echo "Done. Data downloaded under: $PHYSIONET_DIR"
