#!/usr/bin/env bash

PHYSIONET_DIR="./physionet.org/files"
PHYSIONET_USER=""
SKIP_RESTRICTED=0

parse_common_args() {
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
}

check_wget() {
  if ! command -v wget >/dev/null 2>&1; then
    echo "wget is required. Install wget, then rerun this script." >&2
    exit 1
  fi
}

download_open() {
  local url="$1"

  check_wget
  mkdir -p "$PHYSIONET_DIR"

  echo "Downloading $url"
  wget -r -N -c -np -nH --cut-dirs=1 -P "$PHYSIONET_DIR" "$url"
}

download_restricted() {
  local url="$1"

  check_wget
  mkdir -p "$PHYSIONET_DIR"

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
