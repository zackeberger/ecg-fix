#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_physionet_download_common.sh"

ECG_ARRHYTHMIA_VERSION="1.0.0"
ECG_ARRHYTHMIA_URL="https://physionet.org/files/ecg-arrhythmia/${ECG_ARRHYTHMIA_VERSION}/"
MAX_REPAIR_ATTEMPTS="${MAX_REPAIR_ATTEMPTS:-10}"

usage() {
  cat <<'USAGE'
Download ECG Arrhythmia using SHA256SUMS.txt as both a file manifest and checksum verifier.

This script:
  1. Downloads SHA256SUMS.txt first.
  2. Reads the list of expected files from SHA256SUMS.txt.
  3. Downloads missing files.
  4. Once all files exist, verifies SHA256 checksum values.
  5. Deletes corrupt files and redownloads them.
  6. Repeats until all files exist and all checksums pass.

Usage:
  bash scripts/download_ecg_arrhythmia.sh [options]

Options:
  --dir PATH             Destination PhysioNet files directory.
                         Default: ./physionet.org/files
  -h, --help             Show this help.

Environment variables:
  MAX_REPAIR_ATTEMPTS    Number of repair attempts.
                         Default: 10

Resulting layout:
  physionet.org/files/ecg-arrhythmia/1.0.0
USAGE
}

check_sha256_tool() {
  if command -v sha256sum >/dev/null 2>&1; then
    SHA256_TOOL="sha256sum"
  elif command -v shasum >/dev/null 2>&1; then
    SHA256_TOOL="shasum"
  else
    echo "sha256sum or shasum is required to verify ECG Arrhythmia." >&2
    exit 1
  fi
}

download_manifest() {
  mkdir -p "$DATASET_DIR"

  echo "Downloading ECG Arrhythmia manifest/checksum file: SHA256SUMS.txt"
  wget -N -c -P "$DATASET_DIR" "${ECG_ARRHYTHMIA_URL}SHA256SUMS.txt"
}

list_expected_files() {
  sed -E 's/^[a-fA-F0-9]{64}[[:space:]]+\*?//' "$DATASET_DIR/SHA256SUMS.txt" |
    sed '/^[[:space:]]*$/d'
}

write_missing_files() {
  : > "$MISSING_LIST"

  while IFS= read -r relpath; do
    [[ -z "$relpath" ]] && continue

    if [[ ! -f "$DATASET_DIR/$relpath" ]]; then
      echo "$relpath" >> "$MISSING_LIST"
    fi
  done < <(list_expected_files)
}

download_files_from_list() {
  local list_file="$1"
  local count

  count="$(wc -l < "$list_file" | tr -d ' ')"

  if [[ "$count" -eq 0 ]]; then
    return 0
  fi

  echo "Downloading $count file(s)..."

  while IFS= read -r relpath; do
    [[ -z "$relpath" ]] && continue

    local local_dir
    local_dir="$DATASET_DIR/$(dirname "$relpath")"
    mkdir -p "$local_dir"

    echo "Downloading: $relpath"
    wget -c -P "$local_dir" "${ECG_ARRHYTHMIA_URL}${relpath}"
  done < "$list_file"
}

run_checksum_check() {
  if [[ "$SHA256_TOOL" == "sha256sum" ]]; then
    (
      cd "$DATASET_DIR"
      sha256sum -c SHA256SUMS.txt
    )
  else
    (
      cd "$DATASET_DIR"
      shasum -a 256 -c SHA256SUMS.txt
    )
  fi
}

verify_checksums_and_write_bad_files() {
  : > "$VERIFY_LOG"
  : > "$BAD_LIST"

  set +e
  run_checksum_check > "$VERIFY_LOG" 2>&1
  local status=$?
  set -e

  if [[ "$status" -eq 0 ]]; then
    return 0
  fi

  awk '
    /: FAILED$/ {
      sub(/: FAILED$/, "")
      print
      next
    }

    /: FAILED open or read$/ {
      sub(/: FAILED open or read$/, "")
      print
      next
    }

    /: No such file or directory$/ {
      sub(/^[^:]+: /, "")
      sub(/: No such file or directory$/, "")
      print
      next
    }

    /: cannot open/ {
      sub(/^[^:]+: /, "")
      sub(/: cannot open.*$/, "")
      print
      next
    }
  ' "$VERIFY_LOG" | sort -u > "$BAD_LIST"

  return 1
}

delete_bad_files() {
  local count
  count="$(wc -l < "$BAD_LIST" | tr -d ' ')"

  if [[ "$count" -eq 0 ]]; then
    echo "Checksum failed, but no bad files could be parsed." >&2
    echo "Last checksum output:" >&2
    cat "$VERIFY_LOG" >&2
    exit 1
  fi

  echo "Deleting $count bad/missing file(s) before redownload..."

  while IFS= read -r relpath; do
    [[ -z "$relpath" ]] && continue
    rm -f "$DATASET_DIR/$relpath"
  done < "$BAD_LIST"
}

parse_common_args "$@"
check_wget
check_sha256_tool

DATASET_DIR="${PHYSIONET_DIR}/ecg-arrhythmia/${ECG_ARRHYTHMIA_VERSION}"
MISSING_LIST="$(mktemp)"
BAD_LIST="$(mktemp)"
VERIFY_LOG="$(mktemp)"
trap 'rm -f "$MISSING_LIST" "$BAD_LIST" "$VERIFY_LOG"' EXIT

download_manifest

attempt=1
while [[ "$attempt" -le "$MAX_REPAIR_ATTEMPTS" ]]; do
  echo
  echo "ECG Arrhythmia repair attempt $attempt of $MAX_REPAIR_ATTEMPTS"

  write_missing_files
  missing_count="$(wc -l < "$MISSING_LIST" | tr -d ' ')"

  if [[ "$missing_count" -gt 0 ]]; then
    echo "Missing files found: $missing_count"
    download_files_from_list "$MISSING_LIST"
    attempt=$((attempt + 1))
    continue
  fi

  echo "No missing files. Verifying SHA256 checksums..."

  if verify_checksums_and_write_bad_files; then
    echo
    echo "Done. ECG Arrhythmia is complete and all checksums passed."
    echo "ECG Arrhythmia location: $DATASET_DIR"
    exit 0
  fi

  echo "Checksum verification found bad file(s)."
  delete_bad_files
  download_files_from_list "$BAD_LIST"

  attempt=$((attempt + 1))
done

echo
echo "ECG Arrhythmia still did not pass after $MAX_REPAIR_ATTEMPTS repair attempts." >&2

echo
echo "Remaining missing files:" >&2
write_missing_files
cat "$MISSING_LIST" >&2

echo
echo "Last checksum output:" >&2
cat "$VERIFY_LOG" >&2

exit 1