#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_physionet_download_common.sh"

CHALLENGE_VERSION="1.0.2"
CHALLENGE_URL="https://physionet.org/files/challenge-2020/${CHALLENGE_VERSION}/"
CPSC_PREFIX="training/cpsc_2018/"
MAX_REPAIR_ATTEMPTS="${MAX_REPAIR_ATTEMPTS:-10}"

usage() {
  cat <<'USAGE'
Download Challenge 2020 CPSC 2018 training data using the root SHA256SUMS.txt.

This script:
  1. Downloads the main Challenge 2020 SHA256SUMS.txt first.
  2. Keeps only files under training/cpsc_2018/.
  3. Downloads missing CPSC 2018 files.
  4. Once all CPSC 2018 files exist, verifies SHA256 checksum values.
  5. Deletes corrupt CPSC 2018 files and redownloads them.
  6. Repeats until all CPSC 2018 files exist and all checksums pass.

Usage:
  bash scripts/download_challenge_2020_cpsc2018.sh [options]

Options:
  --dir PATH             Destination PhysioNet files directory.
                         Default: ./physionet.org/files
  -h, --help             Show this help.

Environment variables:
  MAX_REPAIR_ATTEMPTS    Number of repair attempts.
                         Default: 10

Resulting layout:
  physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018
USAGE
}

check_sha256_tool() {
  if command -v sha256sum >/dev/null 2>&1; then
    SHA256_TOOL="sha256sum"
  elif command -v shasum >/dev/null 2>&1; then
    SHA256_TOOL="shasum"
  else
    echo "sha256sum or shasum is required to verify Challenge 2020 CPSC 2018." >&2
    exit 1
  fi
}

download_manifest() {
  mkdir -p "$CHALLENGE_DIR"

  echo "Downloading Challenge 2020 root manifest/checksum file: SHA256SUMS.txt"
  wget -N -c -P "$CHALLENGE_DIR" "${CHALLENGE_URL}SHA256SUMS.txt"
}

write_cpsc_manifest() {
  : > "$CPSC_MANIFEST"

  # Keep only checksum entries for training/cpsc_2018/.
  grep -E "[[:space:]]+\*?${CPSC_PREFIX}" "$CHALLENGE_DIR/SHA256SUMS.txt" > "$CPSC_MANIFEST" || true

  local count
  count="$(wc -l < "$CPSC_MANIFEST" | tr -d ' ')"

  if [[ "$count" -eq 0 ]]; then
    echo "No entries for ${CPSC_PREFIX} found in SHA256SUMS.txt." >&2
    exit 1
  fi

  echo "CPSC 2018 manifest entries found: $count"
}

list_expected_files() {
  sed -E 's/^[a-fA-F0-9]{64}[[:space:]]+\*?//' "$CPSC_MANIFEST" |
    sed '/^[[:space:]]*$/d'
}

write_missing_files() {
  : > "$MISSING_LIST"

  while IFS= read -r relpath; do
    [[ -z "$relpath" ]] && continue

    if [[ ! -f "$CHALLENGE_DIR/$relpath" ]]; then
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
    local_dir="$CHALLENGE_DIR/$(dirname "$relpath")"
    mkdir -p "$local_dir"

    echo "Downloading: $relpath"
    wget -c -P "$local_dir" "${CHALLENGE_URL}${relpath}"
  done < "$list_file"
}

run_checksum_check() {
  if [[ "$SHA256_TOOL" == "sha256sum" ]]; then
    (
      cd "$CHALLENGE_DIR"
      sha256sum -c "$CPSC_MANIFEST"
    )
  else
    (
      cd "$CHALLENGE_DIR"
      shasum -a 256 -c "$CPSC_MANIFEST"
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
  ' "$VERIFY_LOG" |
    grep -E "^${CPSC_PREFIX}" |
    sort -u > "$BAD_LIST"

  return 1
}

delete_bad_files() {
  local count
  count="$(wc -l < "$BAD_LIST" | tr -d ' ')"

  if [[ "$count" -eq 0 ]]; then
    echo "Checksum failed, but no CPSC 2018 bad files could be parsed." >&2
    echo "Last checksum output:" >&2
    cat "$VERIFY_LOG" >&2
    exit 1
  fi

  echo "Deleting $count bad/missing CPSC 2018 file(s) before redownload..."

  while IFS= read -r relpath; do
    [[ -z "$relpath" ]] && continue

    case "$relpath" in
      "$CPSC_PREFIX"*)
        rm -f "$CHALLENGE_DIR/$relpath"
        ;;
      *)
        echo "Refusing to delete non-CPSC path: $relpath" >&2
        ;;
    esac
  done < "$BAD_LIST"
}

parse_common_args "$@"
check_wget
check_sha256_tool

CHALLENGE_DIR="${PHYSIONET_DIR}/challenge-2020/${CHALLENGE_VERSION}"
CPSC_DIR="${CHALLENGE_DIR}/${CPSC_PREFIX%/}"

MISSING_LIST="$(mktemp)"
BAD_LIST="$(mktemp)"
VERIFY_LOG="$(mktemp)"
CPSC_MANIFEST="$(mktemp)"
trap 'rm -f "$MISSING_LIST" "$BAD_LIST" "$VERIFY_LOG" "$CPSC_MANIFEST"' EXIT

download_manifest
write_cpsc_manifest

attempt=1
while [[ "$attempt" -le "$MAX_REPAIR_ATTEMPTS" ]]; do
  echo
  echo "Challenge 2020 CPSC 2018 repair attempt $attempt of $MAX_REPAIR_ATTEMPTS"

  write_missing_files
  missing_count="$(wc -l < "$MISSING_LIST" | tr -d ' ')"

  if [[ "$missing_count" -gt 0 ]]; then
    echo "Missing CPSC 2018 files found: $missing_count"
    download_files_from_list "$MISSING_LIST"
    attempt=$((attempt + 1))
    continue
  fi

  echo "No missing CPSC 2018 files. Verifying SHA256 checksums..."

  if verify_checksums_and_write_bad_files; then
    echo
    echo "Done. Challenge 2020 CPSC 2018 is complete and all checksums passed."
    echo "CPSC 2018 location: $CPSC_DIR"
    exit 0
  fi

  echo "Checksum verification found bad CPSC 2018 file(s)."
  delete_bad_files
  download_files_from_list "$BAD_LIST"

  attempt=$((attempt + 1))
done

echo
echo "Challenge 2020 CPSC 2018 still did not pass after $MAX_REPAIR_ATTEMPTS repair attempts." >&2

echo
echo "Remaining missing CPSC 2018 files:" >&2
write_missing_files
cat "$MISSING_LIST" >&2

echo
echo "Last checksum output:" >&2
cat "$VERIFY_LOG" >&2

exit 1