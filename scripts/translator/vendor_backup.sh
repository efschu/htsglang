#!/usr/bin/env bash
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
#
# Back up a translator component (weights, pinned wheel, patched lib) to the
# PRIVATE vendor repository.
#
# Why this exists: several selected components are either restrictively
# licensed or published by an entity that no longer exists (Coqui is defunct;
# its XTTS-v2 mirror is exactly the kind of source that vanishes). A local copy
# on one machine is not a backup. This script puts small files in git and large
# weights in GitHub release assets, with a sha256 manifest so a restore can be
# verified rather than hoped at.
#
# The repository is PRIVATE FOREVER. CPML and CC-BY-NC permit private copies,
# not redistribution. The public fork never references it: the fork only knows
# the neutral backend interface.
#
# Usage:
#   scripts/translator/vendor_backup.sh init
#   scripts/translator/vendor_backup.sh add-file  <path> <license> <source-url>
#   scripts/translator/vendor_backup.sh add-asset <path> <tag> <license> <source-url>
#
# The PAT is read one-shot from its file and never printed. Push output is
# redacted.

set -euo pipefail

REPO_NAME="${TRANSLATOR_VENDOR_REPO:-translator-vendor-private}"
OWNER="${TRANSLATOR_VENDOR_OWNER:-efschu}"
WORK="${TRANSLATOR_VENDOR_WORK:-/spinning/llm_stuff/translator-models/vendor-private}"
PAT_FILE="${PAT_FILE:-/root/GITHUB_PAT}"
# GitHub caps a single release asset at 2 GiB. Stay under it with margin.
SPLIT_SIZE="${TRANSLATOR_VENDOR_SPLIT:-1900m}"

die() { echo "error: $*" >&2; exit 1; }

redact() { sed -E 's#https://[^@]*@#https://<redacted>@#g; s#gh[pousr]_[A-Za-z0-9]+#<redacted>#g'; }

read_pat() {
  [ -r "$PAT_FILE" ] || die "no PAT file at $PAT_FILE"
  tr -d '\r\n' < "$PAT_FILE"
}

api() {
  local method="$1" path="$2" body="${3:-}"
  local pat; pat="$(read_pat)"
  if [ -n "$body" ]; then
    curl -sS -m 60 -X "$method" \
      -H "Authorization: Bearer ${pat}" \
      -H "Accept: application/vnd.github+json" \
      -d "$body" "https://api.github.com${path}"
  else
    curl -sS -m 60 -X "$method" \
      -H "Authorization: Bearer ${pat}" \
      -H "Accept: application/vnd.github+json" \
      "https://api.github.com${path}"
  fi
}

cmd_init() {
  mkdir -p "$WORK"
  local response private
  response="$(api GET "/repos/${OWNER}/${REPO_NAME}")"
  if echo "$response" | grep -q '"id"'; then
    echo "repository already exists"
  else
    echo "creating ${OWNER}/${REPO_NAME} as private"
    response="$(api POST "/user/repos" \
      "{\"name\":\"${REPO_NAME}\",\"private\":true,\"description\":\"Private vendor backup for the #466 translator. Never public.\",\"auto_init\":true}")"
  fi

  # HARD GATE: refuse to proceed unless GitHub itself says private:true.
  # Trusting the create request's intent rather than the API's answer is how a
  # non-commercial checkpoint ends up world-readable.
  private="$(echo "$response" | grep -o '"private"[[:space:]]*:[[:space:]]*[a-z]*' | head -1 | grep -o '[a-z]*$')"
  [ "$private" = "true" ] || die "repository is NOT private (private=${private:-unknown}); refusing to push anything"
  echo "verified private=true"

  if [ ! -d "$WORK/.git" ]; then
    local pat; pat="$(read_pat)"
    git clone "https://${pat}@github.com/${OWNER}/${REPO_NAME}.git" "$WORK" 2>&1 | redact
    git -C "$WORK" config user.name efschu
    git -C "$WORK" config user.email efschu@users.noreply.github.com
  fi
  [ -f "$WORK/MANIFEST.md" ] || cat > "$WORK/MANIFEST.md" <<'MANIFEST'
# Vendor manifest — #466 live translator

Private backup of components that are restrictively licensed, gated, or
published by a source that may disappear. **This repository is private
forever.** The licenses here permit private copies, not redistribution.

Restore for a split asset:

    cat <name>.part-* > <name>
    sha256sum -c <name>.sha256

| Component | Kind | Location | sha256 | License | Source |
|---|---|---|---|---|---|
MANIFEST
  echo "vendor repo ready at $WORK"
}

cmd_add_file() {
  local path="$1" license="$2" source="$3"
  [ -f "$path" ] || die "no such file: $path"
  local name sum
  name="$(basename "$path")"
  sum="$(sha256sum "$path" | cut -d' ' -f1)"
  mkdir -p "$WORK/files"
  cp "$path" "$WORK/files/$name"
  printf '| `%s` | file | `files/%s` | `%s` | %s | %s |\n' \
    "$name" "$name" "$sum" "$license" "$source" >> "$WORK/MANIFEST.md"
  git -C "$WORK" add -A
  git -C "$WORK" commit -m "Add ${name} (${license})" 2>&1 | redact
  git -C "$WORK" push 2>&1 | redact
  echo "backed up $name"
}

cmd_add_asset() {
  local path="$1" tag="$2" license="$3" source="$4"
  [ -f "$path" ] || die "no such file: $path"
  local name sum staging pat upload_url
  name="$(basename "$path")"
  sum="$(sha256sum "$path" | cut -d' ' -f1)"
  staging="$(mktemp -d)"
  trap 'rm -rf "$staging"' RETURN

  echo "splitting $name into <= ${SPLIT_SIZE} parts"
  split -b "$SPLIT_SIZE" -d "$path" "$staging/${name}.part-"

  local release
  release="$(api GET "/repos/${OWNER}/${REPO_NAME}/releases/tags/${tag}")"
  if ! echo "$release" | grep -q '"upload_url"'; then
    release="$(api POST "/repos/${OWNER}/${REPO_NAME}/releases" \
      "{\"tag_name\":\"${tag}\",\"name\":\"${tag}\",\"body\":\"Vendor assets for ${tag}. Private.\"}")"
  fi
  upload_url="$(echo "$release" | grep -o '"upload_url"[^,]*' | head -1 | cut -d'"' -f4 | cut -d'{' -f1)"
  [ -n "$upload_url" ] || die "could not resolve the release upload URL for ${tag}"

  pat="$(read_pat)"
  local release_id
  release_id="$(echo "$release" | grep -o '"id"[[:space:]]*:[[:space:]]*[0-9]*' | head -1 | grep -o '[0-9]*$')"

  for part in "$staging/${name}".part-*; do
    # GitHub answers 422 for a name that already exists on the release, which
    # is exactly what a retry after a failed upload hits -- and the stale asset
    # is the truncated one. Delete before re-uploading so a retry converges
    # instead of silently keeping the broken copy.
    local part_name existing_id
    part_name="$(basename "$part")"
    existing_id="$(api GET "/repos/${OWNER}/${REPO_NAME}/releases/${release_id}/assets?per_page=100" \
      | tr ',' '\n' | grep -B1 "\"name\": *\"${part_name}\"" | grep -o '"id": *[0-9]*' \
      | grep -o '[0-9]*$' | head -1)"
    if [ -n "$existing_id" ]; then
      echo "  replacing existing asset ${part_name}"
      api DELETE "/repos/${OWNER}/${REPO_NAME}/releases/assets/${existing_id}" > /dev/null
    fi
  done

  for part in "$staging/${name}".part-*; do
    echo "uploading $(basename "$part") ($(stat -c %s "$part") bytes)"
    # `--data-binary @file` reads the WHOLE file into memory and dies with
    # "out of memory" on a multi-gigabyte checkpoint -- found by running it.
    # `-T` streams from disk; `-X POST` keeps the method GitHub's upload API
    # requires, which `-T` would otherwise turn into a PUT.
    curl -sS -m 7200 -X POST -T "${part}" \
      -H "Authorization: Bearer ${pat}" \
      -H "Content-Type: application/octet-stream" \
      "${upload_url}?name=$(basename "$part")" -o /dev/null -w '  http=%{http_code}\n' | redact
  done

  printf '| `%s` | release asset | tag `%s`, parts `%s.part-*` | `%s` | %s | %s |\n' \
    "$name" "$tag" "$name" "$sum" "$license" "$source" >> "$WORK/MANIFEST.md"
  git -C "$WORK" add -A
  git -C "$WORK" commit -m "Record ${name} in release ${tag} (${license})" 2>&1 | redact
  git -C "$WORK" push 2>&1 | redact
  echo "backed up $name to release $tag"
}

case "${1:-}" in
  init)      cmd_init ;;
  add-file)  shift; [ $# -eq 3 ] || die "add-file <path> <license> <source-url>"; cmd_add_file "$@" ;;
  add-asset) shift; [ $# -eq 4 ] || die "add-asset <path> <tag> <license> <source-url>"; cmd_add_asset "$@" ;;
  *) die "usage: $0 {init|add-file|add-asset}" ;;
esac
