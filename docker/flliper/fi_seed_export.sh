#!/bin/bash
# fi_seed_export.sh -- runs INSIDE an existing htsglang/fLLiper image (the FlashInfer seed), fed on stdin by
# make_flat_ctx.sh --write:
#   ssh proxmox "docker run -i --rm --network none --memory 1g --memory-swap 1g --entrypoint /bin/bash <image-id> -s" \
#     < fi_seed_export.sh | tar -xpf - -C <ctx>/assets/fi-seed
# Read-only towards the image: no build, no GPU, no network; ~30 s, < 100 MiB.
#
# Writes ONE POSIX tar to stdout (all diagnostics go to stderr):
#   flashinfer/...     the image's FlashInfer JIT cache /root/.cache/flashinfer (cached_ops + generated, .ninja_log/.ninja_deps)
#                      with its mtimes -- ninja decides freshness by mtime, so they must travel unchanged;
#   DEPS.manifest      "<sha256> <mtime-epoch-s> <path>" for every header/source OUTSIDE the cache that the ninja deps logs of
#                      the cached modules name (FlashInfer data/csrc/cutlass/cccl, tvm_ffi include, CUDA, libstdc++/libc,
#                      gcc include) -- Dockerfile.flliper step 3a gives every dep whose CONTENT is identical in the new build
#                      the seed's mtime back, so ninja finds the seed state ("no work to do"); a dep whose content differs
#                      keeps its new mtime and ninja rebuilds exactly the modules that include it;
#   PROVENANCE.txt     what the seed was built from (lock, overrides, module list, FlashInfer wheel, torch, nvcc, ninja) and
#                      the seed's own freshness (modules, stale in the seed itself -- must be 0).
# Why mtimes at all: pip writes every installed file with the INSTALL time (it does not keep the wheel's mtime), so in a fresh
# build every FlashInfer header is newer than the seed's .ninja_log/.ninja_deps records and ninja would rebuild all modules.
# Measured on htsglang:cu130-weg2-rc12g-27b-nf-flat (14.10.): 65/65 modules dirty after re-stamping the deps, 0/65 after the
# content-checked restore; one header with changed content -> exactly its 1 module dirty.
set -euo pipefail
C=/root/.cache/flashinfer
O=$(mktemp -d /tmp/fi-seed.XXXXXX)
[ -d "$C" ] || { echo "fi-seed: $C missing in this image" >&2; exit 7; }
command -v ninja >/dev/null || { echo "fi-seed: no ninja on PATH" >&2; exit 7; }
n=0; stale=0; : > "$O/deps.raw"
for b in "$C"/*/*/cached_ops/*/; do
  [ -f "$b/build.ninja" ] || continue
  n=$((n + 1))
  ninja -C "$b" -f build.ninja -n 2>/dev/null | grep -q 'no work to do' || { stale=$((stale + 1)); echo "fi-seed: stale in the seed itself: $b" >&2; }
  if [ -f "$b/.ninja_deps" ]; then ninja -C "$b" -f build.ninja -t deps >> "$O/deps.raw"; fi
done
{ grep -E '^ +/' "$O/deps.raw" || true; } | sed -E 's/^ +//' | sort -u | { grep -v "^$C/" || true; } > "$O/deps.txt"
[ -s "$O/deps.txt" ] || { echo "fi-seed: no dependency records in $C (.ninja_deps)" >&2; exit 7; }
while IFS= read -r f; do
  [ -f "$f" ] || { echo "fi-seed: dep listed but missing in the seed: $f" >&2; continue; }
  printf '%s %s %s\n' "$(sha256sum < "$f" | cut -c1-64)" "$(stat -L -c %Y "$f")" "$f"
done < "$O/deps.txt" > "$O/DEPS.manifest"
fiurl=$(python - <<'PY' 2>/dev/null || echo "?"
import importlib.metadata as m, json
d = m.distribution("flashinfer-python")
u = json.loads(d.read_text("direct_url.json") or "{}")
print(d.version, (u.get("archive_info") or {}).get("hash", "no-direct-url"))
PY
)
vers=$(python -c "import importlib.metadata as m; print(' '.join(f'{d}={m.version(d)}' for d in ('torch', 'apache-tvm-ffi', 'flashinfer-python', 'flashinfer-cubin', 'ninja')))" 2>/dev/null || echo "?")
sh256(){ if [ -f "$1" ]; then sha256sum "$1" | cut -c1-64; else echo missing; fi; }
{
  echo "modules=$n"
  echo "stale_in_seed=$stale"
  echo "deps=$(wc -l < "$O/DEPS.manifest")"
  echo "lock_sha256=$(sh256 /opt/htsglang/lock/requirements.lock)"
  echo "overrides_sha256=$(sh256 /opt/htsglang/lock/overrides.txt)"
  echo "flashinfer_modules_sha256=$(sh256 /opt/htsglang/tools/flashinfer_modules.txt)"
  echo "flashinfer_python=$fiurl"
  echo "versions=$vers"
  echo "nvcc=$(/usr/local/cuda/bin/nvcc --version | grep -oE 'V[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
  echo "cache_bytes=$(du -sb "$C" | cut -f1)"
} > "$O/PROVENANCE.txt"
echo "fi-seed: $n modules (stale in the seed: $stale), $(wc -l < "$O/DEPS.manifest") deps, $(du -sh "$C" | cut -f1)" >&2
tar --format=posix -C /root/.cache -cf - flashinfer -C "$O" DEPS.manifest PROVENANCE.txt
