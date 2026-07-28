#!/bin/bash
# SPDX-License-Identifier: MIT
#
# Baut gpurdma_00_smoke (Stufe-0-Smoke-Test) und gpurdma_04_bench
# (RO-Variante, Leiter-Messungen) fuer den GDR-Fensterlauf.
#
# Folgt /spinning/gdr-uebergabe/BUILD.md: die quelloffenen NVIDIA-Kernelmodul-
# Header werden NICHT im Repo vendored, sondern per Klon aus dem offiziellen
# Repository bezogen, exakt auf die installierte Treiberversion (Layout der
# ioctl-Structs muss passen). Der Klon landet ausserhalb des Repos.
#
# Zusaetzlich: ein Buildzeit-Symbol-Check fuer IBV_ACCESS_RELAXED_ORDERING.
# Das Symbol ist in verbs.h ein enum-Wert, kein Praeprozessor-Makro -- ein
# "#ifdef" im C-Code selbst waere immer falsch, unabhaengig von der
# installierten rdma-core-Version. Der Check laeuft daher hier als kleine
# Testkompilierung; das Ergebnis wird dem echten Build als
# -DHAVE_IBV_ACCESS_RELAXED_ORDERING=(0|1) mitgegeben.
#
# Usage:
#   scripts/probe/build.sh [OGKM_DIR]
#
# Env overrides:
#   OGKM_DIR          Klon-Ziel ausserhalb des Repos (default: /tmp/gdr-window-ogkm)
#   CUDA_HOME         CUDA-Toolkit-Praefix (default: /usr/local/cuda)
#   DRIVER_VERSION    Treiberversion fuer den ogkm-Tag (default: aus
#                     /proc/driver/nvidia/version gelesen)
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT_DIR="$HERE/bin"
mkdir -p "$OUT_DIR"

OGKM_DIR="${1:-${OGKM_DIR:-/tmp/gdr-window-ogkm}}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

case "$OGKM_DIR" in
  "$HERE"*|"$(cd "$HERE/../.." && pwd)"*)
    echo "[FAIL] OGKM_DIR ($OGKM_DIR) liegt innerhalb des Repos -- die Header" >&2
    echo "       duerfen laut BUILD.md nicht vendored werden. Bitte einen" >&2
    echo "       Pfad ausserhalb des Repos angeben (z.B. /tmp/...)." >&2
    exit 1
    ;;
esac

if [ -z "${DRIVER_VERSION:-}" ]; then
  if [ -r /proc/driver/nvidia/version ]; then
    DRIVER_VERSION=$(grep -oE '[0-9]+\.[0-9]+\.[0-9]+' /proc/driver/nvidia/version | head -1)
  fi
fi
if [ -z "${DRIVER_VERSION:-}" ]; then
  echo "[FAIL] Konnte die installierte Treiberversion nicht ermitteln" >&2
  echo "       (/proc/driver/nvidia/version nicht lesbar). Bitte" >&2
  echo "       DRIVER_VERSION=<version> explizit setzen." >&2
  exit 1
fi
echo "[i] installierte Treiberversion: $DRIVER_VERSION"

if [ ! -d "$OGKM_DIR/.git" ]; then
  echo "[i] klone open-gpu-kernel-modules @ $DRIVER_VERSION nach $OGKM_DIR"
  git clone --depth 1 --branch "$DRIVER_VERSION" \
    https://github.com/NVIDIA/open-gpu-kernel-modules.git "$OGKM_DIR"
else
  got=$(git -C "$OGKM_DIR" describe --tags --exact-match 2>/dev/null || echo "?")
  if [ "$got" != "$DRIVER_VERSION" ]; then
    echo "[FAIL] $OGKM_DIR existiert bereits, aber auf Tag '$got', nicht" >&2
    echo "       '$DRIVER_VERSION'. Loeschen oder anderen OGKM_DIR angeben." >&2
    exit 1
  fi
  echo "[i] wiederverwende vorhandenen Klon: $OGKM_DIR @ $got"
fi

INC="-I$CUDA_HOME/include \
     -I$OGKM_DIR/src/common/sdk/nvidia/inc \
     -I$OGKM_DIR/src/nvidia/arch/nvalloc/unix/include \
     -I$OGKM_DIR/kernel-open/common/inc"

for f in \
  "$OGKM_DIR/src/common/sdk/nvidia/inc/nvtypes.h" \
  "$OGKM_DIR/src/common/sdk/nvidia/inc/nvos.h" \
  "$OGKM_DIR/src/nvidia/arch/nvalloc/unix/include/nv-ioctl.h" \
  "$OGKM_DIR/src/nvidia/arch/nvalloc/unix/include/nv_escape.h" \
  "$OGKM_DIR/src/common/sdk/nvidia/inc/class/cl0000.h" \
  "$OGKM_DIR/src/common/sdk/nvidia/inc/class/cl0080.h" \
  "$OGKM_DIR/src/common/sdk/nvidia/inc/ctrl/ctrl0000/ctrl0000gpu.h" \
  "$OGKM_DIR/src/common/sdk/nvidia/inc/ctrl/ctrl0000/ctrl0000unix.h"; do
  [ -r "$f" ] || { echo "[FAIL] erwarteter Header fehlt: $f" >&2; exit 1; }
done

# ---- Buildzeit-Symbol-Check: kennt dieses rdma-core IBV_ACCESS_RELAXED_ORDERING? ----
RO_PROBE=$(mktemp --suffix=.c)
cat > "$RO_PROBE" <<'EOF'
#include <infiniband/verbs.h>
int probe_ro_symbol(void) { return (int)IBV_ACCESS_RELAXED_ORDERING; }
EOF
HAVE_RO=0
if gcc -fsyntax-only "$RO_PROBE" >/dev/null 2>&1; then
  HAVE_RO=1
fi
rm -f "$RO_PROBE"
if [ "$HAVE_RO" = "1" ]; then
  echo "[i] IBV_ACCESS_RELAXED_ORDERING: verfuegbar (--ro wirkt)"
else
  echo "[i] IBV_ACCESS_RELAXED_ORDERING: NICHT verfuegbar in diesem rdma-core"
  echo "    (--ro wird zur Laufzeit sauber deaktiviert, keine Fehlermeldung"
  echo "    bricht den Lauf ab, nur eine Warnung auf stderr)"
fi

BIN="$OUT_DIR/gpurdma_04_bench"
echo "[i] baue $BIN"
gcc -O2 -Wall -DWITH_CUDA -DHAVE_IBV_ACCESS_RELAXED_ORDERING=$HAVE_RO \
    -o "$BIN" "$HERE/gpurdma_04_bench.c" $INC -lcuda -libverbs
echo "[OK] gebaut: $BIN"

SMOKE_BIN="$OUT_DIR/gpurdma_00_smoke"
echo "[i] baue $SMOKE_BIN (Stufe-0-Smoke-Test)"
gcc -O2 -Wall -o "$SMOKE_BIN" "$HERE/gpurdma_00_smoke.c" $INC -lcuda -libverbs
echo "[OK] gebaut: $SMOKE_BIN"

echo "RO_AVAILABLE=$HAVE_RO" > "$OUT_DIR/build_info.env"
echo "DRIVER_VERSION=$DRIVER_VERSION" >> "$OUT_DIR/build_info.env"
echo "OGKM_DIR=$OGKM_DIR" >> "$OUT_DIR/build_info.env"
