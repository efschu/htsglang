#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Uebersetzungs- und Spill-Beleg fuer den gepipelineten BAR1-Kern.
#
# OHNE KARTE. nvcc uebersetzt offline; ausgefuehrt wird nichts. Deshalb
# CUDA_VISIBLE_DEVICES=99 -- ein Torch-Import in der Extraktion darf keine
# echte Karte anfassen.
#
# Gefragt ist STACK: ein Local-Memory-Spill im Kern ist ein echter
# Perf-Bug (Praezedenz: bar1_netz_kernel STACK 64 -> 0). REG und SHARED
# stehen daneben, weil sie die Belegung mitbestimmen.
#
#   scripts/probe/bar1_pipe_spill.sh [arch ...]      # Vorgabe: 86 120
set -euo pipefail

HIER="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-/spinning/htsglang-gpu/.venv/bin/python}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
CUOBJ="${CUOBJ:-/usr/local/cuda/bin/cuobjdump}"
ARCHES=("$@")
[ ${#ARCHES[@]} -eq 0 ] && ARCHES=(86 120)

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

export CUDA_VISIBLE_DEVICES=99
INC="$("$PY" - <<'EOF'
import sysconfig

from torch.utils.cpp_extension import include_paths

pfade = list(include_paths())
# torch/extension.h zieht Python.h herein. Ohne den Interpreterpfad bricht
# die Uebersetzung ab, bevor sie den Kern ueberhaupt sieht.
pfade.append(sysconfig.get_paths()["include"])
print(" ".join("-I" + p for p in pfade))
EOF
)"

# Den Kernelquelltext als TEXT holen, nicht das Modul importieren: gefragt
# ist genau das, was nvcc zu sehen bekommt.
"$PY" - "$HIER" > "$TMP/pipe.cu" <<'EOF'
import ast
import pathlib
import sys

p = (pathlib.Path(sys.argv[1])
     / "python/sglang/srt/distributed/device_communicators/barlink_bar1_pipe_ext.py")
tree = ast.parse(p.read_text(encoding="utf-8"))
for node in tree.body:
    if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_CUDA_SRC":
        sys.stdout.write(node.value.value)
        break
else:
    raise SystemExit("_CUDA_SRC nicht gefunden")
EOF

for a in "${ARCHES[@]}"; do
    echo "=== sm_$a ==="
    # shellcheck disable=SC2086
    "$NVCC" -std=c++17 -cubin -arch="sm_$a" $INC \
        -o "$TMP/pipe_$a.cubin" "$TMP/pipe.cu" 2>&1 | grep -v '^$' || true
    "$CUOBJ" -res-usage "$TMP/pipe_$a.cubin" \
        | grep -E "Function|REG|STACK|SHARED" \
        | sed -e 's/^[[:space:]]*/  /'
done
