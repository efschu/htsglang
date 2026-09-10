#!/bin/bash
# Mutants in the danger direction for #1273 S6-BOUNCE B2/B3.
# Each edits the PRODUCT, runs the smoke, and must go RED; the tree is restored
# from the COMMIT after every mutant (08c27d1994) and the baseline re-checked.
set -u
cd /spinning/wt-desk-s6b || exit 2
MOD=python/sglang/srt/weg2/weight_exchange_bounce.py
T=test/registered/unit/weg2/test_weg2_xchg_bounce_execution_smoke_1273.py
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH=/spinning/wt-desk-s6b/python
export OMP_NUM_THREADS=1

run() { timeout 900 python3 -m pytest "$T" -q 2>&1 \
        | grep -oE '[0-9]+ (passed|failed)' | tr '\n' ' '; }

restore() {
  git checkout -- "$MOD"
  # A restore that silently failed would make every later mutant read green,
  # so the guard is on the FILE and not on the exit code alone.
  git diff --quiet -- "$MOD" || { echo "RESTORE FAILED"; exit 3; }
}

mutate() { python3 - "$1" "$2" <<'PY'
import sys, pathlib
path = pathlib.Path("python/sglang/srt/weg2/weight_exchange_bounce.py")
old, new = sys.argv[1], sys.argv[2]
s = path.read_text()
n = s.count(old)
if n != 1:
    print(f"ANCHOR NOT UNIQUE ({n})"); sys.exit(4)
path.write_text(s.replace(old, new))
PY
}

echo "=== BASELINE ==="; run; echo

echo "M1 collect uses src_off as the destination offset (copy-paste)"
mutate 'dst_ptr = int(desc.dst_ptr) + piece.dst_off' \
       'dst_ptr = int(desc.dst_ptr) + piece.src_off' && run; restore; echo

echo "M2 the buffer is never released (leaked pinned post)"
mutate '        ops.destroy_stream(d_stream)
        ops.destroy_stream(c_stream)' \
       '        ops.destroy_stream(d_stream)
        ops.destroy_stream(c_stream)
        _MUTANT_SKIP_CLOSE = True' && \
mutate '        bounce.close()

    result = BounceResult(' \
       '        pass  # MUTANT: buffer never released

    result = BounceResult(' && run; restore; echo

echo "M3 overlap claimed unconditionally (depth 1 lies about the x4 link)"
mutate 'overlap=("ok" if int(depth) >= 2 else "none")' \
       'overlap="ok"' && run; restore; echo

echo "M4 the coverage bound graded on the MEAN instead of the widest"
mutate '    widest = widest_unit(units)
    if widest.nbytes <= int(slot_bytes):
        return
    if terms is None:' \
       '    widest = widest_unit(units)
    if sum(u.nbytes for u in units) / max(1, len(units)) <= int(slot_bytes):
        return
    if terms is None:' && run; restore; echo

echo "M5 deposit not synchronised before the collect reads the slot"
mutate '                ops.synchronize(d_stream)
                deposit_ms' \
       '                deposit_ms' && run; restore; echo

echo "M6 units keyed by parameter instead of by layer"
mutate 'key = (str(getattr(d, "tag", "")), unit_name(getattr(d, "param_name", "")))' \
       'key = (str(getattr(d, "tag", "")), str(getattr(d, "param_name", "")))' \
       && run; restore; echo

echo "M7 a slot is re-deposited without draining its own collect"
mutate '                if inflight[slot] is not None:' \
       '                if False and inflight[slot] is not None:' && run; restore; echo

echo "M8 the geometry is re-derived here instead of read from the ARM term"
mutate 'return int(terms.buffer_bytes) // int(terms.depth), int(terms.depth)' \
       'return int(terms.mean_layer_bytes), int(terms.depth)' && run; restore; echo

echo "=== BASELINE AFTER ==="; run; echo
