# Review mutants (not for commit): READ-ROOM-D 54ea321c06, dangerous directions not in D1-D13.
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
SRC = os.path.join(ROOT, "python", "sglang", "srt", "weg2", "read_room.py")

AUX_KV = """        for c in aux:   # aux first: never an aux host row without the KV row
            if node.component_data[c.component_type].host_value is not None:
                cache._evict_component_and_detach_lru(node, c, target=EvictLayer.HOST, tracker=None)
        _dev, host_freed = cache._evict_component_and_detach_lru(node, base, target=EvictLayer.HOST, tracker=None)
"""
KV_AUX = """        _dev, host_freed = cache._evict_component_and_detach_lru(node, base, target=EvictLayer.HOST, tracker=None)
        for c in aux:   # MUTANT: KV first, aux after
            if node.component_data[c.component_type].host_value is not None:
                cache._evict_component_and_detach_lru(node, c, target=EvictLayer.HOST, tracker=None)
"""

MUTANTS = {
    "MA_kv_before_aux": (AUX_KV, KV_AUX),
    "MB_no_host_leaf_discard": ("        cache.evictable_host_leaves.discard(node)\n", ""),
    "MC_verdict_before_release": (
        "        node.l3_present = True\n        _r12.record_state(cache, node, why=\"read-room\")\n",
        "        node.l3_present = True\n",
    ),
    "MD_no_staging_check": ("        if int(hv.min()) < staging:\n            continue\n", ""),
    "ME_owt_ignored": ("        if node.id in owt or getattr(", "        if False or getattr("),
    "MF_mamba_anchor_ignored_on_aux_only": (
        "        if mamba is not None and node.component_data[mamba].host_value is not None:",
        "        if False and mamba is not None:"),
}
# MC: record_state moved BEFORE the release (verdict says kv=1, the worker keeps its references)
MUTANTS["MC_verdict_before_release"] = (
    "        for c in aux:   # aux first: never an aux host row without the KV row\n",
    "        _r12.record_state(cache, node, why=\"read-room\")\n"
    "        for c in aux:   # aux first: never an aux host row without the KV row\n",
)

TESTS = [
    os.path.join(HERE, "test_weg2_read_room_d_1015.py"),
]

RUNNER = r"""
import sys, importlib
src = open(sys.argv[1]).read()
old, new = sys.argv[2], sys.argv[3]
old = old.encode().decode('unicode_escape'); new = new.encode().decode('unicode_escape')
assert src.count(old) >= 1, 'mutant anchor not found'
if 'MC_' in sys.argv[4]:
    # MC: move the verdict before the release (and remove the original call)
    src = src.replace('        _r12.record_state(cache, node, why="read-room")\n', '', 1)
src = src.replace(old, new, 1)
import sglang.srt.weg2.read_room as m
exec(compile(src, m.__file__, 'exec'), m.__dict__)
import pytest
sys.exit(pytest.main(['-q', '-x', '-p', 'no:cacheprovider', '-p', 'no:warnings'] + sys.argv[5:]))
"""


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_mutant_is_killed(name):
    old, new = MUTANTS[name]
    enc = lambda s: s.encode("unicode_escape").decode()
    env = dict(os.environ)
    r = subprocess.run([sys.executable, "-c", RUNNER, SRC, enc(old), enc(new), name] + TESTS,
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    tail = (r.stdout + r.stderr)[-1500:]
    print(name, "rc", r.returncode, tail)
    assert "mutant anchor not found" not in tail, tail
    assert r.returncode != 0, f"{name} SURVIVED: {tail}"
