"""efeu-TP14 (01.10. 19:45 crash): teach UnifiedRadixCache.sanity_check the two
tree states the host staging ring creates on purpose.

The on-idle invariant checker killed the scheduler at 19:45:27
("node 22 backed up but parent 21 not backed up"): the ring had dropped the
host copy of resident, L3-persisted node 21 and later backed up its child 22,
which the relaxed parent-first rule allows. sanity_check still enforced the
upstream rule (it already exempts write_back for exactly this pair).

  * child backed up, parent NOT backed up: allowed when the parent is on the
    device and l3_persisted (the same condition write_backup now accepts).
  * ...or when the parent is a ring tombstone (child unreachable, swept by
    host eviction together with the tombstone).
  * node with no Full on either layer: allowed for a ring tombstone (marked
    in _evict_device_leaf), which must still have children -- a childless
    tombstone is a real leak and stays an error.

Applies on top of patch_host_staging_ring.py.
"""
import ast
import os
import shutil
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/root/efeu35q3/sglang_src/python"
URC = os.path.join(ROOT, "sglang/srt/mem_cache/unified_radix_cache.py")


def sub(s, old, new, label):
    n = s.count(old)
    assert n == 1, f"{label}: anchor found {n}x"
    return s.replace(old, new, 1)


s = open(URC).read()
assert "_drop_resident_host_backups" in s, "apply patch_host_staging_ring.py first"
assert "ring_tombstone" not in s, "already patched"

s = sub(s, """                node.component_data[BASE_COMPONENT_TYPE].value = None
                self.evictable_device_leaves.discard(node)
                self._update_evictable_leaf_sets(node)
                self._update_evictable_leaf_sets(node.parent)
                return
""", """                node.component_data[BASE_COMPONENT_TYPE].value = None
                node.ring_tombstone = True
                self.evictable_device_leaves.discard(node)
                self._update_evictable_leaf_sets(node)
                self._update_evictable_leaf_sets(node.parent)
                return
""", "tombstone mark")

s = sub(s, """            if not full_dev and not full_hst:
                E(f"node {nid} dead: no Full device and no Full host")
""", """            if not full_dev and not full_hst:
                if getattr(node, "ring_tombstone", False) and node.children:
                    pass  # efeu-TP14 staging ring: swept with its last child
                else:
                    E(f"node {nid} dead: no Full device and no Full host")
""", "sanity dead")

s = sub(s, """                if full_hst and not p_hst and not write_back:
                    E(f"node {nid} backed up but parent {node.parent.id} not backed up")
""", """                if (
                    full_hst
                    and not p_hst
                    and not write_back
                    # efeu-TP14 staging ring: a resident parent already in L3
                    # needs no host copy (write_backup accepts it the same way).
                    and not (p_dev and getattr(node.parent, "l3_persisted", False))
                    # ...or a ring tombstone: the child is unreachable for
                    # matching and is freed by host eviction, which then sweeps
                    # the tombstone.
                    and not getattr(node.parent, "ring_tombstone", False)
                ):
                    E(f"node {nid} backed up but parent {node.parent.id} not backed up")
""", "sanity parent")
ast.parse(s)

if "--dry" in sys.argv:
    print("dry run ok")
    sys.exit(0)
if not os.path.exists(URC + ".orig-efeu-ring-sanity"):
    shutil.copy(URC, URC + ".orig-efeu-ring-sanity")
open(URC + ".new", "w").write(s)
os.replace(URC + ".new", URC)
print("staging ring sanity rules patched (active at next restart)")
