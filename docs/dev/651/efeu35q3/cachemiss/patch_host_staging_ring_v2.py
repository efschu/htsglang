"""efeu-TP14 (01.10., v2 after the 19:45:27 crash): HiCache host pool as an
L3 staging ring that KEEPS the upstream parent-first backup invariant.

v1 (patch_host_staging_ring.py) relaxed "a backed-up node's parent is backed
up" and killed the scheduler at 19:45:27 in the on-idle sanity_check
("node 22 backed up but parent 21 not backed up"). v2 never creates that state:

  * a host copy is dropped only if the node is on the device, its storage
    backup was acked complete (l3_persisted), it is unlocked (device and
    host), no write-through is pending, it is not protected, AND NO CHILD IS
    BACKED UP -- leaf-first; after a drop its parent may become a leaf and
    is considered next (LRU order among the leaves);
  * write_backup protects the node being backed up and ALL its ancestors
    (the parent-first rule needs them on the host);
  * prefetch_from_storage protects the request's own path and the anchor's
    ancestors;
  * write_backup's parent-first rule, _evict_device_leaf, sanity_check and
    the scheduler are UNCHANGED.

Consequence (documented limit): one path longer than the host pool still
cannot be staged past the host size (its ancestors are protected); what the
ring frees is the copies of OTHER branches and stale tails, which is what
filled the pool in the 18:14 trace (old chain tail 40k-43k, side requests).

Gate: SGLANG_HICACHE_HOST_STAGING_RING (default 1); 0 = upstream behaviour
(every new line is behind the gate except the l3_persisted attribute).
Applies to the PRISTINE unified_radix_cache.py (.orig-efeu-ring); restores
the pristine scheduler.py (v1 had patched its prefetch gate).
"""
import ast
import os
import shutil
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/root/efeu35q3/sglang_src/python"
URC = os.path.join(ROOT, "sglang/srt/mem_cache/unified_radix_cache.py")
SCH = os.path.join(ROOT, "sglang/srt/managers/scheduler.py")


def sub(s, old, new, label):
    n = s.count(old)
    assert n == 1, f"{label}: anchor found {n}x"
    return s.replace(old, new, 1)


def pristine(path):
    orig = path + ".orig-efeu-ring"
    return open(orig if os.path.exists(orig) else path).read()


s = pristine(URC)
assert "_drop_resident_host_backups" not in s, "base is not pristine"

# l3_persisted: storage backup acked complete
s = sub(s, """                entry = self.ongoing_backup.pop(operation.id, None)
                if entry is not None:
                    node, lock_params = entry
                    self.dec_host_lock_ref(node, lock_params)
""", """                entry = self.ongoing_backup.pop(operation.id, None)
                if entry is not None:
                    node, lock_params = entry
                    self.dec_host_lock_ref(node, lock_params)
                    # efeu-TP14 staging ring: all tokens of the node are in L3.
                    _hv = node.component_data[BASE_COMPONENT_TYPE].host_value
                    if _hv is not None and operation.completed_tokens >= len(_hv):
                        node.l3_persisted = True
""", "ack")
# ... nodes inserted by a storage prefetch are in L3 by construction
s = sub(s, """        new_node.hash_value = hash_value
""", """        new_node.hash_value = hash_value
        new_node.l3_persisted = True  # efeu-TP14: it came from L3
""", "insert_host")
# ... both halves of a split inherit it
s = sub(s, """        new_node.creation_time = child.creation_time
""", """        new_node.creation_time = child.creation_time
        new_node.l3_persisted = getattr(child, "l3_persisted", False)
""", "split")

# write_backup: after evict_host falls short, drop leaf copies off this path
s = sub(s, """            evicted = self.evict_host(needed)
            if evicted < needed:
                return 0
""", """            evicted = self.evict_host(needed)
            if evicted < needed and _host_staging_ring():
                evicted += self._drop_resident_host_backups(
                    needed - evicted, protected=self._path_to_root(node), reason="backup"
                )
            if evicted < needed:
                return 0
""", "write_backup evict")

# prefetch: drop leaf copies off the request's path before truncating
s = sub(s, """            if host_indices is None and not symmetric:
                available_size = self.cache_controller.mem_pool_host.available_size()
""", """            if host_indices is None and not symmetric and _host_staging_ring():
                short = prefetch_length - self.cache_controller.mem_pool_host.available_size()
                if short > 0:
                    self._drop_resident_host_backups(
                        short,
                        protected=self._path_to_root(last_host_node)
                        | self._request_path_nodes(last_host_node, prefetch_key),
                        reason="prefetch",
                    )
                    host_indices = self.cache_controller.mem_pool_host.alloc(
                        prefetch_length
                    )
            if host_indices is None and not symmetric:
                available_size = self.cache_controller.mem_pool_host.available_size()
""", "prefetch")

s = sub(s, """    def evict_host(
""", '''    def _path_to_root(self, node: UnifiedTreeNode) -> set:
        out = set()
        while node is not None:
            out.add(node)
            node = node.parent
        return out

    def _request_path_nodes(self, start: UnifiedTreeNode, key: RadixKey) -> set:
        """Nodes from *start* down along *key* (no split): a prefetch's own path."""
        nodes = {start}
        node = start
        while len(key) > 0:
            child = node.children.get(key.child_key(self.page_size))
            if child is None:
                break
            nodes.add(child)
            plen = child.key.match(key, page_size=self.page_size)
            if plen < len(child.key):
                break
            key = key[plen:]
            node = child
        return nodes

    def _drop_resident_host_backups(
        self, num_tokens: int, protected: set, reason: str = ""
    ) -> int:
        """efeu-TP14 staging ring: free host copies that are redundant because
        the node is on the device AND already persisted in L3. LEAF-FIRST: a
        node whose child is still backed up is never dropped, so the
        parent-first backup invariant holds. Returns KV host tokens freed."""
        import heapq

        def candidate(n: UnifiedTreeNode) -> bool:
            return (
                n is not self.root_node
                and n not in protected
                and not n.evicted
                and n.backuped
                and getattr(n, "l3_persisted", False)
                and n.write_through_pending_id is None
                and not any(c.backuped for c in n.children.values())
                and all(
                    cd.lock_ref == 0 and cd.host_lock_ref == 0
                    for cd in n.component_data
                )
            )

        heap = []
        stack = [self.root_node]
        while stack:
            n = stack.pop()
            stack.extend(n.children.values())
            if candidate(n):
                heap.append((n.last_access_time, n.id, n))
        heapq.heapify(heap)
        freed = dropped = 0
        while freed < num_tokens and heap:
            _, _, n = heapq.heappop(heap)
            if not candidate(n):
                continue
            self._record_remove_event(n, medium=StorageMedium.CPU)
            for comp in self._components_tuple:
                _, hf = self._evict_component_and_detach_lru(
                    n, comp, target=EvictLayer.HOST, tracker=None
                )
                if comp.component_type == BASE_COMPONENT_TYPE:
                    freed += hf
            self._update_evictable_leaf_sets(n)
            dropped += 1
            p = n.parent
            if p is not None and candidate(p):
                heapq.heappush(heap, (p.last_access_time, p.id, p))
        if dropped:
            logger.info(
                "HiCache host staging: dropped %d host copies (%d tokens) of "
                "device-resident nodes already in L3 for %s; wanted %d, host free %d",
                dropped, freed, reason, num_tokens,
                self.cache_controller.mem_pool_host.available_size(),
            )
        return freed

    def evict_host(
''', "methods")

s = sub(s, "\nclass UnifiedTreeNode", '''

def _host_staging_ring() -> bool:
    """efeu-TP14: SGLANG_HICACHE_HOST_STAGING_RING (default on)."""
    import os

    return os.environ.get("SGLANG_HICACHE_HOST_STAGING_RING", "1") != "0"


class UnifiedTreeNode''', "flag fn")
ast.parse(s)
c = pristine(SCH)
ast.parse(c)

if "--dry" in sys.argv:
    print("dry run ok")
    sys.exit(0)
for path, text in ((URC, s), (SCH, c)):
    if not os.path.exists(path + ".orig-efeu-ring"):
        shutil.copy(path, path + ".orig-efeu-ring")
    open(path + ".new", "w").write(text)
    os.replace(path + ".new", path)
print("host staging ring v2 (leaf-first, invariant kept) written -- active at next restart")
