"""efeu-TP14 (01.10.): the HiCache host pool as a STAGING RING for L3.

Measured on backend_180118.log (18:04-18:24): the host KV pool (0.15 ->
19662 tokens) filled with write-through copies of DEVICE-RESIDENT nodes in the
first minutes of the boot. Upstream host eviction only frees host leaves that
are already evicted from the device, so from then on
  * every write_backup returned 0 (no host room) -> nothing new reached L3
    (1037 tokens written to /var/lib/hicache/kv in 20 min of a 43k chat), and
  * every storage prefetch hit `host_kv_full` (18:14:43, len=25544 free=199).

A host copy of a node that is still on the device AND already persisted in L3
is redundant. This patch lets the host tier drop exactly those copies (oldest
first) when a backup or a prefetch needs room:

  1. node.l3_persisted: set when the storage backup of the node is acked with
     all its tokens, and on nodes inserted from storage by a prefetch;
     inherited across node splits.
  2. _drop_resident_host_backups(): frees host copies of nodes that are on the
     device, l3_persisted, unlocked (device and host), not pending a
     write-through, and not protected; LRU order.
  3. write_backup: after evict_host falls short, drop such copies; the
     parent-first backup invariant also accepts a device-resident parent that
     is l3_persisted (its KV is on the device, its pages are in L3), so a long
     conversation's tail can be staged although the whole path no longer fits
     on the host.
  4. _evict_device_leaf: a not-backed-up node that still has (host-only)
     children is tombstoned in place instead of being unlinked (unlinking it
     would orphan the children; the tombstone is deleted by the existing
     _iteratively_delete_tombstone_leaf once its last child goes).
  5. prefetch_from_storage: on host_kv_full, drop such copies (never on the
     request's own path) before the existing truncate-to-free fallback.
  6. Scheduler._prefetch_kvcache: an l3_persisted last host node is a valid
     prefetch anchor (its hash chain is intact).

Env SGLANG_HICACHE_HOST_STAGING_RING=0 restores the upstream behaviour.
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


s = open(URC).read()
assert "_drop_resident_host_backups" not in s, "already patched"

# 1a. storage backup ack -> l3_persisted
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

# 1b. prefetched-from-storage nodes are in L3 by construction
s = sub(s, """        new_node.hash_value = hash_value
""", """        new_node.hash_value = hash_value
        new_node.l3_persisted = True  # efeu-TP14: it came from L3
""", "insert_host")

# 1c. split inherits
s = sub(s, """        new_node.creation_time = child.creation_time
""", """        new_node.creation_time = child.creation_time
        new_node.l3_persisted = getattr(child, "l3_persisted", False)
""", "split")

# 3a. write_backup: drop redundant host copies before giving up
s = sub(s, """            evicted = self.evict_host(needed)
            if evicted < needed:
                return 0
""", """            evicted = self.evict_host(needed)
            if evicted < needed and _host_staging_ring():
                evicted += self._drop_resident_host_backups(
                    needed - evicted, protected={node}, reason="backup"
                )
            if evicted < needed:
                return 0
""", "write_backup evict")

# 3b. parent-first invariant accepts a resident, persisted parent
s = sub(s, """        if not write_back and (
            node.parent is not self.root_node and not node.parent.backuped
        ):
""", """        if not write_back and (
            node.parent is not self.root_node
            and not node.parent.backuped
            and not (
                _host_staging_ring()
                and not node.parent.evicted
                and getattr(node.parent, "l3_persisted", False)
            )
        ):
""", "write_backup invariant")

# 4. tombstone instead of unlinking a not-backed-up node with children
s = sub(s, """            else:
                # Write-through: node has no backup, delete entirely.
                self._record_remove_event(node, medium=StorageMedium.GPU)
""", """            elif node.children:
                # efeu-TP14 staging ring: a resident node whose host copy was
                # dropped can carry host-only children. Unlinking it would
                # orphan them; tombstone it in place (dead for matching), the
                # tombstone sweep deletes it when its last child goes.
                self._record_remove_event(node, medium=StorageMedium.GPU)
                for comp in self._components_tuple:
                    self._evict_component_and_detach_lru(
                        node, comp, target=EvictLayer.DEVICE, tracker=tracker
                    )
                node.component_data[BASE_COMPONENT_TYPE].value = None
                self.evictable_device_leaves.discard(node)
                self._update_evictable_leaf_sets(node)
                self._update_evictable_leaf_sets(node.parent)
                return
            else:
                # Write-through: node has no backup, delete entirely.
                self._record_remove_event(node, medium=StorageMedium.GPU)
""", "tombstone")

# 5. prefetch: drop redundant host copies before truncating
s = sub(s, """            if host_indices is None and not symmetric:
                available_size = self.cache_controller.mem_pool_host.available_size()
""", """            if host_indices is None and not symmetric and _host_staging_ring():
                short = prefetch_length - self.cache_controller.mem_pool_host.available_size()
                if short > 0:
                    self._drop_resident_host_backups(
                        short,
                        protected=self._request_path_nodes(last_host_node, prefetch_key),
                        reason="prefetch",
                    )
                    host_indices = self.cache_controller.mem_pool_host.alloc(
                        prefetch_length
                    )
            if host_indices is None and not symmetric:
                available_size = self.cache_controller.mem_pool_host.available_size()
""", "prefetch")

# 2. the methods (before evict_host)
s = sub(s, """    def evict_host(
""", '''    def _request_path_nodes(self, start: UnifiedTreeNode, key: RadixKey) -> set:
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
        the node is on the device AND already persisted in L3. LRU order.
        Returns the KV host tokens freed."""
        import heapq

        def candidate(n: UnifiedTreeNode) -> bool:
            return (
                n is not self.root_node
                and n not in protected
                and not n.evicted
                and n.backuped
                and getattr(n, "l3_persisted", False)
                and n.write_through_pending_id is None
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

c = open(SCH).read()
c = sub(c, """        locally_eligible = (
            last_host_node.backuped or last_host_node is self.tree_cache.root_node
        )
""", """        locally_eligible = (
            last_host_node.backuped
            or last_host_node is self.tree_cache.root_node
            # efeu-TP14 staging ring: a resident node whose redundant host copy
            # was dropped is in L3 with an intact hash chain.
            or (
                not last_host_node.evicted
                and getattr(last_host_node, "l3_persisted", False)
            )
        )
""", "scheduler gate")
ast.parse(c)

if "--dry" in sys.argv:
    print("dry run ok")
    sys.exit(0)
for path, text in ((URC, s), (SCH, c)):
    if not os.path.exists(path + ".orig-efeu-ring"):
        shutil.copy(path, path + ".orig-efeu-ring")
    open(path + ".new", "w").write(text)
    os.replace(path + ".new", path)
print("host staging ring patched (active at next restart)")
