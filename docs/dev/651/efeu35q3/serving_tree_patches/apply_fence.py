import ast
import re

p = "/root/efeu35q3/staged_hybrid_cache_controller.py"
s = open("/root/efeu35q3/sglang_src/python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py").read()

helper = '''

def _mamba_snapshot_fence_needed(pool_transfers) -> bool:
    """MAMBA-SNAPSHOT-FENCE (picked from htsglang 4291b4c5f8 for efeu-TP14):
    does this write op snapshot recurrent state (a MAMBA pool transfer with
    rows)? Switch SGLANG_WEG2_MAMBA_SNAPSHOT_FENCE (absent here = fence on)."""
    if not pool_transfers:
        return False
    import os as _os

    if _os.environ.get("SGLANG_WEG2_MAMBA_SNAPSHOT_FENCE", "1") in ("0", "false", "False"):
        return False
    from sglang.srt.mem_cache.hicache_storage import PoolName

    for t in pool_transfers:
        if getattr(t, "name", None) != PoolName.MAMBA:
            continue
        idx = getattr(t, "device_indices", None)
        if idx is not None and hasattr(idx, "numel") and int(idx.numel()) > 0:
            return True
    return False
'''
anchor = "device_module = get_device_module()\n"
assert s.count(anchor) == 1, s.count(anchor)
s = s.replace(anchor, anchor + helper, 1)

old = '''            self._record_transfer_indices_on_stream(
                self.write_stream, kv_host_indices, kv_device_indices
            )
        self.ack_write_queue.append(HiCacheAck(start_event, finish_event, op.node_ids))'''
new = '''            self._record_transfer_indices_on_stream(
                self.write_stream, kv_host_indices, kv_device_indices
            )
        if _mamba_snapshot_fence_needed(resolved_pool_transfers):
            # MAMBA-SNAPSHOT-FENCE: the recurrent state is MUTABLE in place (a
            # KV row is append-only). The write stream waits for the compute
            # stream at issue (start_event), but nothing made the compute
            # stream wait for the copy: a forward launched after this issue
            # could change the state rows while the async D2H still reads
            # them, and host / L3 would hold a plausible state of ANOTHER
            # position under this node's key -- a cache hit then resumes from
            # it (efeu-TP14: garbage only on hits). The next forward now starts
            # only after the snapshot has been read.
            device_module.current_stream().wait_event(finish_event)
            self._mamba_fence_n = getattr(self, "_mamba_fence_n", 0) + 1
        self.ack_write_queue.append(HiCacheAck(start_event, finish_event, op.node_ids))'''
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new, 1)
ast.parse(s)
open(p, "w").write(s)
print("staged fence ok")
