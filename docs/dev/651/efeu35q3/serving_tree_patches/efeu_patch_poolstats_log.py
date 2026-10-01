import ast
import os
import shutil

P = "/root/efeu35q3/sglang_src/python/sglang/srt/managers/scheduler_components/pool_stats_observer.py"
s = open(P).read()
old = """            parts.append(f"mamba usage: {self.mamba_usage:.2f}")
        if not parts:"""
new = """            parts.append(f"mamba usage: {self.mamba_usage:.2f}")
            # efeu-TP14 01.10.: LOG ONLY. "usage" above excludes what the radix
            # tree holds (evictable); without these a cold re-prefill cannot be
            # explained from the log (were anchors / KV still on the device?).
            parts.append(
                f"mamba evictable: {self.mamba_evictable_size}, "
                f"mamba avail: {self.mamba_available_size}, "
                f"kv evictable: {self.full_evictable_size}"
            )
        if not parts:"""
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new, 1)
ast.parse(s)
if not os.path.exists(P + ".orig-efeu"):
    shutil.copy(P, P + ".orig-efeu")
open(P + ".new", "w").write(s)
os.replace(P + ".new", P)
print("pool-stats log extension staged")
