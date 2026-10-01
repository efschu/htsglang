import ast

U = "/root/efeu35q3/sglang_src/python/sglang/srt/mem_cache/unified_radix_cache.py"
S = "/root/efeu35q3/sglang_src/python/sglang/srt/managers/scheduler.py"

u = open(U).read()
old1 = """        if not eligible and not symmetric:
            return
"""
new1 = """        if not eligible and not symmetric:
            # efeu-TP14: every silent prefetch exit is now named (01.10.: an
            # 18k omp prompt re-prefilled cold with no prefetch line at all).
            logger.info(
                "HiCache prefetch skipped: reason=%s len=%d threshold=%d occupied=%d limit=%d",
                "not_backuped" if not locally_eligible
                else ("short" if prefetch_length < self.prefetch_threshold else "rate_limited"),
                prefetch_length, self.prefetch_threshold,
                self.cache_controller.prefetch_tokens_occupied,
                self.cache_controller.prefetch_capacity_limit,
            )
            return
"""
assert u.count(old1) == 1, "old1"
u = u.replace(old1, new1, 1)
old2 = """                else:
                    self.dec_host_lock_ref(last_host_node, anchor_lock_params)
                    return
            if host_indices is None and not symmetric:
                self.dec_host_lock_ref(last_host_node, anchor_lock_params)
                return
"""
new2 = """                else:
                    logger.info(
                        "HiCache prefetch skipped: reason=host_kv_full len=%d host_free=%d",
                        len(new_input_tokens), available_size,
                    )
                    self.dec_host_lock_ref(last_host_node, anchor_lock_params)
                    return
            if host_indices is None and not symmetric:
                logger.info("HiCache prefetch skipped: reason=host_kv_alloc len=%d", prefetch_length)
                self.dec_host_lock_ref(last_host_node, anchor_lock_params)
                return
"""
assert u.count(old2) == 1, "old2"
u = u.replace(old2, new2, 1)
old3 = """                    if transfers == []:
                        alloc_failed = True
                        break
"""
new3 = """                    if transfers == []:
                        logger.info(
                            "HiCache prefetch skipped: reason=%s_host_slot len=%d",
                            comp.component_type, prefetch_length,
                        )
                        alloc_failed = True
                        break
"""
assert u.count(old3) == 1, "old3"
u = u.replace(old3, new3, 1)
ast.parse(u)

s = open(S).read()
old4 = """        if not locally_eligible and not group_decides:
            return
"""
new4 = """        if not locally_eligible and not group_decides:
            # efeu-TP14: name the silent exit (matched into a host node whose
            # KV is not backed up, so storage is never asked).
            logger.info(
                "HiCache prefetch skipped: reason=last_host_node_not_backuped matched=%d of %d",
                len(req.prefix_indices) + req.host_hit_length,
                len(req.full_untruncated_fill_ids),
            )
            return
"""
assert s.count(old4) == 1, "old4"
s = s.replace(old4, new4, 1)
ast.parse(s)

for p, txt in ((U, u), (S, s)):
    import shutil, os
    if not os.path.exists(p + ".orig-efeu-prefetchlog"):
        shutil.copy(p, p + ".orig-efeu-prefetchlog")
    open(p + ".new", "w").write(txt)
    os.replace(p + ".new", p)
print("prefetch-skip logging staged")
