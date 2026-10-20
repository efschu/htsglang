"""S109b/c/d mutants: each one weakens a rule; the S109 test file must go RED on every one. Serial, under the lock."""
import shutil, subprocess, sys
R = "/spinning/wt-s109/python/flliper/srt/"
M = [
 ("M1 secured counts every COMPLETE page", R+"mem_cache/l3_secured.py",
  "out = ok & _live & (gens >= 0) & (np.asarray(sec_gen[idx]) == gens)", "out = ok & _live & (gens >= 0)"),
 ("M2 claim without the uncopied keep list", R+"mem_cache/pool_host/arena_pool.py",
  "_unsec = _l3s.unsecured_keys_lo(arena, getattr(self, \"_backend\", None))",
  "_unsec = __import__('numpy').zeros(0, dtype=__import__('numpy').uint64)"),
 ("M3 anchor_held stats again", R+"mem_cache/pool_host/arena_mamba_pool.py",
  "return bool(listed([stem])[0])", "return stem in self._backend._stat_stems([stem])"),
 ("M4 write-behind does not publish", R+"mem_cache/hicache_storage.py",
  "        _l3s.publish(arena, slots, gens)\n", "        pass\n"),
 ("M5 host-leaf aux stage off", R+"pdflip/host_only_spill.py",
  "if _kinds and not (allow_aux and all(", "if _kinds and not (False and all("),
 ("M6 S109c a worker decides", R+"mem_cache/unified_radix_cache.py",
  "if role == \"worker\" or not aux_cands:", "if not aux_cands:"),
 ("M7 clock without the uncopied keep list", R+"mem_cache/hicache_storage.py",
  "keep, extra_lo=_unsec_lo)", "keep, extra_lo=None)"),
 ("M8 secure_rows_to_l3 counts pending as copied", R+"mem_cache/pool_host/arena_pool.py",
  "out[\"on_disk\"] = int(_ok.sum())", "out[\"on_disk\"] = int(_ok.shape[0]); _pending = _sl[:0]"),
 ("A1a secured ignores the L3 index (record alone)", R+"mem_cache/l3_secured.py",
  "        if callable(listed) and cand.shape[0]:", "        if False and callable(listed) and cand.shape[0]:"),
 ("A1c census of the uncopied ignores the L3 index", R+"mem_cache/l3_secured.py",
  "\n    if callable(listed) and cand.shape[0]:", "\n    if False and callable(listed) and cand.shape[0]:"),
 ("A1b L3 LRU evict leaves the record", R+"mem_cache/hicache_storage.py",
  "_n = _l3s.invalidate_stems(arena, [stem])", "_n = 0"),
 ("A2a drop_unreferenced drops uncopied anchors", R+"mem_cache/pool_host/arena_mamba_pool.py",
  "keep = {int(s) for s, k in zip(complete, ok.tolist()) if not k}", "keep = set()"),
 ("A2b realpath on every clock/claim call", R+"mem_cache/storage/file/hicache_arena.py",
  "    key = _REALPATH.get(path)\n", "    key = None\n"),
 ("A3a claim frees the uncopied race candidates", R+"mem_cache/pool_host/arena_pool.py",
  "_free = list(cands) if _mask is None else [c for c, k in zip(cands, _mask) if k]", "_free = list(cands)"),
 ("A3b handover frees without a copy", R+"mem_cache/l3_secured.py",
  "        if ok:\n            freed = free_evicted_named(", "        if True:\n            freed = free_evicted_named("),
 ("A4 request does not wake the write-behind", R+"mem_cache/l3_secured.py",
  "    WAKE.set()   # S109-A4", "    pass   # S109-A4"),
 ("A5a pass ignores the shared record", R+"mem_cache/hicache_storage.py",
  "_stale |= np.asarray(_pub[0][slots]) != gens", "pass"),
 ("A5b invalidate leaves the write-behind baseline", R+"mem_cache/l3_secured.py",
  "                base[0][_f] = -1", "                pass"),
 ("A6 release queues no LRU promotion", R+"mem_cache/l3_secured.py",
  "            ent[1][st] = None", "            pass"),
 ("A7 quiesce entrance wakes nothing", R+"managers/pdflip_bubble_publish.py",
  "_l3s.queue_unsecured_all(\"quiesce\")", "pass"),
 ("A8a hand-over copies a reaped / reclaimed slot", R+"mem_cache/l3_secured.py",
  "        if not ok_slot:\n", "        if False:\n"),
 ("A5N1 publish without the index re-read", R+"mem_cache/hicache_storage.py",
  "bad = [int(s_) for s_, ok in zip(np.asarray(slots, dtype=np.int64).tolist(), listed) if not ok]", "bad = []"),
 ("A8N1a hand-over does not stamp its candidate", R+"mem_cache/l3_secured.py",
  "touched = bool(touch(int(slot), gen)) if callable(touch) else True", "touched = True"),
 ("A8N1b C touch does not stamp touched_ms", R+"mem_cache/storage/file/arena.c",
  "    atomic_store(&sh->touched_ms, mono_ms());\n    if (atomic_load(&sh->state) != S_EVICTING) return 0;\n    if (gen >= 0",
  "    if (atomic_load(&sh->state) != S_EVICTING) return 0;\n    if (gen >= 0"),
 ("A8N1c no re-check after the copy", R+"mem_cache/l3_secured.py",
  "still, st2, g2 = _hand_still_ours(arena, cand[0], gen)", "still, st2, g2 = True, 3, gen"),
 ("A8N2 freed counted on a refused free", R+"mem_cache/l3_secured.py",
  "if freed is None or int(freed) >= 1:", "if True:"),
 ("A8b failed hand-over copy not counted dropped", R+"mem_cache/l3_secured.py",
  "            HAND_STATS[\"copy_failed\"] += 1\n            _dropped_without_l3(1)", "            HAND_STATS[\"copy_failed\"] += 1"),
]
TEST = "test/registered/unit/mem_cache/test_s109_backup_wall_aux_leaves_1020.py"
TEST2 = "test/registered/unit/mem_cache/test_review_s108_s109_1020.py"   # the review's R8/R11 count as well
TEST3 = "test/registered/unit/mem_cache/test_rereview_s109_a5a8_1020.py"  # and the re-review's Q1/Q3
res = []
_only = sys.argv[1:]                       # optional name prefixes: run only those mutants
for name, path, old, new in M:
    if open(path).read().count(old) != 1:
        raise SystemExit("mutant %s does not match exactly once" % name)
for name, path, old, new in M:
    if _only and not any(name.startswith(o) for o in _only):
        continue
    src = open(path).read()
    assert src.count(old) == 1, (name, src.count(old))
    shutil.copy(path, path + ".mutbak")
    try:
        open(path, "w").write(src.replace(old, new, 1))
        p = subprocess.run(["flock", "/spinning/gpu-arb/pytest.lock", "timeout", "600", "systemd-run", "--scope", "-q",
                            "-p", "MemoryMax=2G", "-E", "PYTHONPATH=python:test/registered/unit/mem_cache:test/registered/unit/pdflip:test/registered/unit/managers",
                            "-E", "CUDA_VISIBLE_DEVICES=", "/spinning/shvllm/.venv/bin/python", "-m", "pytest", "-q",
                            "-p", "no:cacheprovider", "--color=no", "--tb=no", TEST, TEST2, TEST3],
                           cwd="/spinning/wt-s109", capture_output=True, text=True)
        tail = [l for l in p.stdout.splitlines() if "passed" in l or "failed" in l][-1:]
        killed = "failed" in (tail[0] if tail else "")
        invalid = not tail or " error" in tail[0]          # a mutant that does not even import proves nothing
        res.append((name, "INVALID" if invalid else ("KILLED" if killed else "SURVIVED"),
                    tail[0] if tail else p.stdout[-200:]))
        print("%-48s %-9s %s" % res[-1], flush=True)
    finally:
        shutil.move(path + ".mutbak", path)
for r in res:
    print("%-48s %-9s %s" % r)
sys.exit(0 if all(r[1] == "KILLED" for r in res) else 1)
