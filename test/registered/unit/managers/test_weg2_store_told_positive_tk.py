"""TK (26.09.): the P paths that never ran with told > 0 since 16.09.

Until SGLANG_WEG2_TOLD_PROBE_TREE_KEY=1 (#1416d) the #1416 clamp probe asked
unigram keys on P's bigram tree and every told was 0. From the first agent
boot with the switch on, told > 0 is the normal case. Each test here feeds a
POSITIVE told through the shipped module and names the path it guards:

  path 1  #1419 cap: told counts KEYS; the cap is a RAW-token limit (bigram
          tree: +1), and the #1400 record is span-relative (host insert at
          last_host_node) -- a device head is lost, and a follower with a
          head below a relative told refuses by name (rank exit). Absolute
          told (TW's head + span form, generalised) fixes both.
  path 2  #1416b: PP0 overwrites its own record with a clamped told > 0.
  path 3  #1442: hand-off keys are PP0-authoritative -- the follower adopts
          PP0's key source instead of reading the file at its own time.
  path 4  #1443/#1455: a rid parked in the dormant hold (or the post-wake
          settle) is not dropped from PP0's held set.

The tree double models the one fact that matters here: the registration's
span is [head, limit) and the completion record counts the SPAN (the insert is
rooted at last_host_node, unified_radix_cache.py `_insert_helper_host`).
"""

import types
from types import SimpleNamespace

import pytest

from sglang.srt.managers import weg2_store_told as m


class _SpanTree:
    """Tree double: registration from `head` up to `limit` (raw tokens), a
    completed read records the SPAN it retained (span-relative, #1176)."""

    def __init__(self, is_eagle=False, store_extent=10**9):
        self.is_eagle = is_eagle
        self.store_extent = store_extent  # keys the store can serve
        self._prefetch_completed_tokens = {}
        self.prefetch_loaded_tokens_by_reqid = {}
        self.ongoing = set()

    def register(self, rid, head, limit_raw):
        keys_end = limit_raw - (1 if self.is_eagle else 0)
        end = min(keys_end, self.store_extent)
        span = max(0, end - head)
        self.ongoing.add(rid)
        self._pending = getattr(self, "_pending", {})
        self._pending[rid] = span

    def check_prefetch_progress(self, rid):
        if rid in self.ongoing:
            self.ongoing.discard(rid)
            span = self._pending.pop(rid)
            self._prefetch_completed_tokens[rid] = span
            self.prefetch_loaded_tokens_by_reqid[rid] = span
        return True

    def completed_prefetch_tokens(self, rid):
        return self._prefetch_completed_tokens.get(rid)

    def pop_prefetch_loaded_tokens(self, rid):
        return self.prefetch_loaded_tokens_by_reqid.pop(rid, 0)


class _Sched:
    def __init__(self, pp_rank, head, n_ids, is_eagle=True, store_extent=10**9):
        self.ps = SimpleNamespace(pp_rank=pp_rank, pp_size=3, tp_size=1)
        self.enable_hicache_storage = True
        self.tree_cache = _SpanTree(is_eagle=is_eagle, store_extent=store_extent)
        self.waiting_queue = []
        self.pp_flip_counters = None
        self._weg2_store_told = {}
        self._weg2_store_held = {}
        self._weg2_store_told_armed = True
        self.head = head
        self.n_ids = n_ids
        self.registered = []

    def _prefetch_kvcache(self, req, rematch=True, limit_tokens=None):
        # the real registration: head = device+host match, span up to
        # min(input_len - 1, limit) raw tokens; the head is stamped (#1176)
        req._prefetch_registered_prefix_len = self.head
        req.prefix_indices = [0] * self.head
        req.host_hit_length = 0
        match_end = self.n_ids - 1
        if limit_tokens is not None:
            match_end = min(match_end, int(limit_tokens))
        self.registered.append((req.rid, limit_tokens))
        if match_end - (1 if self.tree_cache.is_eagle else 0) <= self.head:
            return "declined:too_short"
        self.tree_cache.register(req.rid, self.head, match_end)
        return "issued"


def _req(rid, n_ids):
    return SimpleNamespace(rid=rid, prefetch_deferred=None, origin_input_ids=list(range(n_ids)),
                           extra_key=None)


def _note():
    return lambda *a: None


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in (m.ENV_ARMED, m.ENV_TREE_KEY, "SGLANG_WEG2_TOLD_ABSOLUTE", "SGLANG_WEG2_TOLD_PACED",
              "SGLANG_WEG2_P_TWIN_DEFER"):
        monkeypatch.delenv(k, raising=False)
    # no store probe in these doubles: the clamp is the identity (unaskable)
    monkeypatch.setattr(m, "_anchor_clamp", lambda s, r, told: int(told))
    yield


def _ring(p0, f, rid, n_ids):
    """intake on both, PP0 publishes, the follower absorbs; returns the wire."""
    r0, r1 = _req(rid, n_ids), _req(rid, n_ids)
    p0.waiting_queue.append(r0)
    f.waiting_queue.append(r1)
    m.intake(p0, r0, lambda g: None)
    m.intake(f, r1, lambda g: None)
    wire = m.pp0_publish(p0, [])
    rest = m.follower_absorb(f, list(wire))
    assert rest == []
    return r0, r1, wire


# ---- path 1: #1419 cap, keys vs raw tokens --------------------------------


def test_path1_cap_reaches_told_keys_on_a_bigram_tree():
    """told counts KEYS (follower_limit_tokens adds the bigram +1 for the same
    reason); the #1419 cap is the raw-token `limit` of RadixKey, whose bigram
    view has limit-1 keys. A cap of `told` raw tokens stops one key short of
    the anchor that told names -- the match falls back to an earlier anchor."""
    from sglang.srt.managers.schedule_batch import _weg2_cap_key_limit
    from sglang.srt.mem_cache.radix_cache import RadixKey

    s = _Sched(pp_rank=1, head=0, n_ids=30000, is_eagle=True)
    req = _req("r1", 30000)
    s._weg2_store_told["r1"] = 22934
    s._weg2_store_told_satisfied = {"r1": 22934}
    assert m.admission(s, req, _note()) == 0
    key = RadixKey(req.origin_input_ids, None, limit=_weg2_cap_key_limit(req, None))
    key, _ = key.maybe_to_bigram_view(True)
    assert len(key) == 22934, "the capped match must be able to reach told keys"


def test_path1_cap_unigram_tree_unchanged_and_told_zero_matches_nothing():
    from sglang.srt.managers.schedule_batch import _weg2_cap_key_limit
    from sglang.srt.mem_cache.radix_cache import RadixKey

    s = _Sched(pp_rank=1, head=0, n_ids=100, is_eagle=False)
    req = _req("u", 100)
    s._weg2_store_told["u"] = 40
    s._weg2_store_told_satisfied = {"u": 40}
    m.admission(s, req, _note())
    assert _weg2_cap_key_limit(req, None) == 40
    sb = _Sched(pp_rank=1, head=0, n_ids=100, is_eagle=True)
    rz = _req("z", 100)
    sb._weg2_store_told["z"] = 0
    sb._weg2_store_told_satisfied = {"z": 0}
    m.admission(sb, rz, _note())
    key, _ = RadixKey(rz.origin_input_ids, None, limit=_weg2_cap_key_limit(rz, None)).maybe_to_bigram_view(True)
    assert len(key) == 0, "told 0 still matches nothing on a bigram tree"


def test_path1_kept_verdict_uses_the_same_cap():
    """H91's kept verdict re-plants the cap on a later visit: same unit."""
    from sglang.srt.weg2 import p_intake

    s = _Sched(pp_rank=1, head=0, n_ids=30000, is_eagle=True)
    req = _req("k1", 30000)
    s._weg2_store_told["k1"] = 22934
    s._weg2_store_told_satisfied = {"k1": 22934}
    s.waiting_queue.append(req)
    assert p_intake.told_admission(s, req, _note(), m.admission) == 0
    req._weg2_prefix_cap = None
    assert p_intake.told_admission(s, req, _note(), m.admission) == 0  # kept
    assert req._weg2_prefix_cap == 22935


def test_path1_relative_told_with_a_follower_head_refuses_by_name():
    """The danger as shipped (switch off): agent turn with a 4096-token device
    head (the shared system prompt, prefilled on P this epoch) and a 98k
    store span. PP0's record is the SPAN (94208); the follower registers up
    to that absolute position from its own head and completes 90112 --
    Weg2StoreToldMismatch, rank exit, group death."""
    p0 = _Sched(0, head=4096, n_ids=100000, store_extent=98304)
    f = _Sched(1, head=4096, n_ids=100000, store_extent=98304)
    r0, r1, wire = _ring(p0, f, "weg2-3-7", 100000)
    assert wire[0].told == 94208
    with pytest.raises(m.Weg2StoreToldMismatch, match="told=94208 own_prefix=90112"):
        m.admission(f, r1, _note())


def test_path1_absolute_told_admits_head_plus_span_on_every_rank(monkeypatch):
    """SGLANG_WEG2_TOLD_ABSOLUTE (default: follows the tree-key switch): told
    = PP0's registered head + span (TW's twin form for every request); every
    rank admits at 98304 and the cap reaches it."""
    monkeypatch.setenv(m.ENV_TREE_KEY, "1")
    p0 = _Sched(0, head=4096, n_ids=100000, store_extent=98304)
    f = _Sched(1, head=4096, n_ids=100000, store_extent=98304)
    r0, r1, wire = _ring(p0, f, "weg2-3-8", 100000)
    assert wire[0].told == 98304 and getattr(wire[0], "absolute", False) is True
    assert m.admission(p0, r0, _note()) == 94208
    assert m.admission(f, r1, _note()) == 94208
    assert r0._weg2_prefix_cap == r1._weg2_prefix_cap == 98305
    assert f.registered == [("weg2-3-8", 98305)]


def test_path1_absolute_follower_with_a_shorter_head_reads_the_difference(monkeypatch):
    """PP0 (fewer layers, bigger pool) kept 4096 on device, the follower only
    2048: the follower reads [2048, 98304) and still lands on told."""
    monkeypatch.setenv(m.ENV_TREE_KEY, "1")
    p0 = _Sched(0, head=4096, n_ids=100000, store_extent=98304)
    f = _Sched(1, head=2048, n_ids=100000, store_extent=98304)
    r0, r1, wire = _ring(p0, f, "weg2-3-9", 100000)
    assert m.admission(p0, r0, _note()) == 94208
    assert m.admission(f, r1, _note()) == 96256


def test_path1_absolute_head_only_when_the_store_has_nothing_beyond(monkeypatch):
    """Multi-turn inside one P epoch: the store holds nothing past the device
    head. PP0 declines (too short) and publishes told = head; a follower that
    holds the head is satisfied locally -- nobody loses the device prefix to
    a cap of 0 (the shipped relative form capped every rank at 0)."""
    monkeypatch.setenv(m.ENV_TREE_KEY, "1")
    p0 = _Sched(0, head=8192, n_ids=9000, store_extent=8192)
    f = _Sched(1, head=8192, n_ids=9000, store_extent=8192)
    r0, r1, wire = _ring(p0, f, "weg2-4-1", 9000)
    assert wire[0].told == 8192
    assert m.admission(p0, r0, _note()) == 0
    assert m.admission(f, r1, _note()) == 0
    assert r0._weg2_prefix_cap == r1._weg2_prefix_cap == 8193


def test_path1_absolute_explicit_off_keeps_the_relative_form(monkeypatch):
    monkeypatch.setenv(m.ENV_TREE_KEY, "1")
    monkeypatch.setenv("SGLANG_WEG2_TOLD_ABSOLUTE", "0")
    p0 = _Sched(0, head=4096, n_ids=100000, store_extent=98304)
    f = _Sched(1, head=4096, n_ids=100000, store_extent=98304)
    _r0, _r1, wire = _ring(p0, f, "weg2-3-10", 100000)
    assert wire[0].told == 94208 and not getattr(wire[0], "absolute", False)


# ---- path 2: #1416b, PP0 overwrites its own record with told > 0 ------------


def test_path2_clamped_positive_told_is_admitted_on_every_rank(monkeypatch):
    """#1416b with told > 0: PP0 completed 53247 (truncated read), the store's
    deepest anchor is at 4095 -> told 4095; PP0's record follows, its own
    admission passes, the follower reads exactly 4095 and passes too."""
    monkeypatch.setattr(m, "_anchor_clamp", lambda s, r, told: min(int(told), 4095))
    p0 = _Sched(0, head=0, n_ids=60000, store_extent=53247)
    f = _Sched(1, head=0, n_ids=60000, store_extent=98000)
    r0, r1, wire = _ring(p0, f, "weg2-5-1", 60000)
    assert wire[0].told == 4095
    assert p0.tree_cache._prefetch_completed_tokens["weg2-5-1"] == 4095
    assert m.admission(p0, r0, _note()) == 53247  # credit = own loaded increment
    assert m.admission(f, r1, _note()) == 4095
    assert r0._weg2_prefix_cap == r1._weg2_prefix_cap == 4096


def test_path2_absolute_clamp_sets_the_absolute_record(monkeypatch):
    monkeypatch.setenv(m.ENV_TREE_KEY, "1")
    monkeypatch.setattr(m, "_anchor_clamp", lambda s, r, told: min(int(told), 20480))
    p0 = _Sched(0, head=4096, n_ids=60000, store_extent=53247)
    f = _Sched(1, head=4096, n_ids=60000, store_extent=98000)
    r0, r1, wire = _ring(p0, f, "weg2-5-2", 60000)
    assert wire[0].told == 20480
    assert p0.tree_cache._prefetch_completed_tokens["weg2-5-2"] == 20480
    m.admission(p0, r0, _note())
    m.admission(f, r1, _note())
    assert r0._weg2_prefix_cap == r1._weg2_prefix_cap == 20481


# ---- path 3: #1442 hand-off keys, PP0-authoritative -------------------------


class _KeySched(_Sched):
    """Registration that resolves the hand-off chain like the scheduler
    (``resolve_chain(req, handoff.read)``, scheduler._prefetch_kvcache)."""

    def __init__(self, pp_rank, files):
        super().__init__(pp_rank, head=0, n_ids=5000, is_eagle=True)
        self.files = files
        self.used = {}

    def _prefetch_kvcache(self, req, rematch=True, limit_tokens=None):
        from sglang.srt.weg2 import handoff
        from sglang.srt.weg2.handoff_keys import resolve_chain

        self.used[req.rid] = resolve_chain(req, handoff.read)
        return super()._prefetch_kvcache(req, rematch, limit_tokens)


@pytest.fixture
def files(monkeypatch):
    """The shared hand-off directory: rid -> {"page_keys": [...]}."""
    from sglang.srt.weg2 import handoff

    d = {}
    monkeypatch.setattr(handoff, "read", lambda rid: d.get(rid))
    return d


def test_path3_follower_adopts_pp0_no_keys_even_if_the_file_appeared(files):
    """xsn331 race on P: PP0 read no hand-off file at its intake (own hashes),
    the file is there when the follower registers one pass later. The
    follower must read with PP0's source, not its own file read."""
    p0, f = _KeySched(0, files), _KeySched(1, files)
    r0, r1 = _req("weg2-7-1", 5000), _req("weg2-7-1", 5000)
    p0.waiting_queue.append(r0)
    f.waiting_queue.append(r1)
    m.intake(p0, r0, lambda g: None)
    m.intake(f, r1, lambda g: None)
    files["weg2-7-1"] = {"page_keys": ["stale%d" % i for i in range(64)]}
    wire = m.pp0_publish(p0, [])
    assert wire[0].keys_digest == ""
    m.follower_absorb(f, list(wire))
    assert p0.used["weg2-7-1"] is None
    assert f.used["weg2-7-1"] is None, "follower read keys PP0 never used"


def test_path3_follower_adopts_pp0_keys_when_digests_agree(files):
    keys = ["P%d" % i for i in range(64)]
    files["weg2-7-2"] = {"page_keys": keys}
    p0, f = _KeySched(0, files), _KeySched(1, files)
    r0, r1 = _req("weg2-7-2", 5000), _req("weg2-7-2", 5000)
    p0.waiting_queue.append(r0)
    f.waiting_queue.append(r1)
    m.intake(p0, r0, lambda g: None)
    m.intake(f, r1, lambda g: None)
    wire = m.pp0_publish(p0, [])
    assert wire[0].keys_digest and wire[0].keys_digest.endswith(":64")
    m.follower_absorb(f, list(wire))
    assert f.used["weg2-7-2"] == keys == p0.used["weg2-7-2"]


def test_path3_follower_without_pp0s_keys_falls_back_to_own_hashes_by_name(files, caplog):
    """PP0 read keys, the follower's file is gone (removed by a hold release)
    or differs: it cannot reproduce PP0's source -- own hashes, named."""
    keys = ["P%d" % i for i in range(64)]
    files["weg2-7-3"] = {"page_keys": keys}
    p0, f = _KeySched(0, files), _KeySched(1, files)
    r0, r1 = _req("weg2-7-3", 5000), _req("weg2-7-3", 5000)
    p0.waiting_queue.append(r0)
    f.waiting_queue.append(r1)
    m.intake(p0, r0, lambda g: None)
    m.intake(f, r1, lambda g: None)
    files["weg2-7-3"] = {"page_keys": ["other%d" % i for i in range(64)]}
    wire = m.pp0_publish(p0, [])
    with caplog.at_level("WARNING"):
        m.follower_absorb(f, list(wire))
    assert f.used["weg2-7-3"] is None
    assert "#1442 HANDOFF-KEYS PP0-DISAGREE" in caplog.text


def test_path3_told_before_request_carries_the_decision_to_intake(files):
    """The told can arrive before the request (re-queue): the decision rides
    with it to the intake registration."""
    p0, f = _KeySched(0, files), _KeySched(1, files)
    r0 = _req("weg2-7-4", 5000)
    p0.waiting_queue.append(r0)
    m.intake(p0, r0, lambda g: None)
    wire = m.pp0_publish(p0, [])
    files["weg2-7-4"] = {"page_keys": ["late%d" % i for i in range(64)]}
    m.follower_absorb(f, list(wire))
    r1 = _req("weg2-7-4", 5000)
    m.intake(f, r1, lambda g: None)
    assert f.used["weg2-7-4"] is None


def test_path3_probe_places_the_handoff_chain_from_token_zero(monkeypatch):
    """#1416d probe hashes the span from position 0; the scheduler's registry
    holds P's keys sliced at the MATCHED length (keys_for_span). A PP0 with a
    matched head put the sliced keys at page 0 and asked the store about
    pages nobody wrote. The probe takes the request's full chain."""
    monkeypatch.undo()
    monkeypatch.setenv(m.ENV_TREE_KEY, "1")
    written = ["P%d" % i for i in range(100)]
    asked = []

    class _B:
        def batch_exists_v2(self, keys, t, e):
            asked.append(list(keys))
            n = 0
            for k in keys:
                if k not in written:
                    break
                n += 1
            return types.SimpleNamespace(kv_hit_pages=n)

    cc = types.SimpleNamespace(page_size=1, storage_backend=_B(),
                               get_hash_str=lambda ids, lh, page_size=1: ["own%d" % i for i in range(len(ids))],
                               _presence_pool_transfers=lambda: ["mamba"])
    req = types.SimpleNamespace(origin_input_ids=list(range(102)), rid="weg2-9-9", extra_key=None,
                                _weg2_handoff_page_keys=written)
    sched = types.SimpleNamespace(cache_controller=cc, tree_cache=types.SimpleNamespace(is_eagle=True))
    # the registry as the scheduler leaves it after a registration at matched=40
    monkeypatch.setattr(m, "_handoff_keys", lambda rid: written[40:])
    assert m._anchor_clamp(sched, req, 100) == 100
    assert asked[0][:100] == written


# ---- path 4: dormant hold (#1443/#1455) --------------------------------------


def test_path4_a_rid_in_the_dormant_hold_is_published_after_the_release():
    """P is dormant (release_memory_occupation sets weg2_dormant on either
    group): PP0's intake registers the read and the rid goes into the
    dormant hold, not the waiting queue. pp0_publish used to drop it from
    held ('left the queue'); after the wake nothing published a told and every
    rank skipped it as weg2_store_told_pending for ever."""
    p0 = _Sched(0, head=0, n_ids=5000, is_eagle=True, store_extent=4095)
    f = _Sched(1, head=0, n_ids=5000, is_eagle=True, store_extent=4095)
    r0, r1 = _req("weg2-8-1", 5000), _req("weg2-8-1", 5000)
    m.intake(p0, r0, lambda g: None)
    m.intake(f, r1, lambda g: None)
    p0.weg2_dormant_hold = [r0]
    f.weg2_dormant_hold = [r1]
    assert m.pp0_publish(p0, []) == [], "nothing published while held"
    assert "weg2-8-1" in p0._weg2_store_held
    # the wake releases the hold into the waiting queue on every rank
    p0.weg2_dormant_hold, p0.waiting_queue = [], [r0]
    f.weg2_dormant_hold, f.waiting_queue = [], [r1]
    wire = m.pp0_publish(p0, [])
    assert [w.told for w in wire] == [4095]
    m.follower_absorb(f, list(wire))
    assert m.admission(p0, r0, _note()) == 4095
    assert m.admission(f, r1, _note()) == 4095


def test_path4_a_rid_parked_by_the_post_wake_settle_stays_held():
    p0 = _Sched(0, head=0, n_ids=5000, is_eagle=True, store_extent=4095)
    r0 = _req("weg2-8-2", 5000)
    m.intake(p0, r0, lambda g: None)
    p0.weg2_post_wake_settle = [r0]
    assert m.pp0_publish(p0, []) == []
    assert "weg2-8-2" in p0._weg2_store_held
    p0.weg2_post_wake_settle, p0.waiting_queue = [], []
    m.pp0_publish(p0, [])
    assert "weg2-8-2" not in p0._weg2_store_held, "gone from everywhere: dropped"


def test_path4_paced_form_keeps_a_held_rid_too(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_TOLD_PACED", "1")
    p0 = _Sched(0, head=0, n_ids=5000, is_eagle=True, store_extent=4095)
    p0._weg2_told_paced_on = True
    r0 = _req("weg2-8-3", 5000)
    m.intake(p0, r0, lambda g: None)
    p0.weg2_dormant_hold = [r0]
    m.pp0_publish(p0, [])
    assert "weg2-8-3" in p0._weg2_store_held
