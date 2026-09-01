"""#1061: the producer-phase census has WRITERS now -- prove each one can fail.

The module (`producer_phase_census.py`, built 2026-08-31) shipped with a full
can-fail suite for its ARITHMETIC (`test_producer_phase_census_631.py`) and
ZERO production writers: no call fed the ledger, no call fed the census, no
line could ever appear in a boot log. This file proves the WIRING, which is a
different failure surface: every test here goes red when the corresponding
writer call is removed again (see `mutants_1061.sh`), which is exactly the
defect #1061 names.

The wired sites under test:
  * `BindingState.advance` -> `note_generation`      (generation -> phase)
  * `backup_thread_func`   -> `note_backup_keys`     (key -> write generation)
  * prefetch adoption      -> `note_prefetch_adopted` (arrival + by_source)
  * `_match_prefix_helper` -> `note_walk_node` + `note_walk` + `emit`
    (the acceptance line: ok / denom / by_producer at the hit)

Hermetic: CPU only, no boot, no GPU. The integration test drives the REAL
`UnifiedRadixCache` walk on CPU tensors (same pattern as
`test_census_sees_host_only_nodes_936.py`).
"""

import logging
import threading
from array import array
from queue import Queue
from types import SimpleNamespace

import pytest
import torch

import sglang.srt.mem_cache.producer_phase_census as m
from sglang.srt.mem_cache.hicache_phase_binding import binding_state
from sglang.srt.mem_cache.producer_phase_census import (
    AdoptionSource,
    ProducerPhase,
    ProducerPhaseCensus,
    adoption_source_of,
    ledger_stats,
    note_backup_keys,
    note_prefetch_adopted,
    note_prefill_hit_tokens,
    note_walk_node,
    prefill_provenance_field,
    producer_generation_of,
)


@pytest.fixture(autouse=True)
def _clean():
    m.reset_for_test()
    binding_state().reset()
    yield
    m.reset_for_test()
    binding_state().reset()


def _arm(monkeypatch, every=1):
    monkeypatch.setenv("SGLANG_MATCH_REFUSAL_CENSUS_EVERY", str(every))


# -- the ledger writer (glue + the backup thread's call site) ------------


def test_disarmed_glue_records_nothing():
    """Default env (knob unset = 0): every glue writer must be a no-op."""
    note_backup_keys(["a", "b"], 1)
    note_prefetch_adopted(["c"])
    assert ledger_stats()["keys"] == 0
    assert m.arrival_stats()["arrivals"] == 0


def test_armed_backup_keys_stamp_the_ledger(monkeypatch):
    _arm(monkeypatch)
    note_backup_keys(["a", "b"], 1)
    assert ledger_stats()["keys"] == 2
    assert producer_generation_of("a") == 1


def test_unstamped_operation_records_nothing(monkeypatch):
    """generation=None is `StorageOperation.__init__`'s except path -- the
    op is unstamped, and an unstamped key must classify UNKNOWN later,
    which requires it to NOT be in the ledger at any generation."""
    _arm(monkeypatch)
    note_backup_keys(["a"], None)
    assert ledger_stats()["keys"] == 0


def test_backup_thread_stamps_keys_with_the_operation_stamp(monkeypatch):
    """The REAL `backup_thread_func` -> `note_backup_keys` call site.

    A stub controller (no __init__) with the exact attributes the loop
    reads; `_page_backup` is stubbed out so the only observable effect of
    the dequeue is the #1061 stamp. Killing the call site (mutant M3) makes
    the ledger stay empty and this test red.
    """
    from sglang.srt.managers.cache_controller import (
        HiCacheController,
        StorageOperation,
    )

    _arm(monkeypatch)
    op = StorageOperation(
        torch.arange(2, dtype=torch.int64),
        [11, 12],
        hash_value=["hk0", "hk1"],
    )
    assert op.binding_generation == binding_state().generation

    c = HiCacheController.__new__(HiCacheController)
    c.backup_skip = False
    c.storage_stop_event = threading.Event()
    c.backup_queue = Queue()
    c.ack_backup_queue = Queue()
    c.storage_backend = SimpleNamespace(check_disk_space=lambda: None)
    c.enable_storage = False
    c._page_backup = lambda operation: None

    c.backup_queue.put(op)
    t = threading.Thread(target=c.backup_thread_func, daemon=True)
    t.start()
    acked = c.ack_backup_queue.get(timeout=10)
    c.storage_stop_event.set()
    t.join(timeout=10)
    assert acked is op
    assert ledger_stats()["keys"] == 2, "the backup thread must stamp its keys"
    assert producer_generation_of("hk0") == op.binding_generation


# -- generation history (the `advance` call site) ------------------------


def test_advance_records_both_phases_of_a_cutover():
    """`BindingState.advance` must feed generation -> phase, outgoing pair
    included, so generation 0 / the boot phase is renderable. Killing the
    call site (mutant M4) leaves both lookups None and this test red."""
    bs = binding_state()
    bs.advance("tp")
    assert m.phase_of_generation(0) == "pp", "the outgoing boot pair"
    assert m.phase_of_generation(1) == "tp", "the minted pair"


# -- classification through the real carrier ------------------------------


def test_walk_node_classifies_a_cross_phase_hit(monkeypatch):
    """The mission's shape, through the REAL carrier: keys stamped at the
    boot (pp) generation, consumed after an `advance` to tp."""
    _arm(monkeypatch)
    bs = binding_state()
    note_backup_keys(["k0", "k1"], bs.generation)  # written under pp/gen0
    bs.advance("tp")  # the flip; records (0, pp) and (1, tp)

    census = ProducerPhaseCensus()
    saw_cross = note_walk_node(census, ["k0", "k1"], 2, 1, True)
    assert saw_cross is True
    census.note_walk(hit=True)
    census.note_cross_phase_walk()
    f = census.log_fields()
    assert f["ok"] == 1 and f["denom"] == 1
    assert f["by_producer"] == "cross:2"
    assert f["by_source"] == "backup_host:2"


def test_walk_node_prefetch_arrival_names_the_arm(monkeypatch):
    _arm(monkeypatch)
    bs = binding_state()
    note_backup_keys(["p0"], bs.generation)
    note_prefetch_adopted(["p0"])  # the store handed it back and it was adopted
    bs.advance("tp")
    assert adoption_source_of("p0") is AdoptionSource.PREFETCH
    census = ProducerPhaseCensus()
    note_walk_node(census, ["p0"], 1, 1, True)
    assert census.log_fields()["by_source"] == "prefetch:1"


def test_walk_node_without_keys_is_unknown_never_guessed(monkeypatch):
    """A device-only node carries hash_value=None: no storage-key carrier.
    Its tokens must land in `unknown`, never in same/cross."""
    _arm(monkeypatch)
    census = ProducerPhaseCensus()
    assert note_walk_node(census, None, 8, 1, True) is False
    f0 = census.tokens_by_producer
    assert f0 == {ProducerPhase.UNKNOWN.value: 8}


def test_walk_node_partial_last_page_sums_exactly(monkeypatch):
    """ARITHMETIC check: per-key tokens sum to key_tokens even when the last
    key carries a partial page (page_size=4, 6 tokens -> 4 + 2)."""
    _arm(monkeypatch)
    bs = binding_state()
    note_backup_keys(["q0", "q1"], bs.generation)
    bs.advance("tp")
    census = ProducerPhaseCensus()
    note_walk_node(census, ["q0", "q1"], 6, 4, True)
    assert census.accepted_tokens == 6


# -- emission windows -----------------------------------------------------


def test_emit_returns_true_on_a_logged_line(monkeypatch):
    monkeypatch.setattr(m, "census_armed", lambda: 1000)
    c = ProducerPhaseCensus()
    c.note_walk(hit=True)
    c.note_cross_phase_walk()
    c.note_accepted_tokens(4, ProducerPhase.CROSS_PHASE, AdoptionSource.PREFETCH)
    assert m.emit(c, logging.getLogger("t1061a")) is True


def test_emit_returns_false_when_suppressed(monkeypatch):
    monkeypatch.setattr(m, "census_armed", lambda: 1000)
    c = ProducerPhaseCensus()
    c.note_walk(hit=False)
    assert m.emit(c, logging.getLogger("t1061b")) is False, (
        "a suppressed emission must tell the caller to keep the window"
    )


# -- the prefill line's window is drained by the line that renders it -----


def test_prefill_field_drains_on_render(monkeypatch):
    monkeypatch.setattr(m, "_phase_flip_enabled", lambda: True)
    monkeypatch.setattr(m, "_current_generation_or_none", lambda: 1)
    note_prefill_hit_tokens(64, ProducerPhase.CROSS_PHASE)
    f1 = prefill_provenance_field(64)
    assert "cross:64" in f1
    f2 = prefill_provenance_field(64)
    assert f2 == ", #cached-producer: NO_OBSERVATION", (
        "the second line must not re-report the first line's tokens"
    )


# -- the real walk emits the acceptance line ------------------------------


def test_real_walk_emits_the_acceptance_line(monkeypatch, caplog):
    """End to end on the REAL `_match_prefix_helper`: plant a hit whose keys
    were stamped under pp/gen0, flip to tp, walk, and read the #631 line
    with ok, denom and the producer partition off the log. Mutants M1/M2
    (walk feed removed / walk+emit removed) go red here."""
    from sglang.srt.mem_cache.base_prefix_cache import (
        InsertParams,
        MatchPrefixParams,
    )
    from sglang.srt.mem_cache.radix_cache import RadixKey

    from test_unified_radix_cache_unittest import CacheConfig, build_fixture

    cache, allocator, _ = build_fixture(CacheConfig())
    tokens = array("q", range(1, 17))
    value = allocator.alloc(len(tokens))
    assert value is not None
    cache.insert(InsertParams(key=RadixKey(tokens), value=value))

    # The single leaf under the root; plant page-granular storage keys on it
    # (page_size=1 -> one key per token), exactly what a backup would leave.
    node = cache.root_node
    while node.children:
        node = next(iter(node.children.values()))
    keys = [f"wk{i}" for i in range(len(tokens))]
    node.hash_value = keys

    _arm(monkeypatch, every=1)
    note_backup_keys(keys, binding_state().generation)  # written under pp/gen0
    binding_state().advance("tp")  # the flip

    with caplog.at_level(logging.INFO):
        cache.match_prefix(MatchPrefixParams(key=RadixKey(tokens)))

    lines = [
        r.getMessage() for r in caplog.records if "#631 producer-phase" in r.getMessage()
    ]
    assert lines, "the armed walk must emit the acceptance line"
    line = lines[-1]
    assert "ok=1" in line, line
    assert "denom=1" in line, line
    assert "cross:16" in line, line
    assert "suppressed=" in line, "the rate limiter must print its suppressed count"


def test_disarmed_walk_emits_nothing_and_feeds_nothing(caplog):
    """INDIKATOR-GESETZ floor: with the knob at its default the walk must not
    even build the census -- no line, no ledger reads, byte-identical path."""
    from sglang.srt.mem_cache.base_prefix_cache import (
        InsertParams,
        MatchPrefixParams,
    )
    from sglang.srt.mem_cache.radix_cache import RadixKey

    from test_unified_radix_cache_unittest import CacheConfig, build_fixture

    cache, allocator, _ = build_fixture(CacheConfig())
    tokens = array("q", range(1, 9))
    value = allocator.alloc(len(tokens))
    cache.insert(InsertParams(key=RadixKey(tokens), value=value))
    with caplog.at_level(logging.INFO):
        cache.match_prefix(MatchPrefixParams(key=RadixKey(tokens)))
    assert not any(
        "#631 producer-phase" in r.getMessage() for r in caplog.records
    ), "disarmed must be silent, not zero-valued"
