"""#1243 step 1 -- hermetic tests for the fp8-body / bf16-tail probe harness.

No GPU, no server, no model.  Run with:

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=/spinning/wt-weg2-tail/python \
      /spinning/htsglang-gpu/.venv/bin/python -m pytest \
      test/registered/unit/mem_cache/test_kv_tail_probe_1243.py -q -p no:cacheprovider

The load-bearing tests, in order of what they protect:

  1. THE EMULATION IS THE REAL THING.  ``test_n0_emulation_is_bit_exact_...``
     asserts the probe's round trip is bitwise identical to what the real fp8
     storage path stores, on random K/V blocks.  Without this the whole probe
     measures a fiction.
  2. THE EMULATION CANNOT DRIFT AWAY FROM IT.  ``test_real_fp8_write_path_...``
     reads the source of ``MHATokenToKVPool.set_kv_buffer`` and asserts the two
     lines the emulation mirrors are still there.  If upstream changes how fp8
     is stored, this goes red instead of the probe silently lying.
  3. THE DEFAULT PATH IS UNTOUCHED.  ``test_probe_is_inert_by_default``.
  4. IT REFUSES RATHER THAN GUESSES.  Every configuration the emulation cannot
     carry faithfully raises ``KvTailProbeRefused``.
"""

from __future__ import annotations

import inspect
import json
import os
import types

import pytest
import torch

from sglang.srt.mem_cache import kv_tail_probe as ktp
from sglang.srt.mem_cache.kv_tail_probe import KvTailProbeRefused

FP8 = torch.float8_e4m3fn


# ---------------------------------------------------------------- fakes ----


class FakePool:
    """The narrow surface ``after_write`` touches on an MHA KV pool."""

    def __init__(self, n_slots=512, heads=2, dim=4, dtype=torch.bfloat16, layers=(0,)):
        self.dtype = dtype
        self.k = {
            lid: torch.randn(n_slots, heads, dim, dtype=torch.float32).to(dtype)
            for lid in layers
        }
        self.v = {
            lid: torch.randn(n_slots, heads, dim, dtype=torch.float32).to(dtype)
            for lid in layers
        }

    def _get_key_buffer(self, layer_id):
        return self.k[layer_id]

    def _get_value_buffer(self, layer_id):
        return self.v[layer_id]


def fake_layer(layer_id=0, k_scale=None):
    return types.SimpleNamespace(
        layer_id=layer_id, k_scale_float=k_scale, v_scale_float=k_scale
    )


def fake_batch(seq_len, n_req=1):
    return types.SimpleNamespace(seq_lens=torch.tensor([seq_len] * n_req))


def real_fp8_store(x: torch.Tensor, scale) -> torch.Tensor:
    """Byte-for-byte the expression ``MHATokenToKVPool.set_kv_buffer`` runs.

    Quoted from the tree (see ``test_real_fp8_write_path_still_matches_...``):

        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            cache_k = cache_k.to(self.dtype)
    """
    cache_k = x.clone()
    if scale is not None:
        cache_k.div_(scale)
    return cache_k.to(FP8)


@pytest.fixture(autouse=True)
def _clean_probe():
    saved = {k: os.environ.get(k) for k in (ktp.ENV_N, ktp.ENV_ACK)}
    for k in saved:
        os.environ.pop(k, None)
    ktp.reset_for_test()
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    ktp.reset_for_test()


# ------------------------------------------------- 1. bit-exactness --------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_n0_emulation_is_bit_exact_against_the_real_fp8_store(seed):
    """The N=0 arm must reproduce the real fp8 path EXACTLY, not approximately.

    fp8_e4m3fn has 3 mantissa bits and its whole exponent range sits inside
    bf16's, so ``bf16(fp8(x)) == fp8(x)`` with no loss -- which is why a bf16
    pool can carry an fp8 body at all.  Asserted on the fp8 bit pattern, so a
    one-ulp difference cannot hide behind a tolerance.
    """
    torch.manual_seed(seed)
    x = (torch.randn(64, 4, 8) * 3.0).to(torch.bfloat16)
    buf = x.clone()
    slots = torch.arange(64)
    ktp.fp8_round_trip_(buf, slots, None)

    want = real_fp8_store(x, None)
    got = buf.to(FP8)
    assert torch.equal(got.view(torch.uint8), want.view(torch.uint8)), (
        "the probe's round trip does not land on the same fp8 bits the real "
        "storage path writes -- the emulation is not an emulation"
    )
    # and the value attention sees is identical too
    assert torch.equal(buf, want.to(torch.bfloat16))


def test_round_trip_is_idempotent_so_demote_once_equals_round_on_every_read():
    """The whole storage-side substitution rests on this.

    Read-side rounding would round at EVERY read; the probe rounds ONCE, when
    the token leaves the tail.  The two are the same value sequence only if the
    round trip is idempotent.
    """
    torch.manual_seed(7)
    x = (torch.randn(256, 3, 6) * 5.0).to(torch.bfloat16)
    once = x.clone()
    ktp.fp8_round_trip_(once, torch.arange(256), None)
    twice = once.clone()
    ktp.fp8_round_trip_(twice, torch.arange(256), None)
    assert torch.equal(once, twice), "round trip is not idempotent"


def test_round_trip_touches_only_the_named_slots():
    torch.manual_seed(3)
    x = (torch.randn(32, 2, 4) * 4.0).to(torch.bfloat16)
    buf = x.clone()
    ktp.fp8_round_trip_(buf, torch.tensor([1, 5, 9]), None)
    untouched = [i for i in range(32) if i not in (1, 5, 9)]
    assert torch.equal(buf[untouched], x[untouched])
    assert not torch.equal(buf[1], x[1]) or torch.equal(
        buf[1], real_fp8_store(x[1], None).to(torch.bfloat16)
    )


def test_round_trip_on_empty_slot_set_is_a_noop():
    x = torch.randn(8, 2, 2).to(torch.bfloat16)
    buf = x.clone()
    ktp.fp8_round_trip_(buf, torch.tensor([], dtype=torch.long), None)
    assert torch.equal(buf, x)


# ------------------------------------- 2. drift guard on the real path -----


def test_real_fp8_write_path_still_matches_the_expression_the_probe_mirrors():
    """If upstream changes how fp8 KV is stored, this test -- not the probe --
    is what notices."""
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    src = inspect.getsource(MHATokenToKVPool.set_kv_buffer)
    for needle in (
        "if cache_k.dtype != self.dtype:",
        "cache_k.div_(k_scale)",
        "cache_k = cache_k.to(self.dtype)",
    ):
        assert needle in src, (
            f"MHATokenToKVPool.set_kv_buffer no longer contains {needle!r}. "
            "kv_tail_probe.fp8_round_trip_ mirrors that expression; re-derive "
            "it before trusting any #1243 number."
        )


def test_the_probe_hook_is_wired_into_the_single_physical_write_site():
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    src = inspect.getsource(MHATokenToKVPool.set_kv_buffer)
    assert "kv_tail_probe.ACTIVE" in src and "kv_tail_probe.after_write" in src


def test_the_probe_hook_is_wired_into_radix_attention():
    from sglang.srt.layers.radix_attention import RadixAttention

    src = inspect.getsource(RadixAttention.forward)
    assert "kv_tail_probe.ACTIVE" in src and "kv_tail_probe.begin_attention" in src


# -------------------------------------------- 3. inert by default ----------


def test_probe_is_inert_by_default():
    assert os.environ.get(ktp.ENV_N) is None
    assert ktp.ACTIVE is False
    # both hooks return immediately and touch nothing
    pool = FakePool()
    before = pool.k[0].clone()
    ktp.begin_attention(fake_layer(), fake_batch(10))
    ktp.after_write(pool, 0, torch.arange(4))
    assert torch.equal(pool.k[0], before)


def test_env_absent_means_config_inactive_even_with_ack_set():
    os.environ[ktp.ENV_ACK] = ktp.ACK_TOKEN
    ktp.reset_for_test()
    assert ktp.ACTIVE is False


# ----------------------------------------------------- 4. refusals ---------


def test_refuses_without_the_acknowledgement_token():
    ktp.configure_for_test(1024, acked=False)
    with pytest.raises(KvTailProbeRefused, match="EXPERIMENT-ONLY"):
        ktp.begin_attention(fake_layer(), fake_batch(10))


def test_refuses_a_batch_with_more_than_one_request():
    ktp.configure_for_test(1024)
    with pytest.raises(KvTailProbeRefused, match="only.*exact for ONE sequence"):
        ktp.begin_attention(fake_layer(), fake_batch(10, n_req=2))


def test_refuses_a_non_unit_kv_scale():
    ktp.configure_for_test(1024)
    with pytest.raises(KvTailProbeRefused, match="bit-exact"):
        ktp.begin_attention(fake_layer(k_scale=0.5), fake_batch(10))


def test_accepts_scale_none_and_scale_one():
    ktp.configure_for_test(1024)
    ktp.begin_attention(fake_layer(k_scale=None), fake_batch(10))
    ktp.begin_attention(fake_layer(k_scale=1.0), fake_batch(11))


def test_refuses_an_fp8_pool_because_that_would_double_quantize():
    ktp.configure_for_test(4)
    ktp.begin_attention(fake_layer(), fake_batch(8))
    pool = FakePool(dtype=FP8)
    with pytest.raises(KvTailProbeRefused, match="double-quantize"):
        ktp.after_write(pool, 0, torch.arange(8))


def test_refuses_a_write_that_arrives_without_an_attention_context():
    ktp.configure_for_test(4)
    pool = FakePool()
    with pytest.raises(KvTailProbeRefused, match="without an attention context"):
        ktp.after_write(pool, 0, torch.arange(8))


def test_refuses_a_write_longer_than_the_sequence():
    ktp.configure_for_test(4)
    ktp.begin_attention(fake_layer(), fake_batch(3))
    pool = FakePool()
    with pytest.raises(KvTailProbeRefused, match="longer than the sequence"):
        ktp.after_write(pool, 0, torch.arange(8))


def test_refuses_a_bad_env_value():
    os.environ[ktp.ENV_N] = "-3"
    with pytest.raises(KvTailProbeRefused):
        ktp.reset_for_test()
    os.environ[ktp.ENV_N] = "nonsense"
    with pytest.raises(KvTailProbeRefused):
        ktp.reset_for_test()
    os.environ.pop(ktp.ENV_N)


def test_env_parses_all_and_integers():
    for raw, want in (("all", None), ("ALL", None), ("0", 0), ("4096", 4096)):
        os.environ[ktp.ENV_N] = raw
        os.environ[ktp.ENV_ACK] = ktp.ACK_TOKEN
        ktp.reset_for_test()
        assert ktp.ACTIVE is True
        assert ktp.probe_stats()["tail_n"] == ("all" if want is None else want)


# ------------------------------------------- 5. the tail rule itself -------


def _drive(pool, layer, tail_n, chunks, layer_id=0, masks=None):
    """Feed writes of the given sizes and return the slots per chunk."""
    ktp.configure_for_test(tail_n)
    pos = 0
    all_slots = []
    for i, n in enumerate(chunks):
        pos += n
        slots = torch.arange(pos - n, pos)
        all_slots.append(slots)
        ktp.begin_attention(layer, fake_batch(pos))
        ktp.after_write(pool, layer_id, slots, None if masks is None else masks[i])
    return all_slots


def test_n_all_is_the_identity_nothing_is_ever_rounded():
    torch.manual_seed(11)
    pool = FakePool(n_slots=64)
    before_k, before_v = pool.k[0].clone(), pool.v[0].clone()
    _drive(pool, fake_layer(), None, [8] * 8)
    assert torch.equal(pool.k[0], before_k), "N=all rounded a token"
    assert torch.equal(pool.v[0], before_v), "N=all rounded a token"
    assert ktp.probe_stats()["rounded_rows_total"] == 0


def test_n0_rounds_every_token_that_is_no_longer_the_current_write():
    torch.manual_seed(12)
    pool = FakePool(n_slots=64)
    before = pool.k[0].clone()
    _drive(pool, fake_layer(), 0, [8] * 8)
    # every slot except the LAST chunk (evicted only at the next write)
    for s in range(56):
        assert torch.equal(
            pool.k[0][s], real_fp8_store(before[s], None).to(torch.bfloat16)
        ), f"slot {s} was not demoted under N=0"
    assert torch.equal(pool.k[0][56:], before[56:]), "the last chunk was demoted early"


def test_exactly_the_youngest_n_tokens_stay_bf16():
    """The rule under test: positions < seq_len - N are fp8, the rest untouched."""
    torch.manual_seed(13)
    tail_n = 16
    pool = FakePool(n_slots=128)
    before = pool.k[0].clone()
    _drive(pool, fake_layer(), tail_n, [1] * 100)  # decode, one token per step
    seq_len = 100
    boundary = seq_len - tail_n  # 84
    for s in range(boundary):
        assert torch.equal(
            pool.k[0][s], real_fp8_store(before[s], None).to(torch.bfloat16)
        ), f"position {s} is older than the tail and was not demoted"
    for s in range(boundary, seq_len):
        assert torch.equal(pool.k[0][s], before[s]), (
            f"position {s} is inside the youngest {tail_n} and must stay bf16"
        )


def test_v_buffer_is_demoted_alongside_k():
    torch.manual_seed(14)
    pool = FakePool(n_slots=64)
    before_v = pool.v[0].clone()
    _drive(pool, fake_layer(), 4, [1] * 32)
    assert torch.equal(
        pool.v[0][0], real_fp8_store(before_v[0], None).to(torch.bfloat16)
    )
    assert torch.equal(pool.v[0][30], before_v[30])


def test_a_dcp_owner_mask_restricts_demotion_to_this_ranks_rows():
    """Under uneven DCP a rank writes only the slots it owns; the probe must
    never round a slot this rank did not write."""
    torch.manual_seed(15)
    pool = FakePool(n_slots=64)
    before = pool.k[0].clone()
    masks = [torch.tensor([i % 3 == 0 for i in range(8)]) for _ in range(6)]
    _drive(pool, fake_layer(), 4, [8] * 6, masks=masks)
    for s in range(40):  # everything older than the tail
        owned = (s % 8) % 3 == 0
        same = torch.equal(pool.k[0][s], before[s])
        assert same != owned or not owned, "unowned slot must be untouched"
        if not owned:
            assert same, f"slot {s} is not owned by this rank but was rounded"


def test_layers_are_independent():
    torch.manual_seed(16)
    pool = FakePool(n_slots=64, layers=(0, 1))
    before1 = pool.k[1].clone()
    _drive(pool, fake_layer(layer_id=0), 4, [1] * 32, layer_id=0)
    assert torch.equal(pool.k[1], before1), "writing layer 0 demoted layer 1"


def test_a_new_shorter_sequence_resets_the_rings():
    torch.manual_seed(17)
    pool = FakePool(n_slots=64)
    _drive(pool, fake_layer(), 4, [1] * 20)
    resets_before = ktp.probe_stats()["sequence_resets"]
    # a second request starts at position 1 again
    ktp.begin_attention(fake_layer(), fake_batch(1))
    assert ktp.probe_stats()["sequence_resets"] == resets_before + 1


def test_residual_is_zero_when_the_write_is_no_longer_than_the_tail():
    """This is the number the plan drives to zero with --chunked-prefill-size
    <= min(N).  A run reporting a non-zero residual has to quote it."""
    pool = FakePool(n_slots=4096, heads=1, dim=2)
    _drive(pool, fake_layer(), 1024, [1024] * 4)
    assert ktp.probe_stats()["residual_rows_max"] == 0


def test_residual_is_reported_when_a_chunk_is_longer_than_the_tail():
    pool = FakePool(n_slots=256)
    _drive(pool, fake_layer(), 4, [16] * 4)
    # first chunk: boundary = 16-4 = 12, first_pos = 0 -> 12 rows unwritten-stale
    assert ktp.probe_stats()["residual_rows_max"] == 12


def test_probe_stats_names_its_denominators():
    pool = FakePool(n_slots=64)
    _drive(pool, fake_layer(), 4, [1] * 20)
    st = ktp.probe_stats()
    for key in (
        "active",
        "tail_n",
        "writes",
        "rings",
        "rounded_rows_total",
        "residual_rows_max",
        "sequence_resets",
    ):
        assert key in st
    assert st["writes"] == 20
    assert st["rounded_rows_total"] == 16  # positions 0..15, tail 4 of 20


# ----------------------------------------- 6. the runner's KLD math --------


def _runner():
    import importlib.util

    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))),
        "scripts",
        "kvtail_1243",
        "probe_kv_tail.py",
    )
    spec = importlib.util.spec_from_file_location("probe_kv_tail", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_runner_self_test_passes():
    """The runner carries its own 26 hermetic checks (KLD closed form,
    renormalisation invariance, denominators, verifiers)."""
    assert _runner().self_test() == 0


# --------------------------------------- 7. the corpus builder ------------
# desk-written-never-executed: the corpus builder is the one part of the
# harness that cannot be exercised on the remote desk against the REAL sources
# (the club-3090 packs and the long documents live only on the rig box, and
# that box was at 95.6 GiB of a ~96 GiB cgroup reap while this was written).
# So it is exercised here against a synthetic source tree -- the SHAPE of the
# output is proven, the real sources are proven at WINDOW GO by the manifest
# the run itself writes.


def test_corpus_builder_produces_passages_probes_and_a_cited_manifest(tmp_path):
    mod = _runner()

    packdir = tmp_path / "packs"
    packdir.mkdir()
    meta = {
        "__meta__": True,
        "pack_id": "reasonmath-15",
        "license": "MIT",
        "upstream_repo": "stevibe/ReasonMath-15",
        "upstream_commit": "deadbeef",
    }
    row = {
        "id": "RM-01",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "2+2?"},
        ],
        "verifier": {
            "asserts": [
                {
                    "canonical_answer": "ANSWER: 4",
                    "accepted_answers": ["ANSWER: four"],
                }
            ]
        },
    }
    (packdir / "reasonmath-15.jsonl").write_text(
        json.dumps(meta) + "\n" + json.dumps(row) + "\n"
    )

    doc = tmp_path / "doc.md"
    doc.write_text("# A real document\n" + ("Realer Fliesstext. " * 40000))

    mod.PACK_DIR = str(packdir)
    mod.SHORT_PACKS = ("reasonmath-15.jsonl",)
    mod.LONG_SOURCES = (str(doc),)
    mod.LONG_TARGET_TOKENS = (128, 512)

    corpus = mod.build_corpus(None)  # None => no tokenizer, char calibration

    # manifest cites every source with a hash and a size -- never a bare name
    assert len(corpus["manifest"]) == 2
    for entry in corpus["manifest"]:
        assert len(entry["sha256"]) == 64
        assert entry["bytes"] > 0
        assert os.path.exists(entry["path"])
    packs = [m for m in corpus["manifest"] if m["kind"] == "quality-pack"]
    assert packs[0]["license"] == "MIT" and packs[0]["upstream_commit"] == "deadbeef"

    groups = {p["group"] for p in corpus["passages"]}
    assert groups == {"short", "long-128", "long-512"}

    # short passages are scored whole; long passages over their last window
    for p in corpus["passages"]:
        if p["group"] == "short":
            assert p["logprob_start_len"] == 0
        else:
            assert p["logprob_start_len"] == max(0, p["approx_tokens"] - mod.SCORE_WINDOW)
        assert p["text"], "no empty passage"

    kinds = {pr["kind"] for pr in corpus["probes"]}
    assert kinds == {"reason_math", "needle"}

    # the needle sits EARLY and is really present in the haystack exactly once
    for pr in corpus["probes"]:
        if pr["kind"] != "needle":
            continue
        hay = pr["messages"][0]["content"]
        assert hay.count(pr["expect_token"]) == 1
        assert pr["needle_offset_fraction"] == mod.NEEDLE_OFFSET_FRACTION
        assert pr["needle_char_offset"] < len(hay) * 0.5, "needle must be EARLY"
        assert mod.verify_probe(pr, f"TOKEN: {pr['expect_token']}") is True

    rm = [pr for pr in corpus["probes"] if pr["kind"] == "reason_math"][0]
    assert "ANSWER: 4" in rm["accepted"] and "ANSWER: four" in rm["accepted"]


def test_compare_refuses_to_fold_mismatched_records_into_a_mean():
    """A record whose scoring window or prompt length differs is NOT the same
    prompt; folding it in is how a denominator lies."""
    mod = _runner()
    top = [[-0.1, 5, "a"], [-1.0, 7, "b"], [-3.0, 9, "c"]]
    base = {
        "id": "p1",
        "group": "short",
        "logprob_start_len": 0,
        "measured_prompt_tokens": 10,
        "scored_positions": 1,
        "input_top_logprobs": [top],
        "input_token_logprobs": [[-0.5, 5, None]],
    }
    prov = {"corpus_sha256": "c" * 64, "harness_commit": "abc123", "harness_dirty": False}
    ref = dict(prov, arm="A0", teacher_forced=[base], determined=[])
    mismatched = dict(base, scored_positions=2)
    rep = mod.compare_arms(
        ref, [dict(prov, arm="X", teacher_forced=[mismatched], determined=[])], None
    )
    assert rep["arms"]["X"]["overall"]["kld_truncated_nats"]["n"] == 0
    rep2 = mod.compare_arms(
        ref, [dict(prov, arm="Y", teacher_forced=[dict(base)], determined=[])], None
    )
    assert rep2["arms"]["Y"]["overall"]["kld_truncated_nats"]["n"] == 1


def test_compare_says_so_loudly_when_no_A_A_floor_was_captured():
    mod = _runner()
    prov = {"corpus_sha256": "c" * 64, "harness_commit": "abc123", "harness_dirty": False}
    ref = dict(prov, arm="A0", teacher_forced=[], determined=[])
    rep = mod.compare_arms(ref, [], None)
    assert rep["A_vs_A_noise_floor"] is None
    assert "NO A/A FLOOR" in rep["decision_rule"]
    rep2 = mod.compare_arms(ref, [], dict(ref, arm="A0b"))
    assert rep2["A_vs_A_noise_floor"] is not None
    assert "gain <= floor => do not build" in rep2["decision_rule"]


# ------------- 8. cross-slot provenance: floors banked in another slot -----
# The plan's "run between bases" section banks the A/A floor in a DIFFERENT
# gpuq slot from the arm it gates.  "Same tree, same corpus" therefore cannot
# be a doc instruction -- these tests are the enforcement.


def _prov(**over):
    base = {
        "corpus_sha256": "c" * 64,
        "harness_commit": "fdbf4add84",
        "harness_dirty": False,
        "teacher_forced": [],
        "determined": [],
    }
    base.update(over)
    return base


def test_a_floor_from_a_different_corpus_is_REFUSED_not_silently_used():
    mod = _runner()
    ref = _prov(arm="A0")
    other_corpus = _prov(arm="A0b", corpus_sha256="d" * 64)
    rep = mod.compare_arms(ref, [], other_corpus)
    assert rep["A_vs_A_noise_floor"] is None
    assert "A/A FLOOR REFUSED" in rep["decision_rule"]
    assert "corpus differs" in rep["decision_rule"]
    assert "Re-capture, do not reinterpret" in rep["decision_rule"]


def test_a_floor_from_a_different_harness_commit_is_REFUSED():
    mod = _runner()
    rep = mod.compare_arms(
        _prov(arm="A0"), [], _prov(arm="A0b", harness_commit="deadbeef")
    )
    assert rep["A_vs_A_noise_floor"] is None
    assert "harness commit differs" in rep["decision_rule"]


def test_a_capture_from_a_dirty_worktree_cannot_be_paired():
    mod = _runner()
    rep = mod.compare_arms(_prov(arm="A0"), [], _prov(arm="A0b", harness_dirty=True))
    assert rep["A_vs_A_noise_floor"] is None
    assert "DIRTY worktree" in rep["decision_rule"]


def test_a_capture_predating_provenance_recording_cannot_be_paired():
    """An old capture has no corpus_sha256 at all -- that is a refusal, not a
    pass-by-absence."""
    mod = _runner()
    old = {"arm": "A0b", "teacher_forced": [], "determined": []}
    rep = mod.compare_arms(_prov(arm="A0"), [], old)
    assert rep["A_vs_A_noise_floor"] is None
    assert "not recorded" in rep["decision_rule"]


def test_an_ARM_from_a_different_corpus_is_refused_rather_than_scored():
    mod = _runner()
    rep = mod.compare_arms(
        _prov(arm="A0"), [_prov(arm="N0", corpus_sha256="d" * 64)], None
    )
    assert "REFUSED" in rep["arms"]["N0"]
    assert "corpus differs" in rep["arms"]["N0"]["REFUSED"]


def test_floor_geometry_reports_same_boot_and_the_groups_it_covers():
    """A floor only gates the groups it covers -- a short-set floor may not
    judge a long-context delta."""
    mod = _runner()
    top = [[-0.1, 5, "a"], [-1.0, 7, "b"], [-3.0, 9, "c"]]
    rec = {
        "id": "p1",
        "group": "short",
        "logprob_start_len": 0,
        "measured_prompt_tokens": 10,
        "scored_positions": 1,
        "input_top_logprobs": [top],
        "input_token_logprobs": [[-0.5, 5, None]],
    }
    ref = _prov(arm="A0", teacher_forced=[rec], boot_log="/tmp/a0.log")
    a0b = _prov(arm="A0b", teacher_forced=[dict(rec)], boot_log="/tmp/a0b.log")
    rep = mod.compare_arms(ref, [], a0b)
    geo = rep["floor_geometry"]
    assert geo["same_boot"] is False, "two boot logs must read as a cross-boot floor"
    assert geo["groups_covered"] == ["short"]
    assert "short-set floor cannot" in geo["note"]

    same = mod.compare_arms(ref, [], dict(a0b, boot_log="/tmp/a0.log"))
    assert same["floor_geometry"]["same_boot"] is True


def test_harness_provenance_reports_this_worktrees_commit():
    mod = _runner()
    prov = mod.harness_provenance()
    assert prov["commit"] is None or len(prov["commit"]) == 40
    assert "dirty" in prov
