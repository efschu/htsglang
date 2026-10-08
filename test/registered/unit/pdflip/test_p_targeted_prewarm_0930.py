"""P-PREWARM (30.09.): targeted boot prewarm of the first P forward, no forward.

Metal (y4k 09301110 / y4l 09301150, P logs, 16384-token forwards, warm
median n=216):

* (a) PP0 ple_ms 334.2 / 339.3 vs 39: the first request's admission found no
  gather yet -- 'PLE-PREFETCH admit rid=pdflip-0-5 skipped: no prefill gather in
  this process yet' -- so chunk 0 read inside its forward ('ready=none
  wait_ms=195.1 host_ms=228.9'), the workers spawned in that same gather;
* (c) gate_ms 112/103/92 (y4k) vs 8/15/10 per stage, once per process: the
  router's first call (Triton ``_router_triton_kernel`` cold, '[nan-49]' line);
* (d) NOT a cold effect: PP0 fetch per spill expert on forward 1 is 192.3 us
  (y4k) / 192.9 (y4l) vs warm median 192.8 / 192.9 -- forward 1 simply routes
  more spill experts (7557 vs median 5969; 14 warm forwards per boot do the
  same, fetch ~1460 ms). No host-store touch is built.

Hermetic (no CUDA): real pread worker processes on real files for (a), fake
routers for (c), the scheduler wiring by AST.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import ast
import importlib
import logging
import pathlib
import re
import tempfile
import time
import types
from array import array
from collections import namedtuple

import pytest
import torch

from flliper.srt.environ import envs
from flliper.srt.models import qwen4_exp_ple_prefetch as pf
from flliper.srt.models import qwen4_exp_ple_table as pt

REPO = pathlib.Path(__file__).resolve().parents[4]
ADM_LOGGER = "flliper.srt.models.qwen4_exp_ple_admit"
PF_LOGGER = "flliper.srt.models.qwen4_exp_ple_prefetch"
ROUTER_LOGGER = "flliper.srt.layers.moe.router_prewarm"

Range = namedtuple("Range", "start end")
ROWS_PER_SHARD = 64
SHARDS = 4
DIM = 160
RB = DIM * 2
HEADER = 100
VOCAB = ROWS_PER_SHARD * SHARDS


def _adm():
    return importlib.import_module("flliper.srt.models.qwen4_exp_ple_admit")


def _router():
    return importlib.import_module("flliper.srt.layers.moe.router_prewarm")


def _switch(on):
    return envs.FLLIPER_PDFLIP_ENABLE_TARGETED_PREWARM.override(on)


# ---------------------------------------------------------------- (a) PLE ----


@pytest.fixture
def table():
    with tempfile.TemporaryDirectory() as tmp:
        files, shard_files, shard_offsets = [], [], []
        for f in range(2):
            path = os.path.join(tmp, f"ple-{f}.safetensors")
            with open(path, "wb") as fh:
                fh.write(os.urandom(HEADER + 2 * ROWS_PER_SHARD * RB))
            files.append(path)
            for s in range(2):
                shard_files.append(path)
                shard_offsets.append(HEADER + s * ROWS_PER_SHARD * RB)
        yield pt.CheckpointMappedPleTable(
            [0] * SHARDS, ROWS_PER_SHARD, VOCAB, torch.bfloat16, DIM,
            keepalive=[], files=files, shard_files=shard_files, shard_offsets=shard_offsets,
        )


def _identity_hasher(tokens, lead):
    return tokens[lead:].clone()


class _Emb:
    """The ``Qwen4ExpPinnedHostEmbedding`` side: the gather and its vocab pair."""

    def __init__(self, gather, start=0, end=VOCAB):
        self._ckpt_pread = gather
        self.shard_indices = types.SimpleNamespace(org_vocab_start_index=start, org_vocab_end_index=end)


class _Model:
    def __init__(self, *mods):
        self._mods = mods

    def modules(self):
        return iter(self._mods)


class _Req:
    def __init__(self, rid, ids):
        self.rid = rid
        self.ids = ids
        self.origin_input_ids = ids.tolist()
        self.full_untruncated_fill_ids = array("q", self.origin_input_ids)
        self.extend_range = None


def _gather(table, hasher=_identity_hasher, procs=2, threads=4, delay_s=0.0):
    adm = _adm()
    return adm.PleAdmitPrefetchGather(
        pt.PleCheckpointPreadGather(table, min_rows=16, workers=4), table, hasher,
        procs=procs, threads=threads, delay_s=delay_s,
    )


def _forward_chunk0(g, req, chunk):
    adm = _adm()
    n = len(req.origin_input_ids)
    req.extend_range = Range(0, min(chunk, n))
    pf.publish_ple_next_chunk([req], chunk)
    adm.note_ple_batch([req])
    ids = req.ids[: min(chunk, n)]
    out = torch.empty((ids.numel(), DIM), dtype=torch.bfloat16)
    g.gather_into(ids, out, vocab_start=0, vocab_end=VOCAB)
    return ids, out


def _serial(table, ids):
    base = pt.PleCheckpointPreadGather(table, min_rows=16, workers=4)
    out = torch.empty((ids.numel(), DIM), dtype=torch.bfloat16)
    base.gather_into(ids, out, vocab_start=0, vocab_end=VOCAB)
    base.close()
    return out


def test_base_behaviour_the_first_admission_is_skipped_cold(table):
    """What the metal shows without the prewarm (pins the defect)."""
    adm = _adm()
    g = _gather(table)
    try:
        req = _Req("pdflip-0-5", torch.randint(0, VOCAB, (150,)))
        assert adm.admit_ple_request(req, 200) == "skipped:cold"
    finally:
        g.close()


def test_boot_prewarm_arms_the_first_admission_and_its_forward_joins(table, caplog):
    adm = _adm()
    g = _gather(table, delay_s=0.2)
    try:
        with _switch(True), caplog.at_level(logging.INFO, logger=ADM_LOGGER):
            res = adm.run_boot_prewarm(model=_Model(object(), _Emb(g)))
        assert res is not None and not res.skipped
        assert res.gathers == 1 and tuple(res.verdicts) == ("warm",)
        assert g._last_vocab == (0, VOCAB)
        assert g._workers is not None and g._workers.n_procs == 2
        assert any("P-PREWARM PLE-ADMIT gathers=1" in r.getMessage() for r in caplog.records)
        req = _Req("pdflip-0-5", torch.randint(0, VOCAB, (150,)))
        caplog.clear()
        with caplog.at_level(logging.INFO):
            assert adm.admit_ple_request(req, 200) == "started"
            time.sleep(0.35)  # P wakes / schedules; the read finishes meanwhile
            t = time.monotonic()
            ids, out = _forward_chunk0(g, req, 200)
            fwd_s = time.monotonic() - t
        msgs = [r.getMessage() for r in caplog.records]
        assert not any("no prefill gather in this process yet" in m for m in msgs)
        chunk = [m for m in msgs if m.startswith("PLE-PREFETCH chunk=0")]
        assert chunk and "ready=yes" in chunk[-1] and "hit_rows=150 read_rows=0" in chunk[-1], chunk
        assert fwd_s < 0.15  # the forward did not pay the 0.2 s read
        assert torch.equal(out.view(torch.int16), _serial(table, ids).view(torch.int16))
    finally:
        g.close()


def test_the_hash_constants_are_on_the_host_after_the_prewarm(table):
    """The real hasher (``ple_next_chunk_hasher``): ready only after the boot copy."""
    adm = _adm()
    emb = types.SimpleNamespace(
        layer_multipliers=torch.tensor([3, 5, 7], dtype=torch.long),
        ngram_heads_vocab_sizes=torch.tensor([61, 67, 71, 73], dtype=torch.long),
        ngram_heads_offsets=torch.tensor([0, 61, 128, 199], dtype=torch.long),
        heads_per_ngram=2, ngram_size=3, eos_token_id=1,
    )
    hasher = pf.ple_next_chunk_hasher(emb)
    g = _gather(table, hasher=hasher, procs=1)
    try:
        assert not hasher.ple_hash_ready()
        with _switch(True):
            res = adm.run_boot_prewarm(model=_Model(_Emb(g)))
        assert tuple(res.verdicts) == ("warm",)
        assert hasher.ple_hash_ready()
        # the copy is exact: the host hash equals a fresh copy of the buffers
        toks = torch.randint(0, 50, (40,))
        ref = pf.ple_ngram_lookup_ids(pf.ple_chunk_windows(toks, 0, 3, 1), pf.PleHashParams.of(emb)).reshape(-1)
        assert torch.equal(hasher(toks, 0), ref)
    finally:
        g.close()


def test_switch_off_leaves_the_gather_lazy(table, caplog):
    adm = _adm()
    assert envs.FLLIPER_PDFLIP_ENABLE_TARGETED_PREWARM.get() is False  # default off until metal
    g = _gather(table)
    try:
        with caplog.at_level(logging.INFO, logger=ADM_LOGGER):
            res = adm.run_boot_prewarm(model=_Model(_Emb(g)))
        assert res.skipped == "FLLIPER_PDFLIP_ENABLE_TARGETED_PREWARM off"
        assert g._last_vocab is None and g._workers is None
        assert any("P-PREWARM PLE-ADMIT skipped: FLLIPER_PDFLIP_ENABLE_TARGETED_PREWARM off" in r.getMessage()
                   for r in caplog.records)
    finally:
        g.close()


def test_a_rank_without_an_admitting_gather_skips_named(table):
    adm = _adm()
    plain = pf.PlePrefetchGather(pt.PleCheckpointPreadGather(table, min_rows=16, workers=4),
                                 table, _identity_hasher, procs=1)
    try:
        with _switch(True):
            res = adm.run_boot_prewarm(model=_Model(_Emb(plain), object()))
        assert res.skipped == "no admitting PLE gather on this rank"
        assert plain._workers is None
    finally:
        plain.close()


def test_the_ple_prewarm_never_raises():
    adm = _adm()

    class Broken:
        def modules(self):
            raise RuntimeError("boom")

    with _switch(True):
        assert adm.run_boot_prewarm(model=Broken()) is None


def test_boot_cost_of_the_ple_prewarm_is_small(table):
    """Serving shape: 4 worker processes x 4 threads (FLLIPER_QWEN4_PLE_PREFETCH_PROCS/
    THREADS defaults). The desk measure is the spawn plus the hash copy."""
    adm = _adm()
    g = _gather(table, procs=4, threads=4)
    try:
        with _switch(True):
            res = adm.run_boot_prewarm(model=_Model(_Emb(g)))
        print(f"P-PREWARM PLE-ADMIT desk ms={res.ms:.1f}")
        assert res.ms < 400.0
    finally:
        g.close()


# ------------------------------------------------------------- (c) router ----


class _Gate:
    def __init__(self, hidden, experts, log):
        self.input_size, self.output_size = hidden, experts
        self.weight = torch.zeros((experts, hidden), dtype=torch.bfloat16)
        self.log = log

    def __call__(self, h):
        self.log.append(("gate", tuple(h.shape), h.dtype))
        return torch.zeros((h.shape[0], self.output_size), dtype=h.dtype), None


class _TopK:
    def __init__(self, log, top_k=8, scoring_func="softmax"):
        self.topk_config = types.SimpleNamespace(top_k=top_k, renormalize=True, scoring_func=scoring_func)
        self.log = log

    def __call__(self, h, logits):
        self.log.append(("topk", tuple(h.shape), tuple(logits.shape), logits.dtype))
        return None


class _Block:
    def __init__(self, log, hidden=64, experts=32, **kw):
        self.gate = _Gate(hidden, experts, log)
        self.topk = _TopK(log, **kw)


@pytest.fixture
def not_form_a(monkeypatch):
    from flliper.srt import rank_role

    monkeypatch.setattr(rank_role, "this_rank_is_form_a_worker", lambda: False)


def test_the_router_is_called_through_the_serving_entry_at_both_keys(not_form_a, caplog):
    rp = _router()
    log = []
    model = _Model(object(), _Block(log), _Block(log), _Block(log))  # one router, three layers
    with _switch(True), caplog.at_level(logging.INFO, logger=ROUTER_LOGGER):
        res = rp.run_boot_prewarm(model=model, dtype=torch.bfloat16, device="cpu")
    assert res is not None and not res.skipped and res.launched == 2 and len(res.routers) == 1
    assert log == [
        ("gate", (16, 64), torch.bfloat16), ("topk", (16, 64), (16, 32), torch.bfloat16),
        ("gate", (17, 64), torch.bfloat16), ("topk", (17, 64), (17, 32), torch.bfloat16),
    ]
    assert any(r.getMessage().startswith("P-PREWARM MOE-ROUTER routers=[hidden=64 experts=32 top_k=8")
               for r in caplog.records)


def test_distinct_routers_each_get_their_call(not_form_a):
    rp = _router()
    log = []
    model = _Model(_Block(log), _Block(log, experts=48), _Block(log, top_k=4), _Block(log))
    with _switch(True):
        res = rp.run_boot_prewarm(model=model, dtype=torch.bfloat16, device="cpu")
    assert len(res.routers) == 3 and res.launched == 6


def test_router_named_skips(monkeypatch):
    rp = _router()
    from flliper.srt import rank_role

    log = []
    res = rp.run_boot_prewarm(model=_Model(_Block(log)), dtype=torch.bfloat16, device="cpu")
    assert res.skipped == "FLLIPER_PDFLIP_ENABLE_TARGETED_PREWARM off" and log == []
    with _switch(True):
        monkeypatch.setattr(rank_role, "this_rank_is_form_a_worker", lambda: True)
        res = rp.run_boot_prewarm(model=_Model(_Block(log)), dtype=torch.bfloat16, device="cpu")
        assert res.skipped == "Form-A worker (routes nothing)" and log == []
        monkeypatch.setattr(rank_role, "this_rank_is_form_a_worker", lambda: False)
        res = rp.run_boot_prewarm(model=_Model(object()), dtype=torch.bfloat16, device="cpu")
        assert res.skipped == "no routed MoE block on this rank"


def test_the_router_prewarm_never_raises(not_form_a):
    rp = _router()
    blk = _Block([])
    blk.topk = _TopK([])
    blk.topk.__class__ = type("Bad", (_TopK,), {"__call__": lambda self, h, l: (_ for _ in ()).throw(RuntimeError("x"))})
    with _switch(True):
        assert rp.run_boot_prewarm(model=_Model(blk), dtype=torch.bfloat16, device="cpu") is None


# ------------------------------------------------------------ the wiring -----


def test_the_scheduler_prewarms_before_the_sampling_barrier():
    src = (REPO / "python/flliper/srt/managers/scheduler.py").read_text()
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
    meths = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    delegate = ast.unparse(meths["warm_targeted_prewarm"])
    assert "router_prewarm import run_boot_prewarm" in delegate
    assert "qwen4_exp_ple_admit import run_boot_prewarm" in delegate
    caller = [m for m in meths.values()
              if "self.warm_sampling_backend()" in ast.unparse(m) and m.name != "warm_sampling_backend"]
    assert len(caller) == 1
    body = ast.unparse(caller[0])
    assert body.index("self.warm_qsa_mqa_tilelang()") < body.index("self.warm_targeted_prewarm()")
    assert body.index("self.warm_targeted_prewarm()") < body.index("self.warm_sampling_backend()")


def test_no_host_store_touch_is_built_for_moe_fetch():
    """(d) is refuted on the metal (per-expert fetch equal on forward 1); the
    prewarm touches no expert store."""
    src = (REPO / "python/flliper/srt/layers/moe/router_prewarm.py").read_text()
    assert not re.search(r"expert_offload|host_store|madvise|mlock", src.split('"""', 2)[2])
