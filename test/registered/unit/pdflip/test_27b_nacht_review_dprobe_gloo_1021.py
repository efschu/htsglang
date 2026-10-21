# SPDX-License-Identifier: Apache-2.0
"""27B NACHT REVIEW 21.10. -- D-PROBE GROUP-MIN (8fc257e6a5) on a REAL three-process gloo group.

The author's suite (test_27b_dprobe_group_min_1021) emulates the MIN in-process. This file drives the real
``Scheduler.handle_pdflip_match_probe`` on three forked processes with a real gloo CPU group, interleaved with the
intake's own broadcast (the control request reaches every rank through it) and a trailing sentinel collective, so a
rank that skips or doubles the probe's reduce shows up as a hang (timeout) or a misaligned sentinel, not as a mocked
call count. Cases: a coalesced batch, a rank whose read-only walk raises for one item only, a rank whose padding
raises, an empty prompt, ranks with different tree depths (the group answers the MIN).
"""
import multiprocessing as mp
import os
import tempfile
import types
from datetime import timedelta

import pytest
import torch

WORLD = 3
ROUNDS = 4


def _walk_for(rank):
    """Per-rank tree: dev/host depth differ per rank; rank 1 raises for the ids that start with 7 (round 1, item 1)."""

    def _peek(tc, ids, extra_key, limit):
        n = len(ids)
        if rank == 1 and n and int(ids[0]) == 7:
            raise RuntimeError("walk failed on rank 1")
        dev = max(0, n - 4 - rank)          # rank 2 holds the least on device
        host = 2 if rank != 0 else 1         # rank 0 holds the least on the host tier
        return dev, host, max(0, n - 3 - (rank % 2))

    return _peek


def _child(rank, init_file, out_q):
    import torch.distributed as dist

    from flliper.srt.managers import scheduler as S
    from flliper.srt.managers import tp_match_floor as T
    from flliper.srt.managers.io_struct import PdFlipMatchProbeReqInput, unwrap_from_pickle, wrap_as_pickle
    from flliper.srt.pdflip import d_match_probe as dmp

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD,
                            timeout=timedelta(seconds=40))
    grp = dist.new_group(list(range(WORLD)), backend="gloo", timeout=timedelta(seconds=40))
    T.form_a_follow_active = lambda: False
    dmp.peek_admission_match = _walk_for(rank)

    class _Tree:
        def _hicache_prefetch_symmetric(self):
            return True

    def _get_mm(mm):
        if rank == 2 and mm == "boom":
            raise ValueError("padding failed on rank 2")
        return None

    fake = types.SimpleNamespace(
        tree_cache=_Tree(), tp_cpu_group=grp, model_config=types.SimpleNamespace(vocab_size=1000),
        pad_input_ids_func=None, _get_multimodal_inputs=_get_mm,
        _try_apply_padded_mm_input_ids=lambda recv, req, mm: False,
    )
    for m in ("_pdflip_match_probe_one", "_pdflip_match_probe_pad", "_pdflip_match_probe_local",
              "_pdflip_match_probe_group_min", "handle_pdflip_match_probe"):
        setattr(fake, m, (lambda n: lambda *a: getattr(S.Scheduler, n)(fake, *a))(m))
    # mm_inputs "boom" must reach _get_multimodal_inputs: unwrap_from_pickle passes a plain str through
    S_unwrap = S.__dict__.get("unwrap_from_pickle")  # noqa: F841 -- the handler imports it locally

    results = []
    try:
        for rnd in range(ROUNDS):
            # the intake broadcast: TP0 holds the list, every rank receives it
            obj = [None]
            if rank == 0:
                items = [
                    {"rid": f"r{rnd}-1", "input_ids": [7 if rnd == 1 else 5] + list(range(20 + rnd)),
                     "mm_inputs": None, "extra_key": None, "tokenize_ms": 0.0},
                    {"rid": f"r{rnd}-2", "input_ids": [1], "mm_inputs": None, "extra_key": None,
                     "tokenize_ms": 0.0},
                    {"rid": f"r{rnd}-3", "input_ids": list(range(40)),
                     "mm_inputs": ("boom" if rnd == 2 else None), "extra_key": None, "tokenize_ms": 0.0},
                ]
                req = PdFlipMatchProbeReqInput(rid=f"r{rnd}-0", input_ids=list(range(30)), mm_inputs=None,
                                               extra_key=None, tokenize_ms=0.0)
                req.batch = wrap_as_pickle(items)
                obj = [req]
            dist.broadcast_object_list(obj, src=0, group=grp)
            out = fake.handle_pdflip_match_probe(obj[0])
            ans = [(out.rid, bool(out.success), int(out.device_tokens or 0), int(out.host_tokens or 0),
                    int(out.key_match_depth or 0))]
            for d in unwrap_from_pickle(out.batch) or ():
                ans.append((d["rid"], bool(d["success"]), int(d["device_tokens"] or 0),
                            int(d["host_tokens"] or 0), int(d["key_match_depth"] or 0)))
            # sentinel: every rank must be at the same collective position after the probe round
            t = torch.tensor([rnd * 1000 + 1], dtype=torch.int64)
            dist.all_reduce(t, op=dist.ReduceOp.SUM, group=grp)
            assert int(t.item()) == WORLD * (rnd * 1000 + 1), (rank, rnd, int(t.item()))
            results.append(ans)
        out_q.put((rank, "ok", results))
    except BaseException as e:  # noqa: BLE001
        out_q.put((rank, "err", f"{type(e).__name__}: {e}"))
    finally:
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


@pytest.mark.timeout(180)
def test_group_min_probe_on_real_gloo_three_ranks_stays_aligned():
    ctx = mp.get_context("fork")
    with tempfile.TemporaryDirectory() as d:
        init_file = os.path.join(d, "init")
        q = ctx.Queue()
        ps = [ctx.Process(target=_child, args=(r, init_file, q)) for r in range(WORLD)]
        for p in ps:
            p.start()
        got = {}
        for _ in range(WORLD):
            rank, status, payload = q.get(timeout=150)
            got[rank] = (status, payload)
        for p in ps:
            p.join(timeout=30)
    for r in range(WORLD):
        assert got[r][0] == "ok", (r, got[r])
    res = [got[r][1] for r in range(WORLD)]
    # every rank answers the SAME group number (rank-uniform), round by round
    assert res[0] == res[1] == res[2]
    by_rid = {a[0]: a for rnd in res[0] for a in rnd}
    # head probe, 30 ids: dev MIN = 30-4-2 = 24, total MIN = min(25+1, 25+2, 24+2) = 26 -> host 2, key MIN 30-3-1
    assert by_rid["r0-0"] == ("r0-0", True, 24, 2, 26)
    # round 1 item 1: rank 1's walk raised -> the whole group refuses (no rank answers its own number)
    assert by_rid["r1-1"][1] is False
    assert by_rid["r0-1"][1] is True and by_rid["r2-1"][1] is True
    # round 2 item 3: rank 2's padding raised -> the group refuses; the next probe of the same batch is fine
    assert by_rid["r2-3"][1] is False
    assert by_rid["r3-3"][1] is True
    # single-token prompt: no walk, still one reduce, success with zero
    assert by_rid["r0-2"] == ("r0-2", True, 0, 0, 0)
