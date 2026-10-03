"""dynpf-Praefix 0929: the #1481 END-ANCHOR mark follows the CLAIM ANCHOR.

METAL (NF dynpf boot 0929_102750, 424346f693, rid weg2-24-38, N=16449, chunks
[0, 8192) + [8192, 16449)): D read 8192 of 16384 tokens back --
``#1028B FETCH CAP n=7: kv=256 claimed=128 lost=128 caps={mamba: 128,
qsa_indexer: 256}`` -- D prefilled the other 8257 itself, leg 2 served
``cached_tokens=8192`` (W16), the front credited 8192 for the follow-up
(``X-EXACT-PRICE rid=weg2-26-44 pending=9055 credit=8192``) and the next turn
went LONG to P: 5-6 flips, flip P->D 18.4 s instead of 3.8 s.

80fa726f31 (CLAIM ANCHOR, in z30y) moved P's last extend track to the store
reader's claim floor_page(N-2) = 16384. The #1233/#1481 END-ANCHOR probe kept
asking for N-1 (16448 units under the exact bigram key) -- one page past that
anchor. On the base therefore:

* dynpf shape: the probe finds the claim anchor at 16384 < 16448, prints
  ``ok=False``, counts a short and marks NOTHING -- the one anchor the reader
  can reach has no un-backed eviction hold (#1469 HOLD-END-ANCHOR), no carrier
  hold across P's reset (H81) and no arena-victim exemption (H19);
* fixed-chunk shape ([0, 16384) + [16384, N), the claim track does not fire):
  the mark sits on the N-1 leaf at 16448 that no reader reaches, while the
  interior chunk anchor at 16384 -- the one D resumes from -- is unmarked,
  exactly the #1481 loss form (interior anchor evicted under a second prompt).

Fix: one predicate (``tail_handoff.claim_anchor_end``) for the track and the
probe; the probe asks for the claim, the mark lands on the anchor the reader
reaches. RED on 895559fed2 (the dynpf and fixed-chunk shapes, N % 64 == 1, and
the invariant), GREEN with the fix. Controls identical on both: N % 64 != 1
(N % 64 == 0 already probes the claim), group D / no group, the upstream keying
(27B), a P-trim request.

Hermetic, CPU: the REAL extend track (``_mamba_radix_cache_v2_req_prepare_for_extend``),
the REAL retention key (``bigram_anchor_key``), the REAL probe
(``UnifiedRadixCache._weg2_note_end_anchor``) on a page-64 tree whose anchors
sit at the node ends those inserts file.
"""

import logging
from array import array
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache import unified_radix_cache as urc
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.weg2 import p_trim_end_anchor as pt
from sglang.srt.weg2 import tail_handoff as th

PAGE = 64


@pytest.fixture
def group_p(monkeypatch):
    import sglang.srt.managers.schedule_batch as sb

    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    monkeypatch.delenv("SGLANG_WEG2_FORK_ANCHOR_TOKEN", raising=False)
    monkeypatch.setattr(sb, "get_server_args", lambda: SimpleNamespace(
        mamba_cache_chunk_size=PAGE, mamba_checkpoint_interval=None,
        enable_mamba_extra_buffer_lazy=lambda: False))
    return monkeypatch, sb


def _last_step_track(sb, tree, req, prefix):
    """The REAL extend track of the step [prefix, N): the depth the finish
    insert retains (``mamba_last_track_seqlen`` -> ``cache_len``)."""
    n = len(req.origin_input_ids)
    step = SimpleNamespace(
        rid=req.rid, origin_input_ids=req.origin_input_ids, output_ids=[],
        return_logprob=False, input_embeds=None, session_id=None, multimodal_inputs=None,
        prefix_indices=list(range(prefix)), mamba_ping_pong_track_buffer=torch.tensor([0, 1]),
        mamba_next_track_idx=0, mamba_branching_seqlen=None, mamba_last_track_seqlen=None,
        extend_range=SimpleNamespace(start=prefix, end=n, length=n - prefix))
    if getattr(req, pt.TRIM_ATTR, None) is not None:
        setattr(step, pt.TRIM_ATTR, getattr(req, pt.TRIM_ATTR))
    batch = SimpleNamespace(
        tree_cache=tree,
        req_to_token_pool=SimpleNamespace(get_mamba_ping_pong_other_idx=lambda i: 1 - i))
    sb.ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend(batch, step)
    return step.mamba_last_track_seqlen


def _prefill(sb, n, chunks=(), exact=True, trim=False, rid=None):
    """One finished P prompt of ``n`` tokens, request chunk ends ``chunks``:
    anchors at every chunk end and at the finish's retained depth, keyed by the
    real ``bigram_anchor_key``; then the REAL END-ANCHOR probe. The match
    returns the deepest anchor at or below the probe (the mamba validator)."""
    prompt = array("q", range(1000, 1000 + n))
    ids = prompt[:-1] if trim else prompt
    root = SimpleNamespace(name="root")
    probes = []
    anchors = []
    nodes = {}

    def match_prefix(params):
        units = len(params.key)
        probes.append(units)
        depth = max([a for a in anchors if a <= units], default=0)
        return SimpleNamespace(device_indices=torch.empty(depth),
                               last_device_node=nodes.get(depth, root))

    tree = SimpleNamespace(is_eagle=True, page_size=PAGE, bigram_anchor_exact=exact,
                           match_prefix=match_prefix, root_node=root)
    req = SimpleNamespace(rid=rid or f"weg2-cm-{n}", extra_key=None, origin_input_ids=ids)
    if trim:
        setattr(req, pt.TRIM_ATTR, prompt[-1:])
    last_start = chunks[-1] if chunks else 0
    retain = _last_step_track(sb, tree, req, last_start)
    if retain is None or retain <= last_start:
        retain = len(ids)
    ends = list(chunks) + [retain]

    def key_units(cache_len):
        return len(urc.bigram_anchor_key(ids, cache_len, None, is_bigram=True,
                                         exact=exact, page_size=PAGE))

    anchors.extend(sorted({key_units(c) for c in ends}))
    nodes.update({a: SimpleNamespace(name=f"node@{a}") for a in anchors})
    UnifiedRadixCache._weg2_note_end_anchor(tree, req, ids)
    return SimpleNamespace(nodes=nodes, anchors=anchors, probes=probes, rid=req.rid)


def _line(caplog, rid):
    lines = [r.getMessage() for r in caplog.records
             if "WEG2 END-ANCHOR n=" in r.getMessage() and f"rid={rid[:12]} " in r.getMessage()]
    assert len(lines) == 1, lines
    return lines[0]


def _marked(res):
    return sorted(a for a, node in res.nodes.items() if getattr(node, "_weg2_end_anchor", False))


def _reader_reaches(res, n):
    """What D's store read claims: the deepest anchor at or below its claim
    (``#1028B FETCH CAP`` -- anchors_in_range, mamba)."""
    claim = th.reader_claim_end(n, PAGE, True)
    return max([a for a in res.anchors if a <= claim], default=0)


# -- the metal shapes ----------------------------------------------------------


@pytest.mark.parametrize("chunks", [(8192,), (16256,), (12160,)],
                         ids=["weg2-24-38", "weg2-24-40", "weg2-24-41"])
def test_dynpf_metal_prompt_marks_the_claim_anchor(group_p, caplog, chunks):
    _mp, sb = group_p
    with caplog.at_level(logging.WARNING):
        res = _prefill(sb, 16449, chunks=chunks)
    assert res.anchors[-1] == 16384, "80fa726f31: the last step tracks at the claim"
    assert res.probes == [16384], "the probe asks for the reader's claim, not N-1 (16448)"
    line = _line(caplog, res.rid)
    assert "tokens=16449 anchor=16384 target=16384 units=16384/16384 ok=True" in line, line
    assert line.endswith("claim=16384"), line
    assert _marked(res) == [16384] == [_reader_reaches(res, 16449)], "#1481 on the anchor D reads"


def test_fixed_chunk_shape_marks_the_interior_claim_anchor_not_the_n_minus_1_leaf(group_p, caplog):
    """weg2-10-14 form: [0, 16384) + [16384, 16449) -- the claim track does not
    fire, the chunk anchor at 16384 is what D resumes from (``X-GATE
    uncached=65``); the base marked the unreachable leaf at 16448."""
    _mp, sb = group_p
    with caplog.at_level(logging.WARNING):
        res = _prefill(sb, 16449, chunks=(16384,), rid="weg2-10-14")
    assert res.anchors == [16384, 16448]
    assert res.probes == [16384]
    assert "anchor=16384 target=16384 units=16384/16384 ok=True" in _line(caplog, res.rid)
    assert _marked(res) == [16384], "interior anchor at the claim marked, the leaf is not"


def test_control_page_multiple_prompt_already_probes_the_claim(group_p, caplog):
    """N % 64 == 0: the N-1 probe's page floor IS the claim (16447 units ->
    16384), so the base already marked the claim anchor -- no cut, no change."""
    _mp, sb = group_p
    with caplog.at_level(logging.WARNING):
        res = _prefill(sb, 16448, chunks=(8192,))
    assert res.anchors[-1] == 16384 and res.probes == [16384]
    line = _line(caplog, res.rid)
    assert "tokens=16448 anchor=16384 target=16447 units=16384/16384 ok=True" in line, line
    assert "claim=" not in line
    assert _marked(res) == [16384]


@pytest.mark.parametrize("n", [1025, 9537, 16448, 16449, 16450, 17247, 20033, 23361, 32833, 32834])
@pytest.mark.parametrize("width", [4096, 8192, 16384])
def test_the_mark_is_always_what_the_reader_claims(group_p, caplog, n, width):
    """Invariant over prompt lengths and fixed/dynamic chunk grids: the probe
    reports ok=True and the #1481 mark sits on exactly the anchor D's store
    read claims (never deeper, never none)."""
    _mp, sb = group_p
    chunks = tuple(range(width, n, width))
    # a chunk ending past the reader claim leaves only the N-1 retain above it
    with caplog.at_level(logging.WARNING):
        res = _prefill(sb, n, chunks=chunks, rid=f"weg2-inv-{n}-{width}")
    reach = _reader_reaches(res, n)
    assert reach > 0, (res.anchors, n)
    assert "ok=True" in _line(caplog, res.rid), (res.anchors, res.probes)
    assert _marked(res) == [reach], (res.anchors, res.probes, reach)


# -- controls: identical on the base and with the fix ----------------------------


def test_control_n_not_0_or_1_mod_page_is_unchanged(group_p, caplog):
    _mp, sb = group_p
    with caplog.at_level(logging.WARNING):
        res = _prefill(sb, 16450, chunks=(8192,))
    assert res.probes == [16448]
    line = _line(caplog, res.rid)
    assert "tokens=16450 anchor=16448 target=16449 units=16448/16448 ok=True" in line, line
    assert "claim=" not in line
    assert _marked(res) == [16448]


@pytest.mark.parametrize("group", ["D", None])
def test_control_other_groups_probe_n_minus_1(group_p, caplog, group):
    mp, sb = group_p
    if group is None:
        mp.delenv("SGLANG_WEG2_GROUP", raising=False)
    else:
        mp.setenv("SGLANG_WEG2_GROUP", group)
    with caplog.at_level(logging.WARNING):
        res = _prefill(sb, 16449, chunks=(8192,), rid=f"weg2-g{group}")
    assert res.anchors == [8192, 16448], "no claim track off group P"
    assert res.probes == [16448]
    line = _line(caplog, res.rid)
    assert "target=16448 units=16448/16448 ok=True" in line and "claim=" not in line, line
    assert _marked(res) == [16448]


def test_control_upstream_keying_is_unchanged(group_p, caplog):
    """27B profile (exact off): the upstream slice, N-2 units -- already the
    reader's claim, no claim cut."""
    _mp, sb = group_p
    with caplog.at_level(logging.WARNING):
        res = _prefill(sb, 1025, exact=False, rid="weg2-27b")
    assert res.probes == [960]
    line = _line(caplog, res.rid)
    assert "tokens=1025 anchor=961 target=1024 units=960/960 ok=True" in line, line
    assert "claim=" not in line


def test_control_p_trim_form_is_unchanged(group_p, caplog):
    _mp, sb = group_p
    with caplog.at_level(logging.WARNING):
        res = _prefill(sb, 1025, trim=True, rid="weg2-trim")
    assert res.probes == [960]
    line = _line(caplog, res.rid)
    assert "tokens=1025 anchor=960 target=1024 units=960/960 ok=True" in line, line
    assert line.endswith("trim=1"), line


def test_track_and_probe_share_one_predicate(group_p):
    """One mechanism: the claim track and the probe ask the same function."""
    import inspect

    import sglang.srt.managers.schedule_batch as sb

    assert "claim_anchor_end" in inspect.getsource(sb._weg2_claim_track)
    assert "claim_anchor_end" in inspect.getsource(UnifiedRadixCache._weg2_note_end_anchor)
    req = SimpleNamespace(origin_input_ids=array("q", range(16449)))
    tree = SimpleNamespace(page_size=PAGE, bigram_anchor_exact=True)
    assert th.claim_anchor_end(req, tree) == 16384 == th.reader_claim_end(16449, PAGE, True)
