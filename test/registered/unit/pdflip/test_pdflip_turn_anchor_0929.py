"""TURN ANCHOR (NF y3m boot ...dauer09292136, 375f44975e).

P 22:02:22, pdflip-50-71: ``#1028B FETCH CAP kv=1005 claimed=989 lost=16
MAMBA (11, 988)``. The KV of the shared prefix reached the fork 64352
(front SESSION-PREFIX pdflip-50-66 vs pdflip-46-61: common=64352 of 64515), the
recurrent state did not: 46-61's only anchor past its chunk anchor was the
END anchor 64512, 163 tokens behind the fork -- the start of 46-61's LAST
message, where the next turn and the client's side requests leave it.

Hermetic, CPU. Pinned:
  A. the boundary is the ``<|im_start|>`` of the prompt's last message (the one
     before the generation prompt's), floored to the page; a step plans its
     second track only strictly inside itself, below its own track, on the
     FLA grid;
  B. the scheduler side draws a slot per planned row (no eviction), frees a
     stale plan, skips without a free slot;
  C. the GDN backend adds exactly the ``h`` row and conv window the upstream
     track math picks for a track at t (checked against the REAL
     ``_init_track_ssm_indices`` / ``_init_track_conv_indices``), and the
     copy lands the state after t tokens in the turn slot;
  D. on the REAL UnifiedRadixCache (unigram and exact bigram): the next turn
     resumes AT the turn boundary with the turn state (red without the
     insert: it resumes from nothing), the step's own END/chunk anchor is
     unchanged, an unmarked plan (GDN or PLE side state not written) inserts
     nothing and frees its slot;
  E. the launcher adds the env to group P only, {} when off.
"""

from array import array
from types import SimpleNamespace

import pytest
import torch

from flliper.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from flliper.srt.environ import envs
from flliper.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from flliper.srt.managers.schedule_batch import Req
from flliper.srt.mem_cache.allocator import TokenToKVPoolAllocator
from flliper.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from flliper.srt.mem_cache.cache_init_params import CacheInitParams
from flliper.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from flliper.srt.mem_cache.radix_cache import RadixKey
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from flliper.srt.sampling.sampling_params import SamplingParams
from flliper.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

TOKEN_ENV = "FLLIPER_PDFLIP_TURN_ANCHOR_TOKEN"   # the switch, by name
IM, IM_END, NL = 248045, 248046, 198           # <|im_start|> <|im_end|> \n (Qwen3.8 / NF)
USER, ASSISTANT, THINK = 872, 74455, 248068
GEN = [IM, ASSISTANT, NL, THINK, NL]          # "<|im_start|>assistant\n<think>\n"
SYS = [IM, 8948, NL] + list(range(3000, 3030)) + [IM_END, NL]
U1 = [IM, USER, NL] + list(range(4000, 4010)) + [IM_END, NL]
A1 = [IM, ASSISTANT, NL] + list(range(5000, 5010)) + [IM_END, NL]
LAST = [IM, USER, NL] + list(range(6000, 6012)) + [IM_END, NL]
PROMPT = SYS + U1 + A1 + LAST + GEN
N = len(PROMPT)
I_LAST = len(SYS) + len(U1) + len(A1)          # index of LAST's <|im_start|>
# the next turn: the same chat up to LAST's <|im_start|>, then the assistant's
# re-rendered reply instead of LAST (the 50-66-from-46-61 shape: common = i + 1)
NEXT = PROMPT[: I_LAST + 1] + [ASSISTANT, NL] + list(range(7000, 7020)) + [IM_END, NL] + GEN
RID = "pdflip-46-61"
S_EARLY, S_TURN, S_END = 0.5, 3.25, 7.5


# -- A. the pure rule ----------------------------------------------------------


def test_the_boundary_is_the_last_messages_turn_start():
    from flliper.srt.pdflip import turn_anchor as ta

    assert ta.boundary(PROMPT, IM) == I_LAST
    assert ta.boundary(PROMPT, None) is None, "switch off"
    assert ta.boundary(PROMPT[:-5], IM) is None, "no generation prompt: not a chat prompt"
    assert ta.boundary(SYS[:-2] + GEN, IM) == 0, "one message: its start is 0"
    assert ta.anchor_pos(SYS[:-2] + GEN, IM, 1) is None, "position 0 anchors nothing"
    # the page floor: a reader sharing ids[:i+1] claims at most floor_page(i)
    assert ta.anchor_pos(PROMPT, IM, 1) == I_LAST
    assert ta.anchor_pos(PROMPT, IM, 16) == I_LAST // 16 * 16
    # the metal numbers: fork 64352 = i + 1 -> anchor 64320 = kv=1005 pages
    ids = [7] * 64351 + [IM] + [USER] + [9] * 160 + GEN
    assert ta.boundary(ids, IM) == 64351 and ta.anchor_pos(ids, IM, 64) == 64320 == 1005 * 64


def test_a_step_plans_only_inside_itself_below_its_track_on_the_grid():
    from flliper.srt.pdflip import turn_anchor as ta

    # 46-61's step [63296, 64515): main track floor_page(N) = 64512, turn 64320
    assert ta.step_target(63296, 64515, 64320, 64, 64512) == 64320
    assert ta.step_target(64320, 64515, 64320, 64, 64512) is None, "at the step start: anchored already"
    assert ta.step_target(0, 64320, 64320, 64, 64320) is None, "at the step end: the chunk anchor"
    assert ta.step_target(63296, 64515, 64320, 64, 64256) is None, "not below the main track"
    assert ta.step_target(63300, 64515, 64320, 64, 64512) is None, "off the step's FLA grid"
    assert ta.step_target(63296, 64515, None, 64, 64512) is None
    assert ta.step_target(63296, 64515, 64320, 64, None) is None, "no main track: no second"


# -- B. the scheduler side -------------------------------------------------------


class _Alloc:
    def __init__(self, free):
        self.free_ids = list(free)
        self.freed = []

    def alloc(self, n):
        if len(self.free_ids) < n:
            return None
        out, self.free_ids = self.free_ids[:n], self.free_ids[n:]
        return torch.tensor(out, dtype=torch.int64)

    def free(self, t):
        self.freed.extend(int(x) for x in t.view(-1).tolist())


def _sreq(ids, rid=RID, cpl=0):
    return SimpleNamespace(rid=rid, origin_input_ids=array("q", ids), output_ids=[],
                           cache_protected_len=cpl)


def _plan(desc, req, alloc, prefix, end, main, mask=True):
    from flliper.srt.pdflip import turn_anchor as ta

    batch = SimpleNamespace(reqs=[req], req_to_token_pool=SimpleNamespace(mamba_allocator=alloc))
    return ta.note_step(batch=batch, desc=desc, req=req, row=0, prefix=prefix, end=end,
                        track_mask=mask, main_track=main, chunk=1, page=1, tok=IM)


def test_the_plan_requires_every_slot_state_the_pool_carries():
    """NF's slot also holds the PLE short-conv and n-gram states: a plan
    requires them from the start, so a forward whose PLE code never ran in
    Python (a captured graph) can not satisfy it with the GDN mark alone."""
    from flliper.srt.pdflip import turn_anchor as ta

    on, off = SimpleNamespace(enabled=True), SimpleNamespace(enabled=False)
    alloc = _Alloc([11])
    req = _sreq(PROMPT)
    batch = SimpleNamespace(reqs=[req], req_to_token_pool=SimpleNamespace(
        mamba_allocator=alloc, short_conv_pool=on, ngram_pool=on))
    desc = ta.note_step(batch=batch, desc=None, req=req, row=0, prefix=0, end=N,
                        track_mask=True, main_track=N - 1, chunk=1, page=1, tok=IM)
    assert desc.need == {"gdn", "ple_conv", "ple_ngram"}
    desc.done.add("gdn")
    assert not desc.complete()
    assert ta.slot_state_kinds(SimpleNamespace(short_conv_pool=off, ngram_pool=off)) == ()


def test_a_stage_without_a_ple_layer_does_not_require_the_ngram_history():
    """P-TURN-REUSE (y3u 30.09.): NF's P is PP3 and only PP0 owns the PLE
    layer. The pool still carries the n-gram rows on PP1/PP2
    (``ngram_context_len`` is not stage-filtered, the short-conv ids are), but
    no forward there writes them -- requiring them refused EVERY turn anchor
    on PP1/PP2 ('TURN-ANCHOR SKIP reason=unmarked:ple_ngram'). Red on the base:
    the plan required ('gdn', 'ple_ngram')."""
    from flliper.srt.pdflip import turn_anchor as ta

    on, off = SimpleNamespace(enabled=True), SimpleNamespace(enabled=False)
    pp1 = SimpleNamespace(short_conv_pool=off, ngram_pool=on)
    assert ta.slot_state_kinds(pp1) == ()
    pp0 = SimpleNamespace(short_conv_pool=on, ngram_pool=on)
    assert ta.slot_state_kinds(pp0) == ("ple_conv", "ple_ngram"), \
        "the stage that keeps the history still requires it"
    alloc = _Alloc([11])
    req = _sreq(PROMPT)
    batch = SimpleNamespace(reqs=[req], req_to_token_pool=SimpleNamespace(
        mamba_allocator=alloc, short_conv_pool=off, ngram_pool=on))
    desc = ta.note_step(batch=batch, desc=None, req=req, row=0, prefix=0, end=N,
                        track_mask=True, main_track=N - 1, chunk=1, page=1, tok=IM)
    desc.done.add("gdn")    # what PP1's forward leaves: the GDN rows, no PLE code
    assert desc.complete()


def test_the_scheduler_plans_one_slot_per_turn_row_and_frees_a_stale_one():
    from flliper.srt.pdflip import turn_anchor as ta

    alloc = _Alloc([11, 12])
    req = _sreq(PROMPT)
    desc = _plan(None, req, alloc, 0, N, N - 1)
    assert desc is not None and desc.rows == [0] and desc.targets == [I_LAST]
    t, slot, d, s0, s1 = getattr(req, ta.PENDING_ATTR)
    assert (t, int(slot), d, s0, s1) == (I_LAST, 11, desc, 0, N)
    assert not desc.complete(), "nothing written yet"
    # a second plan while the first was never consumed: the first slot goes back
    desc2 = _plan(None, req, alloc, 0, N, N - 1)
    assert alloc.freed == [11] and int(getattr(req, ta.PENDING_ATTR)[1]) == 12
    assert desc2 is not desc


@pytest.mark.parametrize("why,kw", [
    ("no slot free", dict(alloc=[], prefix=0, end=N, main=N - 1)),
    ("step after the boundary", dict(alloc=[5], prefix=I_LAST, end=N, main=N - 1)),
    ("step before the boundary", dict(alloc=[5], prefix=0, end=I_LAST, main=I_LAST - 1)),
    ("no default track this step", dict(alloc=[5], prefix=0, end=N, main=N - 1, mask=False)),
])
def test_the_scheduler_plans_nothing(why, kw):
    from flliper.srt.pdflip import turn_anchor as ta

    alloc = _Alloc(kw.pop("alloc"))
    req = _sreq(PROMPT)
    desc = _plan(None, req, alloc, kw["prefix"], kw["end"], kw["main"], kw.get("mask", True))
    assert desc is None and getattr(req, ta.PENDING_ATTR, None) is None, why
    assert alloc.freed == []


def test_a_protected_prefix_at_the_boundary_plans_nothing():
    alloc = _Alloc([5])
    req = _sreq(PROMPT, cpl=I_LAST)
    assert _plan(None, req, alloc, 0, N, N - 1) is None


def test_the_switch_needs_group_p_extra_buffer_and_no_overlap(monkeypatch):
    from flliper.srt.pdflip import turn_anchor as ta

    sa = SimpleNamespace(enable_mamba_extra_buffer=lambda: True,
                         mamba_checkpoint_interval=None, disable_overlap_schedule=True)
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "P")
    assert ta.armed(sa) is None, "off without the token"
    monkeypatch.setenv(TOKEN_ENV, str(IM))
    assert ta.armed(sa) == IM
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    assert ta.armed(sa) is None, "group D: overlap schedule, no END anchor"
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "P")
    assert ta.armed(SimpleNamespace(**{**sa.__dict__, "disable_overlap_schedule": False})) is None
    assert ta.armed(SimpleNamespace(**{**sa.__dict__, "mamba_checkpoint_interval": 4096})) is None
    assert ta.armed(SimpleNamespace(**{**sa.__dict__,
                                       "enable_mamba_extra_buffer": lambda: False})) is None


# -- C. the GDN backend's rows ------------------------------------------------------


@pytest.fixture
def server_args_64():
    sa = ServerArgs(model_path="dummy", page_size=1)
    sa._mamba_cache_chunk_size = FLA_CHUNK_SIZE
    set_global_server_args_for_scheduler(sa)
    return sa


def _fb(prefixes, ext, track_seqlens, track_slots, turn=None):
    from flliper.srt.model_executor.forward_batch_info import ForwardMode

    bs = len(ext)
    return SimpleNamespace(
        batch_size=bs, forward_mode=ForwardMode.EXTEND,
        extend_prefix_lens_cpu=list(prefixes), extend_seq_lens_cpu=list(ext),
        extend_prefix_lens=torch.tensor(prefixes, dtype=torch.int32),
        extend_seq_lens=torch.tensor(ext, dtype=torch.int32),
        mamba_track_mask=torch.ones(bs, dtype=torch.bool),
        mamba_track_indices=torch.tensor(track_slots, dtype=torch.int64),
        mamba_track_seqlens=torch.tensor(track_seqlens, dtype=torch.int64),
        pdflip_turn_tracks=turn,
    )


def _backend(conv_len=3):
    from flliper.srt.layers.attention.linear.gdn_backend import GDNAttnBackend

    be = object.__new__(GDNAttnBackend)
    be.device = torch.device("cpu")
    be.conv_states_shape = (8, conv_len)
    be.req_to_token_pool = SimpleNamespace(translate_mamba_indices=lambda x: x)
    return be


def _metadata(be, fb):
    from flliper.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
    from flliper.srt.model_executor.forward_batch_info import ForwardBatch

    fb.mamba_track_aligned_lens = lambda: ForwardBatch.mamba_track_aligned_lens(fb)
    qsl = torch.tensor([0] + list(torch.cumsum(torch.tensor(fb.extend_seq_lens_cpu), 0)),
                       dtype=torch.int32)
    h_src, h_dst, f_src, f_dst = be._init_track_ssm_indices(
        torch.tensor([90 + i for i in range(fb.batch_size)]), fb)
    md = ForwardMetadata(query_start_loc=qsl, mamba_cache_indices=torch.arange(fb.batch_size))
    md.track_ssm_h_src, md.track_ssm_h_dst = h_src, h_dst
    md.track_ssm_final_src, md.track_ssm_final_dst = f_src, f_dst
    md.track_conv_indices = be._init_track_conv_indices(qsl, fb)
    md.has_mamba_track_mask = True
    md.conv_states_mask_indices = fb.mamba_track_indices
    return md


def test_the_backend_adds_the_h_row_and_conv_window_the_upstream_math_picks(server_args_64):
    """Two rows, the turn on the second: its second track must be exactly the
    row the REAL upstream functions compute for a (default) track at t."""
    from flliper.srt.pdflip import turn_anchor as ta

    prefixes, ext = [0, 63296], [300, 1219]          # 46-61's step is row 1
    t = 64320
    turn = ta.TurnTracks(2)
    turn.add(1, torch.tensor([7]), t, 63296, 1219)
    fb = _fb(prefixes, ext, [256 + 1, 64512 + 1], [40, 41], turn)
    be = _backend()
    be.forward_metadata = _metadata(be, fb)
    base_h = be.forward_metadata.track_ssm_h_src.clone()
    base_conv = be.forward_metadata.track_conv_indices.clone()
    be._pdflip_turn_rows(fb)
    md = be.forward_metadata
    assert "gdn" in turn.done
    # the reference: a default track AT t on the same row, through upstream code
    ref_fb = _fb(prefixes, ext, [256 + 1, t + 1], [40, 7])
    ref = _metadata(_backend(), ref_fb)
    assert md.track_ssm_h_src[: len(base_h)].tolist() == base_h.tolist(), "default rows first, unchanged"
    assert md.track_ssm_h_src[-1].item() == ref.track_ssm_h_src[-1].item()
    assert md.track_ssm_h_dst[-1].item() == 7
    assert md.track_conv_indices[: len(base_conv)].tolist() == base_conv.tolist()
    assert md.track_conv_indices[-1].tolist() == ref.track_conv_indices[-1].tolist()
    assert md.conv_states_mask_indices.tolist() == [40, 41, 7]
    assert turn.rows_dev.tolist() == [1] and turn.offsets_dev.tolist() == [t - 63296]

    # and the copy the forward runs lands the state after t tokens in slot 7
    n_h = sum((L - 1) // 64 + 1 for L in ext)
    h = torch.arange(n_h, dtype=torch.float32).view(1, n_h, 1, 1, 1).expand(1, n_h, 1, 2, 2)
    ssm = torch.zeros(64, 1, 2, 2)
    be._track_mamba_state_extend(fb, h, ssm, md)
    k = (300 - 1) // 64 + 1 + (t - 63296) // 64        # row 1's first state + chunks into it
    assert ssm[7].flatten().tolist() == [float(k)] * 4


def test_a_split_or_unplanned_batch_gets_no_rows(server_args_64):
    from flliper.srt.pdflip import turn_anchor as ta

    turn = ta.TurnTracks(2)
    turn.add(1, torch.tensor([7]), 64320, 63296, 1219)
    fb = _fb([0], [300], [257], [40], turn)          # one row: not the planned batch
    be = _backend()
    be.forward_metadata = _metadata(be, fb)
    before = be.forward_metadata.track_ssm_h_src.tolist()
    be._pdflip_turn_rows(fb)
    assert be.forward_metadata.track_ssm_h_src.tolist() == before
    assert "gdn" not in turn.done and not turn.complete()


def test_a_ple_side_state_is_required_once_registered():
    from flliper.srt.models import qwen4_exp as q
    from flliper.srt.pdflip import turn_anchor as ta

    turn = ta.TurnTracks(1)
    turn.add(0, torch.tensor([7]), 64, 0, 100)
    fb = SimpleNamespace(pdflip_turn_tracks=turn)
    assert q._ple_turn_rows(fb, "ple_conv") is None, "no GDN rows yet: nothing to write"
    assert "ple_conv" in turn.need
    turn.done.add("gdn")
    assert not turn.complete(), "the PLE conv state was registered and not written"
    turn.dst_phys = torch.tensor([7])
    assert q._ple_turn_rows(fb, "ple_conv") is turn
    turn.done.add("ple_conv")
    assert turn.complete()


# -- D. the real tree --------------------------------------------------------------

MAMBA_SLOTS = 20
KV_SIZE = 512
NUM_LAYERS = 8
FULL_LAYER_IDS = (3, 7)
MAMBA_LAYER_IDS = [i for i in range(NUM_LAYERS) if i not in FULL_LAYER_IDS]


@pytest.fixture
def group_p(monkeypatch):
    from flliper.srt.mem_cache import unified_radix_cache as urc

    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "P")
    monkeypatch.setenv("FLLIPER_PDFLIP_BIGRAM_ANCHOR_EXACT", "1")
    monkeypatch.setattr(urc, "_PDFLIP_END_ANCHOR", True)
    return monkeypatch


def _fixture(bigram, ple=None):
    server_args = ServerArgs(model_path="dummy", page_size=1)
    server_args._mamba_cache_chunk_size = FLA_CHUNK_SIZE
    server_args.chunked_prefill_size = 16
    set_global_server_args_for_scheduler(server_args)
    with envs.FLLIPER_MAMBA_SSM_DTYPE.override("bfloat16"):
        shape = Mamba2StateShape.create(
            tp_world_size=1, intermediate_size=256, n_groups=1, num_heads=2,
            head_dim=16, state_size=16, conv_kernel=4,
        )
        cache_params = Mamba2CacheParams(shape=shape, layers=MAMBA_LAYER_IDS)
    pool = HybridReqToTokenPool(
        size=10, mamba_size=MAMBA_SLOTS, mamba_spec_state_size=10, max_context_len=256,
        device="cpu", enable_memory_saver=False, cache_params=cache_params,
        mamba_layer_ids=MAMBA_LAYER_IDS, enable_mamba_extra_buffer=False,
        speculative_num_draft_tokens=3, **(ple or {}),
    )
    kv_pool = HybridLinearKVPool(
        size=KV_SIZE, dtype=torch.bfloat16, page_size=1, head_num=2, head_dim=64,
        full_attention_layer_ids=list(FULL_LAYER_IDS), device="cpu",
        enable_memory_saver=False, mamba_pool=pool.mamba_pool,
    )
    allocator = TokenToKVPoolAllocator(
        size=KV_SIZE, dtype=torch.bfloat16, device="cpu", kvcache=kv_pool, need_sort=False,
    )
    params = CacheInitParams(
        req_to_token_pool=pool, token_to_kv_pool_allocator=allocator, page_size=1,
        disable=False, sliding_window_size=None,
        tree_components=(ComponentType.FULL, ComponentType.MAMBA),
        enable_mamba_extra_buffer=False, enable_kv_cache_events=False,
        eviction_policy="lru", is_eagle=bigram,
    )
    cache = UnifiedRadixCache(params=params)
    cache.cache_init_params = params
    return SimpleNamespace(cache=cache, allocator=allocator, pool=pool)


def _key(fx, ids):
    return RadixKey(array("q", ids), is_bigram=fx.cache.is_eagle)


def _req(fx, ids):
    req = Req(rid=RID, origin_input_text="", origin_input_ids=array("q", ids),
              sampling_params=SamplingParams(temperature=0, max_new_tokens=1))
    fx.pool.alloc([req])
    req.output_ids = array("q")
    req.full_untruncated_fill_ids = array("q", ids)
    req.swa_uuid_for_lock = None
    req.extra_key = None
    mr = fx.cache.match_prefix(MatchPrefixParams(key=_key(fx, ids)))
    req.prefix_indices = mr.device_indices.to(torch.int64)
    req.cache_protected_len = len(req.prefix_indices)
    req.last_node = mr.last_device_node
    fx.cache.inc_lock_ref(req.last_node)
    return req


def _state(fx, slot, value):
    fx.pool.mamba_pool.mamba_cache.temporal[:, slot] = value


def _write_kv(fx, req, end):
    start = len(req.prefix_indices)
    if end > start:
        fx.pool.write((req.req_pool_idx, slice(start, end)), fx.allocator.alloc(end - start))
    req.set_extend_range(start, end)


def _plan_turn(fx, req, start, end, marks=("gdn",), need=()):
    """What prepare_for_extend + the forward leave behind: the slot the second
    track wrote (state S_TURN) and the batch's marks."""
    from flliper.srt.pdflip import turn_anchor as ta

    slot = fx.pool.mamba_allocator.alloc(1)
    _state(fx, slot, S_TURN)
    desc = ta.TurnTracks(1)
    desc.add(0, slot, I_LAST, start, end - start)
    desc.need.update(need)
    desc.done.update(marks)
    setattr(req, ta.PENDING_ATTR, (I_LAST, slot, desc, start, end))
    return slot


def _finish(fx, req):
    req.kv_committed_len = N
    req.kv_allocated_len = N
    fx.cache.cache_finished_req(req, is_insert=True)


def _claim(fx, ids):
    claim = Req._compute_max_prefix_len(
        SimpleNamespace(return_logprob=False, logprob_start_len=0), len(ids))
    mr = fx.cache.match_prefix(MatchPrefixParams(key=_key(fx, ids[:claim])))
    node = mr.last_device_node
    slot = node.component_data[ComponentType.MAMBA].value
    state = None if slot is None else float(
        fx.pool.mamba_pool.mamba_cache.temporal[:, slot].float().mean())
    return len(mr.device_indices), state


def _one_forward_prefill(fx, marks=("gdn",), need=()):
    """46-61's shape: the whole prompt in ONE step, END state S_END in the
    request's own slot, the second track at the last message's start."""
    req = _req(fx, PROMPT)
    _write_kv(fx, req, N)
    slot = _plan_turn(fx, req, 0, N, marks=marks, need=need)
    _state(fx, req.mamba_pool_idx, S_END)
    _finish(fx, req)
    return req, slot


@pytest.mark.parametrize("bigram", [False, True], ids=["unigram", "bigram"])
def test_the_next_turn_resumes_at_the_turn_boundary(group_p, bigram):
    fx = _fixture(bigram)
    _one_forward_prefill(fx)
    depth, state = _claim(fx, NEXT)
    # red without the turn insert: the one-step prompt holds only its END
    # anchor, 50-66 resumes from nothing (on the boot: the older chunk anchor)
    assert (depth, state) == (I_LAST, pytest.approx(S_TURN)), (depth, state)


@pytest.mark.parametrize("bigram", [False, True], ids=["unigram", "bigram"])
def test_the_end_anchor_is_unchanged(group_p, bigram):
    """A reader of the whole prompt (identical re-send, a side request past the
    generation prompt) resumes where it did without the turn anchor: at the
    finish insert, with the END state."""
    later = PROMPT + [USER, 1, 2]
    fx = _fixture(bigram)
    _one_forward_prefill(fx)
    armed = _claim(fx, later)
    fx0 = _fixture(bigram)
    req = _req(fx0, PROMPT)
    _write_kv(fx0, req, N)
    _state(fx0, req.mamba_pool_idx, S_END)
    _finish(fx0, req)
    assert armed == _claim(fx0, later)
    assert armed == (N - (1 if bigram else 0), pytest.approx(S_END))


def test_an_inner_chunk_step_also_inserts_below_its_chunk_anchor(group_p):
    fx = _fixture(False)
    req = _req(fx, PROMPT)
    end = I_LAST + 8                                  # an inner chunk end past the boundary
    _write_kv(fx, req, end)
    _plan_turn(fx, req, 0, end)
    _state(fx, req.mamba_pool_idx, S_EARLY)
    fx.cache.cache_unfinished_req(req, chunked=True)
    assert req.cache_protected_len == end, "the chunk's own insert still lands at its end"
    assert _claim(fx, NEXT) == (I_LAST, pytest.approx(S_TURN))
    assert _claim(fx, PROMPT[:end] + [1, 2]) == (end, pytest.approx(S_EARLY))


@pytest.mark.parametrize("marks,need,why", [
    ((), (), "the GDN rows were never added (graph replay, split batch)"),
    (("gdn",), ("ple_conv",), "the PLE conv state was required and not written"),
])
def test_an_unmarked_plan_inserts_nothing_and_frees_its_slot(group_p, marks, need, why):
    fx = _fixture(False)
    free_before = fx.pool.mamba_allocator.available_size()
    req, slot = _one_forward_prefill(fx, marks=marks, need=need)
    depth, state = _claim(fx, NEXT)
    assert state != pytest.approx(S_TURN), why
    # the request's own slot went to the tree as its END anchor; the turn slot back
    assert fx.pool.mamba_allocator.available_size() == free_before - 1, why


#: NF's P stages as the model runner builds their pools: the short-conv ids are
#: filtered to the stage's layers, the n-gram window is not.
PLE_PP0 = dict(short_conv_layer_ids=[MAMBA_LAYER_IDS[0]], short_conv_state_shape=(4, 3),
               ngram_context_len=2, ngram_eos_token_id=IM_END)
PLE_PP1 = dict(short_conv_layer_ids=[], short_conv_state_shape=(4, 3),
               ngram_context_len=2, ngram_eos_token_id=IM_END)


def _planned_one_forward_prefill(fx, marks):
    """The scheduler's own plan (note_step against the stage's real pool),
    then the forward's marks, then the finish -- pdflip-0-6's shape."""
    from flliper.srt.pdflip import turn_anchor as ta

    req = _req(fx, PROMPT)
    _write_kv(fx, req, N)
    batch = SimpleNamespace(reqs=[req], req_to_token_pool=fx.pool)
    desc = ta.note_step(batch=batch, desc=None, req=req, row=0, prefix=0, end=N,
                        track_mask=True, main_track=N - 1, chunk=1, page=1, tok=IM)
    assert desc is not None, "the step holds the boundary: a plan is drawn"
    slot = getattr(req, ta.PENDING_ATTR)[1]
    _state(fx, slot, S_TURN)
    desc.done.update(marks)
    _state(fx, req.mamba_pool_idx, S_END)
    _finish(fx, req)
    return req


@pytest.mark.parametrize("bigram", [False, True], ids=["unigram", "bigram"])
def test_a_pp_stage_without_the_ple_layer_inserts_its_turn_anchor(group_p, bigram):
    """y3u pdflip-0-6 -> pdflip-2-8: PP1/PP2 planned the turn anchor, their forward
    wrote the GDN rows (no PLE layer, no PLE code), the insert refused it as
    'unmarked:ple_ngram' -- so the arena never held the 69696 anchor on every
    rank and the next turn read only to 60224 (9528 tokens re-prefilled).
    Red on the base: the claim stops below the boundary."""
    fx = _fixture(bigram, ple=PLE_PP1)
    assert fx.pool.ngram_pool.enabled and not fx.pool.short_conv_pool.enabled
    _planned_one_forward_prefill(fx, marks=("gdn",))
    assert _claim(fx, NEXT) == (I_LAST, pytest.approx(S_TURN))


def test_the_ple_stage_still_refuses_an_unwritten_ple_state(group_p):
    """PP0 keeps the PLE states: a forward that wrote only the GDN rows (its
    PLE code did not run in Python) is still refused there."""
    fx = _fixture(False, ple=PLE_PP0)
    assert fx.pool.ngram_pool.enabled and fx.pool.short_conv_pool.enabled
    _planned_one_forward_prefill(fx, marks=("gdn",))
    assert _claim(fx, NEXT)[1] != pytest.approx(S_TURN)
    fx = _fixture(False, ple=PLE_PP0)
    _planned_one_forward_prefill(fx, marks=("gdn", "ple_conv", "ple_ngram"))
    assert _claim(fx, NEXT) == (I_LAST, pytest.approx(S_TURN))


def test_an_unarmed_request_is_untouched(group_p):
    fx = _fixture(False)
    req = _req(fx, PROMPT)
    _write_kv(fx, req, N)
    _state(fx, req.mamba_pool_idx, S_END)
    _finish(fx, req)
    depth, state = _claim(fx, NEXT)
    # today's shape (the gap itself): a one-step prompt anchors only at its
    # END, so the next turn resumes BELOW its fork -- here from nothing
    assert depth < I_LAST and state != pytest.approx(S_TURN), (depth, state)


# -- E. the launcher --------------------------------------------------------------


def test_the_launcher_env_is_group_p_only_and_empty_when_off():
    from flliper.srt.pdflip import launcher as L

    assert L.turn_anchor_env(None) == {}
    assert L.turn_anchor_env(IM) == {TOKEN_ENV: str(IM)}
    with pytest.raises(SystemExit):
        L.turn_anchor_env(0)
