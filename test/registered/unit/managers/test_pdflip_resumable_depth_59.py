"""#59 (D side): a finished leg 2 names the depth it can be RESUMED from.

The front credits a D leg by its response's ``cached_tokens`` (what D found on
ARRIVAL). A hybrid GDN/Mamba D resumes the next turn only from a recurrent
anchor, and a finish anchors on the track grid: an off-grid finish keeps KV
above its last anchor that no admission resumes from (RETAIN off-grid). The
front has no number for that -- it prices the next turn's D prefill low.

``pdflip_resumable_depth`` rides the finishing output: time_stats ->
meta_info (/generate) -> usage (OpenAI chat/completions, Anthropic messages).
Group-uniform: the Form A host decides (workers follow, H98), one rank is
trivially uniform, a classic TP group MIN-reduces, anything else sends none.

RED on the base: no module, no time_stats field, no meta_info/usage field.
"""

from __future__ import annotations

import contextlib
import inspect
import types
import unittest
from array import array

import torch

from flliper.srt import rank_role
from flliper.srt.mem_cache.base_prefix_cache import MatchResult

ROLES = ("host", "worker", "worker")
GRID = 25600  # the finish's last track anchor
PROMPT = 25611
OUTPUT = 300  # the finish lands off-grid at 25911


def _mod():
    from flliper.srt.managers import pdflip_resumable_depth as m

    return m


@contextlib.contextmanager
def _as_rank(rank, roles=ROLES):
    prev = (rank_role._INSTALLED_PLAN, rank_role._INSTALLED_RANK)
    plan = None if roles is None else rank_role.RankRolePlan(tuple(roles))
    rank_role.set_form_a_role_plan(plan, rank)
    try:
        yield
    finally:
        rank_role._INSTALLED_PLAN, rank_role._INSTALLED_RANK = prev


def _node(name, anchor=True):
    host = torch.tensor([3]) if anchor else None
    return types.SimpleNamespace(
        name=name,
        component_data=[None, None, types.SimpleNamespace(value=None, host_value=host)],
    )


class _FinishedTree:
    """The tree after a finish's insert: the key reaches the whole sequence,
    the recurrent state only the deepest anchor at or below it (a match ends
    on the deepest mamba-valued node). ``anchor=False``: that node's state is
    gone (no device, no host value -- #928 (a))."""

    def __init__(self, anchor_depth=GRID, anchor=True):
        self.anchor_depth, self.anchor = anchor_depth, anchor
        self.root_node = _node("root", anchor=False)
        self.cache_controller = None
        self.is_eagle = False
        self.supports_mamba = lambda: True
        self.swa_reprefill_tail_tokens = lambda: 0
        self.limits = []

    def match_prefix(self, params):
        n = len(params.key)
        self.limits.append(n)
        d = min(self.anchor_depth, n)
        node = _node(f"n{d}", anchor=self.anchor)
        return MatchResult(
            device_indices=torch.arange(d, dtype=torch.int64),
            last_device_node=node,
            last_host_node=node,
            best_match_node=node,
        )


def _ts():
    from flliper.srt.observability.req_time_stats import SchedulerReqTimeStats

    return SchedulerReqTimeStats()


def _req(rid="pdflip-59-1", n=PROMPT, out=OUTPUT, finished=True, finished_output=False):
    return types.SimpleNamespace(
        rid=rid,
        origin_input_ids=array("q", range(n)),
        output_ids=list(range(n, n + out)),
        extra_key=None,
        positional_embed_overrides=None,
        cached_tokens=GRID,
        time_stats=_ts(),
        finished=lambda: finished,
        finished_output=finished_output,
        _compute_max_prefix_len=lambda k: max(k - 1, 0),
    )


def _ps(tp=1, dp=1, pp=1, rank=0):
    return types.SimpleNamespace(tp_size=tp, attn_dp_size=dp, pp_size=pp, attn_tp_rank=rank)


class TestProbe(unittest.TestCase):
    def test_off_grid_finish_resumes_at_its_anchor(self):
        m = _mod()
        with _as_rank(0, roles=None):
            self.assertEqual(m.local_depth(_FinishedTree(GRID), _req()), GRID)

    def test_next_turn_view_takes_the_whole_sequence(self):
        """A finish ON the grid resumes at all of it: this turn's ``len - 1``
        admission limit is not the next turn's."""
        m = _mod()
        tree = _FinishedTree(anchor_depth=GRID)
        with _as_rank(0, roles=None):
            depth = m.local_depth(tree, _req(n=GRID - 100, out=100))
        self.assertEqual(depth, GRID)
        self.assertEqual(tree.limits[0], GRID)

    def test_gone_anchor_is_a_measured_zero(self):
        m = _mod()
        with _as_rank(0, roles=None):
            self.assertEqual(m.local_depth(_FinishedTree(GRID, anchor=False), _req()), 0)


class TestGroupUniform(unittest.TestCase):
    def test_form_a_host_decides(self):
        m = _mod()
        reqs = [_req()]
        with _as_rank(0):
            mode = m.stamp_finished(_FinishedTree(), reqs, _ps(tp=3, dp=3))
        self.assertEqual(mode, m.MODE_HOST)
        self.assertEqual(reqs[0].time_stats.pdflip_resumable_depth, GRID)

    def test_form_a_worker_stamps_nothing(self):
        m = _mod()
        reqs = [_req()]

        def _no_reduce(values):
            raise AssertionError("a Form A worker never enters a reduce")

        with _as_rank(1):
            mode = m.stamp_finished(_FinishedTree(), reqs, _ps(tp=3, dp=3, rank=0),
                                    reduce_min=_no_reduce)
        self.assertEqual(mode, m.MODE_FOLLOW)
        self.assertEqual(reqs[0].time_stats.pdflip_resumable_depth, m.UNSET)

    def test_classic_tp_takes_the_group_min(self):
        m = _mod()
        reqs = [_req("a"), _req("b")]
        seen = []

        def _reduce(values):
            seen.append(list(values))
            return [min(v, 16384) for v in values]  # a peer realizes 16384 only

        with _as_rank(0, roles=None):
            mode = m.stamp_finished(_FinishedTree(), reqs, _ps(tp=2), reduce_min=_reduce)
        self.assertEqual(mode, m.MODE_MIN)
        self.assertEqual(seen, [[GRID, GRID]])
        self.assertEqual([r.time_stats.pdflip_resumable_depth for r in reqs], [16384, 16384])

    def test_dp_attention_without_form_a_sends_nothing(self):
        m = _mod()
        reqs = [_req()]
        with _as_rank(0, roles=None):
            mode = m.stamp_finished(_FinishedTree(), reqs, _ps(tp=2, dp=2))
        self.assertEqual(mode, m.MODE_NONE)
        self.assertEqual(reqs[0].time_stats.pdflip_resumable_depth, m.UNSET)

    def test_empty_list_enters_no_reduce(self):
        m = _mod()

        def _no_reduce(values):
            raise AssertionError("no finishing request -> no collective")

        with _as_rank(0, roles=None):
            self.assertIsNone(m.stamp_finished(_FinishedTree(), [], _ps(tp=2),
                                               reduce_min=_no_reduce))


class TestStreamHook(unittest.TestCase):
    def test_only_finishing_outputs_are_stamped(self):
        m = _mod()
        done, running, twice = _req("d"), _req("r", finished=False), _req("t", finished_output=True)
        skip = _req("s")
        with _as_rank(0, roles=None):
            m.stamp_stream(_FinishedTree(), [done, running, twice, skip], skip, _ps())
        self.assertEqual(done.time_stats.pdflip_resumable_depth, GRID)
        for r in (running, twice, skip):
            self.assertEqual(r.time_stats.pdflip_resumable_depth, m.UNSET)

    def test_streamer_calls_the_hook_on_a_pdflip_d_group(self):
        from flliper.srt.managers.scheduler_components import output_streamer

        src = inspect.getsource(output_streamer.SchedulerOutputStreamer._stream_output_generation)
        self.assertIn("pdflip_resumable_depth.stamp_stream", src)


class TestCarrier(unittest.TestCase):
    def test_getstate_carries_the_depth_zero_included(self):
        ts = _ts()
        self.assertNotIn("pdflip_resumable_depth", ts.__getstate__())
        ts.pdflip_resumable_depth = 0
        self.assertEqual(ts.__getstate__().get("pdflip_resumable_depth"), 0)
        ts.pdflip_resumable_depth = GRID
        self.assertEqual(ts.__getstate__().get("pdflip_resumable_depth"), GRID)

    def test_tokenizer_puts_it_into_meta_info(self):
        m = _mod()
        from flliper.srt.managers import tokenizer_manager

        ts = _ts()
        ts.pdflip_resumable_depth = 0
        self.assertEqual(m.meta_value(ts), 0)
        self.assertIsNone(m.meta_value(_ts()))
        src = inspect.getsource(tokenizer_manager)
        self.assertIn("pdflip_resumable_depth.meta_value", src)

    def test_meta_infos_take_the_min_and_skip_absent(self):
        m = _mod()
        self.assertIsNone(m.from_meta_infos([{"cached_tokens": 4}, None]))
        self.assertEqual(m.from_meta_infos([{"pdflip_resumable_depth": 0}]), 0)
        self.assertEqual(
            m.from_meta_infos([{"pdflip_resumable_depth": GRID}, {"pdflip_resumable_depth": 16384}, {}]),
            16384,
        )

    def test_openai_sglext_carries_it_zero_included(self):
        from flliper.srt.entrypoints.openai.protocol import SglExt

        self.assertEqual(SglExt(pdflip_resumable_depth=0).model_dump(), {"pdflip_resumable_depth": 0})
        self.assertEqual(SglExt().model_dump(), {})

    def test_chat_and_completion_responses_carry_it(self):
        """Body and stream: the 27B front reads ``sglext`` (d_resumable_depth,
        and the stream tail's last data line that names the field)."""
        from flliper.srt.entrypoints.openai import serving_chat, serving_completions

        for mod in (serving_chat, serving_completions):
            src = inspect.getsource(mod)
            self.assertIn("pdflip_resumable_depth=", src, mod.__name__)
            self.assertIn("resumable_depths[index]", src, mod.__name__)

    def test_anthropic_message_and_delta_carry_it(self):
        from flliper.srt.entrypoints.anthropic.protocol import (
            AnthropicMessageEndDelta,
            AnthropicMessagesResponse,
            AnthropicSglExt,
            AnthropicUsage,
            MessageDeltaEvent,
            TextBlock,
        )

        ev = MessageDeltaEvent(
            delta=AnthropicMessageEndDelta(stop_reason="end_turn"),
            usage=AnthropicUsage(output_tokens=2),
            sglext=AnthropicSglExt(pdflip_resumable_depth=GRID),
        )
        self.assertEqual(
            ev.model_dump(exclude_none=True)["sglext"], {"pdflip_resumable_depth": GRID}
        )
        plain = MessageDeltaEvent(
            delta=AnthropicMessageEndDelta(stop_reason="end_turn"),
            usage=AnthropicUsage(output_tokens=2),
        )
        self.assertNotIn("sglext", plain.model_dump(exclude_none=True))
        msg = AnthropicMessagesResponse(
            content=[TextBlock(text="x")], model="m",
            sglext=AnthropicSglExt(pdflip_resumable_depth=0),
        )
        self.assertEqual(msg.model_dump(exclude_none=True)["sglext"], {"pdflip_resumable_depth": 0})

    def test_anthropic_serving_forwards_it(self):
        from flliper.srt.entrypoints.anthropic import serving

        src = inspect.getsource(serving)
        self.assertIn("AnthropicSglExt(pdflip_resumable_depth=", src)


if __name__ == "__main__":
    unittest.main()
