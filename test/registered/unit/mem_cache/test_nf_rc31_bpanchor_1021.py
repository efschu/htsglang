"""BP-ANCHOR (NF rc31 82784e1138, boot ..._1021_202608, D TP0): a finished turn's page-aligned end holds a mamba state.

METAL (D log, TP0, 20:29:54-20:39:34): 22 matches recomputed KV the tree held (``MATCH-CENSUS-DEEP ... branching ==
reached > prefix``); 16 of them stopped at the PAGE-ALIGNED END of the previous turn of the same chat and resumed at
that turn's last mamba state below it. pdflip-2-7: reached=19520 = floor_page(19567 - 1) of pdflip-0-2 (``#59
RESUMABLE rid=pdflip-0-2 depth=19456 seq=19567 last_track=-``, ``mamba checkpoint interval: no tracked position at
finish``), resumed at 19456, recomputed 64. S109R named it ``branch_point_never_anchored`` (rc30: ``store_miss``).

Root: the decode tracks a state only where the sequence crosses a multiple of ``--mamba-track-interval`` (256), while
the tree's pages are 64 tokens. Fix: ``ServerArgs.mamba_track_page_grid`` puts the decode track grid on the tree grid
(lcm(page, chunk) = 64 on NF) where nothing else owns it.

The behaviour test drives the REAL boundary rule (``SchedulerBatchResultProcessor._mamba_check_track_boundary``) with the
interval the REAL ``ServerArgs`` handler leaves, over the metal turn shapes; it is red on 82784e1138 (interval 256:
tails 64..448 recomputed) and green with the fix (tail 0).
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from flliper.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from flliper.srt.server_args import ServerArgs

PAGE = 64

#: (rid, previous turn, its prompt N, its seq at finish (#59 RESUMABLE seq), its resumable depth, reached, recomputed)
#: -- every case of class A in the rc31 D log (TP0 census + RESUMABLE), exact numbers.
METAL_A = [
    ("pdflip-2-7", "pdflip-0-2", 19460, 19567, 19456, 19520, 64),
    ("pdflip-2-8", "pdflip-2-7", 21052, 21158, 20992, 21120, 128),
    ("pdflip-2-14", "pdflip-2-10", 126723, 126840, 126720, 126784, 64),
    ("pdflip-2-16", "pdflip-2-13", 174853, 175016, 174848, 174976, 128),
    ("pdflip-2-21", "pdflip-2-18", 30099, 30207, 30080, 30144, 64),
    ("pdflip-2-40", "pdflip-2-39", 139905, 140028, 139904, 139968, 64),
]


def _args(page=PAGE, **kw):
    args = ServerArgs(model_path="dummy", page_size=page)
    args._mamba_cache_chunk_size = max(FLA_CHUNK_SIZE, page)
    for k, v in kw.items():
        object.__setattr__(args, k, v)
    return args


def _view(page=PAGE, track=256, strategy="extra_buffer", spec="EAGLE", draft=4, disable_radix=False):
    return SimpleNamespace(
        page_size=page, mamba_track_interval=track, mamba_radix_cache_strategy=strategy,
        speculative_algorithm=spec, speculative_num_draft_tokens=draft, disable_radix_cache=disable_radix,
    )


def _resolve(args, view):
    """What the launch leaves: the BP-ANCHOR handler where it exists (82784e1138 has none -> the field stays)."""
    handler = getattr(args, "_handle_mamba_track_page_grid", None)
    if handler is not None:
        handler(view)
    return int(args.mamba_track_interval)


def _last_track(interval, prompt, seq, accepts):
    """The decode of one turn through the REAL ``_mamba_check_track_boundary`` (spec path): the extend ends at
    ``prompt``, each verify round accepts ``a`` drafts (+1 bonus). Returns ``mamba_last_track_seqlen`` at finish."""
    from flliper.srt.managers.scheduler_components.batch_result_processor import (
        SchedulerBatchResultProcessor,
    )

    check = SchedulerBatchResultProcessor._mamba_check_track_boundary
    sa = SimpleNamespace(mamba_track_interval=interval)
    batch = SimpleNamespace(spec_algorithm=SimpleNamespace(is_none=lambda: False))
    last = None
    seqlen = prompt + 1          # the first output token (from the extend) is unfed
    i = 0
    with mock.patch(
        "flliper.srt.managers.scheduler_components.batch_result_processor.get_server_args",
        return_value=sa,
    ):
        while seqlen < seq:
            a = accepts[i % len(accepts)]
            i += 1
            new = min(seq, seqlen + a + 1)
            correct = new - seqlen - 1          # drafts accepted this round (the bonus is the +1)
            seqlen = new
            req = SimpleNamespace(seqlen=seqlen, kv_committed_len=seqlen - 1)
            res = SimpleNamespace(num_correct_drafts_per_req_cpu=[correct])
            hit, at = check(None, req, batch, res, 0)
            if hit:
                last = at
    return last


class TestPageGridRule(unittest.TestCase):
    def test_nf_d_eagle_page64_gets_page_grid(self):
        args = _args()
        self.assertEqual(args.mamba_track_page_grid(_view()), 64)
        self.assertEqual(_resolve(args, _view()), 64)

    def test_nf_p_no_spec_gets_page_grid(self):
        args = _args()
        self.assertEqual(_resolve(args, _view(spec=None, draft=None)), 64)

    def test_idempotent(self):
        args = _args()
        _resolve(args, _view())
        # a later legacy slot sees the overridden value: nothing more happens
        self.assertIsNone(args.mamba_track_page_grid(_view(track=args.mamba_track_interval)))
        self.assertEqual(args.mamba_track_interval, 64)

    def test_explicit_interval_is_kept(self):
        args = _args(mamba_track_interval=128)
        self.assertEqual(_resolve(args, _view(track=128)), 128)

    def test_dflash_keeps_default(self):
        """DFLASH's verify does not rebuild the track targets (l15-endanchor-1006 §0.2): never densified."""
        args = _args(page=1)
        self.assertEqual(_resolve(args, _view(page=1, spec="DFLASH", draft=8)), 256)

    def test_checkpoint_interval_owns_the_grid(self):
        args = _args(mamba_checkpoint_interval=2048)
        self.assertIsNone(args.mamba_track_page_grid(_view()))

    def test_lazy_and_no_buffer_untouched(self):
        args = _args()
        self.assertIsNone(args.mamba_track_page_grid(_view(strategy="extra_buffer_lazy")))
        self.assertIsNone(args.mamba_track_page_grid(_view(strategy="no_buffer")))
        self.assertIsNone(args.mamba_track_page_grid(_view(disable_radix=True)))

    def test_grid_clears_two_overlap_rounds(self):
        """A grid below 2 * (draft + 1) would let a verify and the overlap's extra forward both cross a boundary."""
        args = _args(page=1)
        self.assertEqual(args.mamba_track_page_grid(_view(page=1, draft=40)), 128)
        args = _args(page=1)
        self.assertIsNone(args.mamba_track_page_grid(_view(page=1, draft=200)))   # >= 256: default stays

    def test_validators_accept_the_grid(self):
        """The grid passes the extra_buffer rules: % page, >= draft tokens, >= mamba chunk."""
        g = _args().mamba_track_page_grid(_view())
        self.assertEqual(g % PAGE, 0)
        self.assertGreaterEqual(g, 4)
        self.assertGreaterEqual(g, 64)


class TestMetalTurnsResumeAtTheirPageEnd(unittest.TestCase):
    """Repro of the rc31 class: the next turn reaches the previous turn's page end; the deepest state below it decides
    what is recomputed. Red on 82784e1138 (interval 256), green with BP-ANCHOR (interval 64)."""

    def _tail(self, interval, prompt, seq, depth, reached, accepts):
        last = _last_track(interval, prompt, seq, accepts)
        # the finish keeps KV to the tracked point when there is one (#1469 RETAIN cache_len), else to the page end
        kv_end = last if last is not None else ((seq - 1) // PAGE) * PAGE
        state = max(depth, last or 0)
        reach = min(reached, kv_end)
        return reached - state, reached - reach

    def test_metal_cases_recompute_nothing(self):
        args = _args()
        interval = _resolve(args, _view())
        bad = []
        for rid, prev, n, seq, depth, reached, metal in METAL_A:
            self.assertEqual(((seq - 1) // PAGE) * PAGE, reached, rid)    # the class: reached = prev page end
            for accepts in ([3], [0], [1, 3, 2], [3, 3, 0, 1]):
                tail, lost_kv = self._tail(interval, n, seq, depth, reached, accepts)
                if tail or lost_kv:
                    bad.append((rid, accepts, tail, lost_kv, metal))
        self.assertEqual(bad, [], "interval %d: (rid, accepts, recomputed, kv cut, metal) %s" % (interval, bad[:6]))

    def test_base_interval_reproduces_the_metal_numbers(self):
        """Pins the class itself: at 256, pdflip-0-2 -> 2-7 recomputes exactly the 64 tokens of the metal."""
        rid, prev, n, seq, depth, reached, metal = METAL_A[0]
        tail, _ = self._tail(256, n, seq, depth, reached, [3])
        self.assertEqual(tail, metal)


if __name__ == "__main__":
    unittest.main()
