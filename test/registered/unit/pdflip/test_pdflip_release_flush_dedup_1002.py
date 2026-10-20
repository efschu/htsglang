"""DP-NACHLAUF 02.10.: the kv release's flush skips the second tree/pool reset
when the quiesce /flush_cache reset this rank moments ago and nothing ran.

N5u (fcf0366fdb), every P>D: rpc_flush[passed alloc_clear 37-66] followed by
release_flush[passed alloc_clear 30-52] -- the same reset twice. Pinned (red
before): only the release flush (zero_kv False) may skip; it skips only with
no forward since the last reset, empty queues and the tree as the reset left
it (the #243 hand-off chain may remain); any change, the switch off, or a
helper error resets as before; the flush branch marks the same segments.
"""
from __future__ import annotations

import inspect
import os
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers import scheduler as S  # noqa: E402


class _Tree:
    def __init__(self, ev=0, pr=0, kids=0):
        self.ev, self.pr = ev, pr
        self.root_node = SimpleNamespace(children={i: i for i in range(kids)})

    def evictable_size(self):
        return self.ev

    def protected_size(self):
        return self.pr


def _sched(**kw):
    s = SimpleNamespace(forward_ct=41, waiting_queue=[], running_batch=SimpleNamespace(is_empty=lambda: True),
                        tree_cache=_Tree(ev=98875, kids=1))
    s.__dict__.update(kw)
    return s


def test_release_skips_only_after_an_untouched_reset(monkeypatch):
    monkeypatch.delenv(S._PDFLIP_RELEASE_DEDUP_ENV, raising=False)
    s = _sched()
    assert not S._pdflip_release_reset_redundant(s, False)     # no reset noted yet
    S._pdflip_note_reset(s)
    assert S._pdflip_release_reset_redundant(s, False)          # release, untouched
    assert not S._pdflip_release_reset_redundant(s, None)       # the /flush_cache RPC never skips
    assert not S._pdflip_release_reset_redundant(s, True)


def test_any_change_resets(monkeypatch):
    monkeypatch.delenv(S._PDFLIP_RELEASE_DEDUP_ENV, raising=False)
    for change in (lambda s: setattr(s, "forward_ct", 42),
                   lambda s: s.waiting_queue.append(object()),
                   lambda s: setattr(s, "running_batch", SimpleNamespace(is_empty=lambda: False)),
                   lambda s: setattr(s.tree_cache, "ev", 5),
                   lambda s: s.tree_cache.root_node.children.update({9: 9})):
        s = _sched()
        S._pdflip_note_reset(s)
        change(s)
        assert not S._pdflip_release_reset_redundant(s, False)


def test_switch_off_and_errors_reset(monkeypatch):
    s = _sched()
    S._pdflip_note_reset(s)
    monkeypatch.setenv(S._PDFLIP_RELEASE_DEDUP_ENV, "0")
    assert not S._pdflip_release_reset_redundant(s, False)
    monkeypatch.delenv(S._PDFLIP_RELEASE_DEDUP_ENV)
    s.tree_cache = SimpleNamespace()                           # no size helpers
    assert not S._pdflip_release_reset_redundant(s, False)


def test_flush_branch_wiring():
    src = inspect.getsource(S.Scheduler.flush_cache)
    i_q = src.index("if _pdflip_release_reset_redundant(self, zero_kv):")
    i_reset = src.index("self.tree_cache.reset()", i_q)
    i_note = src.index("_pdflip_note_reset(self)", i_reset)
    assert i_q < i_reset < i_note
    # NF has no PDFLIP-SLEEP-SUB segment clock (_fsub, 27B PDFLIP-L): the skip
    # branch only has to leave the tree and both pools untouched
    seg = src[i_q:i_reset]
    for m in (".reset()", ".clear()"):
        assert m not in seg
