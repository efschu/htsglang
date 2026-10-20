"""1304 ANCHOR-LOST on a Form A worker is a false alarm: TP0 alone speaks.

Hermetic (no CUDA). Metal (NF y9nf5, boot 1004_045005, 3e7fe33519): 42 sleeps
x (TP1 + TP2) = 84 ``PDFLIP-ANCHOR-LOST at=flush`` lines, on TP0 none; at the
first one (04:54:46) TP2/TP1 named ``depths=[64, 29568, 34304, 34496, 52544]``
while TP0 printed ``#1470 FLUSH-PUBLISH ... unbacked_left=0``. A Form A
worker holds no GDN/mamba state (``Mamba Cache is allocated ... ssm_state
size: 0.00GB`` on TP1/TP2, 2.06GB on TP0) and no arena: its tree anchors are
byteless bookkeeping whose write-through is refused by construction
(``BACKUP-REFUSED why=anchor_only_no_arena`` 28 / ``mamba_pin`` 18 on TP1).
The group fence unions every rank's list, so the front retracted presence
credit (PRESENCE-ANCHOR-LOST, 42 events, 161 spans) for anchors TP0 had
secured; the depths came back through the L3 index afterwards.

What these cases pin:

* a Form A worker does not run the probe, notes nothing and answers [];
* the Form A host (TP0) still notes what its probe names;
* a classic boot (no plan) notes exactly as before;
* the union the fence builds over a host and two workers is the host's list.
"""

import contextlib
import logging
from types import SimpleNamespace

from flliper.srt import rank_role
from flliper.srt.managers.scheduler import Scheduler

ROLES = ("host", "worker", "worker")


@contextlib.contextmanager
def _as_rank(rank, roles=ROLES):
    prev = (rank_role._INSTALLED_PLAN, rank_role._INSTALLED_RANK)
    rank_role.set_form_a_role_plan(rank_role.RankRolePlan(roles) if roles else None, rank)
    try:
        yield
    finally:
        rank_role._INSTALLED_PLAN, rank_role._INSTALLED_RANK = prev


def _sched(lost):
    calls = []

    def probe():
        calls.append(1)
        return list(lost)

    return SimpleNamespace(tree_cache=SimpleNamespace(pdflip_unbacked_anchors=probe)), calls


LOST = [(29568, None), (34304, None), (52544, None)]


def test_a_form_a_worker_notes_nothing(caplog):
    sched, calls = _sched(LOST)
    with _as_rank(1), caplog.at_level(logging.INFO):
        Scheduler._pdflip_note_lost_anchors(sched)
    assert not calls  # the worker's tree is not asked
    assert Scheduler.pdflip_take_anchors_lost(sched) == []
    assert "PDFLIP-ANCHOR-LOST at=flush n=" not in caplog.text


def test_the_form_a_host_still_notes_its_anchors(caplog):
    sched, calls = _sched(LOST)
    with _as_rank(0), caplog.at_level(logging.WARNING):
        Scheduler._pdflip_note_lost_anchors(sched)
    assert calls
    assert "PDFLIP-ANCHOR-LOST at=flush n=3 depths=[29568, 34304, 52544]" in caplog.text
    assert Scheduler.pdflip_take_anchors_lost(sched) == [29568, 34304, 52544]


def test_a_classic_boot_notes_exactly_as_before(caplog):
    sched, calls = _sched(LOST)
    with _as_rank(0, roles=None), caplog.at_level(logging.WARNING):
        Scheduler._pdflip_note_lost_anchors(sched)
    assert calls
    assert "PDFLIP-ANCHOR-LOST at=flush n=3 depths=[29568, 34304, 52544]" in caplog.text
    assert Scheduler.pdflip_take_anchors_lost(sched) == [29568, 34304, 52544]


def test_the_union_over_host_and_workers_is_the_hosts_list():
    """y9nf5 04:54:46: TP0 backed everything (nothing lost), TP1 and TP2 named
    five depths each. The fence's union (``lost = sorted({... for v in votes})``)
    must stay empty so the front retracts nothing."""
    worker_view = [(64, None), (29568, None), (34304, None), (34496, None), (52544, None)]
    ledgers = []
    for rank, view in ((0, []), (1, worker_view), (2, worker_view)):
        sched, _ = _sched(view)
        with _as_rank(rank):
            Scheduler._pdflip_note_lost_anchors(sched)
        ledgers.append(Scheduler.pdflip_take_anchors_lost(sched))
    assert sorted({d for led in ledgers for d in led}) == []
    # and a real TP0 loss still reaches the front, workers or not
    host, _ = _sched([(34304, "pdflip-0-10")])
    with _as_rank(0):
        Scheduler._pdflip_note_lost_anchors(host)
    assert Scheduler.pdflip_take_anchors_lost(host) == [34304]
