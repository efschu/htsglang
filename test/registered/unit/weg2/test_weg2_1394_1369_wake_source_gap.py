# SPDX-License-Identifier: Apache-2.0
"""#1369/#1394 (DESK10 Paket B): the completeness refusal and weights_draft's
own wake source.

USER ORDER 2026-09-14, verbatim: "auf der festplatte liegt ein snapshot. das
ist auch ein rueckfall. aber wenn es korrekt implementiert ist braucht es
NIEMALS einen rueckfall....!" -- the host ring (46.40 GiB) is redundancy to a
net (the checkpoint on disk) that is already there. Turning the ring off for
the weights region under an armed, AUTHORITATIVE exchange is safe ONLY if two
gaps are closed first:

1. A tag whose ring backup is off must have SOMETHING that will actually
   write its bytes at wake -- either the exchange (and only if it is
   AUTHORITATIVE: shadow mode compares, it never writes, #1391's own
   lesson generalised) or, for weights_draft specifically, a dedicated
   disk-reload path (#1394: `_weg2_shadow_plan` resolves region_tag from
   the main runner ALONE, on every call, so the draft/MTP shard's
   descriptors are structurally unreachable through the exchange -- proven
   by execution in #1391 round 2, `test_weg2_undrained_lane_refusal_1391.py
   ::test_the_draft_region_is_structurally_unreachable_from_either_call_site`).
2. Silently missing either is worse than the wall it replaces: "still wrong
   weights" reports no symptom, only bad text later. Both gaps must refuse
   BY NAME, at the sleep leg, before the pause that would make the gap real.

THIS FILE covers Paket B's two halves only (weight_updater.py /
weg2_memory_saver.py): the completeness refusal
(`Weg2XchgWakeSourceGapRefused`, W106, verified free by manual grep since
the wcode census tool's own regex caps at 2 digits and cannot see 3-digit
codes -- W100/W101/.../W105/W107 are ALL invisible to it, a sixth instance of
the #1263 blind-spot class) and weights_draft's disk-reload wake source

SAME-NIGHT RENUMBER (2026-09-14, after this file's own tests were already
green): TRAIN2's census against the MERGED tree -- something this file's own
free-number check could not see, since it only ever grepped this branch --
found `Weg2XchgLaneNeverDrainedRefused` (#1391, `weg2_memory_saver.py` /
`weight_updater.py`) colliding with host_ledger.py's pre-existing W100
(`Weg2SleepLegCushionDeficit`, from `8799c7f945`). Renumbered to W108 (also
invisible to the 2-digit census, same blind spot) in the same commit
sequence as this file's own tests; the two `test_weg2_undrained_lane_refusal_1391.py`
assertions that pinned the digit now pin `Weg2XchgLaneNeverDrainedRefused`
by name instead, which is the actual fix against the next such collision
(#1265/#1306 precedent) -- W106 above is unaffected, it was free against
the merged tree from the start.
(`_weg2_xchg_draft_reload_from_disk`). The flag/publication/ledger halves are
DESK9's (Paket A, weight_exchange.py/launcher.py/host_ledger.py) and
model_runner.py/tms_csrc's enable_cpu_backup binding is DESK12's (Paket C) --
this file imports DESK9's contract (`weights_cpu_backup_armed`) but does not
test it.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.scheduler_components import weight_updater as wu
from sglang.srt.managers.weg2_memory_saver import (
    GPU_MEMORY_TYPE_WEIGHTS_DRAFT,
    Weg2WakeRefused,
    Weg2XchgWakeSourceGapRefused,
)
from sglang.srt.weg2 import weight_exchange as wx
from sglang.srt.weg2 import weight_exchange_shadow as sh

Manager = wu.SchedulerWeightUpdaterManager


# ---------------------------------------------------------------------------
# Fakes -- the same shapes test_weg2_xchg_inject_wiring_1342.py drives the
# product with.
# ---------------------------------------------------------------------------


class _FakeServerArgs:
    def __init__(self, model_path="/models/main", draft_path=None):
        self.enable_memory_saver = True
        self.enable_weights_cpu_backup = True
        self.enable_draft_weights_cpu_backup = False
        self.speculative_draft_model_path = draft_path
        self.model_path = model_path
        self.load_format = None


class _FakeRunner:
    def __init__(self):
        self.model = None
        self.model_config = None


class _FakeWorker:
    def __init__(self):
        self.model_runner = _FakeRunner()


class _FakeDraftWorker:
    def __init__(self, *, success=True, message=""):
        self.calls = []
        self._success = success
        self._message = message

    def update_weights_from_disk(self, recv_req):
        self.calls.append(recv_req)
        return self._success, self._message


class _FakeDesc:
    def __init__(self, name, tag="weights_0", src_rank=0, dst_rank=1, nbytes=1024):
        self.param_name = name
        self.tag = tag
        self.kind = "param"
        self.nbytes = int(nbytes)
        self.rows = 1
        self.run_bytes = int(nbytes)
        self.spitch = int(nbytes)
        self.src_rank = int(src_rank)
        self.dst_rank = int(dst_rank)


class _FakePlan:
    def __init__(self, descs):
        self.descs = tuple(descs)


class _FakeTerms:
    total_bytes = 402653184
    lane_slots = 8
    n_lanes = 8
    lanes_concurrent = 0

    def expression(self):
        return "(depth+1) x slot = 3 x 134217728"


def _manager(monkeypatch, *, group="D", rank=0, device=0, draft_worker=None,
            server_args=None):
    monkeypatch.setattr(Manager, "_weg2_server_args",
                        lambda self: server_args or _FakeServerArgs(),
                        raising=True)
    monkeypatch.setattr(Manager, "_weg2_group_name", lambda self: group,
                        raising=True)
    monkeypatch.setattr(Manager, "_weg2_rank", lambda self: rank, raising=True)
    monkeypatch.setattr(Manager, "_weg2_device_index", lambda self: device,
                        raising=True)
    m = Manager(
        tp_worker=_FakeWorker(), draft_worker=draft_worker, tp_cpu_group=None,
        memory_saver_adapter=None, flush_cache=lambda *a, **k: True,
        is_fully_idle=lambda *a, **k: True,
    )
    return m


@pytest.fixture()
def exchange_armed(monkeypatch):
    monkeypatch.setenv(wx.WEIGHT_SOURCE_ENV, wx.WEIGHT_SOURCE_EXCHANGE)
    assert wx.exchange_armed() is True


@pytest.fixture()
def authoritative(monkeypatch, exchange_armed):
    monkeypatch.setenv(wx.INJECT_ENV, wx.INJECT_AUTHORITATIVE)
    assert wx.inject_authoritative() is True


@pytest.fixture()
def shadow_mode(monkeypatch, exchange_armed):
    monkeypatch.setenv(wx.INJECT_ENV, wx.INJECT_SHADOW)
    assert wx.inject_authoritative() is False


@pytest.fixture()
def ring_off(monkeypatch):
    """`auto`: under the CORRECTED contract (a033f2926a, DESK9) this is
    `not (exchange_armed() and inject_authoritative())` -- the ring falls
    ONLY under exchange+authoritative. Tests that need a genuinely-off ring
    under `shadow` (case 1's own scenario) use the explicit `off` mode
    below instead, since `auto` deliberately keeps the ring on there."""
    monkeypatch.setenv(wx.WEIGHTS_CPU_BACKUP_ENV, wx.WEIGHTS_CPU_BACKUP_AUTO)


@pytest.fixture()
def ring_on(monkeypatch):
    monkeypatch.setenv(wx.WEIGHTS_CPU_BACKUP_ENV, wx.WEIGHTS_CPU_BACKUP_ON)


@pytest.fixture()
def ring_off_explicit(monkeypatch):
    """The hard kill switch (`off`), regardless of arm -- the only way to
    reach case 1 (ring off, shadow) now that `auto` itself refuses to turn
    the ring off under shadow (the bug DESK9's own correction fixed)."""
    monkeypatch.setenv(wx.WEIGHTS_CPU_BACKUP_ENV, wx.WEIGHTS_CPU_BACKUP_OFF)


# ===========================================================================
# 1. `_weg2_xchg_wake_source_gap` -- the pure decision, both cases.
# ===========================================================================


def test_ring_on_is_never_a_gap_regardless_of_arm(monkeypatch, ring_on):
    """Default `on`: byte-identical to every pre-#1369 boot -- no refusal,
    no matter what the exchange/plan state is."""
    m = _manager(monkeypatch)
    assert m._weg2_xchg_wake_source_gap("weights_0", cdescs_present=False) is None
    assert m._weg2_xchg_wake_source_gap("weights_0", cdescs_present=True) is None
    assert m._weg2_xchg_wake_source_gap(
        GPU_MEMORY_TYPE_WEIGHTS_DRAFT, cdescs_present=False) is None


def test_ring_off_shadow_is_a_gap_even_with_a_full_plan(monkeypatch,
                                                        ring_off_explicit,
                                                        shadow_mode):
    """CASE 1: shadow only compares, never writes. A perfect plan changes
    nothing -- this must refuse regardless of `cdescs_present`. Uses the
    explicit `off` mode: under the corrected `auto` this combination
    (ring off + shadow) cannot arise from `auto` alone any more -- `auto`
    itself now keeps the ring on there (DESK9's own fix for exactly this
    danger) -- but an operator's explicit `off` can still produce it, and
    this check must catch that regardless of which knob produced it."""
    m = _manager(monkeypatch)
    gap = m._weg2_xchg_wake_source_gap("weights_0", cdescs_present=True)
    assert gap is not None
    assert "shadow" in gap.lower() or "authoritative" in gap.lower()


def test_auto_itself_keeps_the_ring_on_under_shadow_no_gap_possible(
        monkeypatch, ring_off, shadow_mode):
    """THE CORRECTED CONTRACT'S OWN GUARANTEE, checked from Paket B's side:
    under `auto` + exchange + shadow, `weights_cpu_backup_armed()` must
    already answer True (ring stays on), so this check never even reaches
    case 1 for that combination -- confirming Paket A's fix and Paket B's
    check agree rather than each silently compensating for the other."""
    m = _manager(monkeypatch)
    assert m._weg2_xchg_wake_source_gap("weights_0", cdescs_present=False) is None


def test_ring_off_authoritative_empty_plan_is_a_gap(monkeypatch, ring_off,
                                                     authoritative):
    """CASE 2: a real writer is armed, but THIS rank's own plan has nothing
    for this tag -- nobody deposits it."""
    m = _manager(monkeypatch)
    gap = m._weg2_xchg_wake_source_gap("weights_0", cdescs_present=False)
    assert gap is not None
    assert "descriptor" in gap.lower()


def test_ring_off_authoritative_real_plan_is_clean(monkeypatch, ring_off,
                                                   authoritative):
    m = _manager(monkeypatch)
    assert m._weg2_xchg_wake_source_gap("weights_0", cdescs_present=True) is None


def test_weights_draft_exempt_from_case_2_but_not_case_1(monkeypatch,
                                                          ring_off_explicit):
    """weights_draft's empty plan is EXPECTED (case 2 exempt), but it still
    needs an authoritative writer somewhere -- shadow mode is still a gap
    for it too, because case 1 is not tag-specific. Uses the explicit `off`
    mode for the same reason as the test above: `auto` alone no longer
    produces "ring off + shadow" at all."""
    draft = _FakeDraftWorker()
    m = _manager(monkeypatch, draft_worker=draft)

    monkeypatch.setenv(wx.WEIGHT_SOURCE_ENV, wx.WEIGHT_SOURCE_EXCHANGE)
    monkeypatch.setenv(wx.INJECT_ENV, wx.INJECT_SHADOW)
    gap = m._weg2_xchg_wake_source_gap(GPU_MEMORY_TYPE_WEIGHTS_DRAFT,
                                       cdescs_present=False)
    assert gap is not None, "shadow mode must still be a gap for weights_draft"

    monkeypatch.setenv(wx.INJECT_ENV, wx.INJECT_AUTHORITATIVE)
    gap = m._weg2_xchg_wake_source_gap(GPU_MEMORY_TYPE_WEIGHTS_DRAFT,
                                       cdescs_present=False)
    assert gap is None, "an empty plan for weights_draft is the EXPECTED shape"


def test_weights_draft_with_no_draft_worker_is_not_a_gap(monkeypatch, ring_off,
                                                          authoritative):
    """No draft shard in this process -- nothing to cover, nothing to gap."""
    m = _manager(monkeypatch, draft_worker=None)
    assert m._weg2_xchg_wake_source_gap(
        GPU_MEMORY_TYPE_WEIGHTS_DRAFT, cdescs_present=False) is None


def test_an_unreadable_contract_answers_no_gap_not_a_refusal(monkeypatch):
    """`weights_cpu_backup_armed` raising (contract not on this tree, or an
    unparseable env) must not manufacture a refusal from an absence -- the
    same rule `_weg2_xchg_undrained_lanes` follows for an unopened sems."""
    m = _manager(monkeypatch)

    def _raise():
        raise RuntimeError("no contract here")

    monkeypatch.setattr(wx, "weights_cpu_backup_armed", lambda **k: _raise())
    assert m._weg2_xchg_wake_source_gap("weights_0", cdescs_present=False) is None


# ===========================================================================
# 2. WIRED -- the real production callsite. RED-FIRST reproduction of the
# defect this refusal exists to prevent: a tag with no source, silently
# accepted by the deposit wrapper before this round.
# ===========================================================================


def test_the_deposit_leg_refuses_by_name_when_the_plan_is_empty_and_the_ring_is_off(
        monkeypatch, ring_off, authoritative):
    """Drives the REAL `_weg2_xchg_deposit_before_sleep` (the production
    call the sleep leg actually makes), not a hand-built call to the gap
    check alone -- the #1342 lesson: an emitter/guard with no wired caller
    is not an instrument."""
    monkeypatch.setenv("SGLANG_WEG2_XCHG_REGION_BOOT", "1394boot")
    from sglang.srt.weg2 import weight_exchange_region as xr

    monkeypatch.setenv(xr.ENV_REGION_BOOT, "1394boot")
    m = _manager(monkeypatch, group="D", rank=0)
    monkeypatch.setattr(
        Manager, "_weg2_shadow_plan",
        lambda self, hook, g, r, agreed=None, require_agreement=None:
            (_FakePlan([_FakeDesc("a.w", tag="weights_9", src_rank=0)]), ""),
        raising=True)
    monkeypatch.setattr(Manager, "_weg2_xchg_device_ops", lambda self: object(),
                        raising=True)

    from sglang.srt.weg2 import xchg_bounce as xb

    monkeypatch.setattr(xb, "read_published_terms", lambda: _FakeTerms(),
                        raising=False)

    with pytest.raises(Weg2XchgWakeSourceGapRefused) as exc:
        m._weg2_xchg_deposit_before_sleep(flip_index=0, tag="weights_0")
    msg = str(exc.value)
    assert "W106" in msg
    assert "weights_0" in msg
    # NUTZER-ORDER 2026-09-14 (PRONTO): "jeder pausierte Tag ... muss den
    # Boot mit NAMEN UND BYTE-ZAHL toeten" -- the name was already there,
    # this checks the count joined it. No fake adapter is wired in
    # `_manager`, so `_weg2_tag_bytes` genuinely cannot answer here --
    # exactly the case NULL-NUR-BEI-ERREICHTEM-EMITTER exists for: the
    # message must say so by name, never print a bare "0" that reads as a
    # real, measured zero-byte tag.
    assert "expected_bytes=unmeasurable" in msg


class _FakeAdapterWithTagBytes:
    """A `memory_saver_adapter` double whose `tag_bytes` genuinely answers,
    for the byte-count-in-the-refusal test below -- the REAL instrument
    (`_weg2_tag_bytes`) is driven, not reimplemented."""

    def __init__(self, mapping):
        self._mapping = dict(mapping)

    def tag_bytes(self, tag):
        return self._mapping.get(str(tag), 0)


def test_the_refusal_names_the_real_measured_byte_count(
        monkeypatch, ring_off, authoritative):
    """The other half of the same order: when `_weg2_tag_bytes` CAN answer,
    the refusal carries the actual number, not the unmeasurable text --
    driven through `self._weg2_tag_bytes(tag)`, the SAME instrument
    `resume_memory_occupation`'s own credit-publish call already reads for
    this tag (weight_updater.py's `tag_bytes` dict), not a new derivation."""
    from sglang.srt.weg2 import weight_exchange_region as xr

    monkeypatch.setenv(xr.ENV_REGION_BOOT, "1394boot-bytes")
    m = _manager(monkeypatch, group="D", rank=0)
    m.memory_saver_adapter = _FakeAdapterWithTagBytes({"weights_0": 3057647616})
    monkeypatch.setattr(
        Manager, "_weg2_shadow_plan",
        lambda self, hook, g, r, agreed=None, require_agreement=None:
            (_FakePlan([_FakeDesc("a.w", tag="weights_9", src_rank=0)]), ""),
        raising=True)
    monkeypatch.setattr(Manager, "_weg2_xchg_device_ops", lambda self: object(),
                        raising=True)

    from sglang.srt.weg2 import xchg_bounce as xb

    monkeypatch.setattr(xb, "read_published_terms", lambda: _FakeTerms(),
                        raising=False)

    with pytest.raises(Weg2XchgWakeSourceGapRefused) as exc:
        m._weg2_xchg_deposit_before_sleep(flip_index=0, tag="weights_0")
    msg = str(exc.value)
    assert "expected_bytes=3057647616" in msg
    assert "unmeasurable" not in msg


def test_the_deposit_leg_stays_a_no_op_when_the_ring_still_covers_it(
        monkeypatch, ring_on):
    """Regression guard: the SAME empty plan, but the ring is on -- must
    stay the pre-#1369 silent no-op, never a refusal."""
    from sglang.srt.weg2 import weight_exchange_region as xr

    monkeypatch.setenv(xr.ENV_REGION_BOOT, "1394boot2")
    monkeypatch.setenv(wx.WEIGHT_SOURCE_ENV, wx.WEIGHT_SOURCE_EXCHANGE)
    m = _manager(monkeypatch, group="D", rank=0)
    monkeypatch.setattr(
        Manager, "_weg2_shadow_plan",
        lambda self, hook, g, r, agreed=None, require_agreement=None:
            (_FakePlan([_FakeDesc("a.w", tag="weights_9", src_rank=0)]), ""),
        raising=True)
    monkeypatch.setattr(Manager, "_weg2_xchg_device_ops", lambda self: object(),
                        raising=True)
    from sglang.srt.weg2 import xchg_bounce as xb

    monkeypatch.setattr(xb, "read_published_terms", lambda: _FakeTerms(),
                        raising=False)

    calls = {}
    monkeypatch.setattr(Manager, "_weg2_xchg_bounce_leg",
                        lambda self, **kw: calls.setdefault("kw", kw),
                        raising=True)
    m._weg2_xchg_deposit_before_sleep(flip_index=0, tag="weights_0")
    # An empty-plan tag returns BEFORE calling the leg at all (the pre-#1369
    # "pieces=0, no-op" shape) -- the leg is simply never reached.
    assert "kw" not in calls, calls


# ===========================================================================
# 3. `_weg2_xchg_draft_reload_from_disk` -- weights_draft's own wake source.
# ===========================================================================


def test_draft_reload_is_a_no_op_with_no_draft_worker(monkeypatch, ring_off,
                                                       authoritative):
    m = _manager(monkeypatch, draft_worker=None)
    assert m._weg2_xchg_draft_reload_from_disk() is False


def test_draft_reload_is_a_no_op_when_the_ring_still_covers_it(monkeypatch,
                                                               ring_on):
    draft = _FakeDraftWorker()
    m = _manager(monkeypatch, draft_worker=draft)
    assert m._weg2_xchg_draft_reload_from_disk() is False
    assert draft.calls == []


def test_draft_reload_calls_the_draft_workers_own_update_from_disk(
        monkeypatch, ring_off, authoritative):
    draft = _FakeDraftWorker(success=True)
    args = _FakeServerArgs(model_path="/models/main",
                           draft_path="/models/draft")
    m = _manager(monkeypatch, draft_worker=draft, server_args=args)
    monkeypatch.setattr(Manager, "_weg2_pcie_lock",
                        lambda self, *a, **k: _NullCtx(), raising=True)
    import sglang.srt.managers.weg2_memory_saver as ms

    monkeypatch.setattr(ms, "weights_region",
                        lambda *a, **k: _NullCtx(), raising=True)

    assert m._weg2_xchg_draft_reload_from_disk() is True
    assert len(draft.calls) == 1
    assert draft.calls[0].model_path == "/models/draft"


def test_draft_reload_refuses_by_name_on_failure(monkeypatch, ring_off,
                                                  authoritative):
    draft = _FakeDraftWorker(success=False, message="disk unreachable")
    m = _manager(monkeypatch, draft_worker=draft)
    monkeypatch.setattr(Manager, "_weg2_pcie_lock",
                        lambda self, *a, **k: _NullCtx(), raising=True)
    import sglang.srt.managers.weg2_memory_saver as ms

    monkeypatch.setattr(ms, "weights_region",
                        lambda *a, **k: _NullCtx(), raising=True)

    with pytest.raises(Weg2WakeRefused) as exc:
        m._weg2_xchg_draft_reload_from_disk()
    assert "disk unreachable" in str(exc.value)


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ===========================================================================
# 4. THE DANGER-DIRECTION MUTANT (operator order): a tag without a wake
# source must never pass silently -- still-wrong weights are worse than the
# refusal.
# ===========================================================================


def test_M_removing_the_gap_check_lets_a_sourceless_tag_pass_silently(
        monkeypatch, ring_off, authoritative):
    """THE MUTANT: patch `_weg2_xchg_wake_source_gap` to always answer
    `None` (as if the check did not exist) and show the deposit leg then
    proceeds with an EMPTY plan for a tag that has no other source --
    exactly the still-wrong-weights shape this refusal exists to prevent.
    This test passes ONLY because the assertion below is written against
    the MUTATED (unsafe) behaviour on purpose, to prove the real check (in
    the test above) is what stands between this and a silent no-op."""
    from sglang.srt.weg2 import weight_exchange_region as xr

    monkeypatch.setenv(xr.ENV_REGION_BOOT, "1394boot3")
    m = _manager(monkeypatch, group="D", rank=0)
    monkeypatch.setattr(
        Manager, "_weg2_shadow_plan",
        lambda self, hook, g, r, agreed=None, require_agreement=None:
            (_FakePlan([_FakeDesc("a.w", tag="weights_9", src_rank=0)]), ""),
        raising=True)
    monkeypatch.setattr(Manager, "_weg2_xchg_device_ops", lambda self: object(),
                        raising=True)
    from sglang.srt.weg2 import xchg_bounce as xb

    monkeypatch.setattr(xb, "read_published_terms", lambda: _FakeTerms(),
                        raising=False)
    # THE MUTATION:
    monkeypatch.setattr(Manager, "_weg2_xchg_wake_source_gap",
                        lambda self, tag, *, cdescs_present: None, raising=True)

    calls = {}
    monkeypatch.setattr(Manager, "_weg2_xchg_bounce_leg",
                        lambda self, **kw: calls.setdefault("kw", kw),
                        raising=True)
    # With the guard mutated away, this call that SHOULD refuse (W106,
    # proven above) instead reaches the pre-#1369 silent "pieces=0" no-op --
    # no exception, no deposit, no evidence beyond a log line nobody greps.
    m._weg2_xchg_deposit_before_sleep(flip_index=0, tag="weights_0")
    assert "kw" not in calls, (
        "the mutant should have reached the silent no-op the real guard "
        "forbids (bounce_leg never even called) -- if this assertion "
        "fails, re-check the mutant is actually wired to the call path")


# ===========================================================================
# 5. DESK12's find (2026-09-14, relayed by the coordinator): `main_carried`
# in `_weg2_wake_weight_carrier` used to read the RAW
# `server_args.enable_weights_cpu_backup` bit -- always True, the launcher
# sets it unconditionally (launcher.py:2702) -- instead of the PREDICATE
# `weights_cpu_backup_armed()`. This is an INDEPENDENT defect from W106
# above, not a second instance of it: W106 refuses at the SLEEP leg, and
# ONLY for a tag whose exchange is armed at all
# (`_weg2_xchg_deposit_before_sleep` returns immediately at :1080-1081 when
# `not wx.exchange_armed()` -- ring mode was never W106's half to guard).
# `--weg2-weights-cpu-backup off` under plain `ring` mode (no exchange at
# all) is therefore NOT caught by W106 -- and the flag-vs-predicate bug
# below is exactly what would have made the wake pick `CARRIER_TMS_BACKUP`
# ("the TMS restore already carried the bytes") for a tag nothing backed
# up, silently, at the WAKE side. The fix is a different mechanism
# (correct wake ROUTING, not a refusal): reading the predicate makes the
# same wake fall through to `CARRIER_DISK` instead -- the checkpoint on
# disk, the user's own "das ist das Netz" -- which is a genuine, always-
# available, always-correct source, so no new W-code is needed here: there
# is nothing left to refuse once the routing is honest.
# ===========================================================================


def test_main_carried_reads_the_predicate_not_the_raw_flag(
        monkeypatch, exchange_armed, authoritative, ring_off_explicit):
    """The exact divergence DESK12 named: flag SET (True, as the launcher
    always sets it) + predicate FALSE (explicit `off`) must make
    `main_carried` FALSE too, not True. Proven here from the OUTSIDE: force
    `exchange_armed()`/`inject_authoritative()` to read False downstream of
    the try-block's own early return by using `ring` weight-source instead,
    so the only thing this call can be deciding is the `main_carried`
    fallback at the end of `_weg2_wake_weight_carrier`."""
    monkeypatch.delenv(wx.WEIGHT_SOURCE_ENV, raising=False)  # ring, not exchange
    assert wx.exchange_armed() is False
    args = _FakeServerArgs()
    assert args.enable_weights_cpu_backup is True  # the raw bit: always set
    m = _manager(monkeypatch, server_args=args)
    carrier = m._weg2_wake_weight_carrier()
    assert carrier == Manager.CARRIER_DISK, (
        f"expected the honest disk fallback for a tag the explicit `off` "
        f"kill switch leaves genuinely unbacked, got {carrier!r} -- a "
        f"`tms-backup` answer here means main_carried read the raw flag "
        f"again and this rank would silently serve undefined VRAM")


def test_main_carried_still_true_when_ring_genuinely_armed(
        monkeypatch, ring_on):
    """Sanity companion: the fix must not flip the ANSWER for the ordinary
    case, only its SOURCE. `on` (or plain unarmed-exchange `auto`) still
    carries via the ring."""
    monkeypatch.delenv(wx.WEIGHT_SOURCE_ENV, raising=False)
    args = _FakeServerArgs()
    m = _manager(monkeypatch, server_args=args)
    assert m._weg2_wake_weight_carrier() == Manager.CARRIER_TMS_BACKUP


def test_M_main_carried_reading_the_raw_flag_again_hides_the_gap(
        monkeypatch, exchange_armed, authoritative, ring_off_explicit):
    """THE DANGER-DIRECTION MUTANT (coordinator order): revert the read to
    the raw flag DESK12 found, and show the wake then silently picks
    `tms-backup` for a tag the explicit `off` kill switch left genuinely
    unbacked -- exactly the still-wrong-weights shape neither W106 nor this
    fix may let through. This test passes ONLY because it asserts the
    UNSAFE, mutated answer on purpose, to prove the real fix (the test
    above) is what stands between this and the silent misroute."""
    monkeypatch.delenv(wx.WEIGHT_SOURCE_ENV, raising=False)
    args = _FakeServerArgs()
    m = _manager(monkeypatch, server_args=args)
    # THE MUTATION: back to the pre-fix read.
    monkeypatch.setattr(
        wx, "weights_cpu_backup_armed",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("mutated out")),
        raising=True)
    # With the predicate call itself made to raise, the code's own
    # exception fallback (documented alongside the fix) reverts to the raw
    # flag -- reproducing DESK12's exact defect shape for this assertion.
    carrier = m._weg2_wake_weight_carrier()
    assert carrier == Manager.CARRIER_TMS_BACKUP, (
        "the mutant should reproduce DESK12's silent misroute (tms-backup "
        "chosen for a tag nothing backed up) -- if this assertion fails, "
        "the mutant is not wired to the code path it claims to be")


if __name__ == "__main__":
    import unittest

    unittest.main()
