#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#1330 B4n -- EXECUTION SMOKE OF THE PRODUCT CALL SITE, both hooks.

Desk-written-never-executed is a defect class this campaign has paid for four
times, so this drives the REAL method bodies of
``SchedulerWeightUpdaterManager`` -- ``_weg2_shadow_plan`` and the two address
books -- over SIX REAL MANIFEST FILES on disk, with no model runner, no GPU and
no boot.

WHY A STUB ``self`` AND NOT A CONSTRUCTED MANAGER: the manager is a
``slots=True`` dataclass that cannot exist without a model runner, a process
group and a device -- which is the tree's own argument for why the derivation
lives in free functions.  The stub therefore BORROWS the real unbound methods
rather than re-implementing them, so what runs here is byte-for-byte the code
the flip leg runs.

WHAT IT ASSERTS, and both are the things weg2xsn20 could not produce:
  * ``WEG2-XCHG-POINTER-PROFILE ... src_resolved=N/N dst_resolved=N/N`` on BOTH
    hooks -- the source hook resolves what it must SUPPLY, the destination hook
    what it must RECEIVE, each from its own manifest, neither from a peer
    pointer;
  * the descriptors survive ``refuse_if_plan_exceeds_slot`` -- the site where
    weg2xsn9 died -- graded against THIS CHECKPOINT'S REAL CLASS WIDTHS
    (widest layer 756,323,776 B = 721 MiB, ``lm_head`` 2425 MiB,
    ``embed_tokens`` 1212 MiB, from ``checkpoint_census``), never against the
    fixture's own sizes, which would be the smoke measuring its own invention.
"""

from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from sglang.srt.managers.scheduler_components import weight_updater as wu  # noqa: E402
from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402
from sglang.srt.weg2 import weight_exchange_bounce as wb  # noqa: E402
from sglang.srt.weg2 import weight_exchange_region as xr  # noqa: E402
from sglang.srt.weg2 import xchg_manifest as xm  # noqa: E402

P_CUT = (44, 10, 10)
D_VECTOR = (17, 7, 8)
CARDS = (0, 1, 2)
TAG = "weights_0"
ITEMSIZE = 1                       # INT8-W8A8

#: MEASURED FROM THIS RIG'S CHECKPOINT by `weg2/checkpoint_census.py`, quoted
#: rather than re-derived: a smoke that sized its slot from its own fixture
#: would grade nothing.
WIDEST_LAYER_BYTES = 756_323_776   # 721 MiB
LM_HEAD_MIB = 2425
EMBED_MIB = 1212

#: Small enough to run at the desk, shaped like the real classes.  The RATIO
#: that matters for the slot check is layer-vs-slot, and that is asserted
#: against the REAL widest layer above, not against these.
CLASSES = (("self_attn.qkv_proj.weight", 512, 256, "rows"),
           ("mlp.down_proj.weight", 256, 384, "cols"))
LAYERS = 6                          # a short stack; the cut is scaled below


def stage_of_layer(layer: int, cut) -> int:
    acc = 0
    for stage, n in enumerate(cut):
        acc += n
        if layer < acc:
            return stage
    raise AssertionError(layer)


def split(total: int) -> tuple:
    parts = [total * w // sum(D_VECTOR) for w in D_VECTOR]
    parts[-1] += total - sum(parts)
    return tuple(parts)


def scaled_cut(n_layers: int) -> tuple:
    """The 44,10,10 SHAPE at a desk-sized layer count, never a new rule."""
    parts = [max(1, n_layers * c // sum(P_CUT)) for c in P_CUT]
    parts[-1] += n_layers - sum(parts)
    return tuple(parts)


def build(tmp: str):
    """Six manifests on disk + this rank's live tensors, per group and rank."""
    cut = scaled_cut(LAYERS)
    models = {}
    for group in ("P", "D"):
        for rank in range(len(CARDS)):
            pieces, params = [], []
            for layer in range(LAYERS):
                for suffix, rows, cols, axis in CLASSES:
                    name = f"model.layers.{layer}.{suffix}"
                    if group == "P":
                        if stage_of_layer(layer, cut) != rank:
                            continue
                        r, c = rows, cols
                    else:
                        widths = split(rows if axis == "rows" else cols)
                        r = widths[rank] if axis == "rows" else rows
                        c = cols if axis == "rows" else widths[rank]
                    pieces.append(xm.ManifestPiece(
                        param_name=name,
                        tensor_class=name.rsplit(".", 2)[-2],
                        rows_full=r, cols_full=c, itemsize=ITEMSIZE,
                        tag=TAG, nbytes=r * c * ITEMSIZE))
                    params.append((name, torch.nn.Parameter(
                        torch.zeros(r, c, dtype=torch.int8),
                        requires_grad=False)))
            # THE AXES OF THE REAL FORM (weg2xsn22): group P is
            # `--tp-size 1 --pp-size 3`, group D is TP3. Keyed on `tp_rank`
            # alone all three P ranks wrote one file and the join refused.
            xm.write_rank_manifest(xm.RankManifest(
                group=group, rank=rank, card=CARDS[rank], region_tag=TAG,
                boot_token="smoke",
                tp_rank=(rank if group == "D" else 0),
                pp_rank=(rank if group == "P" else 0),
                pieces=tuple(pieces)), tmp)
            models[(group, rank)] = params
    return models, cut


class _Model:
    def __init__(self, params):
        self._p = params

    def named_parameters(self):
        return list(self._p)


class _Runner:
    def __init__(self, model):
        self.model = model


class _Worker:
    def __init__(self, model):
        self.model_runner = _Runner(model)


class _Stub:
    """Borrows the REAL unbound methods -- nothing here is re-implemented."""

    # BORROWED, INCLUDING THE NEW TWO-RUNNER TABLE: the address books read it,
    # and a stub that omitted it would exercise a path the product does not
    # have (weg2xsn24's `fc.weight has no destination pointer` came from the
    # single-runner table this method replaced).
    _weg2_rank_param_table = (
        wu.SchedulerWeightUpdaterManager._weg2_rank_param_table)
    _weg2_join_src_addr = wu.SchedulerWeightUpdaterManager._weg2_join_src_addr
    _weg2_join_dst_addr = wu.SchedulerWeightUpdaterManager._weg2_join_dst_addr
    _weg2_shadow_plan = wu.SchedulerWeightUpdaterManager._weg2_shadow_plan

    def __init__(self, model):
        self.tp_worker = _Worker(model)


def main() -> int:
    failures = []
    lines = []

    def check(ok, what):
        print(f"  {'PASS' if ok else 'FAIL'}  {what}")
        if not ok:
            failures.append(what)

    print(f"WEG2-XCHG-PROVIDER-SMOKE p_cut={','.join(map(str, P_CUT))} "
          f"d_vector={','.join(map(str, D_VECTOR))} cards={list(CARDS)} "
          f"quant=w8a8_int8 layers={LAYERS}")

    with tempfile.TemporaryDirectory(prefix="weg2-b4n-prov") as tmp:
        models, cut = build(tmp)
        print(f"  manifests written: {sorted(os.listdir(tmp))[:2]} ... "
              f"({len(os.listdir(tmp))} files, cut={cut})")
        os.environ[xm.DIR_ENV] = tmp
        os.environ[xr.ENV_REGION_BOOT] = "smoke"
        os.environ["SGLANG_WEG2_WEIGHT_SOURCE"] = "exchange"

        # THE xsn22 SHAPE IS ASSERTED, not assumed: three distinct P files.
        p_files = [f for f in os.listdir(tmp) if f.startswith("phase_manifest_P_")]
        check(len(p_files) == len(CARDS),
              f"group P (tp=1, pp=3) wrote {len(p_files)} distinct files "
              f"(weg2xsn22 wrote 1 and lost every leg)")
        check(wx.exchange_armed(),
              "the arm is armed, so the join is the producer (not the "
              "derivation)")

        # -- 1. THE PRODUCT CALL SITE, BOTH HOOKS --------------------------
        print("\n[1] _weg2_shadow_plan -- the real method body, both hooks")
        plans = {}
        for hook, group in (("source", "P"), ("destination", "D")):
            rank = 0
            stub = _Stub(_Model(models[(group, rank)]))
            plan, reason = stub._weg2_shadow_plan(
                hook, group, rank, agreed=None, require_agreement=False)
            check(plan is not None,
                  f"hook={hook} group={group}: a plan was produced "
                  f"({reason or 'no reason'})")
            if plan is None:
                continue
            plans[(hook, group)] = plan
            prof = wx.pointer_profile(plan.descs)
            line = wx.pointer_profile_line(prof, hook=hook,
                                           is_source=(hook == "source"))
            print("  " + line)
            lines.append(line)
            check(prof.src_resolved == prof.descs_total > 0
                  if hook == "source" else
                  prof.dst_resolved == prof.descs_total > 0,
                  f"hook={hook}: the side this rank OWNS is fully resolved "
                  f"(src {prof.src_resolved}/{prof.descs_total}, dst "
                  f"{prof.dst_resolved}/{prof.descs_total})")
            check(plan.facts.source == xm.JOIN_PLAN_SOURCE,
                  f"hook={hook}: the plan's provenance names the MANIFESTS, "
                  f"not the derivation")

        # -- 2. THE SLOT CHECK, AT THE SITE WHERE XSN9 DIED ----------------
        print("\n[2] refuse_if_plan_exceeds_slot -- real class widths")
        for (hook, group), plan in plans.items():
            units = wb.plan_units(plan.descs)
            layered = [u for u in units if u.is_layer]
            widest = max((u.nbytes for u in layered), default=0)
            print(f"  hook={hook} units={len(units)} layered={len(layered)} "
                  f"widest_unit={widest} B against slot="
                  f"{WIDEST_LAYER_BYTES} B (721 MiB, the checkpoint's own "
                  f"widest layer)")
            try:
                wb.refuse_if_plan_exceeds_slot(WIDEST_LAYER_BYTES, plan.descs)
                check(True, f"hook={hook}: the plan fits the real slot")
            except BaseException as exc:  # noqa: BLE001
                check(False, f"hook={hook}: {type(exc).__name__}: {exc}")
            # ... and the same call REFUSES on a slot that cannot hold it,
            # so the check is proven able to fire rather than merely green.
            if widest > 1:
                try:
                    wb.refuse_if_plan_exceeds_slot(widest - 1, plan.descs)
                    check(False, f"hook={hook}: an undersized slot passed")
                except BaseException:  # noqa: BLE001
                    check(True, f"hook={hook}: an undersized slot REFUSES "
                                f"(the check can fire)")

        # -- 3. NO FALLBACK TO THE DIAGONAL --------------------------------
        print("\n[3] a missing peer manifest refuses BY FILE NAME")
        victim = os.path.join(tmp, xm.manifest_filename(
            2, "P", TAG, tp_rank=0, pp_rank=2))
        os.rename(victim, victim + ".hidden")
        stub = _Stub(_Model(models[("D", 0)]))
        plan, reason = stub._weg2_shadow_plan(
            "destination", "D", 0, agreed=None, require_agreement=False)
        print(f"  refusal: {reason[:170]}")
        check(plan is None, "no plan was produced from an incomplete set")
        check("manifest-missing" in reason and ".json" in reason,
              "the refusal names the missing FILE, not merely the condition")
        os.rename(victim + ".hidden", victim)

        # -- 4. THE LEG ITSELF, THROUGH THE ADAPTER, BOTH DIRECTIONS -------
        #
        # THE POINT OF THIS SECTION: slices 1-3 are only worth something if the
        # descriptors the provider produced actually reach `run_bounce_leg`
        # with the right phase and survive the slot check. weg2xsn9 died at
        # `refuse_if_plan_exceeds_slot` and weg2xsn20 at `_missing_pointer`;
        # both are on this path.
        print("\n[4] the adapter -> run_bounce_leg, both hooks")
        from sglang.srt.managers.scheduler_components import weight_updater as wu
        import sglang.srt.weg2.weight_exchange_bounce as _wb

        seen = []
        orig = _wb.run_bounce_leg

        def _record(descs, ops, nonce, **kw):
            seen.append((kw.get("phase"), kw.get("rendezvous") is not None,
                         len(descs)))
            # The two refusals that killed two boots, on the REAL descriptors.
            hole = _wb._missing_pointer(descs, kw.get("phase", "both"))
            if hole is not None:
                raise AssertionError(f"_missing_pointer: {hole}")
            _wb.refuse_if_plan_exceeds_slot(WIDEST_LAYER_BYTES, descs)
            return None

        class _LegStub:
            _weg2_xchg_bounce_leg = (
                wu.SchedulerWeightUpdaterManager._weg2_xchg_bounce_leg)

        class _Sems:
            def timedwait(self, *a, **k):
                return True

            def post(self, *a, **k):
                pass

            def diagonal_timedwait(self, *a, **k):
                return True

            def diagonal_post(self, *a, **k):
                pass

        class _Region:
            boot_nonce = "smoke"

            def publish(self, *a, **k):
                pass

            def read_slot(self, pair, slot):
                class _R:
                    bytes_filled = 0
                return _R()

        _wb.run_bounce_leg = _record
        try:
            for hook, group in (("source", "P"), ("destination", "D")):
                plan = plans.get((hook, group))
                if plan is None:
                    continue
                seen.clear()
                try:
                    _LegStub()._weg2_xchg_bounce_leg(
                        descs=list(plan.descs), ops=None, boot_nonce="smoke",
                        slot_bytes=WIDEST_LAYER_BYTES, depth=1,
                        mode=wx.INJECT_AUTHORITATIVE, hook=hook,
                        region=_Region(), sems=_Sems())
                    ok, why = True, ""
                except BaseException as exc:  # noqa: BLE001
                    ok, why = False, f"{type(exc).__name__}: {exc}"
                want = (_wb.PHASE_DEPOSIT if hook == "source"
                        else _wb.PHASE_COLLECT)
                phases = {p for p, _r, _n in seen}
                print(f"  hook={hook} legs={len(seen)} phases={sorted(phases)} "
                      f"descs={[n for _p, _r, n in seen]}")
                check(ok, f"hook={hook}: the leg ran through the adapter "
                          f"({why or 'no refusal'})")
                check(want in phases or phases == {_wb.PHASE_BOTH},
                      f"hook={hook}: the adapter passed {want} for its cross "
                      f"pairs (saw {sorted(phases)})")
                cross = [s for s in seen if s[0] != _wb.PHASE_BOTH]
                check(all(s[1] for s in cross) if cross else True,
                      f"hook={hook}: every cross leg carries a rendezvous")
        finally:
            _wb.run_bounce_leg = orig

    total = 19
    print(f"\nWEG2-XCHG-PROVIDER-SMOKE verdict="
          f"{'PASS' if not failures else 'FAIL'} "
          f"checks={total - len(failures)}/{total}")
    for f in failures:
        print(f"  FAILED: {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
