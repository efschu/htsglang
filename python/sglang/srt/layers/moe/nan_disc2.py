"""Task #49 (20.09.): the SECOND NaN discriminator, '[nan-disc2]'.

The first discriminator (``qwen4_exp._nan_discriminate``, 19.09.) answers three
questions -- input already bad / transient / persistent -- and on the 259k
needle it always answers the same one: input finite, output non-finite for the
rows of ONE expert (periodic filler, row stride 170), recompute on the same
input clean. 'TRANSIENT' is where that instrument stops; it does not say WHICH
of the five candidate classes produced it, and fn8aj refuted the cheapest of
them (a copy still in flight -- ``SGLANG_MOE_OFFLOAD_FETCH_SYNC=1`` did not
prevent the hit).

This module is the evidence the next needle boot has to bring back. On the
FIRST hit of a process it collects, as one coherent log group:

  (1) the top-k expert ids of the BAD rows against those of the GOOD rows, and
      the intersection -- 'which expert is in EVERY bad row'.
  (2) that expert's slot, per the wave trace the offload cache recorded while
      the forward ran (wave index, slot_of_needed, and what ``_scratch_holds``
      said the slot held during that wave).
  (3) a byte fingerprint of the slot's content per resident tensor
      (qweight / scales / zeros) BEFORE the recompute and AFTER it.
  (4) the same fingerprint of the HOST original of that expert's row in the
      pinned spill pool -- the bytes the slot is supposed to hold.
  (5) the Marlin lock workspace's non-zero count before and after.
  (6) the wave index and ``partials_mode``.

and then names ONE of five classes. The classes and the log picture each one
produces are in ``classify`` below; that function is pure and is what the desk
tests exercise, so the verdict logic is proven without a GPU.

Honest limits, printed with the evidence rather than left for the reader:
  * scratch slots are REUSED by later waves of the same forward, so a 'before'
    fingerprint may already be a different expert's bytes. ``holds_now`` and
    ``resident`` say whether that is the case; a slot below ``resident_count``
    is stable and its fingerprint is directly comparable.
  * the recompute re-runs ``planner.resolve`` + ``_fetch``, so the 'after'
    fingerprint is the state AFTER a refetch. That is the point: it separates
    'the bytes in the slot were wrong' from 'the bytes were right and the
    kernel still produced NaN'.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

_STATE: Dict[str, Any] = {"on": None, "armed": True}

# The five candidate classes of Task #49, and the verdict code each gets.
CLASS_LUT = "A-LUT-MISASSIGN"
CLASS_BYTES = "B-SLOT-BYTES"
CLASS_KERNEL = "C-KERNEL-OR-WORKSPACE"
CLASS_ROUTER = "D-ROUTER-ID-OUTSIDE-LUT"
CLASS_PARTIALS = "E-WAVE-PARTIALS"
CLASS_UNRESOLVED = "UNRESOLVED"


def disc2_on() -> bool:
    """SGLANG_NAN_DISC2=0 turns the second discriminator off; default on (it
    only ever runs behind a nan-guard hit, so off costs nothing either way)."""
    if _STATE["on"] is None:
        _STATE["on"] = str(os.environ.get("SGLANG_NAN_DISC2", "1")).strip().lower() not in (
            "",
            "0",
            "off",
            "false",
        )
    return bool(_STATE["on"])


def arm_once() -> bool:
    """True exactly once per process: the FIRST hit of a boot pays the cost."""
    if not _STATE["armed"]:
        return False
    _STATE["armed"] = False
    return True


def _reset_for_tests(on: bool = True, armed: bool = True) -> None:
    _STATE.update({"on": on, "armed": armed})


# --- pure analysis -------------------------------------------------------


def fingerprint(t) -> str:
    """A stable 64-bit hex digest of a tensor's BYTES.

    Byte-exact (not a float reduction), so two rows that differ in a single
    quantised nibble differ here. Falls back to a numeric checksum when the
    uint8 re-view is not available for the dtype/layout, and never raises --
    a fingerprint that could not be taken is reported as such, not guessed.
    """
    if t is None:
        return "none"
    try:
        import hashlib

        import torch

        c = t.detach().contiguous().reshape(-1)
        if c.device.type != "cpu":
            c = c.cpu()
        try:
            raw = c.view(torch.uint8).numpy().tobytes()
        except Exception:  # noqa: BLE001 -- dtype without a uint8 re-view
            raw = c.to(torch.float64).numpy().tobytes()
        return hashlib.blake2b(raw, digest_size=8).hexdigest()
    except Exception as exc:  # noqa: BLE001
        return f"unhashable({type(exc).__name__})"


def row_expert_sets(ids_list: Sequence[Sequence[int]], rows: Sequence[int]) -> List[Set[int]]:
    """The routed expert set of each named row (negative padding dropped)."""
    out: List[Set[int]] = []
    n = len(ids_list)
    for r in rows:
        r = int(r)
        if 0 <= r < n:
            out.append({int(e) for e in ids_list[r] if int(e) >= 0})
    return out


def common_experts(sets: Sequence[Set[int]]) -> List[int]:
    """The experts present in EVERY set. Empty list for an empty input --
    an intersection over nothing is not 'all experts'."""
    if not sets:
        return []
    acc = set(sets[0])
    for s in sets[1:]:
        acc &= s
        if not acc:
            break
    return sorted(acc)


def union_experts(sets: Sequence[Set[int]]) -> Set[int]:
    acc: Set[int] = set()
    for s in sets:
        acc |= s
    return acc


def suspect_experts(
    bad_sets: Sequence[Set[int]], good_sets: Sequence[Set[int]]
) -> Tuple[List[int], List[int]]:
    """(common, exclusive): common = in every bad row; exclusive = in every bad
    row AND in none of the sampled good rows. The exclusive set is the strong
    signal -- an expert that every good row also uses cannot be the one whose
    weights are wrong."""
    common = common_experts(bad_sets)
    good = union_experts(good_sets)
    return common, [e for e in common if e not in good]


def locate_expert(waves: Sequence[Dict[str, Any]], expert: int) -> List[Dict[str, Any]]:
    """Every wave of the recorded trace that asked for ``expert``, with the slot
    it was mapped to and what the scratch holds said that slot carried."""
    hits: List[Dict[str, Any]] = []
    for w in waves:
        slot_of = w.get("slot_of_needed") or {}
        if int(expert) not in slot_of:
            continue
        slot = int(slot_of[int(expert)])
        holds = w.get("holds") or {}
        hits.append(
            {
                "wave": int(w.get("wave", -1)),
                "slot": slot,
                "held_by_then": holds.get(slot, holds.get(str(slot))),
            }
        )
    return hits


def hold_mismatches(waves: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Every (wave, expert, slot) where the planner mapped ``expert -> slot``
    but the scratch holds recorded for that same wave say the slot carries a
    DIFFERENT expert.

    This is the sound test for class A. It compares two independent records of
    the same fact: ``slot_of_needed``, which the LUT is built from, and
    ``_scratch_holds``, which ``_fetch`` writes as it claims slots. Resident
    slots never appear in the holds and are skipped rather than guessed at."""
    out: List[Dict[str, Any]] = []
    for w in waves:
        slot_of = w.get("slot_of_needed") or {}
        holds = w.get("holds") or {}
        for e, slot in slot_of.items():
            held = holds.get(int(slot), holds.get(str(slot)))
            if held is None:
                continue  # resident slot: not tracked in the holds
            if int(held) != int(e):
                out.append(
                    {
                        "wave": int(w.get("wave", -1)),
                        "expert": int(e),
                        "slot": int(slot),
                        "held": int(held),
                    }
                )
    return out


def slot_collisions(waves: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Waves in which two distinct experts were mapped to the SAME slot -- the
    planner's invariant broken; the second expert's rows read the first's
    weights."""
    out: List[Dict[str, Any]] = []
    for w in waves:
        by_slot: Dict[int, List[int]] = {}
        for e, slot in (w.get("slot_of_needed") or {}).items():
            by_slot.setdefault(int(slot), []).append(int(e))
        for slot, experts in by_slot.items():
            if len(experts) > 1:
                out.append({"wave": int(w.get("wave", -1)), "slot": slot, "experts": sorted(experts)})
    return out


def waves_of_rows(
    waves: Sequence[Dict[str, Any]], ids_list: Sequence[Sequence[int]], rows: Sequence[int]
) -> List[int]:
    """The wave indices that touched any expert of the named rows."""
    want = union_experts(row_expert_sets(ids_list, rows))
    out = []
    for w in waves:
        needed = {int(e) for e in (w.get("needed") or [])}
        if needed & want:
            out.append(int(w.get("wave", -1)))
    return sorted(set(out))


def _is_digest(v) -> bool:
    return isinstance(v, str) and len(v) == 16 and all(c in "0123456789abcdef" for c in v)


def _comparable(a: Dict[str, str], b: Dict[str, str]) -> List[str]:
    """The attrs for which BOTH sides carry a real digest. A fingerprint that
    could not be taken ('unreadable(...)', 'none') is UNKNOWN -- counting it as
    a difference would manufacture a corruption finding out of a missing read,
    which is exactly the kind of false positive this instrument exists to
    avoid."""
    return sorted(k for k in a if k in b and _is_digest(a[k]) and _is_digest(b[k]))


def _any_differs(a: Dict[str, str], b: Dict[str, str]) -> List[str]:
    """The attrs on which two fingerprint maps genuinely disagree."""
    return [k for k in _comparable(a, b) if a[k] != b[k]]


def classify(obs: Dict[str, Any]) -> Tuple[str, str]:
    """Name ONE of the five candidate classes from the collected observations.

    The order of the tests is the order of specificity, not of suspicion:
    a slot that holds the wrong expert explains a byte mismatch, so it is
    decided first; bytes that differ from the host original explain a wrong
    GEMM, so they are decided before the kernel is blamed.

    Log picture per class (this is the contract the next boot is read against):

      D  ROUTER-ID-OUTSIDE-LUT -- an expert is in every bad row but in NO wave
         of the trace: its LUT entry was -1, the remap handed the kernel a
         negative slot and it read whatever sits at that offset.
      A  LUT-MISASSIGN -- the suspect expert HAS a slot, but the scratch holds
         say that slot carried a different expert during the wave: the first
         compute read another expert's weights; the recompute rebuilds the LUT
         and is clean.
      B  SLOT-BYTES -- the slot's fingerprint differs from the host original's.
         'refetch repaired' (after == host) is a fetch/visibility fault on the
         gather over the UVA view; 'still differs' points at the pinned pool
         itself (host-side corruption or a wrong pool row).
      E  WAVE-PARTIALS -- NO expert is common to all bad rows, but the bad rows
         all live in one wave: the fault is in the per-wave partial reduction
         (partials_mode 'stream' index_add_ vs 'table' combine), not in an
         expert's weights.
      C  KERNEL-OR-WORKSPACE -- every fingerprint agrees (slot == host, before
         == after) and the slot assignment is consistent, yet the first compute
         produced NaN and the recompute did not: identical inputs, different
         output. ``ws_nonzero_before > 0`` additionally says the Marlin lock
         workspace was NOT clean when the failing GEMM started, which is the
         reason SGLANG_MOE_MARLIN_PRIVATE_WORKSPACE=1 exists.
    """
    common = list(obs.get("common") or [])
    located = list(obs.get("located") or [])
    suspect = obs.get("suspect")
    fp_before = dict(obs.get("fp_before") or {})
    fp_after = dict(obs.get("fp_after") or {})
    fp_host = dict(obs.get("fp_host") or {})
    ws_before = obs.get("ws_nonzero_before")
    mism = list(obs.get("hold_mismatches") or [])
    coll = list(obs.get("slot_collisions") or [])

    if common and not located:
        return (
            CLASS_ROUTER,
            "expert(s) %s are in every bad row but in NO wave of the trace -- the LUT "
            "entry was -1 and the remap handed the kernel a negative slot" % (common,),
        )

    if coll:
        return (
            CLASS_LUT,
            "the planner mapped two distinct experts to ONE slot: %s -- the second "
            "expert's rows read the first's weights" % (coll[:4],),
        )
    if mism:
        return (
            CLASS_LUT,
            "planner and scratch holds disagree about who owns a slot: %s (expert -> "
            "slot per slot_of_needed, 'held' per _scratch_holds) -- the LUT points at "
            "a slot the fetch never claimed for that expert" % (mism[:4],),
        )

    # The byte comparison is only meaningful while the slot still belongs to the
    # suspect. A spill slot reused by a later wave of the same forward carries
    # someone else's bytes by design -- saying 'corrupt' there would be an
    # artefact of reading late, not a finding.
    valid = bool(obs.get("fp_valid"))
    diff_host = _any_differs(fp_before, fp_host) if valid else []
    if diff_host:
        if not _any_differs(fp_after, fp_host):
            return (
                CLASS_BYTES,
                "slot bytes differed from the host original on %s BEFORE the recompute "
                "and match it after -- the refetch repaired it: a fetch/visibility "
                "fault on the gather over the UVA view of the pinned pool" % (diff_host,),
            )
        return (
            CLASS_BYTES,
            "slot bytes differ from the host original on %s before AND after the "
            "recompute -- the pinned pool row itself, not the copy" % (diff_host,),
        )

    if not common and obs.get("common_wave") is not None:
        return (
            CLASS_PARTIALS,
            "no expert is common to all bad rows, but they all live in wave %s "
            "(partials_mode=%s) -- the per-wave partial reduction, not an expert"
            % (obs.get("common_wave"), obs.get("partials")),
        )

    if not valid:
        return (
            CLASS_UNRESOLVED,
            "the slot of suspect expert %s was reused before the group could be read "
            "(holds_now=%s, resident_slot=%s), so the byte comparison proves nothing; "
            "everything else is consistent. Re-run with "
            "SGLANG_MOE_OFFLOAD_WAVE_ORDER=token or a larger scratch to keep the slot"
            % (suspect, obs.get("holds_now"), obs.get("resident_slot")),
        )

    if _comparable(fp_before, fp_after) and not _any_differs(fp_before, fp_after):
        ws = ""
        if isinstance(ws_before, int) and ws_before > 0:
            ws = (
                " AND the Marlin lock workspace held %d non-zero entries when the "
                "failing GEMM started (SGLANG_MOE_MARLIN_PRIVATE_WORKSPACE=1 is the "
                "counter-probe)" % ws_before
            )
        host = (
            "slot == host original, "
            if _comparable(fp_before, fp_host)
            else "no comparable host fingerprint (resident expert, or the pool row "
            "could not be read -- see fp_host), "
        )
        return (
            CLASS_KERNEL,
            "every fingerprint agrees (%sbefore == after) and the slot assignment is "
            "consistent -- identical inputs, different output%s" % (host, ws),
        )

    return (
        CLASS_UNRESOLVED,
        "no class fits: common=%s located=%s fp_before=%s fp_after=%s fp_host=%s "
        "-- report the raw group, do not guess" % (common, located, fp_before, fp_after, fp_host),
    )


# --- collection (needs the live objects; still CPU-runnable with stubs) ---


def find_cache(layer):
    """(moe_module, offload_cache) for a decoder layer, or (module, None)."""
    mlp = getattr(layer, "mlp", layer)
    experts = getattr(mlp, "experts", mlp)
    cache = getattr(experts, "_expert_offload", None)
    if cache is not None:
        return experts, cache
    try:
        members = vars(experts).values()
    except TypeError:
        return experts, None
    for obj in members:
        if isinstance(getattr(obj, "_resident", None), dict):
            return experts, obj
    return experts, None


def _workspace_nonzero(experts) -> Optional[int]:
    ws = getattr(experts, "workspace", None)
    if ws is None:
        return None
    try:
        return int((ws != 0).sum().item())
    except Exception:  # noqa: BLE001
        return None


def _host_row(cache, expert: int) -> Optional[int]:
    """The pinned spill pool row of a global expert id, or None when the expert
    is RESIDENT (it has no pool row -- its slot is its home)."""
    R = int(getattr(cache, "resident_count", 0))
    idx = getattr(cache, "_spill_pool_index", None)
    if idx is not None:
        return int(idx[expert]) if int(expert) in idx else None
    row = int(expert) - R
    return row if row >= 0 else None


def slot_fingerprints(cache, slot: int) -> Dict[str, str]:
    resident = getattr(cache, "_resident", None) or {}
    out: Dict[str, str] = {}
    for attr, buf in resident.items():
        try:
            out[attr] = fingerprint(buf[int(slot)])
        except Exception as exc:  # noqa: BLE001
            out[attr] = f"unreadable({type(exc).__name__})"
    return out


def host_fingerprints(cache, expert: int) -> Dict[str, str]:
    row = _host_row(cache, int(expert))
    if row is None:
        return {}
    pinned = getattr(cache, "_pinned", None) or {}
    out: Dict[str, str] = {}
    for attr, pool in pinned.items():
        try:
            out[attr] = fingerprint(pool[row])
        except Exception as exc:  # noqa: BLE001
            out[attr] = f"unreadable({type(exc).__name__})"
    return out


def snapshot(experts, cache, bad_rows: Sequence[int], good_rows: Sequence[int]) -> Optional[Dict[str, Any]]:
    """Everything that must be read BEFORE the recompute. None when there is no
    routing trace to read (the guard was off while the forward ran, or this
    layer is fully resident and never went through run_waves)."""
    trace = getattr(cache, "_nan_trace", None) if cache is not None else None
    if not isinstance(trace, dict):
        return None
    # a [T][K] list (list route) or a [T, K] int64 array (H20c vector route);
    # `or []` would ask an array for its truth value and raise
    ids_list = trace.get("ids_list")
    if ids_list is None:
        ids_list = []
    waves = trace.get("waves") or []
    bad_sets = row_expert_sets(ids_list, bad_rows)
    good_sets = row_expert_sets(ids_list, good_rows)
    common, exclusive = suspect_experts(bad_sets, good_sets)
    suspect = exclusive[0] if exclusive else (common[0] if common else None)
    located = locate_expert(waves, suspect) if suspect is not None else []
    hit_waves = waves_of_rows(waves, ids_list, bad_rows)
    slot = located[0]["slot"] if located else None
    holds_now = dict(getattr(cache, "_scratch_holds", None) or {})
    R = int(getattr(cache, "resident_count", 0))
    resident_slot = slot is not None and slot < R
    held_now = holds_now.get(slot) if slot is not None else None
    # The slot still belongs to the suspect (or is a stable resident slot), so
    # its bytes can be held against the host original.
    fp_valid = slot is not None and (
        resident_slot or (held_now is not None and suspect is not None and int(held_now) == int(suspect))
    )
    return {
        "bad_rows": [int(r) for r in bad_rows],
        "good_rows_sampled": len(good_sets),
        "bad_row_experts": [sorted(s) for s in bad_sets[:6]],
        "good_row_experts": [sorted(s) for s in good_sets[:6]],
        "common": common,
        "exclusive": exclusive,
        "suspect": suspect,
        "located": located,
        "slot": slot,
        "resident_slot": resident_slot,
        "resident_count": R,
        "holds_now": held_now,
        "fp_valid": fp_valid,
        "hold_mismatches": hold_mismatches(waves),
        "slot_collisions": slot_collisions(waves),
        "hit_waves": hit_waves,
        "common_wave": hit_waves[0] if len(hit_waves) == 1 else None,
        "partials": trace.get("partials"),
        "n_waves": len(waves),
        "fp_before": slot_fingerprints(cache, slot) if slot is not None else {},
        "fp_host": host_fingerprints(cache, suspect) if suspect is not None else {},
        "ws_nonzero_before": _workspace_nonzero(experts),
    }


def finish(experts, cache, snap: Dict[str, Any], recompute_bad_rows: int) -> Dict[str, Any]:
    """The AFTER half: the same fingerprints once the recompute (and its
    refetch) has run, plus the verdict."""
    slot = snap.get("slot")
    snap["fp_after"] = slot_fingerprints(cache, slot) if slot is not None else {}
    snap["ws_nonzero_after"] = _workspace_nonzero(experts)
    snap["recompute_bad_rows"] = int(recompute_bad_rows)
    code, text = classify(snap)
    snap["verdict"] = code
    snap["verdict_text"] = text
    return snap


def render(layer_id, snap: Dict[str, Any]) -> str:
    """The one coherent '[nan-disc2]' log group. One string, one record, so a
    grep on the tag returns the whole finding and never half of it."""
    L = []
    a = L.append
    a("[nan-disc2] layer %s: FIRST hit of this process, one group." % (layer_id,))
    a(
        "[nan-disc2]  rows: %d bad %s | %d good sampled; bad-row expert sets %s"
        % (
            len(snap.get("bad_rows") or []),
            (snap.get("bad_rows") or [])[:12],
            snap.get("good_rows_sampled"),
            snap.get("bad_row_experts"),
        )
    )
    a("[nan-disc2]  good-row expert sets %s" % (snap.get("good_row_experts"),))
    a(
        "[nan-disc2]  experts in EVERY bad row: %s; of those in NO sampled good row: %s"
        " -> suspect %s"
        % (snap.get("common"), snap.get("exclusive"), snap.get("suspect"))
    )
    a(
        "[nan-disc2]  slot per LUT/planner: %s (resident_slot=%s, resident_count=%s); "
        "wave hits %s; scratch holds that slot NOW: %s"
        % (
            snap.get("located"),
            snap.get("resident_slot"),
            snap.get("resident_count"),
            snap.get("hit_waves"),
            snap.get("holds_now"),
        )
    )
    a(
        "[nan-disc2]  waves=%s partials_mode=%s common_wave=%s"
        % (snap.get("n_waves"), snap.get("partials"), snap.get("common_wave"))
    )
    a(
        "[nan-disc2]  planner-vs-holds mismatches %s; two-experts-one-slot %s"
        % ((snap.get("hold_mismatches") or [])[:4], (snap.get("slot_collisions") or [])[:4])
    )
    a(
        "[nan-disc2]  byte comparison valid: %s (a reused spill slot makes it "
        "meaningless; a resident slot is always comparable)" % (snap.get("fp_valid"),)
    )
    a("[nan-disc2]  slot fingerprint BEFORE recompute: %s" % (snap.get("fp_before"),))
    a("[nan-disc2]  slot fingerprint AFTER  recompute: %s" % (snap.get("fp_after"),))
    a("[nan-disc2]  host original (pinned pool row):   %s" % (snap.get("fp_host"),))
    a(
        "[nan-disc2]  marlin lock workspace non-zero entries: before=%s after=%s"
        % (snap.get("ws_nonzero_before"), snap.get("ws_nonzero_after"))
    )
    a(
        "[nan-disc2]  recompute bad rows: %s"
        % (snap.get("recompute_bad_rows"),)
    )
    a("[nan-disc2]  VERDICT %s: %s" % (snap.get("verdict"), snap.get("verdict_text")))
    return "\n".join(L)
