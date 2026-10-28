"""Module-level helpers moved out of the source module (refactor, behaviour-neutral): bodies are byte-identical,
the source module re-exports every name (tests and callers keep ``<module>.<name>``). Logger name unchanged."""

from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
import os

from flliper.srt.compat_shims import operator_dir as _operator_dir
from flliper.srt.pdflip import host_ledger
from flliper.srt.pdflip import rank_state as rank_state_mod
from flliper.srt.pdflip import shared_cache_release as shared_cache_release_mod
from flliper.srt.pdflip import weight_exchange
from flliper.srt.pdflip import weight_exchange_shadow
from flliper.srt.pdflip import weight_exchange_transport
from flliper.srt.pdflip import xchg_bounce
from flliper.srt.pdflip import xchg_manifest
from flliper.srt.pdflip import xchg_residency
from flliper.srt.pdflip.launcher_d_mid import ONCARD_MODE_DEFAULT
from flliper.srt.pdflip.launcher_d_p_r2_L0 import EVIDENCE_DIR
from flliper.srt.pdflip.launcher_d_p_r2_L0 import GPU_ARB
from flliper.srt.pdflip.launcher_d_p_r2_L0 import boot_state_write
from flliper.srt.pdflip.launcher_d_p_r2_L0 import dc_measured_d_mib
from flliper.srt.pdflip.launcher_d_p_r2_L0 import helper_registry_dir
from flliper.srt.pdflip.launcher_d_p_r2_L0 import load_xchg_census_for_cards
from flliper.srt.pdflip.launcher_d_p_r2_L0 import p_group_has_draft_producer
from flliper.srt.pdflip.launcher_log_plan import Log
from flliper.srt.pdflip.launcher_p_l3_leaf import WEIGHT_SOURCE_EXCHANGE
from flliper.srt.pdflip.launcher_prepare_sweep_r2_L0_1 import prepare_xchg_region
from flliper.srt.pdflip.launcher_prepare_sweep_r2_L0_1 import rank_state_dir_for
from flliper.srt.pdflip.launcher_prepare_sweep_r2_L0_1 import state_path
from flliper.srt.pdflip.launcher_refuse_proc_leaf import WEIGHT_SOURCE_ARMED
from flliper.srt.pdflip.launcher_refuse_proc_leaf import _proc_start_ticks
from flliper.srt.pdflip.launcher_types import BootState
from flliper.srt.pdflip.launcher_types import Card
from flliper.srt.pdflip.launcher_types import GroupSpec
from flliper.srt.pdflip.launcher_types import PdFlipLaunchRefused
from flliper.srt.pdflip.launcher_xchg_vision_leaf import xchg_census_is_reserve
import json

#: HICACHE-DRAFT-TIER (user order 2026-09-24 14:15Z, verbatim: "und schreiben
#: wir in D auch draft context in den hicache? das muesste raus, weil draft ja
#: keinen hicacheplatz mehr bekommt"). The operator's input, default ``auto``:
#: off when group P has no draft producer (p_group_has_draft_producer), on
#: otherwise; ``on``/``off`` force it (A/B). The RESOLVED ``off`` rides BOTH
#: groups' environment (a rank cannot resolve auto: group D does not know P's
#: form); a resolved ``on`` writes nothing, so the producer form's environment
#: stays byte-identical. Same name as the NF line; code and values per line.
HICACHE_DRAFT_TIER_ENV = "FLLIPER_PDFLIP_HICACHE_DRAFT_TIER"
HICACHE_DRAFT_TIER_DEFAULT = "auto"
HICACHE_DRAFT_TIER_VALUES = ("auto", "on", "off")

def resolve_hicache_draft_tier() -> Tuple[str, str]:
    """``(tier, provenance)``: ``tier`` is ``on`` or ``off``."""
    raw = os.environ.get(HICACHE_DRAFT_TIER_ENV, HICACHE_DRAFT_TIER_DEFAULT)
    value = str(raw or HICACHE_DRAFT_TIER_DEFAULT).strip().lower()
    if value not in HICACHE_DRAFT_TIER_VALUES:
        raise PdFlipLaunchRefused(
            f"W151 PdFlipHicacheDraftTierInvalid: {HICACHE_DRAFT_TIER_ENV}={raw!r} -- "
            f"expected one of {', '.join(HICACHE_DRAFT_TIER_VALUES)}"
        )
    if value == "off" and p_group_has_draft_producer():
        # A producer that computes into a tier that does not exist: the DFlash
        # producer's first publish raises (DFlashDraftKvProduceError, no
        # registered draft pool), the NEXTN one computes for nothing. Named,
        # before a rank starts -- switch the producer off instead.
        raise PdFlipLaunchRefused(
            f"W152 PdFlipHicacheDraftTierConflict: {HICACHE_DRAFT_TIER_ENV}=off but group P "
            "produces draft pages (FLLIPER_PDFLIP_DRAFT_ON_P=1 / --draft-kv-on-p on: P carries "
            "the MTP head as draft-KV producer) -- the draft would get no HiCache space to publish into. "
            "Drop the override (auto follows P's form) or switch P's producer off."
        )
    if value in ("on", "off"):
        return value, f"{HICACHE_DRAFT_TIER_ENV}={value} (operator)"
    if p_group_has_draft_producer():
        return "on", f"auto: group P produces draft pages"
    return "off", "auto: group P has no draft producer"

def prepare_xchg_env(log: Log, boot_nonce: str, weight_source: str,
                       hook_mode: int = 0, dry: bool = False,
                       hop_bound_ms: Optional[float] = None,
                       oncard_mode: str = ONCARD_MODE_DEFAULT,
                       oncard_slot_mib: int =
                       weight_exchange_transport.ONCARD_SLOT_MIB_DEFAULT,
                       bounce_terms: Optional[xchg_bounce.BounceTerms] = None,
                       inject_mode: str = weight_exchange.INJECT_SHADOW,
                       legs: str = weight_exchange.LEGS_BOTH,
                       manifest_dir: str = "",
                       waves: Optional[Sequence[Sequence[str]]] = None,
                       ) -> Dict[str, str]:
    """#1273 S5/S6: arm the region for EVERY non-default arm and publish it.

    RENAMED FROM ``prepare_shadow_env`` IN STEP 7, because the old name became
    a lie the moment the exchange needed the same publication: one publisher,
    one name.  The old name is deliberately NOT kept as an alias -- two names
    for one publisher is how two readers begin.

    ``{}`` on the DEFAULT arm, and that empty dict is what keeps the default
    boot byte-identical: no region file, no semaphores, no environment, and
    :func:`build_env` pops the names so nothing can be inherited.

    THE GUARD USED TO READ ``weight_source != "shadow"`` AND THAT WAS THE
    BLOCKER.  Under ``--pdflip-weight-source exchange`` it returned ``{}``, so no
    region was armed and nothing was published -- not the on-card arm, not the
    slot, and not step 6's two variables.  Three silent wrong answers followed:
    a rank read ``inject_mode()``'s DEFAULT rather than a published value, so
    ``--pdflip-xchg-inject authoritative`` would have been ignored and the boot's
    mode was unprovable from its own argv; ``read_published_terms()`` answered
    ``None`` and the injector refused W4; and with no region there was no Gate
    0 and no transport.  The set is now derived from
    :data:`WEIGHT_SOURCE_ARMED`, so an arm added later cannot be missed by a
    comparison nobody remembered to widen.

    THE SHADOW NEEDS EXACTLY WHAT THE EXCHANGE NEEDS, MINUS THE AUTHORITY: the
    same 385 MiB staging region, the same 24 semaphores, the same env names.
    What differs is the MODE SWITCH INSIDE THE RANKS, and it is published as
    the arm that was asked for: ``FLLIPER_PDFLIP_WEIGHT_SOURCE=shadow`` leaves
    ``exchange_armed()`` False, so ``model_runner`` still opens the weights
    region with ``enable_cpu_backup`` and the ring still refills every byte,
    while ``=exchange`` is what makes the exchange the weight source at all.

    W71 IS NOT ARMED HERE, on purpose.  W71 prices the co-residency peak of a
    real exchange, and under ``shadow`` no tag is ever exchanged: both images
    stay where they are and the only new VRAM is the shadow's own bounded
    subset, which is priced per card, per leg, against the LIVE NVML free
    column by ``weight_exchange_shadow.price_shadow`` -- inside the rank that
    will allocate it, at the moment it would.  Pricing it at launch instead
    would grade a peak this arm does not have.
    """
    if str(weight_source) not in WEIGHT_SOURCE_ARMED:
        return {}
    got = prepare_xchg_region(log, boot_nonce, hook_mode, dry=dry)
    env = dict(got.get("env") or {})
    # THE ARM THAT WAS ASKED FOR, not a literal.  This read
    # `env[...] = "shadow"` -- correct while `shadow` was the only arm that
    # reached this line, and the SECOND HALF of step 7's blocker once
    # `exchange` did: it would have published `shadow` to every rank, leaving
    # `exchange_armed()` False in all six, so `model_runner` would still open
    # the weights region with `enable_cpu_backup` and the ring would still
    # refill every byte -- an `exchange` boot that silently ran as `shadow`,
    # with nothing in its own log to say so.
    env["FLLIPER_PDFLIP_WEIGHT_SOURCE"] = str(weight_source)
    # S5b: the launcher-given hop bound, published by the same mechanism and
    # for the same reason as the region names -- a rank reads a flag it never
    # sees on its own argv, and a bound nobody published is a bound the rank
    # would have to invent.
    env[weight_exchange_shadow.ENV_HOP_BOUND_MS] = repr(
        float(weight_exchange_shadow.hop_bound_ms(hop_bound_ms)))
    # S6: the on-card arm, published by the same mechanism and for the same
    # reason.  It is what decides whether a deposit is possible at all, and it
    # is the same string ``choose_host_ledger`` charges the deposit's host
    # bytes on -- so the arm a rank enforces and the arm the ledger paid for
    # cannot be two different readings.
    env[weight_exchange_transport.ENV_ONCARD_MODE] = str(oncard_mode)
    # S6 fix E: the slot CEILING, published by the same mechanism and for the
    # same reason as the arm -- it decides the deposit's size in the ranks AND
    # the worst case the #1269 ledger charged at launch, so the two may not be
    # two readings.  Validated here, so an illegal value dies at the launcher
    # with W82 rather than at six rank imports.
    slot_mib = weight_exchange_transport.validate_oncard_slot_mib(
        oncard_slot_mib)
    env[weight_exchange_transport.ENV_ONCARD_SLOT_MIB] = str(slot_mib)
    # #1273 S6 step 6c: THE INJECTION MODE, published by the same mechanism
    # and for the same reason as the arm -- it decides whether the bounce legs
    # at the wake seam OWN the weight bytes or merely grade themselves against
    # the refill, and a rank that read a different answer than the boot was
    # launched with would either serve ungraded bytes or silently keep paying
    # for a refill nobody reads.
    env[weight_exchange.INJECT_ENV] = str(inject_mode)
    # #1330 B4n: THE DIRECTION KNOB, published by the same mechanism and for
    # the same reason as the arm and the inject mode.  It decides WHICH legs a
    # rank runs at all, so a rank that read a different answer than the boot
    # was launched with would either run a direction the boot did not ask for
    # or silently skip one it did -- and a skipped leg looks in a log exactly
    # like a leg that ran clean, which is the shape four boots of this campaign
    # were unreadable for.
    env[weight_exchange.XCHG_LEGS_ENV] = str(legs)
    # #1330 B4n: WHERE THE PLACEMENT MANIFESTS LAND.
    #
    # A SEPARATE NAME WAS NOT THE FIRST CHOICE, and the reason it is the right
    # one is measured rather than argued.  The manifest was written first
    # against `FLLIPER_PHASE_FOOTPRINT_DUMP`, the directory the #1292/#1348
    # coverage dumps share -- but this launcher only ever puts that variable in
    # a rank's environment under `--xchg-coverage-diff` (`coverage_dump_dir`,
    # which returns "" otherwise so the OFF state is byte-identical).  A writer
    # keyed on it would therefore have found an EMPTY directory on every boot
    # that did not also arm the coverage tracer, returned None, and left no
    # file and no line: built, green at the desk, and never executed -- the
    # exact class this slice exists to remove, re-entered from the publisher
    # side.
    #
    # The TOKEN is still not new: the join keys on the region's own boot nonce
    # (`FLLIPER_PDFLIP_XCHG_BOOT`, published four statements up), so a stale
    # manifest from a previous boot in the same evidence directory is filtered
    # rather than merged.  One new name for one fact that had none, not a
    # second answer to one that did.
    env[xchg_manifest.DIR_ENV] = str(manifest_dir or EVIDENCE_DIR)
    # B4d: THE PRICED WAVE PARTITION, published by the same channel and for a
    # sharper version of the same reason.  A rank cannot derive this one at
    # all: a chunk tag is a LAYER BAND, so the map needs the layer count of
    # EVERY PP stage and a rank holds only its own -- `derive_leg_plan` says so
    # itself and passes `{}`, which yields ONE wave.  The launcher knows every
    # stage, so it prices the per-card partition (THREE waves on this rig), and
    # MEASURED at the desk the difference is not cosmetic: the uniform
    # partition puts both images on the 5090 at once, 39454 MiB against a
    # 32607 MiB board, W71 both directions.
    #
    # ABSENT means "derive as before", not "no waves": a boot with no census
    # publishes no key and `published_waves()` answers None, under which
    # `waves_for_plan` returns the derivation byte for byte.
    if waves:
        env[weight_exchange.WAVES_ENV] = weight_exchange.publish_waves(waves)
    # #1273 S6 step 5: THE PRICED TERM ITSELF, published to the ranks.
    #
    # The seam that injects the weight bytes needs the bounce geometry, and a
    # rank may not re-derive it: the sizing inputs come from a CHECKPOINT
    # CENSUS, and a second reader of that census would be a second sizing
    # authority -- the exact defect AMENDMENT 5 retired one instance of.  So
    # the launcher publishes the scalars it PRICED WITH and the rank rebuilds
    # the term through `bounce_terms`, the one sizing function, which makes the
    # two objects identical by construction instead of by agreement.
    #
    # ABSENT means NO AUTHORITY TO INJECT, not "size it yourself": an unarmed
    # boot publishes nothing and `read_published_terms` answers None, which the
    # seam turns into a named refusal rather than a default.
    if bounce_terms is not None:
        env[xchg_bounce.ENV_BOUNCE_TERMS] = xchg_bounce.publish_terms(
            bounce_terms)
        log(f"PDFLIP-XCHG-BOUNCE PUBLISHED {xchg_bounce.publish_terms(bounce_terms)} "
            f"-- the ranks rebuild this through bounce_terms(); the ARM line "
            f"above priced the same inputs")
    log(f"PDFLIP-XCHG-SHADOW ARMED epoch={boot_nonce} path={got.get('path', '')} "
        f"sems={got.get('sems', 0)} ring=AUTHORITATIVE exchange=OBSERVER "
        f"-- the ring refills every weight byte as it does today; the exchange "
        f"runs beside it into raw cudaMalloc buffers nothing reads")
    return env

def rank_vram_display(spec: GroupSpec, log: Log, states: Optional[list] = None) -> int:
    """VRAM-Vertrag M2, DISPLAY ONLY: the ranks' RankState ``vram`` blocks
    (pdflip/vram_actual.py) as one human line per rank, and -- when re-read
    later than the W7/W10 gate -- the fresh records into
    ``groups.<G>.ranks`` of state.json. Nothing here decides; a rank without
    a block (FLLIPER_PDFLIP_VRAM_ACTUAL off) is simply not shown. Returns the
    number of blocks shown."""
    fresh = states is None
    if fresh:
        states, _bad = rank_state_mod.read_group_states(rank_state_dir_for(spec))
    from flliper.srt.pdflip import vram_actual as vram_actual_mod

    n = 0
    for s in sorted(states, key=lambda x: x.rank_key):
        if s.vram_refused is not None:
            # a finding, never a refusal: the record's other facts stand
            log(f"VRAM-ACTUAL-BLOCK-REFUSED group {spec.name} {s.rank_key}: {s.vram_refused}")
            boot_state_write(log, event=("vram_actual_block_refused", {
                "group": spec.name, "rank": s.rank_key, "code": "VRAM_ACTUAL_BLOCK_REFUSED",
                "why": s.vram_refused}))
        if s.vram is None:
            continue
        n += 1
        log(f"VRAM-ACTUAL group {spec.name} {s.rank_key}: "
            f"{vram_actual_mod.VramBlock.from_dict(s.vram).line()}")
    if fresh and n:
        boot_state_write(log, fields={f"groups.{spec.name}.ranks": [json.loads(s.to_json()) for s in states]})
    return n

def release_shared_cache(spec_d: GroupSpec, log: Log) -> Optional[dict]:
    """W98 (z30u): BOOTZEIT 3 Stufe 2b's kept ranges dropped AFTER group D is
    ready -- D's take-over is over, so the boot time does not pay for it. One
    measured line (advised vs evicted by mincore, memory.current before/after)
    plus the event ``shared_cache_release`` in state.json's events. No manifest
    = 2b was off = nothing to do, nothing said."""
    d = shared_cache_release_mod.manifest_dir_for(spec_d.log)
    if not os.path.isdir(d):
        return None
    try:
        rec = shared_cache_release_mod.release(d, read_pressure=host_ledger.read_cgroup_pressure)
    except Exception as e:  # noqa: BLE001 -- a lost release costs cache, never the boot
        log(f"{shared_cache_release_mod.MARKER} FAILED ({type(e).__name__}): {e}")
        return None
    if rec is None:
        return None
    log(shared_cache_release_mod.line(rec))
    boot_state_write(log, event=("shared_cache_release", rec))
    return rec

def register_helper(tag: str, name: str, pid: int, log: Optional[Log] = None, suffix: str = "pid") -> str:
    """Write the registry entry for helper ``name`` (a session leader, pgid ==
    pid). Never raises: a registry that cannot be written costs the arm its
    group stop for this helper, not the boot -- and the line says so.
    ``suffix`` "grp" writes a BOOT GROUP entry instead (H135b, see
    :func:`register_boot_group`); the line is identical."""
    path = f"{helper_registry_dir(tag)}/{name}.{suffix}"
    owner = os.getppid()
    line = f"{pid} {_proc_start_ticks(pid)} {owner} {_proc_start_ticks(owner)} {name}\n"
    kind = "HELPER" if suffix == "pid" else "BOOT-GROUP"
    try:
        os.makedirs(helper_registry_dir(tag), exist_ok=True)
        tmp = f"{helper_registry_dir(tag)}/.{name}.{suffix}.tmp"
        with open(tmp, "w") as f:
            f.write(line)
        os.replace(tmp, path)
        msg = f"PDFLIP-{kind} registered {name} pgid={pid} owner={owner} -> {path}"
    except OSError as e:
        msg = f"PDFLIP-{kind} NOT registered {name} pgid={pid}: {e} (the arm cannot stop it by group)"
    if log is not None:
        log(msg)
    return msg

def unregister_helper(tag: str, name: str, suffix: str = "pid") -> None:
    try:
        os.unlink(f"{helper_registry_dir(tag)}/{name}.{suffix}")
    except OSError:
        pass

def xchg_form_dormant_reserve(
    cards: List[Card],
    census_path: str,
    *,
    log: Optional[Log] = None,
    profile: Optional[str] = None,
) -> Tuple[Dict[str, int], List[str]]:
    """``({uuid: the reserve W19 grades against}, printed lines)``.

    W19 graded boot weg2xsn14 at EPOCH 0 -- before the first flip -- against a
    reserve built from ``DC_MEASURED_D_*``, which are MEASURED ON A SERVING
    BOOT (weg2ls1b2) and therefore contain none of the exchange lane's own
    device residency.  Those constants are never hand-raised (operator ruling,
    and the VRAM-corridor law), so the exchange form carries its own measured
    triple (:data:`DC_MEASURED_D_XCHG_MIB`) and this function REPORTS it beside
    the other reading of the SAME instrument, per card, on one line.

    **B4g's PRICED RESERVE AND ITS 773 / 935 / 773 MiB "UNATTRIBUTED RESIDUAL"
    ARE RETRACTED, AND THE RETRACTION IS THE REASON THIS FUNCTION IS THIS
    SHORT.**  That residual was manufactured by mixing two populations: the
    staging region (``xchg.bin``, 385 MiB) and the on-card deposit slots
    (3x32 = 96 MiB) are ``/dev/shm`` allocations, while ``PDFLIP-DC`` is
    PER-PROCESS NVML at sleep.  They were never inside the measured number, so
    subtracting them produced the gap exactly: ``1254 - 481 = 773`` and
    ``1416 - 481 = 935``.  A line that carries a host term next to a device
    reading invites that subtraction again, so it carries neither.

    SAME-INSTRUMENT ARITHMETIC, from boot weg2xsn15's per-card term table --
    the two numbers on this line are both ``PDFLIP-DC`` at sleep, one per form::

        XSN15  D @ sleep   3084 / 2590 / 2588   the EXCHANGE form (the constant)
        sn5b   D @ sleep   1668 / 1334 / 1334   the SERVING form (the census)
        delta              1416 / 1256 / 1254
        weights_draft tag  1440 / 1280 / 1280   tms_tag_bytes, resident
        unexplained          -24 /  -24 /   -26 instrument margin

    So the whole difference IS the resident ``weights_draft`` tag, which group
    D holds and the exchange never moves because the tag is out of the weights
    family.  The answer is therefore B4k (the draft head joins the family and is
    exchanged like every other layer), not a wider reserve and not a residual
    field -- and once it lands, this constant must be RE-DERIVED rather than
    hand-lowered.

    WHAT IT DOES NOT DO: change any budget.  ``main``'s ``dc_expect_d`` is the
    one consumer of ``dc_measured_d_mib`` and this function reads the same
    selector, so there is no parallel reserve object; it exists so a boot log
    says WHAT the reserve is and WHERE both numbers came from.
    """
    census = load_xchg_census_for_cards(census_path, cards, log)
    out: Dict[str, int] = {}
    lines: List[str] = []
    census_rules = xchg_census_is_reserve(profile)
    for c in cards:
        entry = census.cards.get(c.uuid)
        if entry is None:
            raise xchg_residency.PdFlipXchgResidencyUnarmable(
                f"W71 PdFlipXchgResidencyUnarmable: card {c.uuid} "
                f"(nvml{c.nvml_index} {c.name}) is not in the census "
                f"{census_path!r}, so the exchange form's dormant reserve for "
                "it cannot be priced and the serving constants must not stand "
                "in -- they contain none of the exchange lane's residency"
            )
        # PLATZTAUSCH / Nutzer 22.09. ("diesen wert nicht doppelt oder gar
        # nicht nehmen"): die Census DIESES Modells IST die Messung. Die
        # Konstante weg2xsn14 (2588/3084/2588) ist ein 27B-Wert, dessen ganzer
        # Ueberschuss der residente DFlash2-Draft war -- seit B4k schlaeft der
        # Draft mit (w132: weights_draft in pause_order). Unter Next Flash kostete
        # sie P 1203 MiB auf der 5090 und 1114-1512 MiB je 3080. Die Konstante
        # bleibt nur, wo keine Census-Zeile existiert.
        _measured = int(getattr(entry, "dormant_proc_used_mib", 0) or 0)
        if not census_rules:
            # THE 27B LINE (bfc6bd87e2 form, byte-identical line): THE ONE
            # SELECTOR, never a second copy of the triple -- this is the number
            # W19 grades against, read where every other consumer reads it.
            out[c.uuid] = dc_measured_d_mib(c, WEIGHT_SOURCE_EXCHANGE)
            lines.append(
                f"PDFLIP-XCHG-RESERVE nvml{c.nvml_index} {c.name} "
                f"reserved_mib={out[c.uuid]} source=measured:weg2xsn14 "
                f"measured_mib={int(entry.dormant_proc_used_mib)} "
                f"instrument=PDFLIP-DC-at-sleep "
                f"delta_mib={out[c.uuid] - int(entry.dormant_proc_used_mib)} "
                f"delta_is=resident-weights_draft-tag-until-B4k "
                f"census_source={entry.dormant_source[:120]}"
            )
            continue
        if _measured > 0:
            out[c.uuid] = _measured
            _src = f"census:{entry.dormant_source[:60]}"
        else:
            out[c.uuid] = dc_measured_d_mib(c, WEIGHT_SOURCE_EXCHANGE)
            _src = "measured:weg2xsn14 (27B-Konstante, keine Census-Zeile)"
        lines.append(
            f"PDFLIP-XCHG-RESERVE nvml{c.nvml_index} {c.name} "
            f"reserved_mib={out[c.uuid]} source={_src} "
            f"measured_mib={_measured} "
            f"instrument=PDFLIP-DC-at-sleep "
            f"constant_27b_mib={dc_measured_d_mib(c, WEIGHT_SOURCE_EXCHANGE)} "
            f"census_source={entry.dormant_source[:120]}"
        )
    if log is not None:
        for ln in lines:
            log(ln)
    return out, lines

def _write_state(state: BootState) -> None:
    # IPC (27B requirement 8, 28.09.): atomic -- a reader (teardown, the
    # arms, a watcher) must never see a half-written boot file.
    os.makedirs(_operator_dir(GPU_ARB), exist_ok=True)
    path = state_path(state)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(state.__dict__, f, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
