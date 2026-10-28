"""Module-level helpers moved out of the source module (refactor, behaviour-neutral): bodies are byte-identical,
the source module re-exports every name (tests and callers keep ``<module>.<name>``). Logger name unchanged."""

import os

from flliper.srt.compat_shims import operator_dir as _operator_dir
from flliper.srt.pdflip import DEFAULT_PP_ORDERED_CUT
from flliper.srt.pdflip import form as pdflip_form
from flliper.srt.pdflip import xchg_bounce
from flliper.srt.pdflip.launcher_d_leaf import D_BARLINK_BAR1_WINDOW_MIB
from flliper.srt.pdflip.launcher_d_leaf import P_BARLINK_BAR1_WINDOW_MIB
from flliper.srt.pdflip.launcher_d_p_r2_L0 import EVIDENCE_DIR
from flliper.srt.pdflip.launcher_d_p_r2_L0 import GPU_ARB
from flliper.srt.pdflip.launcher_d_p_r2_L1 import ARMING_FLOOR_MIB
from flliper.srt.pdflip.launcher_d_p_r2_L1 import BARLINK_BAR1_CAP_CYCLES
from flliper.srt.pdflip.launcher_d_p_r2_L1 import COLLECTIVE_CENSUS_INTERVAL
from flliper.srt.pdflip.launcher_d_p_r2_L1 import RANDOM_SEED
from flliper.srt.pdflip.launcher_d_p_r2_L4 import BARLINK_BUILD_WINDOW_CAP_S
from flliper.srt.pdflip.launcher_d_p_r2_L4 import MATCH_REFUSAL_CENSUS_EVERY
from flliper.srt.pdflip.launcher_d_p_r2_L4 import PP_CHAIN_RECV_STALL_S
from flliper.srt.pdflip.launcher_d_p_r2_L4 import PP_OCCUPANT_HORIZON_S
from flliper.srt.pdflip.launcher_d_set_r3_L0 import MEASURED_MS_PER_LAYER
from flliper.srt.pdflip.launcher_refuse_proc_leaf import PROFILE_QWEN27B
from flliper.srt.pdflip.launcher_xchg_vision_leaf import VISION_PLACE_AUTO
from flliper.srt.pdflip.launcher_xchg_vision_leaf import VISION_SOURCE_DISK
from flliper.srt.pdflip.launcher_xchg_vision_leaf import VISION_SOURCE_PINNED
from flliper.srt.pdflip.launcher_xchg_vision_leaf import VISION_SOURCE_RAM
import argparse

#: The step-0 metal probe's RECORD (C0, WEG2_BUILD_DECISIONS_0906 section 1p).
#: A FILE, not a number: the per-card duplex ratios C12/C13 gate on are parsed
#: out of its own measured rows and printed with this path beside them.
DUPLEX_PROBE_DEFAULT = _operator_dir(GPU_ARB, "PROBE_RING_0907.md")
MODEL_DEFAULT = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-gdncov-vocabembed"
#: FLLIPER_PDFLIP_VENV: see the rig-path block at the top (--venv still wins).
VENV_DEFAULT = os.environ.get("FLLIPER_PDFLIP_VENV") or "/spinning/htsglang-gpu/.venv"
VISION_SOURCES = (VISION_SOURCE_DISK, VISION_SOURCE_RAM, VISION_SOURCE_PINNED)
#: VISION-WEIGHTS (09.10.): `weights` = the tower borrows victim weight memory
#: of P's PP0 (pdflip.vision_victim); off until the metal proof
VISION_PLACES = (VISION_PLACE_AUTO, "kvtail", "free", "weights")
PROFILES = tuple(pdflip_form.PROFILES)

def _add_args_tree(ap: argparse.ArgumentParser) -> None:
    """Part of :func:`build_parser` (split behaviour-neutrally, refactor 28.10.): flags from ``--tree`` on."""
    ap.add_argument("--tree", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--venv", default=VENV_DEFAULT)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--profile", choices=list(PROFILES), default=PROFILE_QWEN27B,
                    help="Task #47 Scheibe 6a: checkpoint profile of the two groups; "
                         "nextflash = P PP3 + D Form A, uneven-DCP env per group, no DCP flag half.")
    # PDFLIP-FORM: --form-arch/--form-experts/--form-draft/--form-p-draft/--form-kv
    pdflip_form.add_form_arguments(ap)
    ap.add_argument(
        "--tokenizer-path", default="",
        help="27B line G2: the tokenizer directory BOTH groups load "
             "(--tokenizer-path on P and D). Default empty: nothing is added and "
             "the server reads the tokenizer from --model -- byte-identical argv. "
             "Needed for a GGUF --model whose directory holds no tokenizer files "
             "(qwen35 GGUFs load their tokenizer only from sibling files); the "
             "launcher refuses such a GGUF without it (W163) before any card is "
             "touched.",
    )
    # #1360: THE NAMED DEVIATION, flag only. The user's standing rule of
    # 2026-09-12 makes the reap mark soft -- crossing it is allowed WITH a
    # reason and a runtime latch, never silently -- and until now the launcher
    # had no way to say so: a refused arm simply stopped the boot. NO ENV TWIN
    # (#781 fossil class): one spelling, on the command line, where it lands in
    # the argv echo and in the form key.
    ap.add_argument("--host-ledger-deviation", default="", metavar="REASON",
                    help="Accept a REFUSED host-ledger fundability verdict under "
                         "a named reason. Requires --host-riegel-gib. Converts "
                         "the verdict only: no term, no ring size and no guard "
                         "is re-priced.")
    ap.add_argument("--host-riegel-gib", type=float, default=None, metavar="GIB",
                    help="The RUNTIME LATCH the deviation is accepted under, in "
                         "GiB non-reclaimable. Must be BELOW the hard bound, or "
                         "it cannot fire before the bound is crossed. Required "
                         "with --host-ledger-deviation.")
    # #1378 Stage 2 (W105): the PRIOR boot's own cited cushion_min/bounce,
    # operator-supplied like --host-riegel-gib -- there is no automatic
    # sidecar for this pair yet, so a boot that wants the cross-boot gate
    # cites its predecessor's own printed numbers (PDFLIP-HOST-LEDGER TERMS /
    # the v3 sampler's cushion_min). BOTH OR NEITHER (W105, checked inside
    # host_ledger.choose, not re-validated here): a half citation is a
    # guessed input printed as a measured one.
    ap.add_argument("--prior-cushion-min-gib", type=float, default=None,
                    metavar="GIB",
                    help="The PRIOR boot's own measured cushion minimum "
                         "(the v3 sampler's cushion_gib column, MINIMUM over "
                         "that boot). Feeds the cross-boot cushion_headroom "
                         "gate (#1378 Stage 2): ONE-TERM MODEL, measured "
                         "non-monotonic against real boots -- RED (negative "
                         "headroom) MAY refuse this boot's arm, GREEN funds "
                         "nothing by itself. Requires --prior-bounce-gib.")
    ap.add_argument("--prior-bounce-gib", type=float, default=None,
                    metavar="GIB",
                    help="The PRIOR boot's own priced xchg_bounce (the "
                         "PDFLIP-HOST-LEDGER ARM line's xchg_bounce= field). "
                         "Required with --prior-cushion-min-gib.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--force", action="store_true",
        help="PROFIL-EDITOR S1: start although the planner/launcher REFUSES the VALUES (capacity, VRAM, "
             "card count, calibration of the inventory, disk floor -- the 'value' class of pdflip/refusals.py). Every refusal passed "
             "is printed as 'FORCED-PAST <CODE> <reason>'. A forced boot writes NO records -- unless every refusal it "
             "passed is a proof gap of its own form (#227: HW-COUNT [METAL-UNPROVEN] alone, the PP-CUT crossing-width "
             "HW-UNCALIBRATED); then its records carry forced_past (line FORCED-RECORDS). NOT lifted: the occupation "
             "check (another process or window on a card), a missing or broken model/file, an unsupported "
             "architecture. HOST RAM is not refused at all any more (HOSTWARN2, user rule 06.10.): the host ledger, "
             "the preflight and the pinned-host registry print 'HOST-RAM-WARN ...' and the boot goes on, with or "
             "without --force. Container: FLLIPER_FORCE=1.")
    ap.add_argument(
        "--profile-inventory", default="",
        help="HW-GENERIC 1002: the card inventory (calibration-class labels in card order, "
             "pdflip/card_identity.py, e.g. 'RTX5090,RTX3080,RTX3080') the profile's positional "
             "vectors were measured on. Default: the inventory the profile's measured records "
             "declare (profile_records_data/<profile>.json 'inventory'). A live inventory that "
             "differs is refused BY NAME (HW-UNCALIBRATED) -- positional measurements of other "
             "cards are never borrowed.")
    ap.add_argument(
        "--cards", default="",
        help="HW-P1a 1003: the NVML indices this launch may use, e.g. '1,0' (the 5090 and one "
             "3080 of the reference rig). Default: every NVML device. The launcher reads NVML "
             "directly and ignores CUDA_VISIBLE_DEVICES (it sets that itself per group, by UUID), "
             "so a bare-metal subset is chosen here; the card ORDER still comes from the card "
             "properties (order_cards), never from this list. An index NVML does not report, or "
             "a duplicate, is refused by name (HW-CARDS).")
    ap.add_argument("--debug-hold", choices=["none", "P", "D", "both"], default="none")

def _add_args_ring_form(ap: argparse.ArgumentParser) -> None:
    """Part of :func:`build_parser` (split behaviour-neutrally, refactor 28.10.): flags from ``--ring-form`` on."""
    ap.add_argument(
        "--ring-form", choices=["auto", "none", "MAP_SHARED"], default="auto",
        help="C18/R16: the cross-process registration form for the shared host "
             "granule ring. MAP_SHARED (cudaHostRegister on a /dev/shm MAP_SHARED "
             "file) is the form the step-0 metal probe PROVED on this rig "
             "(WEG2_BUILD_DECISIONS_0906 section 1p); it is the only form built, "
             "and the memfd candidate is deleted because that probe retired it. "
             "'auto' (the default) arms it when the launch checks fund it, and "
             "otherwise prints the failing check by name (W32 / W34) and REFUSES "
             "the boot, exit 2 -- spec Amendment A1-3: the OLD flip form is "
             "infeasible on this host budget, so an un-armed ring has nothing to "
             "fall back to. 'none' never arms and therefore never launches. Never "
             "a size, never a knob: H and both spans are solved from the previous "
             "boot's own lines.",
    )
    ap.add_argument(
        "--xchg-coverage-diff",
        action="store_true",
        help="#1348 DIAGNOSTIC BOOT ONLY -- record, per rank, which lines of "
             "the exchange lane's modules this boot actually executed, and "
             "print the complement afterwards with "
             "scripts/pdflip/lane_coverage_diff.py. The complement is the list "
             "of seams nothing reached, i.e. the walls not yet found. "
             "OVERHEAD: coverage.py line tracing, which its own numbers put "
             "at 2x-10x ON THE TRACED MODULES ONLY (11 files, allowlisted -- "
             "not the whole tree); the measured per-leg cost of the dump "
             "itself is printed on every PDFLIP-COVERAGE DUMP line, never "
             "estimated. Default OFF and byte-identical off: the environment "
             "variable it publishes is POPPED when this flag is absent. Do "
             "NOT combine with FLLIPER_SEAM_COVERAGE_DIR -- two coverage "
             "tracers in one process measure neither, and the arm refuses by "
             "name if it sees it.")
    ap.add_argument("--evidence-dir", default=EVIDENCE_DIR,
                    help="where ring_table reads the previous boot's logs from")
    ap.add_argument("--duplex-probe", default=DUPLEX_PROBE_DEFAULT,
                    help="C12/C13 + A1-4: the step-0 probe RECORD the per-card PCIe "
                         "duplex ratio is solved from. A measured FILE, never a "
                         "number: the launcher parses its 'PROBE granule ... ratio=' "
                         "rows per card UUID, prints one PDFLIP-PCIE-DUPLEX line per "
                         "card with the R17 gate beside it, and publishes the table "
                         "to the ranks as FLLIPER_PDFLIP_PCIE_DUPLEX. An unreadable or "
                         "rowless file means NO card splits its lock key")
    ap.add_argument(
        "--pdflip-seam-digest", action="store_true",
        help="#1350: arm the SEAM GRADER on both groups. It answers one "
             "question -- 'did the round trip bring MY bytes back' -- by "
             "hashing this rank's own pieces at the last instant before the "
             "weights family is paused and again at the first instant after it "
             "has landed, and refusing by name (W90 PdFlipSeamDigestMismatch) "
             "when they differ. Rank-local: no peer, no collective, no byte "
             "moved between cards, and both moments AWAKE -- at flip time the "
             "sleeping group holds no weight bytes in VRAM at all, so a "
             "cutover-time compare is impossible rather than merely expensive. "
             "Keyed by the PLACEMENT identity the plan already publishes "
             "(param_name, class, tag, card, storage extents), so a mismatch "
             "NAMES the pieces that moved -- and so a change of arena, pointer "
             "or pitch, which the exchange makes by design, is not mistaken "
             "for a change of content. "
             "DEFAULT OFF and meant for an S6I INSTRUMENT boot: the hash costs "
             "a full read of the shard on every flip leg, so arming it on a "
             "serving boot is a flip-cost regression. Published to the ranks "
             "as FLLIPER_PDFLIP_SEAM_DIGEST, which is launcher OUTPUT and is "
             "POPPED when this flag is absent, so an inherited shell value "
             "cannot arm it. LIMIT: a green digest says the bytes came back, "
             "never that the layout was verified.")
    ap.add_argument(
        "--pdflip-disable-hicache", action="store_true",
        help="#1386: for the ZIELFRAGE MINIMALFORM (two manual flips + "
             "shadow compare, zero requests served) -- drop the "
             "HiCache/Mamba-anchor host tier that exists to reuse KV pages "
             "ACROSS requests, which a boot that serves none cannot do. "
             "ONE SWITCH, ONE TRUTH: this bool reaches BOTH "
             "`host_ledger.charge_terms` (anchors_gib/rings_gib/overhead_gib "
             "price at 0.00) AND `common_flags` (no --enable-hierarchical-"
             "cache, no --hicache-* flag on either group's argv) -- never "
             "two readers of two copies of the same decision, which is the "
             "class of defect that under-priced #1385's bounce buffer against "
             "the allocator xsn31/4 and xsn31/5 actually ran. "
             "DEFAULT OFF, and OFF is byte-identical to every boot before "
             "this flag existed: it is for the Minimalform's own use, not "
             "for the serving basis, and it is NOT a smaller "
             "--hicache-size/--hicache-mamba-host-mib -- those flags are "
             "omitted outright, not passed as 0, so no HiCache code path is "
             "entered at all.")

def _add_args_xchg_bounce_depth(ap: argparse.ArgumentParser) -> None:
    """Part of :func:`build_parser` (split behaviour-neutrally, refactor 28.10.): flags from ``--xchg-bounce-depth`` on."""
    ap.add_argument(
        # #1358 THE ONE HOST LEVER FOR xsn28, AS A FLAG AND NOT A CONSTANT.
        # weg2xsn27 died in `shm,pinned`, and the #1358 reader attributed the
        # WHOLE 10.565 GiB of that region to five slot holders x 2.113 GiB from
        # one emit site. The buffer is `widest x assemble_slots(depth)` and
        # `slots = depth + 1` under shadow, so one step of depth is one widest
        # layer per holder: 721.3 MiB x 5 = 3.52 GiB of non-reclaimable, at the
        # sleep leg, in the region the boot died in.
        #
        # DEFAULT 2 IS BYTE-IDENTICAL to every boot so far -- the flag exposes
        # the constant, it does not move it. The value is PUBLISHED with the
        # rest of the bounce terms, so P, D and the host ledger read ONE
        # source: a second spelling of the depth is how `2.16 GiB` came to
        # have three causes (#1361 [23a]).
        #
        # THE COST IS NOT PRICED HERE, and that is deliberate: depth=1 leaves
        # one pipeline stage fewer, and nothing on this rig has measured what
        # that does to flip duration. Operator order 2026-09-13: measured at
        # the metal, never estimated.
        "--xchg-bounce-depth", type=int,
        default=xchg_bounce.ASSEMBLE_DEPTH_DEFAULT,
        help="#1358: depth-slots of the assemble buffer (default 2, "
             "byte-identical). `assemble_slots` adds ONE more under "
             "inject_mode=shadow, so the buffer is widest_layer x (depth+1) "
             "there. depth=1 saves one widest layer per slot holder -- "
             "measured on weg2xsn27: 721.3 MiB x 5 holders = 3.52 GiB of "
             "non-reclaimable at the sleep leg. The value is published to the "
             "ranks and read back by the host ledger, so all three agree.",
    )
    ap.add_argument(
        # #1385 (Wand 11b): THE LANE-COUNT LEVER, as a flag and not a
        # constant, exactly the way #1358 exposed depth. xsn31/3 measured
        # `lanes=5 source=measured` and `xchg_bounce=15.75` GiB (5 lanes x
        # 3.00 GiB + 0.75 GiB staging) at the sleep leg the boot dies in --
        # 13.50 GiB more non-reclaimable than xsn31/2's 78.90 GiB peak, which
        # is exactly the growth the cushion floor (1.50 GiB) has no room for
        # (cushion measured 0.20, W98 PdFlipHostRateLatched). ALL FIVE lanes are
        # pair/diagonal buffers of the SAME boot, created by different ranks
        # (host_ledger.xchg_lane_count is BOOT-WIDE) -- this cap does not
        # change which lanes exist, only how many of their buffers may be
        # pinned in tmpfs AT ONCE; the rest are SERIALISED (named on
        # `PDFLIP-XCHG-LANES-CONCURRENT`, never a silent flip-time cost).
        #
        # DEFAULT UNSET (None) IS BYTE-IDENTICAL: `None` reaches
        # `xchg_bounce.resolve_lanes_concurrent` as "the flag was never
        # given" and resolves to the 0 sentinel, which prices every measured
        # lane exactly as every boot before this flag did. `0` is NOT the
        # default -- an operator who explicitly types
        # `--xchg-lanes-concurrent 0` is refused (W103) rather than silently
        # handed the "uncapped" behaviour their argv says the opposite of.
        "--xchg-lanes-concurrent", type=int, default=None,
        help="#1385: cap on how many of this boot's measured exchange lanes "
             "(host_ledger.xchg_lane_count, boot-wide) may hold their "
             "assemble buffer in tmpfs at once. Unset (default) is "
             "byte-identical -- every lane prices, exactly as before this "
             "flag. `--xchg-lanes-concurrent 2` on a 5-lane cut charges "
             "min(5, 2) buffers instead of 5 (measured on xsn31/3's "
             "checkpoint: 15.75 GiB falls to 6.75 GiB) and serialises the "
             "remaining 3 -- named and counted on PDFLIP-XCHG-LANES-CONCURRENT, "
             "because a lane that waits for an earlier one's collector to "
             "drain costs flip time the size line alone cannot show. A value "
             "below 1 is refused (W103 PdFlipXchgLanesConcurrentInvalid), "
             "never silently read as 'uncapped'.",
    )

def _add_args_xchg_band_credit(ap: argparse.ArgumentParser) -> None:
    """Part of :func:`build_parser` (split behaviour-neutrally, refactor 28.10.): flags from ``--xchg-band-credit`` on."""
    ap.add_argument(
        # #1397 OPTION 3 (DESIGN_option3_band_credit_0914.md section 9):
        # a CROSS lane's collector gate names a DIFFERENT rank than a
        # DIAGONAL lane's does (the design doc's own "why cross but never
        # diagonal" finding), so a cross lane may reuse a slot WITHIN one
        # tag under a per-band drain credit instead of sizing the whole
        # tag in -- the diagonal share stays at the #1374 floor,
        # unconditionally, because relaxing it would re-enter boot
        # weg2xsn30's circular wait.
        #
        # ARMING REDUCES TO THIS ONE BOOLEAN (design doc section 9.2): the
        # cross-lane COUNT (`n_cross_lanes`) is never a second flag here --
        # `xchg_bounce.resolve_cross_lanes` derives a SAFE, structural lower
        # bound from `n_lanes` (measured, #1358) and
        # `weight_exchange_region.N_CARDS` (a pigeonhole argument: at most
        # N_CARDS of any `n_lanes` active lanes can be diagonal, so the
        # remainder is a hard floor on how many must be cross, regardless of
        # WHICH specific lanes those are). A caller with a real, measured
        # split may still override it -- `bounce_terms(n_cross_lanes=...)` --
        # but no CLI flag exposes that override: this switch alone is the
        # whole arming surface, on purpose.
        #
        # DEFAULT OFF IS BYTE-IDENTICAL: `band_credit=False` makes
        # `n_cross_lanes_priced` 0 by construction (`BounceTerms.
        # n_cross_lanes_priced`), so `total_bytes` takes its pre-#1397
        # branch untouched -- no existing boot's price moves by a byte.
        #
        # NEEDS `max_tag_bytes > 0` (Option 1 / #1374) OR REFUSES: the
        # cross-lane price is the small ONCARD slot unit, which
        # `run_bounce_leg` only ever allocates under Option 1 -- pricing the
        # small number while Option 1 is absent would allocate the WIDEST-
        # LAYER-sized slot instead, the #1385-round-3 mismatch class
        # (measured 5.6x on boot weg2xsn31/4). `bounce_terms` itself refuses
        # this combination at ARM time, before either group starts.
        "--xchg-band-credit", action="store_true", default=False,
        help="#1397 (Option 3): let a CROSS lane (a directed card pair, "
             "never the diagonal) reuse a slot within one tag under a "
             "per-band drain credit, instead of sizing the whole tag into "
             "the buffer -- the diagonal share is unaffected. Default off, "
             "byte-identical. On a 5-lane cut with weight_exchange_region."
             "N_CARDS=3, the safe auto-derived cross count is 2 "
             "(xchg_bounce.resolve_cross_lanes); at slot=128 MiB/depth=2 "
             "this falls the pinned shm from 15.75 GiB (5 x 3.00 diag + "
             "0.75 staging) to 10.5 GiB (3 x 3.00 diag + 2 x 384 MiB cross "
             "+ 0.75 staging). Needs --xchg-bounce-oncard-slot-mib's Option "
             "1 (max_tag_bytes measured > 0) to price safely -- refuses "
             "(ValueError, xchg_bounce.bounce_terms) rather than silently "
             "pricing the small number while the leg allocates the large "
             "widest-layer one.",
    )
    ap.add_argument(
        "--pdflip-vision-source", choices=list(VISION_SOURCES), default=VISION_SOURCE_DISK,
        help="H125: where the TRANSIENT tower's bytes come from (only with "
             "--pdflip-vision transient). `disk` (default): the checkpoint shard, "
             "O_DIRECT, per image request. `ram`: P's PP0 stages the tower extent "
             "once into a tmpfs image (NF 856 MiB, 27B 879 MiB of host RAM for the "
             "life of the boot) and every stage reads it from RAM.",
    )
    ap.add_argument(
        "--pdflip-vision-place", choices=list(VISION_PLACES), default=VISION_PLACE_AUTO,
        help="H125: where the TRANSIENT tower sits on PP0's card (only with "
             "--pdflip-vision transient). `auto` (default): the KV tail when it is "
             "wholly free (the phase start), else the card's free VRAM, else a "
             "named refusal. `kvtail`: the 27B stage exactly. `free`: always the "
             "card's free VRAM. `weights` (VISION-WEIGHTS, not yet a profile default): "
             "the tower's parameters are views on victim WEIGHT memory of PP0 "
             "(27B flip: MLP storages; dual: only the hull parts D does not share); "
             "the victim bytes go to a host image (freed after every image) and come "
             "back with a device checksum before the admission -- a mismatch stops "
             "the group (W110c). No KV page, no free VRAM, always synchronous. "
             "`kvtail` or `auto` TOLD on the command line together with --dual-layout "
             "is refused (W111): the dual P pool is born trimmed to 0 tokens, so the "
             "dual's place is `weights`.",
    )

def _add_args_ring_table_boot(ap: argparse.ArgumentParser) -> None:
    """Part of :func:`build_parser` (split behaviour-neutrally, refactor 28.10.): flags from ``--ring-table-boot`` on."""
    ap.add_argument("--ring-table-boot", default="",
                    help="pin the ring table to ONE boot instead of the newest usable "
                         "one. Matched as a SUBSTRING of the log stem, so the boot TAG "
                         "('weg2zr2') is enough; it must match exactly one boot, and a "
                         "pin that matches none or several returns the R22 reason "
                         "rather than raising an OSError -- and since A1-3 leaves no "
                         "feasible form to fall back to, that reason is a named "
                         "refusal (W20) and exit 2, verified on the rig")
    # -- group P: the layer cut ------------------------------------------
    ap.add_argument(
        "--pp-stage-ratio", default=None,
        help="OVERRIDE the solved layer cut for group P (e.g. '32,18,14'). "
             "Passing it is announced as 'PINNED (user override)' in the "
             "PP-CUT provenance line and priced on the same two axes as the "
             "solved cut. Unset = the solver decides. Next-Flash arm: this is "
             "the ONE cut variable, PP_RATIO in arm_fnFL2_long.sh (default "
             "29,11,8), with PP_ATTN_RATIO = the attention count that cut "
             "lands on (fnFL2 H57). The 'PP-CUT RATEN-SCHNITT' line prints the "
             "cut the measured stage rates recommend for the CURRENT power "
             "limit (or 'keine Raten fuer <Limit>, Schnitt bleibt gepinnt'); "
             "it is never applied.",
    )
    ap.add_argument(
        "--pp-attn-stage-ratio", default=None,
        help="OVERRIDE the solved full-attention split for group P (e.g. "
             "'8,4,4'). Same PINNED provenance. Unset = the attention axis is "
             "resolved for the chosen layer cut by pp_cut.best_attention_split.",
    )

def _add_args_pp_solve_pool_floor(ap: argparse.ArgumentParser) -> None:
    """Part of :func:`build_parser` (split behaviour-neutrally, refactor 28.10.): flags from ``--pp-solve-pool-floor`` on."""
    ap.add_argument(
        "--pp-solve-pool-floor", type=int, default=None,
        help="#1286b. A HARD LOWER BOUND, in WORLD KV TOKENS, on the priced "
             "pool of whatever --pp-solve-objective ships. UNSET = the SHIPPED "
             "DEFAULT, which since the user order of "
             "2026-09-09 -- verbatim: '39,13,12 mit bs2 im pp layout soll "
             "standard werden vorerst' -- is the ORDERED CUT "
             f"{','.join(str(n) for n in DEFAULT_PP_ORDERED_CUT)} "
             "(DEFAULT_PP_ORDERED_CUT): the solver READS THE FLOOR OFF THIS "
             "BOOT'S OWN FRONTIER as that cut's priced pool (#1305), so "
             "'fastest above F' selects it exactly and the floor cannot drift "
             "off a moved frontier the way a typed constant did on boot "
             "weg2sn5pre; a cut that is not on the frontier is a W67 refusal, "
             "never a neighbour -- the cut stays SOLVED, "
             "this only bounds the pool. PASS 0 to turn the floor OFF and get "
             "the pre-#1286b behaviour exactly: the objective then ranks over "
             "the same feasible set it always did and nothing moves. SET to a "
             "positive number = the objective still ranks, but only over the "
             "cuts whose PRICED "
             "world pool clears this number -- so 'makespan' becomes 'the "
             "FASTEST cut that still holds N tokens'. It is a CONSTRAINT ON "
             "the one objective knob, not a second objective: the standing law "
             "is that trades live behind one knob, and 'fastest above N' is "
             "the same ranking over a smaller set. WHY IT EXISTS: #1286 "
             "repriced every candidate against the boot's own sizing formula "
             "and the makespan winner came out at 304,946 tokens against the "
             "incumbent's 715,089 -- 43 %% of the capacity for the speed cut -- "
             "and before this flag the only way to put a bound under that was "
             "to pin a cut by hand, which is the thing the solver replaces. "
             "The pool it is compared against is the SAME number "
             "PP-POOL-JOIN publishes and group P then sizes (within 0.1 %% on "
             "both reference boots), NOT a second estimate. NO SILENT "
             "FALLBACK: if no servable cut clears it the launch is a W40 "
             "REFUSAL that prints the frontier -- the fastest cuts that DO "
             "clear it and the best pool anywhere in the field -- so the next "
             "move is a number, not a re-solve by hand. Distinct from "
             "--max-kv-per-request, which is the PHYSICAL floor (one "
             "full-context prompt must fit); both are checked and each refusal "
             "names its own flag. Read the curve on the PP-CUT FRONTIER: line "
             "of any boot to choose a value.",
    )
    ap.add_argument(
        "--pp-cut-measured-ms-per-layer", default=MEASURED_MS_PER_LAYER,
        help=f"Per-stage prefill ms per layer for the cost model. Default "
             f"'{MEASURED_MS_PER_LAYER}' is MEASURED on boot bsscale "
             f"(/spinning/gpu-arb/weg2/BSSCALE_0907.md, tip 37c884b0b0, "
             f"per full 4096-token chunk at bs6: PP0 259.1 ms/32 layers, PP1 "
             f"632.9/18, PP2 470.2/14). Used as the ANCHOR when the measured "
             f"card-rate library supplies the per-card ratios.",
    )
    ap.add_argument(
        "--pp-cut-arming-floor-mib", type=float, default=ARMING_FLOOR_MIB,
        help=f"Per-rank arming floor subtracted before KV in the pool model. "
             f"Default {ARMING_FLOOR_MIB} = the top of the rig's VRAM "
             f"corridor (819-1229 MiB NVML-free per card under load). "
             f"pp_cut.PhasePoolModel names the gap this stands in for: the "
             f"solved per-layout floor only exists for a layout that has "
             f"booted, and the proxy carries about +-500 MiB.",
    )
    ap.add_argument(
        "--p-microbatch-depth", type=int, default=None,
        help="OVERRIDE group P's solved --pp-async-batch-depth (#692). Unset = "
             "DERIVED from the previous boot's own PP-BUBBLE line: a stage busy "
             "forward/(gap+forward) of the time needs ceil(1/(1-stall_share)) "
             "passes in flight to cover its gap, and the flag counts the passes "
             "BEYOND the one being forwarded. Priced against P's KV pool -- one "
             "extra pass costs one chunk of KV rows plus one crossing "
             "activation frame per stage -- and refused as W42 rather than "
             "lowered when the pool cannot fund it. Passing it is announced as "
             "PINNED and priced on the same axis.",
    )
    ap.add_argument(
        "--p-host-overlap", action="store_true", default=True,
        help="Group P: let each stage's host work for the NEXT prefill chunk run "
             "while the forward it just launched computes (managers/"
             "pdflip_p_overlap.py). Adds FLLIPER_PDFLIP_P_HOST_OVERLAP=1 and "
             "upstream's FLLIPER_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM=1 to group P "
             "only: a middle chunk exchanges no output (the last rank stops "
             "waiting on its own forward), the last rank fences its schedule "
             "stream device-side instead, and the chunked-prefill HiCache "
             "publish moves from the plan to right after the next launch. "
             "Measured reason (weg2xsn420, 4x98k): PP2 pass 873 ms vs 733 "
             "gpu-ms, ~95-140 ms card idle per 4096 chunk, flat over depth. "
             "DEFAULT ON since S43 (NF rc12 1018: PP0 duty 57 %% in its busy "
             "runs, 80 s of PP0 commit_ms on middle-chunk output exchanges "
             "alone -- the output ring, not L3; user rule 14.10. new features "
             "on); an UNSET flag takes the booted row (p_host_overlap: qwen27b "
             "on since LS6 30.09.). --no-p-host-overlap = the old argv/env, "
             "byte-identical.",
    )

def _add_args_random_seed(ap: argparse.ArgumentParser) -> None:
    """Part of :func:`build_parser` (split behaviour-neutrally, refactor 28.10.): flags from ``--random-seed`` on."""
    # -- #1235: the literals and env gates that used to be nobody's ------
    ap.add_argument(
        "--random-seed", type=int, default=RANDOM_SEED,
        help=f"Sampling seed for BOTH groups. Default {RANDOM_SEED} is "
             f"ARBITRARY AND FIXED, and that is the whole provenance -- it is "
             f"not measured and not solved, and it is a flag so that no reader "
             f"has to wonder which it was. What it buys is that two boots of "
             f"one tip sample identically, so a decode difference between them "
             f"is a difference in the code.",
    )
    ap.add_argument(
        "--barlink-bar1-cap-cycles", type=int, default=BARLINK_BAR1_CAP_CYCLES,
        help=f"Ceiling on barlink's build-time cycle budget, both groups. "
             f"Default {BARLINK_BAR1_CAP_CYCLES} is large BY CONSTRUCTION: it "
             f"is a ceiling, not a target, and the gate that actually binds on "
             f"this rig is the BAR1 aperture priced at "
             f"--barlink-bar1-window-mib (#1234 C1), not this number.",
    )
    ap.add_argument(
        "--collective-census-interval", type=int, default=COLLECTIVE_CENSUS_INTERVAL,
        help=f"How often the collective census prints, both groups. Default "
             f"{COLLECTIVE_CENSUS_INTERVAL} is a STRIDE and not a measurement: "
             f"larger is quieter, smaller costs log volume, and nothing about "
             f"the boot depends on it except how much of it a reader sees.",
    )
    ap.add_argument(
        "--d-barlink-bar1-window-mib", default=D_BARLINK_BAR1_WINDOW_MIB,
        help=f"Group D's BAR1 windows. Default {D_BARLINK_BAR1_WINDOW_MIB!r} is the N = 3 / 256 MiB "
             f"BAR measurement (#1234 C1, see argv_d); on another card count or BAR1 size the "
             f"launcher derives it (HW-P1c, pdflip/bar1_windows.py) unless this flag names one.",
    )
    ap.add_argument(
        "--p-barlink-bar1-window-mib", default=P_BARLINK_BAR1_WINDOW_MIB,
        help=f"Group P's BAR1 windows. Default {P_BARLINK_BAR1_WINDOW_MIB!r} "
             f"shares group D's provenance from the other side (#1234 C1): the "
             f"two groups' windows are sized to fit TOGETHER -- P 24+96 and D "
             f"16+32+40 = 208 of the 224 MiB usable per 3080, measured Used "
             f"224/256 including the RM carve-out. That sentence lived only in "
             f"the declared-deviations list while D's window carried the whole "
             f"derivation in a comment.",
    )
    ap.add_argument(
        "--barlink-build-window-cap-s", type=int, default=BARLINK_BUILD_WINDOW_CAP_S,
        help=f"FLLIPER_BARLINK_BUILD_WINDOW_CAP_S, both groups. Default "
             f"{BARLINK_BUILD_WINDOW_CAP_S} s. A TIMEOUT, not a measurement. "
             f"It is now launcher OUTPUT: it used to be written as "
             f"env.get(KEY, default), so an inherited export from the "
             f"launcher's own shell won silently and the boot ran a number no "
             f"flag, record or log line ever named (the R19 shape).",
    )
    ap.add_argument(
        "--pp-chain-recv-stall-s", type=int, default=PP_CHAIN_RECV_STALL_S,
        help=f"FLLIPER_PP_CHAIN_RECV_STALL_S. Default {PP_CHAIN_RECV_STALL_S} s, "
             f"a timeout, launcher OUTPUT for the same reason as above.",
    )
    ap.add_argument(
        "--pp-occupant-horizon-s", type=int, default=PP_OCCUPANT_HORIZON_S,
        help=f"FLLIPER_PP_OCCUPANT_HORIZON_S. Default {PP_OCCUPANT_HORIZON_S} s, "
             f"a timeout, launcher OUTPUT for the same reason as above.",
    )
    ap.add_argument(
        "--match-refusal-census-every", type=int, default=MATCH_REFUSAL_CENSUS_EVERY,
        help=f"FLLIPER_MATCH_REFUSAL_CENSUS_EVERY. Default "
             f"{MATCH_REFUSAL_CENSUS_EVERY} is a census STRIDE. Launcher "
             f"OUTPUT for the same reason as above -- and note the denominator "
             f"law applies to what it emits: a rate-limited counter's zero is "
             f"not a zero.",
    )
    ap.add_argument(
        "--no-arming-floor-solved", dest="arming_floor_solved",
        action="store_false",
        help="Drop FLLIPER_ARMING_FLOOR_SOLVED=1 (default ON, unchanged "
             "shipped behaviour). A boolean env gate with no flag was "
             "indistinguishable from a hard-coded constant (#1235).",
    )
    ap.add_argument(
        "--no-hicache-bigram-keys", dest="hicache_bigram_keys",
        action="store_false",
        help="Drop FLLIPER_HICACHE_BIGRAM_KEYS=1 (default ON). MEASURED reason "
             "for ON, boot weg2ls3b3 (#1233): group D (NEXTN) keys the store "
             "by BIGRAM page hashes and group P (no spec) by UNIGRAM, so the "
             "two built disjoint chains for the same prompt and D never read "
             "P's pages. One key scheme for both groups; a no-op on D.",
    )
