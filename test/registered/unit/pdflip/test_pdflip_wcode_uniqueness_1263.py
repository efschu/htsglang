# SPDX-License-Identifier: Apache-2.0
"""One W-code, one exception name -- enumerated, not remembered.

THE CLASS, three instances in one day and every one found by a READER:

* W31 named both this rig's serving-path re-route (``PdFlipTpPrefillExceeded``)
  and ``PdFlipHostRingExhausted``, the host-ring exhaustion that killed boot
  weg2tr2. A census that greps ``W31`` reports one number for a fatal
  exhaustion and a routing event -- how a postmortem merges a killer into
  noise. Train fix 5 renumbered it.
* It renumbered it to W47, which the #1235 argv slice had ALREADY assigned to
  ``PdFlipTpObjectiveRefused``. One collision traded for another, because the
  new number was picked rather than enumerated.
* Renumbered again here to W50, this time by enumerating the used set.

A third hand-picked number would repeat it, so the enumeration is the guard
rather than the fix: this file recomputes the census on every run.

FOURTH INSTANCE, 2026-09-09 (#1257), and the reason the SCAN is hardened here
rather than only the label fixed: the #1257 corridor pass claimed ``W52``,
which #1290 already held at the base commit -- and this census reported 7/0/0
anyway. Its pattern was ``W<nn>`` + WHITESPACE + ``PdFlip<Name>``, and #1290
writes the code in the two forms that have no whitespace after it:

* CONCATENATED, ``front.py:561``: the name is built from the bare code plus a
  separate marker constant, so a QUOTE follows the code, not a space.
* COUNTER KEY, ``front.py:1981``/``:2603``: ``W52_PdFlipNoServiceableRoute``, an
  UNDERSCORE after the code.

Both are read by a human grepping a boot log for ``W52`` and by neither of the
regexes that were supposed to prevent the clash -- and ``front.py:557``'s own
"W52 is free" comment was written from this same blind instrument, which is
how the wrong number looked enumerated. So the census now reads all three
forms and RESOLVES the concatenated one through the marker constant; an
operand it cannot resolve is a loud failure, never a silent miss.

Hardening the scan also surfaced one PRE-EXISTING collision that was invisible
before (``W22``: the host-watermark breach vs. the span-unknown counter key,
both at the base commit) -- recorded in ``KNOWN_COLLISIONS`` below with its
locations, not fixed here.

WHY A CENSUS AND NOT A REGISTRY. There is no table of W-codes to keep in sync
-- the code IS the message text -- so a registry would be second bookkeeping
beside the refusals themselves, and would drift exactly the way the front's
docstring drifted from its own constant. Scanning the messages is the one
reading that cannot go stale.

WHAT IS ASSERTED, and what is deliberately NOT:

* NEW collisions fail. That is the whole point.
* The FOUR pre-existing collisions are pinned as a named, dated set rather
  than silently allowed. Pinning them exactly means fixing one also fails
  here, which forces the list to shrink deliberately instead of rotting into
  a permanent allow-list. They are NOT fixed on this branch: it is the
  serving-boot base, and renumbering four more codes across five modules and
  their tests is a bigger diff than a label defect justifies before a boot.
  Each carries the two names and where they live so the next pass has the
  work already done.
"""

import os
import pathlib
import re
import unittest
from collections import defaultdict

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.test.test_utils import CustomTestCase

#: Where a Weg-2 refusal can be raised, logged or re-routed from. Enumerated
#: from the modules that actually carry ``W<nn> PdFlip<Name>`` text today; a new
#: module is picked up by the directory walks, not by an edit here.
ROOTS = (
    "python/flliper/srt/pdflip",
    "python/flliper/srt/managers",
    "python/flliper/srt/mem_cache",
    # serve-next4 train (2026-09-09), refuter A finding 7 / refuter B R7: these
    # two were OUTSIDE the walk, so a collision confined to them was invisible
    # and the tree's answer for ``W18 PdFlipPhaseFootprintCollision``
    # (``mem_ledger/activation_probe.py``) and ``W40 PdFlipPPCutRefused``
    # (``planner/pp_cut_launch.py``) rested on a manual grep. Adding them
    # changes no verdict at this tip -- the collision set is byte-identical --
    # which is the point: the guard now DEFENDS what the grep asserted.
    "python/flliper/srt/mem_ledger",
    "python/flliper/srt/planner",
    "scripts/pdflip",
)
FILES = ("python/flliper/srt/server_args.py",)

#: THE THREE FORMS A W-CODE IS WRITTEN IN. Reading only the first is what let
#: #1257 claim a taken number while this file reported no collision.
#:
#: 1. PLAIN, in a message or a docstring: ``W12 PdFlipSomething``. The letter
#:    suffix (``W11b``, ``W40b``) is part of the code.
ASSIGNMENT = re.compile(r"\b(W\d{1,2}[a-z]?)\s+(PdFlip[A-Za-z0-9_]+)")
#: 2. COUNTER KEY: ``self.counters["W28_PdFlipLeg2Unpriced"]``. The exception
#:    name is CamelCase, so a trailing ``_lowercase`` part is a sub-key
#:    ("..._stream_served") and not part of the name -- hence no underscore in
#:    the captured group, which truncates the suffix instead of inventing a
#:    second holder for the code.
COUNTER_KEY = re.compile(r"\b(W\d{1,2}[a-z]?)_(PdFlip[A-Za-z0-9]+)")
#: 3. CONCATENATED: ``NAME = "W50 " + X_REFUSAL_MARKER``. The code's holder is
#:    then the VALUE of that operand, so the scan has to resolve it; capturing
#:    the identifier itself would report the variable name as the holder and
#:    turn every such site into a phantom collision.
CONCAT = re.compile(r"[\"'](W\d{1,2}[a-z]?) [\"']\s*\+\s*([A-Za-z_][A-Za-z0-9_]*)")
#: ``X_REFUSAL_MARKER = "PdFlipTpPrefillExceeded"`` -- resolves form 3's operand.
MARKER_DEF = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*[\"'](PdFlip[A-Za-z0-9]+)[\"']\s*$"
)
#: 4. SIGNATURE DEFAULT: ``code: str = "W88",`` with ``name: str =
#:    "PdFlipStoreLoadNotProgressing",`` on the NEXT line -- one terminal exit
#:    parameterised by the code, so neither half is on the same line as the
#:    other and no per-line pattern above can see it.
#:
#:    THIS FORM COST A RENUMBER (operator catch 2026-09-11). Boot seat 2's
#:    #1332 B1b enumerated the "free" codes with a pattern that required an
#:    exception NAME beside the number, read W88 as free, and shipped
#:    ``W88`` for the exception ``PdFlipXchgWidestLayerUnreadable`` into a tip
#:    where  -- written with a word between the two so this very comment
#:    cannot become a phantom holder if ROOTS ever grows to cover tests --
#:    ``scheduler.py:6314`` already held W88 for #1324's store-read standstill.
#:    This census reported no collision, because the holder is a STRING DEFAULT
#:    and not an assignment -- the same blind-spot CLASS as #1257 (form 1 only)
#:    and #1263 (counter keys), one form further out. A textual, word-bounded
#:    grep over ``python/flliper/srt`` found 19 hits for W88; this file found 0.
SIGNATURE_CODE = re.compile(
    r'\bcode\s*(?::\s*str\s*)?=\s*[\"\'](W\d{1,2}[a-z]?)[\"\']')
SIGNATURE_NAME = re.compile(
    r'\bname\s*(?::\s*str\s*)?=\s*[\"\'](PdFlip[A-Za-z0-9]+)[\"\']')
#: How far a signature's ``name=`` may sit from its ``code=``. Four lines is
#: the observed distance (adjacent at scheduler.py:6314/6315) plus slack for a
#: comment between them; a wider window would start pairing a code with the
#: NEXT refusal's name, which invents a holder instead of finding one.
SIGNATURE_WINDOW = 4
#: Written into the census when a signature default's ``name=`` partner is not
#: found inside the window. Asserted empty, exactly like the concat operand: a
#: code whose holder cannot be resolved must FAIL rather than be dropped, which
#: is the whole lesson of the three forms before it.
UNRESOLVED_SIGNATURE = "UNRESOLVED-SIGNATURE-NAME"

#: Written into the census when form 3's operand resolves to nothing in its own
#: file. Asserted to be empty: an unresolvable operand is the blind spot coming
#: back, and it has to fail rather than quietly drop the code.
UNRESOLVED = "UNRESOLVED-CONCAT-OPERAND"

#: THE KNOWN COLLISIONS, with both holders. Deferred, not accepted: this
#: branch renumbers W47 (2026-09-08) and W52 (2026-09-09, #1257) only.
#:
#: W22 joined this list on 2026-09-09 WITHOUT anything changing in the source:
#: it is pre-existing at the base commit ``80de2d31d1`` and was simply
#: invisible to the un-hardened scan, which could not read a counter key.
#: ``W22 PdFlipHostWatermarkBreached`` at ``front.py:1805`` and
#: ``host_ledger.py:688`` vs. the counter ``W22_PdFlipSpanUnknownPricedFull`` at
#: ``front.py:1938``. Recorded here rather than renumbered, for the reason the
#: docstring gives for the other four: this is the serving-boot base.
KNOWN_COLLISIONS = {
    "W10": {"PdFlipCanonicalPageMissing", "PdFlipDrafterIdentityMismatch"},
    "W22": {"PdFlipHostWatermarkBreached", "PdFlipSpanUnknownPricedFull"},
    "W35": {"PdFlipVramCreditRefused", "PdFlipXReQueueLoop"},
    "W36": {"PdFlipAdmitterBarrierExpired", "PdFlipDuplexDecisionRefused"},
    "W46": {"PdFlipPPSplitMapMismatch", "PdFlipTokenVectorRefused"},
}


def _repo_root() -> str:
    here = os.path.abspath(__file__)
    for _ in range(8):
        here = os.path.dirname(here)
        if os.path.isdir(os.path.join(here, "python", "flliper", "srt", "pdflip")):
            return here
    raise AssertionError("could not locate the repo root from this test file")


def census():
    """``{code: {exception_name: [file:line, ...]}}`` over the whole surface.

    Reads the SOURCE, never the imported module: a refusal inside an ``f``
    string is only text at runtime, and half of these are logged rather than
    raised, so there is no object to enumerate.

    Reads all THREE forms (see the patterns above). Marker constants are
    resolved per file, which is where they are defined and used.
    """
    root = _repo_root()
    paths = [os.path.join(root, f) for f in FILES]
    for rel in ROOTS:
        for dirpath, _dirs, names in os.walk(os.path.join(root, rel)):
            paths += [
                os.path.join(dirpath, n) for n in names
                if n.endswith((".py", ".sh"))
            ]
    out = defaultdict(lambda: defaultdict(list))
    for p in sorted(set(paths)):
        try:
            with open(p, encoding="utf-8") as fh:
                lines = fh.read().split("\n")
        except OSError:
            continue
        rel_p = os.path.relpath(p, root)
        markers = {}
        for line in lines:
            m = MARKER_DEF.match(line)
            if m:
                markers[m.group(1)] = m.group(2)
        for i, line in enumerate(lines, 1):
            hits = ASSIGNMENT.findall(line) + COUNTER_KEY.findall(line)
            hits += [
                (code, markers.get(ident, f"{UNRESOLVED}:{ident}"))
                for code, ident in CONCAT.findall(line)
            ]
            # FORM 4: the code is on this line, its holder on a later one.
            for code in SIGNATURE_CODE.findall(line):
                holder = None
                for ahead in lines[i - 1:i - 1 + SIGNATURE_WINDOW + 1]:
                    found = SIGNATURE_NAME.search(ahead)
                    if found:
                        holder = found.group(1)
                        break
                hits.append((code, holder or f"{UNRESOLVED_SIGNATURE}:{code}"))
            for code, name in hits:
                locs = out[code][name]
                if f"{rel_p}:{i}" not in locs:
                    locs.append(f"{rel_p}:{i}")
    return out


class TestOneWCodePerException(CustomTestCase):

    def test_the_census_actually_finds_the_codes(self):
        """DENOMINATOR FIRST. A scan that silently matched nothing would pass
        every assertion below and prove nothing -- the failure mode this whole
        family of guards keeps producing."""
        c = census()
        self.assertGreater(len(c), 20, "the W-code scan found almost nothing")
        self.assertIn("W50", c, "the renumbered prefill refusal must be found")
        self.assertIn("W47", c, "the objective refusal must still be found")
        # THE TWO ROOTS ADDED BY THE serve-next4 TRAIN, named so that dropping
        # one from ROOTS fails here instead of silently shrinking the surface.
        self.assertIn("W18", c, "mem_ledger must be in the walk (W18)")
        self.assertIn("W40", c, "planner must be in the walk (W40)")

    def test_no_new_collision(self):
        """RED-FIRST: this failed at the parent 64a2829afd with W47 in the
        list (two names, PdFlipTpObjectiveRefused + PdFlipTpPrefillExceeded)."""
        c = census()
        collisions = {code: set(names) for code, names in c.items() if len(names) > 1}
        new = {k: v for k, v in collisions.items() if k not in KNOWN_COLLISIONS}
        self.assertEqual(
            new, {},
            "a W-code names more than one exception. Enumerate the used set "
            "(this file's census) and pick a FREE number -- picking the next "
            "one by hand is how W31 became W47 and W47 became a second "
            "collision.",
        )

    def test_the_known_collisions_are_exactly_the_recorded_ones(self):
        """Pinned EXACTLY, so fixing one fails here too and the list shrinks
        deliberately rather than becoming a permanent allow-list."""
        c = census()
        collisions = {code: set(names) for code, names in c.items() if len(names) > 1}
        self.assertEqual(
            collisions, KNOWN_COLLISIONS,
            "the known-collision list no longer matches reality: a collision "
            "was fixed (shrink the list, in the same commit) or the census "
            "changed shape.",
        )

    def test_w47_is_the_objective_refusal_alone(self):
        c = census()
        self.assertEqual(set(c["W47"]), {"PdFlipTpObjectiveRefused"})

    def test_w50_is_the_prefill_refusal_alone_and_the_wire_is_name_keyed(self):
        c = census()
        self.assertEqual(set(c["W50"]), {"PdFlipTpPrefillExceeded"})
        # The durable half: the leg-2 detector matches the NAME, so the label
        # can be renumbered without breaking the wire. Asserted because that
        # property is the reason this renumber is safe at all.
        from flliper.srt.pdflip import front

        self.assertEqual(front.X_REFUSAL_MARKER, "PdFlipTpPrefillExceeded")
        self.assertIsNone(
            re.match(r"W\d", front.X_REFUSAL_MARKER),
            "the wire marker must carry no W-CODE at all -- a protocol "
            "keyed on a renumberable label breaks at the renumber that "
            "fixes the collision",
        )
        self.assertTrue(front.X_REFUSAL_NAME.startswith("W50 "))
        self.assertTrue(
            front.x_refusal_marker_in("... W50 PdFlipTpPrefillExceeded ...")
        )
        # ... and it still matches a body carrying the OLD label, because the
        # match never depended on the number.
        self.assertTrue(
            front.x_refusal_marker_in("... W47 PdFlipTpPrefillExceeded ...")
        )

    def test_w4_is_the_wake_refusal_alone(self):
        """Train2 fix 2 introduced ``W4 PdFlipWakeRefused`` across three modules
        (weight_updater, front, pdflip_memory_saver).

        W4 was FREE among PdFlip names before it -- the other ``W4`` tokens on
        this tree are quantization scheme names (``W4A8``) and PP-obligation
        prose, neither of which this census can match, because the pattern is
        ``W<nn> PdFlip<Name>`` and not the bare number. That distinction is the
        whole reason a bare ticket/code number is never a usable grep pattern.
        """
        c = census()
        self.assertIn("W4", c, "the wake refusal must be found by the census")
        self.assertEqual(set(c["W4"]), {"PdFlipWakeRefused"})
        # It is raised from more than one module, and that is the point of
        # keying the census on the whole surface rather than one file.
        files = {loc.split(":")[0] for loc in c["W4"]["PdFlipWakeRefused"]}
        self.assertGreaterEqual(len(files), 2, sorted(files))

    def test_the_scan_reads_all_three_forms_and_not_just_the_plain_one(self):
        """DENOMINATOR FOR THE HARDENING ITSELF. The un-hardened scan passed
        every other assertion in this file while being blind to two of the
        three forms, so the hardening needs its own proof that it can see
        them -- on the real tree, at the real lines, not on a synthetic
        string."""
        c = census()
        front = "python/flliper/srt/pdflip/front.py"
        with open(os.path.join(_repo_root(), front), encoding="utf-8") as fh:
            lines = fh.read().split("\n")

        def sites(literal):
            """The 1-based lines of ``front.py`` that carry this exact text.

            THE LINE IS LOOKED UP, NOT TYPED. The pins here were four hard
            numbers, and one of them (2804) went stale the moment an unrelated
            front.py change moved the site by a single line in a merge -- a
            RED that says nothing about the scan. What this test is for is
            that the census reads the real SITE in all three forms, so the
            site is named by its own text and the line is derived from it.
            An empty result fails loudly below: a literal that no longer
            exists means the form itself is gone, which IS a finding.
            """
            found = [i for i, ln in enumerate(lines, 1) if literal in ln]
            self.assertTrue(found, f"{front} no longer carries {literal!r}")
            return found

        # form 3, concatenated: front.py builds both of these from a bare code
        # plus a marker constant, and the scan must credit the RESOLVED name.
        for line in sites('X_REFUSAL_NAME = "W50 " + X_REFUSAL_MARKER'):
            self.assertIn(f"{front}:{line}", c["W50"]["PdFlipTpPrefillExceeded"],
                          "the concatenated form must be read AND resolved")
        for line in sites('NO_ROUTE_NAME = "W52 " + NO_ROUTE_MARKER'):
            self.assertIn(f"{front}:{line}", c["W52"]["PdFlipNoServiceableRoute"],
                          "#1290's concatenated claim is the one #1257 "
                          "walked into")
        # form 2, counter key: an underscore where the plain pattern wants a
        # space. BOTH of #1290's counter sites, and the pre-existing W22 one.
        w52_counters = sites('"W52_PdFlipNoServiceableRoute"')
        self.assertEqual(len(w52_counters), 2, w52_counters)
        for line in w52_counters:
            self.assertIn(f"{front}:{line}",
                          c["W52"]["PdFlipNoServiceableRoute"])
        for line in sites('"W22_PdFlipSpanUnknownPricedFull"'):
            self.assertIn(f"{front}:{line}",
                          c["W22"]["PdFlipSpanUnknownPricedFull"])
        # ...and the sub-key suffix is NOT read as a second holder.
        self.assertEqual(set(c["W28"]), {"PdFlipLeg2Unpriced"},
                         "W28_PdFlipLeg2Unpriced_stream_served is a sub-key of "
                         "the same refusal, not a second exception")

    def test_no_concat_operand_is_left_unresolved(self):
        """The one way this hardening could go quietly blind again: a marker
        constant defined somewhere the per-file resolution cannot see it. That
        must FAIL here, not drop the code from the census."""
        c = census()
        unresolved = {
            code: names for code, names in c.items()
            for n in names if n.startswith(UNRESOLVED)
        }
        self.assertEqual(
            unresolved, {},
            "a W-code is concatenated with a marker this scan cannot resolve "
            "in its own file; resolve it (or define the marker beside its "
            "use) rather than letting the code drop out of the census",
        )

    def test_w52_belongs_to_the_no_route_refusal_alone(self):
        """#1257's regression, pinned. W52 is #1290's, claimed at the base
        commit 80de2d31d1 in two whitespace-free forms; the corridor pass was
        renumbered to W54 rather than sharing it."""
        c = census()
        self.assertEqual(set(c["W52"]), {"PdFlipNoServiceableRoute"})

    def test_the_corridor_codes_are_w54_w55_w56_alone(self):
        """#1257c. W53 was claimed on the base by #1291's
        ``W53 PdFlipStoreHandbackFailed`` between #1257's enumeration and its
        merge, so on the merged tree the two contradicted each other: this
        census sees both forms since the CONCAT hardening. The corridor pass
        renumbered to W55 (enumerated free against this same census) and the
        older claim keeps W53. W56 is the new verdict-only floor refusal."""
        c = census()
        self.assertEqual(set(c["W53"]), {"PdFlipStoreHandbackFailed"})
        self.assertEqual(set(c["W54"]), {"PdFlipCorridorBudgetUnpriced"})
        self.assertEqual(set(c["W55"]), {"PdFlipCorridorBudgetWouldBind"})
        self.assertEqual(set(c["W56"]), {"PdFlipCorridorFloorUnmeasured"})
        # and the label is not built by concatenation, because that is what
        # hid the previous claim from this very census.
        from flliper.srt.pdflip import corridor_budget as cb

        self.assertEqual(cb.UNPRICED_NAME, "W54 PdFlipCorridorBudgetUnpriced")
        self.assertEqual(cb.WOULD_BIND_NAME, "W55 PdFlipCorridorBudgetWouldBind")
        self.assertEqual(
            cb.UNMEASURED_FLOOR_NAME, "W56 PdFlipCorridorFloorUnmeasured"
        )

    def test_the_serve_next5_train_renumbered_the_incoming_codes_by_census(self):
        """serve-next5 train (2026-09-09). Two input branches were cut on trees
        where W52-W56 were free: ``pdflip/tp3-decode-dec2-0909`` (base
        4f762260ba, before #1236) claimed W52-W56 for the #1241/#1293
        operating-point refusals and the eager-decode arm, and
        ``pdflip/fix-1298-0909`` claimed W55 for the #580 span STOP. On the
        merged tree every one of those collided with a serve-line holder
        (W52 no-route, W53 handback, W54-W56 corridor). Renumbered to the
        first free numbers above this census's maximum (W60 -> W61..W66),
        exception NAMES unchanged, so a boot log is still grepped by name."""
        c = census()
        self.assertEqual(set(c["W61"]), {"PdFlipTpOperatingPointUnpriced"})
        self.assertEqual(set(c["W62"]), {"PdFlipTpOperatingPointDisablesUnevenAxis"})
        self.assertEqual(set(c["W63"]), {"PdFlipTpOperatingPointNeedsEnvPin"})
        self.assertEqual(set(c["W64"]), {"PdFlipTpOperatingPointInfeasible"})
        self.assertEqual(set(c["W65"]), {"PdFlipPrefetchSpanSplit"})
        self.assertEqual(set(c["W66"]), {"PdFlipEagerDecodeArm"})
        # and the serve-line holders the incoming codes collided with kept
        # their numbers (the pushed serve-next4 tip's boot ticket names them).
        self.assertEqual(set(c["W52"]), {"PdFlipNoServiceableRoute"})
        self.assertEqual(set(c["W53"]), {"PdFlipStoreHandbackFailed"})
        self.assertEqual(set(c["W56"]), {"PdFlipCorridorFloorUnmeasured"})

    def test_the_xchg_replay_renumbered_its_sixteen_codes_by_census(self):
        """#1273 xchg on serve-next5 (2026-09-09). The weight-exchange chain
        was cut on ``3ea18deb95``, whose census maximum is W50, and enumerated
        W51-W66 there -- every one of them free at the time. While it was being
        built the serve line claimed that whole band (W51 host-ring, W52
        no-route, W53 handback, W54-W56 corridor, W57/W58 store, W59
        chunk-card, W60 carrier floor, W61-W66 operating point / prefetch /
        eager decode), so on the replayed tree all sixteen collided at once.

        Renumbered to W67-W82 (and W84 for the coverage refusal, which collided with
        the line's W67 PdFlipPPCutOrderedCutOffFrontier), by census
        maximum W66 -- the same rule the serve-next4 and serve-next5 trains
        applied to their own incoming codes -- with the exception NAMES
        unchanged, so a boot log is still grepped by name.

        W73 has no holder on purpose: it is the chain's old W57, a number its
        prose reserved for the S6 roll-forward that never got an exception.
        It moved with the rest so that no xchg comment points at a serve-line
        code (W57 is ``PdFlipStoreDiskRefused``) that means something else.
        """
        c = census()
        for code, name in (
            ("W84", "PdFlipXchgCoverageRefused"),
            ("W68", "PdFlipXchgPlanDisagree"),
            ("W69", "PdFlipXchgGateTimeout"),
            ("W70", "PdFlipXchgShortPiece"),
            ("W71", "PdFlipXchgResidencyUnarmable"),
            ("W72", "PdFlipXchgOnCardUnavailable"),
            ("W74", "PdFlipXchgSourceMissing"),
            ("W75", "PdFlipXchgShadowMismatch"),
            ("W76", "PdFlipXchgRunnerShapeUnknown"),
            ("W77", "PdFlipXchgShadowUnaffordable"),
            ("W78", "PdFlipXchgSemaphoreNotRearmed"),
            ("W79", "PdFlipXchgShadowRankLocalSkip"),
            ("W80", "PdFlipXchgShadowPlanDiverged"),
            ("W81", "PdFlipXchgDepositUnfundable"),
            ("W82", "PdFlipXchgOncardSlotRefused"),
        ):
            self.assertEqual(set(c[code]), {name}, code)
        # and every serve-line holder the incoming codes collided with KEPT
        # its number -- the half a renumber gets wrong by moving the wrong side.
        for code, name in (
            ("W51", "PdFlipHostRingUnfunded"),
            ("W52", "PdFlipNoServiceableRoute"),
            ("W53", "PdFlipStoreHandbackFailed"),
            ("W54", "PdFlipCorridorBudgetUnpriced"),
            ("W55", "PdFlipCorridorBudgetWouldBind"),
            ("W56", "PdFlipCorridorFloorUnmeasured"),
            ("W57", "PdFlipStoreDiskRefused"),
            ("W58", "PdFlipStoreArcRefused"),
            ("W59", "PdFlipChunkCardMismatch"),
            ("W60", "PdFlipCarrierFloorUnreachable"),
            ("W61", "PdFlipTpOperatingPointUnpriced"),
            ("W66", "PdFlipEagerDecodeArm"),
        ):
            self.assertEqual(set(c[code]), {name}, code)
        # the xchg band is contiguous apart from the holder-less W73: nothing
        # was left behind on an old number by a partial rewrite.
        self.assertEqual(set(c["W73"]), set())

    def test_the_chosen_number_was_free_and_the_free_ones_are_named(self):
        """W50 is not 'the next one': it is the first free number above the
        highest assigned code, and the census can say which others are free."""
        c = census()
        used = {int(m.group(1)) for code in c
                for m in [re.match(r"W(\d{1,2})", code)] if m}
        self.assertNotIn(50, used - {50}, "W50 must have been free before this")
        # #1257 renumbered to W54 by enumerating this same census. The numbers
        # below stayed free through both renumbers and are the next candidates.
        self.assertIn(54, used, "W54 is the corridor pass's number now")
        self.assertIn(55, used, "W55 is the would-bind refusal's number now")
        self.assertIn(56, used, "W56 is the unmeasured-floor refusal's number")
        # #1332 B1b (2026-09-11) TOOK W14 for `PdFlipXchgWidestLayerUnreadable`,
        # and this list shrinks in the SAME commit -- a register that names a
        # free number after it was consumed is the shape the determination law
        # forbids ("the register entry is part of the work").
        self.assertIn(14, used, "W14 is the widest-layer refusal's number now")
        # #1334 (2026-09-11) TOOK W15 for `PdFlipXchgDiagonalHasNoCrossPair`, the
        # refusal that replaced boot weg2xsn9's bare IndexError. Same register
        # discipline as W14 one commit earlier: the list shrinks here, now.
        self.assertIn(15, used, "W15 is the diagonal-lane refusal's number now")
        # #1335 (2026-09-11) TOOK W23 for `PdFlipXchgCardUuidMapUnusable`, the
        # refusal that replaced the card->uuid map's UNPRODUCED default -- the
        # root of boot weg2xsn9's AND weg2xsn10's 36-leg IndexError. The list
        # shrinks here, in the same commit, for the third code running.
        self.assertIn(23, used, "W23 is the card-uuid-map refusal's number now")
        # AND THE LIST IS NOW WORTH MORE THAN IT WAS: `census` reads the
        # SIGNATURE-DEFAULT form as of this commit (form 4), so a number that
        # is only held by `code: str = "Wnn"` can no longer be named free here.
        # That is how W88 was mis-named free by B1b's first enumeration --
        # 19 textual hits under python/flliper/srt, 0 visible to this file.
        # #1273 B4d (2026-09-11) TOOK W39 for `PdFlipXchgWavePartitionDisagree`,
        # the published-vs-derived wave-partition reason.  The list shrinks
        # HERE, in the same commit, for the fourth code running -- the register
        # entry is part of the work.
        self.assertIn(39, used, "W39 is the wave-partition refusal's number now")
        # AND THE REMAINING FREE LIST IS NOW STATED WITH ITS OWN LIMIT, because
        # this census has a BLIND SPOT that named a taken number free:
        # `W19 DormantResidueRefused` is raised by front.py through
        # `do_stop("W19 DormantResidueRefused", ...)`, and NONE of the four
        # forms above can see it -- every one requires a `PdFlip`-prefixed name.
        # So W19 is TAKEN and this file would have offered it.  Same class as
        # the W88 hole B1b hit, one naming convention further out.  Until a
        # fifth form lands, a number from this list is a CANDIDATE and the
        # author greps the tree for it before minting.
        self.assertNotIn(19, used, "the census still cannot see W19's holder")
        front = pathlib.Path(_repo_root(), "python", "flliper", "srt", "pdflip", "front.py")
        self.assertIn(
            "W19 DormantResidueRefused", front.read_text(errors="replace"),
            "W19 IS held in front.py -- if this ever fails the holder moved and "
            "the free list above must be re-derived, not trusted",
        )
        # #1350 seam-digest (2026-09-12, ON THE MERGE TRAIN) TOOK W90 for
        # `PdFlipSeamDigestMismatch`.  The list shrinks HERE, in the same commit
        # that lands the consumer, for the fifth code running -- and this one
        # was caught BY this census rather than remembered: #1273 B4d wrote
        # "W90 is free and is the next candidate" into the loop below while a
        # sibling branch was already minting it.  Two slices, both green alone,
        # colliding only at the train: exactly what a per-merge census is for.
        # ...AND THE SAME CONSTRUCTION HOLDS W90, so this line may NOT rest on
        # the census alone.  MEASURED BY MUTATION while writing this: the ONLY
        # census-visible W90 in the tree is a string inside the CLI HELP TEXT of
        # `--pdflip-seam-digest` (launcher.py), because that sentence happens to
        # spell the number and the name adjacently.  The REFUSAL itself is
        # `REFUSAL_MARKER = W_CODE + " " + REFUSAL_NAME`, invisible like W91's.
        # Rewording that help sentence -- zero behaviour change -- flips this
        # census back to "W90 is free" while the code still holds it.  So the
        # census reading is kept as a record of what the instrument sees, and
        # the LOAD-BEARING guard is the second reading on the minting site.
        self.assertIn(90, used, "W90 is the seam-digest mismatch refusal's number now")
        # AND THE BLIND SPOT BIT AGAIN, A SIXTH TIME, ON THE SAME SLICE -- so it
        # is closed here the way W19's was, not merely mentioned.
        # `seam_digest.py` holds W91 for `PdFlipSeamDigestBudgetExceeded`, but it
        # builds the marker by CONCATENATION at import time:
        #     W_CODE_BUDGET = "W91"
        #     BUDGET_MARKER = W_CODE_BUDGET + " " + BUDGET_NAME
        # so NO literal "W91 PdFlipSeamDigestBudgetExceeded" exists anywhere for a
        # text form to find, and every form above needs the number and the name
        # ADJACENT.  W91 is therefore TAKEN and this file would have offered it
        # to the next author -- the same shape as W19 and W88, one construction
        # further out.  MEASURED CONSEQUENCE, not hypothetical: the #1348
        # coverage-diff slice had independently minted W91 off the same base,
        # and the census could not have caught that half.  Hence the second
        # reading, pinned:
        self.assertNotIn(91, used, "the census still cannot see W91's holder")
        digest = pathlib.Path(_repo_root(), "python", "flliper", "srt", "pdflip",
                              "seam_digest.py")
        digest_src = digest.read_text(errors="replace")
        self.assertIn(
            'W_CODE_BUDGET = "W91"', digest_src,
            "W91 IS held in seam_digest.py -- if this ever fails the holder "
            "moved and the free list above must be re-derived, not trusted",
        )
        self.assertIn(
            'BUDGET_NAME = "PdFlipSeamDigestBudgetExceeded"', digest_src,
            "W91's holder name moved; re-derive the free list rather than "
            "trusting this register",
        )
        self.assertIn(
            'W_CODE = "W90"', digest_src,
            "W90's holder moved out of seam_digest.py; the census sees W90 only "
            "through a help-text sentence, so re-derive the free list rather "
            "than trusting this register",
        )
        self.assertIn(
            'REFUSAL_NAME = "PdFlipSeamDigestMismatch"', digest_src,
            "W90's holder name moved; re-derive the free list rather than "
            "trusting this register",
        )
        for n in (5,):
            self.assertNotIn(n, used, f"W{n} is free and is the next candidate")
        # W92 and W93 are the next candidates ABOVE the band this train filled
        # (#1348's two codes were renumbered onto them at their merge, so by the
        # time that posten lands these two assertions move with it).
        self.assertIn(88, used, "form 4 must see scheduler.py's signature code")


if __name__ == "__main__":
    unittest.main()
