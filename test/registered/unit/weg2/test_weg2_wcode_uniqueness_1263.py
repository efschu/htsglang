# SPDX-License-Identifier: Apache-2.0
"""One W-code, one exception name -- enumerated, not remembered.

THE CLASS, three instances in one day and every one found by a READER:

* W31 named both this rig's serving-path re-route (``Weg2TpPrefillExceeded``)
  and ``Weg2HostRingExhausted``, the host-ring exhaustion that killed boot
  weg2tr2. A census that greps ``W31`` reports one number for a fatal
  exhaustion and a routing event -- how a postmortem merges a killer into
  noise. Train fix 5 renumbered it.
* It renumbered it to W47, which the #1235 argv slice had ALREADY assigned to
  ``Weg2TpObjectiveRefused``. One collision traded for another, because the
  new number was picked rather than enumerated.
* Renumbered again here to W50, this time by enumerating the used set.

A third hand-picked number would repeat it, so the enumeration is the guard
rather than the fix: this file recomputes the census on every run.

FOURTH INSTANCE, 2026-09-09 (#1257), and the reason the SCAN is hardened here
rather than only the label fixed: the #1257 corridor pass claimed ``W52``,
which #1290 already held at the base commit -- and this census reported 7/0/0
anyway. Its pattern was ``W<nn>`` + WHITESPACE + ``Weg2<Name>``, and #1290
writes the code in the two forms that have no whitespace after it:

* CONCATENATED, ``front.py:561``: the name is built from the bare code plus a
  separate marker constant, so a QUOTE follows the code, not a space.
* COUNTER KEY, ``front.py:1981``/``:2603``: ``W52_Weg2NoServiceableRoute``, an
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

FIFTH INSTANCE, 2026-09-09 (#1306), and it is a SCOPE defect, not a form
defect: every reading above was taken through a fixed ``ROOTS`` list of six
weg-2-shaped directories. A W-code emitted anywhere else was invisible, so a
census-driven renumber left it standing and the next merge collided silently.
Three measured shapes of the same gap, all from one day:

* DIRECTORY. ``weg2/kvtail-feature-0909`` @ ``072ac2be5b`` puts four
  ``W58``/``Weg2KvTailFormRefused`` sites in ``layers/attention/
  flashinfer_backend.py``, one in ``model_executor/
  model_runner_kv_cache_mixin.py`` and two ``W54`` sites in
  ``model_executor/pool_configurator.py`` -- none of those three directories
  was walked. The census's answer, "W56 only in ``kv_tail.py``", was a
  statement about the walk, not about the tree: W56 also sits twice in
  ``test/registered/unit/mem_cache/test_kv_tail_1243.py``, because the whole
  TEST TREE was outside the walk too.
* NAME. All three regexes require a ``Weg2`` prefix, so a refusal whose class
  is named anything else holds a number the census reports as FREE. Two live
  ones at this tip: ``W19 DormantResidueRefused`` (``front.py``'s own
  ``do_stop``) and ``W27 PPWidthDivergenceRefused`` (a real ``RuntimeError``
  subclass in ``pp_admission_congruence.py``). A renumber pass enumerating
  "the first free number" from the old census would have handed out either.
* CASE. The xchg replay (record SECTION 1bh) had to hand-fix
  ``reason=w62-stale``, a LOWERCASE code emitted as a log reason VALUE. No
  ``\\bW\\d\\d\\b`` pass sees it, and it reaches a boot log, which is the one
  place a W-code is actually read by a human.

So the walk is the whole ``srt`` tree plus the whole registered test tree, and
the scan reads five forms rather than three. Two mechanisms keep the widened
surface honest rather than merely bigger:

* ``wcode-census: ignore`` -- a per-LINE opt-out for a line that QUOTES a
  historical label instead of assigning one (a test proving the wire detector
  is name-keyed has to write the pre-renumber label on purpose). The marker
  sits at the site, where a reader sees why, and the whole population of them
  is pinned below, so adding one is a visible act.
* the concat operand now resolves against a TREE-GLOBAL marker table, and
  refuses if a constant name carries two different values anywhere. Per-file
  resolution failed on every test that asserted a source constant's shape.

WHY A CENSUS AND NOT A REGISTRY. There is no table of W-codes to keep in sync
-- the code IS the message text -- so a registry would be second bookkeeping
beside the refusals themselves, and would drift exactly the way the front's
docstring drifted from its own constant. Scanning the messages is the one
reading that cannot go stale.

WHAT IS ASSERTED, and what is deliberately NOT:

* NEW collisions fail. That is the whole point.
* The FIVE pre-existing collisions are pinned as a named, dated set rather
  than silently allowed. Pinning them exactly means fixing one also fails
  here, which forces the list to shrink deliberately instead of rotting into
  a permanent allow-list. They are NOT fixed on this branch: it is the
  serving-boot base, and renumbering across five modules and their tests is a
  bigger diff than a label defect justifies before a boot. Each carries the
  two names and where they live so the next pass has the work already done.
* #1306 REPORTS, it does not renumber. Widening the walk to the whole tree
  adds no new collision at ``319e76d60b`` -- every one of the 152 sites it
  newly reaches is a test echoing its own source holder -- and the two
  foreign-named holders it uncovers (W19, W27) collide with nothing today.
  They are pinned as OCCUPIED so the next renumber cannot spend them.
"""

import builtins
import os
import re
import unittest
from collections import defaultdict

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.test.test_utils import CustomTestCase

#: Where a Weg-2 refusal can be raised, logged or re-routed from. #1306: the
#: WHOLE runtime tree and the WHOLE registered test tree, not an enumerated
#: list of weg-2-shaped directories. The old list
#: (``srt/{weg2,managers,mem_cache,mem_ledger,planner}`` + ``scripts/weg2`` +
#: ``server_args.py``) was itself the defect: it grew by one pair of
#: directories per train, always AFTER a reader found the miss by hand, and
#: each addition was a statement that the previous census had been reporting
#: about its walk rather than about the tree.
ROOTS = (
    "python/sglang/srt",
    "test/registered",
    "scripts",
)
#: Kept as a mechanism for a future emitter outside the roots above (a
#: top-level launcher, say). Empty today: ``server_args.py``, the one file the
#: old list named explicitly, is inside ``python/sglang/srt``.
FILES = ()

#: THE FIVE FORMS A W-CODE IS WRITTEN IN. Reading only the first is what let
#: #1257 claim a taken number while this file reported no collision; reading
#: only the first three is what let #1306's foreign-named and lowercase
#: holders read as free.
#:
#: 1. PLAIN, in a message or a docstring: ``W12 Weg2Sample``. The letter  # wcode-census: ignore
#:    suffix (``W11b``, ``W40b``) is part of the code.
ASSIGNMENT = re.compile(r"\b(W\d{1,2}[a-z]?)\s+(Weg2[A-Za-z0-9_]+)")
#: 2. COUNTER KEY: ``self.counters["W28_Weg2Leg2Unpriced"]``. The exception
#:    name is CamelCase, so a trailing ``_lowercase`` part is a sub-key
#:    ("..._stream_served") and not part of the name -- hence no underscore in
#:    the captured group, which truncates the suffix instead of inventing a
#:    second holder for the code.
COUNTER_KEY = re.compile(r"\b(W\d{1,2}[a-z]?)_(Weg2[A-Za-z0-9]+)")
#: 3. CONCATENATED: ``NAME = "W50 " + X_REFUSAL_MARKER``. The code's holder is
#:    then the VALUE of that operand, so the scan has to resolve it; capturing
#:    the identifier itself would report the variable name as the holder and
#:    turn every such site into a phantom collision.
CONCAT = re.compile(r"[\"'](W\d{1,2}[a-z]?) [\"']\s*\+\s*([A-Za-z_][A-Za-z0-9_]*)")
#: ``X_REFUSAL_MARKER = "Weg2TpPrefillExceeded"`` -- resolves form 3's operand.
MARKER_DEF = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*[\"'](Weg2[A-Za-z0-9]+)[\"']\s*$"
)
#: 4. FOREIGN-NAMED HOLDER (#1306): ``do_stop("W19 DormantResidueRefused")``.
#:    Same plain form as 1, but the class is not ``Weg2``-prefixed, so forms
#:    1-3 report the number as unused while a live refusal emits it.
#:
#:    TWO FILTERS, BOTH LOAD-BEARING, both measured against this tree:
#:
#:    * a lowercase letter in the name, which kills the ALL-CAPS log banners
#:      that follow a code (``W10 DRAFTER``, ``W27 KILLED``, ``W30 SEAM``,
#:      ``W33 INVENTORY``, ``W35 CLASS``, ``W38 RETIRED``, ``W8b MOVES``,
#:      ``W11 DRAFT``);
#:    * a VERDICT SUFFIX, which kills the second namespace this tree writes in
#:      the same shape: ``scripts/cert_485/certify_485.py`` numbers its
#:      acceptance criteria ``W1 NVML corridor``, ``W2 Seam census``,
#:      ``W3 Flips``, ``W4 Soak``, ``W5 Ranks``, ``W6 Work-matched`` -- work
#:      items, not refusals, and admitting them collides five codes at once.
#:
#:    THE BOUND, stated rather than hidden: a foreign holder whose name ends
#:    in none of these words is missed. That is a deliberate trade against the
#:    cert_485 namespace, and the suffix list is the place to widen it.
#:    Builtin exception names are excluded separately: ``the W38 IndexError at
#:    pool_host`` is prose ABOUT a Python builtin, not a refusal called W38.
FOREIGN_VERDICT = (
    "Refused|Failed|Error|Exception|Mismatch|Missing|Exhausted|Breached"
    "|Expired|Aborted|Unfundable|Blind|Absent|Timeout|Unknown|Collision"
    "|Denied|Rejected|Stuck|Disagreement|Invalid|Unpriced|Overbudget"
    "|Unreachable|Unarmable|Diverged|Unaccounted|Loop|Split|NoOp"
)
FOREIGN_NAME = re.compile(
    r"\b(W\d{1,2}[a-z]?) ((?!Weg2)[A-Z][A-Za-z0-9]*[a-z][A-Za-z0-9]*"
    rf"(?:{FOREIGN_VERDICT}))\b"
)
#: 5. LOWERCASE REASON TOKEN (#1306): ``reason=w62-stale`` on an emitted leg
#:    result -- a string that reaches a BOOT LOG, which is the one place a
#:    W-code is read by a human, and which no uppercase pass sees.
#:
#:    ANCHORED TWICE, because the bare shape is noise: the leading lookbehind
#:    refuses a token preceded by a word character or a hyphen (which is what
#:    ``mosaicml-mpt-7b-instruct-w4-g128-awq`` is, in ``awq.py``), and the tail
#:    must START with a letter (which is what ``f"w2-1298-m1-{kv}"`` in
#:    ``test_weg2_store_grid_claim_1298.py`` is not). Both were measured
#:    false positives of the unanchored form.
LOWER_REASON = re.compile(
    r"(?<![A-Za-z0-9_-])w(\d{1,2}[a-z]?)-([a-z][a-z0-9]*(?:[-_][a-z][a-z0-9]*)*)"
)
#: Python builtins are not W-code holders; see FOREIGN_NAME. Read from the
#: ``builtins`` MODULE, never from ``__builtins__``: that name is the module
#: only in ``__main__`` and a plain dict everywhere else, so ``dir()`` on it
#: silently returns dict METHODS in an imported test and the filter empties
#: itself exactly where it runs for real.
_BUILTIN_EXC = frozenset(
    n for n in dir(builtins)
    if n.endswith(("Error", "Exception", "Warning"))
)

#: PROSE RESERVATION: a bare code on a line that talks about reserving, freeing
#: or holding a NUMBER. This is the W57 shape from record SECTION 1bh -- the
#: xchg chain reserved W57 for its S6 roll-forward and never gave it an
#: exception class, so eight comments across four modules pointed a reader at a
#: number the census could not see. Anchored on the reservation vocabulary
#: because the bare token alone matches 1888 sites on this tree, almost all of
#: them design-section markers (``#824 W4a``, ``#347 W1``, ``#969 SS W3``) and
#: MoE weight names (``W1``, ``W13``).
RESERVATION_WORDS = re.compile(
    r"\breserv\w*|\bis free\b|\bare free\b|\bnext free\b|\bfirst free\b"
    r"|\bheld by\b|\bunassigned\b|\bnot yet assigned\b",
    re.IGNORECASE,
)
BARE_CODE = re.compile(r"(?<![A-Za-z0-9_])W(\d{1,2}[a-z]?)(?![A-Za-z0-9_])")
#: ``#824 W4a`` / ``DESIGN #347 W1`` / ``#969 SS W3`` are design-section
#: markers, not W-codes. Recognised by what precedes them, not by their value.
TICKET_PREFIX = re.compile(r"(#\d+|§|DESIGN\s+#?\d+)\s*$")

#: A line carrying this marker is skipped by every form. For a line that
#: QUOTES a historical label, or carries a synthetic fixture string, rather
#: than assigning a live one. Pinned below.
#:
#: ANCHORED ON THE ``#``, and that is not cosmetic. The first version matched
#: the bare substring, so every line of THIS file's own documentation of the
#: mechanism suppressed itself -- five docstring lines dropped silently out of
#: the scan, including the line that defines the marker. A suppression
#: mechanism that hides its own description is the exact shape of defect this
#: file exists to catch, so it is only a marker inside a COMMENT.
IGNORE_MARKER = re.compile(r"#\s*wcode-census:\s*ignore")
#: Written into the census when form 3's operand resolves to nothing anywhere
#: in the tree. Asserted to be empty: an unresolvable operand is the blind spot
#: coming back, and it has to fail rather than quietly drop the code.
UNRESOLVED = "UNRESOLVED-CONCAT-OPERAND"
#: Written when one constant name carries two different values in the tree --
#: the one way a tree-global marker table could be worse than a per-file one.
AMBIGUOUS = "AMBIGUOUS-CONCAT-OPERAND"

#: THE KNOWN COLLISIONS, with both holders. Deferred, not accepted: this
#: branch renumbers W47 (2026-09-08) and W52 (2026-09-09, #1257) only.
#:
#: W22 joined this list on 2026-09-09 WITHOUT anything changing in the source:
#: it is pre-existing at the base commit ``80de2d31d1`` and was simply
#: invisible to the un-hardened scan, which could not read a counter key.
#: ``W22 Weg2HostWatermarkBreached`` at ``front.py:1805`` and
#: ``host_ledger.py:688`` vs. the counter ``W22_Weg2SpanUnknownPricedFull`` at
#: ``front.py:1938``. Recorded here rather than renumbered, for the reason the
#: docstring gives for the other four: this is the serving-boot base.
#:
#: #1306 (2026-09-09) widened the walk from six directories to the whole srt
#: and test trees and added two forms, and this set did NOT grow. That is a
#: measurement, not luck: every one of the 152 newly-reached sites is a test
#: echoing the holder its own source module already carries.
KNOWN_COLLISIONS = {
    "W10": {"Weg2CanonicalPageMissing", "Weg2DrafterIdentityMismatch"},
    "W22": {"Weg2HostWatermarkBreached", "Weg2SpanUnknownPricedFull"},
    "W35": {"Weg2VramCreditRefused", "Weg2XReQueueLoop"},
    "W36": {"Weg2AdmitterBarrierExpired", "Weg2DuplexDecisionRefused"},
    "W46": {"Weg2PPSplitMapMismatch", "Weg2TokenVectorRefused"},
}

#: THE FOREIGN-NAMED HOLDERS (#1306), pinned so a renumber cannot spend their
#: numbers. Neither collides with a ``Weg2`` name today; both would have been
#: offered as free by the old census, which is the whole finding.
FOREIGN_HOLDERS = {
    "W19": {"DormantResidueRefused"},
    "W27": {"PPWidthDivergenceRefused"},
}

#: Codes claimed in RESERVATION prose that no holder of any kind carries. The
#: W57 shape. Empty of the xchg chain's own reservation at this tip (that
#: branch is not merged here); these two are the serve line's own.
PROSE_ONLY_RESERVATIONS = {
    "W24": ["python/sglang/srt/managers/kv_backing_relief.py"],
}

#: Every ``wcode-census: ignore`` line in the tree, as ``relpath -> count``.
#: PINNED so the opt-out cannot grow quietly into an allow-list. Line numbers
#: are deliberately absent -- the file's own history has one stale hard-pinned
#: line in it already, and a line number that moves in a merge is a RED that
#: says nothing about the scan.
IGNORED_LINES = {
    # this file's own documentation of form 1, which has to spell a code and a
    # name next to each other to explain the regex.
    "test/registered/unit/weg2/test_weg2_wcode_uniqueness_1263.py": 2,
    # the two tests that PROVE the wire detector is keyed on the name and not
    # the number, which they can only do by writing a pre-renumber label.
    "test/registered/unit/weg2/test_weg2_ring_form_gate_1261.py": 1,
}


def _repo_root() -> str:
    here = os.path.abspath(__file__)
    for _ in range(8):
        here = os.path.dirname(here)
        if os.path.isdir(os.path.join(here, "python", "sglang", "srt", "weg2")):
            return here
    raise AssertionError("could not locate the repo root from this test file")


def _paths():
    root = _repo_root()
    paths = [os.path.join(root, f) for f in FILES]
    for rel in ROOTS:
        for dirpath, _dirs, names in os.walk(os.path.join(root, rel)):
            paths += [
                os.path.join(dirpath, n) for n in names
                if n.endswith((".py", ".sh"))
            ]
    return root, sorted(set(paths))


_SCAN_CACHE = {}


def _scan():
    """One walk of the widened surface, memoised.

    Reads the SOURCE, never the imported module: a refusal inside an ``f``
    string is only text at runtime, and half of these are logged rather than
    raised, so there is no object to enumerate. Memoised because the walk is
    now 5000+ files rather than 400 and this file calls it a dozen times.
    """
    if _SCAN_CACHE:
        return _SCAN_CACHE
    root, paths = _paths()
    contents = {}
    markers = defaultdict(set)
    for p in paths:
        try:
            with open(p, encoding="utf-8") as fh:
                lines = fh.read().split("\n")
        except (OSError, UnicodeDecodeError):
            continue
        contents[os.path.relpath(p, root)] = lines
        for line in lines:
            if IGNORE_MARKER.search(line):
                continue
            m = MARKER_DEF.match(line)
            if m:
                markers[m.group(1)].add(m.group(2))

    def resolve(ident):
        vals = markers.get(ident)
        if not vals:
            return f"{UNRESOLVED}:{ident}"
        if len(vals) > 1:
            return f"{AMBIGUOUS}:{ident}"
        return next(iter(vals))

    named = defaultdict(lambda: defaultdict(list))
    lower = defaultdict(lambda: defaultdict(list))
    prose = defaultdict(lambda: defaultdict(list))
    ignored = defaultdict(int)
    for rel_p in sorted(contents):
        for i, line in enumerate(contents[rel_p], 1):
            if IGNORE_MARKER.search(line):
                ignored[rel_p] += 1
                continue
            hits = ASSIGNMENT.findall(line) + COUNTER_KEY.findall(line)
            hits += [(c, resolve(idn)) for c, idn in CONCAT.findall(line)]
            hits += [
                (c, n) for c, n in FOREIGN_NAME.findall(line)
                if n not in _BUILTIN_EXC
            ]
            for code, name in hits:
                locs = named[code][name]
                if f"{rel_p}:{i}" not in locs:
                    locs.append(f"{rel_p}:{i}")
            for code, tok in LOWER_REASON.findall(line):
                lower["W" + code][f"w{code}-{tok}"].append(f"{rel_p}:{i}")
            if RESERVATION_WORDS.search(line):
                for m in BARE_CODE.finditer(line):
                    if TICKET_PREFIX.search(line[:m.start()]):
                        continue
                    prose["W" + m.group(1)][rel_p].append(i)
    _SCAN_CACHE.update(
        named=named, lower=lower, prose=prose, ignored=dict(ignored),
        nfiles=len(contents), markers={k: sorted(v) for k, v in markers.items()},
    )
    return _SCAN_CACHE


def census():
    """``{code: {exception_name: [file:line, ...]}}`` over the whole surface.

    THE COLLISION AUTHORITY, and keyed on NAMES, never on digits: two sites
    that say the same code and the same name are one holder however many
    modules they live in, and that is what makes a renumber checkable.
    """
    named = _scan()["named"]
    return defaultdict(
        lambda: defaultdict(list),
        {c: defaultdict(list, {n: list(l) for n, l in v.items()})
         for c, v in named.items()},
    )


def lowercase_reasons():
    """``{code: {token: [file:line, ...]}}`` -- form 5, kept OUT of ``census``.

    Deliberately a second map rather than a pseudo-name in the first: a
    lowercase reason token carries no exception name, so folding it in would
    invent a second holder for its code and turn every emitted reason into a
    phantom collision. A renumber pass reads BOTH maps.
    """
    return {c: {t: list(l) for t, l in v.items()}
            for c, v in _scan()["lower"].items()}


def prose_reservations():
    """``{code: {relpath: [line, ...]}}`` -- reservation prose, also separate.

    Same reason as ``lowercase_reasons``: a reservation names a NUMBER and no
    exception, so it cannot key the collision map. It answers the other half
    of a renumber's question -- which sites also have to move.
    """
    return {c: {f: list(l) for f, l in v.items()}
            for c, v in _scan()["prose"].items()}


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

    def test_the_walk_is_the_whole_tree_and_not_a_curated_list(self):
        """#1306, the SCOPE half. The old walk was six directories plus one
        named file, and every train added a pair AFTER a reader found the miss
        by hand. Three properties are pinned here, each of which the old walk
        failed:

        * the roots are the TREE roots, not weg-2-shaped subdirectories;
        * the walk is large enough that a new module is picked up by existing
          code (denominator, so a walk that silently collapsed to one
          directory fails here rather than passing everything);
        * ``server_args.py``, the one file the old list named EXPLICITLY, is
          still reached now that ``FILES`` is empty.
        """
        self.assertEqual(
            ROOTS, ("python/sglang/srt", "test/registered", "scripts"),
            "narrowing the walk is the #1306 defect, not a tuning knob",
        )
        s = _scan()
        self.assertGreater(
            s["nfiles"], 3000,
            f"the widened walk reached only {s['nfiles']} files; the six-root "
            "walk it replaces reached 388, so anything in that range means "
            "the roots collapsed",
        )
        c = census()
        sa = [
            loc for names in c.values() for locs in names.values()
            for loc in locs if loc.startswith("python/sglang/srt/server_args.py")
        ]
        self.assertTrue(
            sa, "server_args.py carried W-codes under the old FILES list and "
                "must still be reached by the root walk",
        )

    def test_the_test_tree_is_in_the_walk(self):
        """The half of the gap that was measured on the kvtail branch: ``W56``
        was reported as "only in kv_tail.py" while it also sat twice in
        ``test/registered/unit/mem_cache/test_kv_tail_1243.py``. The test tree
        is not decoration -- a renumber that misses it leaves a red suite
        behind, and a reader grepping the tree for a code gets a wrong answer.
        """
        c = census()
        in_tests = {
            code for code, names in c.items() for locs in names.values()
            for loc in locs if loc.startswith("test/registered/")
        }
        self.assertGreater(
            len(in_tests), 10,
            f"only {sorted(in_tests)} codes were found under test/registered; "
            "the test tree is out of the walk again",
        )

    def test_no_new_collision(self):
        """RED-FIRST: this failed at the parent 64a2829afd with W47 in the
        list (two names, Weg2TpObjectiveRefused + Weg2TpPrefillExceeded)."""
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
        self.assertEqual(set(c["W47"]), {"Weg2TpObjectiveRefused"})

    def test_w50_is_the_prefill_refusal_alone_and_the_wire_is_name_keyed(self):
        c = census()
        self.assertEqual(set(c["W50"]), {"Weg2TpPrefillExceeded"})
        # The durable half: the leg-2 detector matches the NAME, so the label
        # can be renumbered without breaking the wire. Asserted because that
        # property is the reason this renumber is safe at all.
        from sglang.srt.weg2 import front

        self.assertEqual(front.X_REFUSAL_MARKER, "Weg2TpPrefillExceeded")
        self.assertIsNone(
            re.match(r"W\d", front.X_REFUSAL_MARKER),
            "the wire marker must carry no W-CODE at all -- a protocol "
            "keyed on a renumberable label breaks at the renumber that "
            "fixes the collision",
        )
        self.assertTrue(front.X_REFUSAL_NAME.startswith("W50 "))
        self.assertTrue(
            front.x_refusal_marker_in("... W50 Weg2TpPrefillExceeded ...")
        )
        # ... and it still matches a body carrying the OLD label, because the
        # match never depended on the number.
        self.assertTrue(
            front.x_refusal_marker_in(
                # quotes the PRE-RENUMBER label on purpose:
                "... W47 Weg2TpPrefillExceeded ..."  # wcode-census: ignore
            )
        )

    def test_w4_is_the_wake_refusal_alone(self):
        """Train2 fix 2 introduced ``W4 Weg2WakeRefused`` across three modules
        (weight_updater, front, weg2_memory_saver).

        W4 was FREE among Weg2 names before it -- the other ``W4`` tokens on
        this tree are quantization scheme names (``W4A8``) and PP-obligation
        prose, neither of which this census can match, because the pattern is
        ``W<nn> Weg2<Name>`` and not the bare number. That distinction is the
        whole reason a bare ticket/code number is never a usable grep pattern.
        """
        c = census()
        self.assertIn("W4", c, "the wake refusal must be found by the census")
        self.assertEqual(set(c["W4"]), {"Weg2WakeRefused"})
        # It is raised from more than one module, and that is the point of
        # keying the census on the whole surface rather than one file.
        files = {loc.split(":")[0] for loc in c["W4"]["Weg2WakeRefused"]}
        self.assertGreaterEqual(len(files), 2, sorted(files))

    def test_the_scan_reads_all_three_forms_and_not_just_the_plain_one(self):
        """DENOMINATOR FOR THE HARDENING ITSELF. The un-hardened scan passed
        every other assertion in this file while being blind to two of the
        three forms, so the hardening needs its own proof that it can see
        them -- on the real tree, at the real lines, not on a synthetic
        string."""
        c = census()
        front = "python/sglang/srt/weg2/front.py"
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
            self.assertIn(f"{front}:{line}", c["W50"]["Weg2TpPrefillExceeded"],
                          "the concatenated form must be read AND resolved")
        for line in sites('NO_ROUTE_NAME = "W52 " + NO_ROUTE_MARKER'):
            self.assertIn(f"{front}:{line}", c["W52"]["Weg2NoServiceableRoute"],
                          "#1290's concatenated claim is the one #1257 "
                          "walked into")
        # form 2, counter key: an underscore where the plain pattern wants a
        # space. BOTH of #1290's counter sites, and the pre-existing W22 one.
        w52_counters = sites('"W52_Weg2NoServiceableRoute"')
        self.assertEqual(len(w52_counters), 2, w52_counters)
        for line in w52_counters:
            self.assertIn(f"{front}:{line}",
                          c["W52"]["Weg2NoServiceableRoute"])
        for line in sites('"W22_Weg2SpanUnknownPricedFull"'):
            self.assertIn(f"{front}:{line}",
                          c["W22"]["Weg2SpanUnknownPricedFull"])
        # ...and the sub-key suffix is NOT read as a second holder.
        self.assertEqual(set(c["W28"]), {"Weg2Leg2Unpriced"},
                         "W28_Weg2Leg2Unpriced_stream_served is a sub-key of "
                         "the same refusal, not a second exception")

    def test_the_concat_operand_resolves_tree_globally_not_per_file(self):
        """#1306. Per-file resolution was itself a scope defect, and widening
        the walk is what exposed it: every test that asserts a source
        constant's shape (``self.assertEqual(NO_ROUTE_NAME, "W52 " +
        NO_ROUTE_MARKER)``) carries form 3 with the operand defined in
        ``front.py``, one directory away. Per-file that is five UNRESOLVED
        entries and a red suite; tree-global it is five more sites credited to
        the right holder.

        The refusal is kept, one level up: a constant name that carries two
        different values anywhere in the tree resolves to AMBIGUOUS and fails
        the assertion below, so a global table can never quietly attribute a
        code to the wrong exception.
        """
        s = _scan()
        self.assertEqual(
            sorted(s["markers"]),
            ["HANDBACK_MARKER", "NO_ROUTE_MARKER", "X_REFUSAL_MARKER"],
            "the marker table changed shape; check the new constant resolves",
        )
        for ident, vals in s["markers"].items():
            self.assertEqual(len(vals), 1, f"{ident} is ambiguous: {vals}")
        c = census()
        for f in (
            "test/registered/unit/weg2/test_weg2_no_route_1290.py",
            "test/registered/unit/weg2/test_weg2_handback_rd_1291.py",
        ):
            hit = [
                loc for names in c.values() for locs in names.values()
                for loc in locs if loc.startswith(f + ":")
            ]
            self.assertTrue(hit, f"{f} carries form 3 and must be credited")

    def test_no_concat_operand_is_left_unresolved(self):
        """The one way this hardening could go quietly blind again: a marker
        constant defined somewhere the resolution cannot see it. That must
        FAIL here, not drop the code from the census."""
        c = census()
        bad = {
            code: dict(names) for code, names in c.items()
            for n in names if n.startswith((UNRESOLVED, AMBIGUOUS))
        }
        self.assertEqual(
            bad, {},
            "a W-code is concatenated with a marker this scan cannot resolve; "
            "resolve it (or define the marker beside its use) rather than "
            "letting the code drop out of the census",
        )

    def test_the_foreign_named_holders_are_occupied_not_free(self):
        """#1306, the NAME half of the scope gap, and the one that would
        actually have bitten: a renumber picks "the first free number above
        the maximum" or "a number the census does not show", and the old
        census showed W19 and W27 as unused while both are held by a live
        refusal whose class is simply not ``Weg2``-prefixed.

        * ``W19 DormantResidueRefused`` -- emitted by ``front.py``'s own
          ``do_stop`` when measured dormant residue exceeds the reserve P
          budgeted, i.e. a boot-stopping refusal, in the census's own home
          directory.
        * ``W27 PPWidthDivergenceRefused`` -- a real ``RuntimeError`` subclass
          declared in ``managers/pp_admission_congruence.py`` and raised there.

        Neither collides with a ``Weg2`` name today. Pinned as OCCUPIED so the
        next pass cannot spend the number, which is the whole point of a
        census over a registry.
        """
        c = census()
        foreign = {
            code: {n for n in names if not n.startswith("Weg2")}
            for code, names in c.items()
        }
        foreign = {k: v for k, v in foreign.items() if v}
        self.assertEqual(
            foreign, FOREIGN_HOLDERS,
            "the set of W-codes held by a non-Weg2 refusal changed. A new one "
            "is a number that must NOT be handed out as free; a lost one is "
            "the scan going blind again.",
        )
        # and the load-bearing consequence, stated as an assertion rather than
        # left to the reader: these numbers are not in the free set.
        used = {int(re.match(r"W(\d{1,2})", code).group(1)) for code in c}
        self.assertIn(19, used, "W19 is held by DormantResidueRefused")
        self.assertIn(27, used, "W27 is held by PPWidthDivergenceRefused")

    def test_the_all_caps_banner_shape_is_not_read_as_a_holder(self):
        """PRECISION FOR THE PREVIOUS TEST, without which it is noise. The
        foreign-name form matches ``W<nn> <CamelCase>``, and this tree writes
        ``W10 DRAFTER``, ``W27 KILLED``, ``W30 SEAM``, ``W33 INVENTORY``,
        ``W35 CLASS``, ``W38 RETIRED`` and ``W11 DRAFT`` as LOG BANNERS. They
        are instrument names, not refusals, and admitting them would invent
        holders for half the numbering. The filter is "the name contains a
        lowercase letter"; the builtin filter is separate, because
        ``the W38 IndexError at pool_host/base.py:344`` is prose about a
        Python builtin.
        """
        for banner in ("W10 DRAFTER", "W27 KILLED", "W30 SEAM",
                       "W33 INVENTORY", "W35 CLASS", "W8b MOVES"):
            self.assertEqual(
                FOREIGN_NAME.findall(f'logger.info("{banner} ...")'), [],
                f"{banner!r} is a log banner and must not become a holder",
            )
        # THE SECOND NAMESPACE, and the one that made the verdict-suffix
        # filter necessary: certify_485.py numbers its acceptance criteria in
        # exactly this shape. Admitting them collided five codes at once.
        for item in ("W2 Seam census", "W3 Flips", "W4 Soak", "W5 Ranks",
                     "W6 Work-matched samples", "W1 NVML corridor"):
            self.assertEqual(
                FOREIGN_NAME.findall(f"  {item}: 0 breaches."), [],
                f"{item!r} is a cert_485 work item, not a refusal",
            )
        self.assertTrue(
            FOREIGN_NAME.findall('self.do_stop("W19 DormantResidueRefused",'),
            "a CamelCase refusal name after a code IS a holder",
        )
        c = census()
        self.assertNotIn("IndexError", set(c["W38"]))
        self.assertNotIn("IndexError", set(c["W40"]))

    def test_the_scan_reads_the_lowercase_reason_form(self):
        """#1306, the CASE half. ``reason=w62-stale`` is an emitted reason
        VALUE that reaches a boot log, and record SECTION 1bh had to hand-fix
        it during the xchg renumber because no uppercase pass sees it.

        CAN-FAIL PROOF ON THE REGEX, because the tree carries zero of these at
        this tip (the xchg branch is not merged here) and an empty result
        would otherwise pass whether the form works or not. Both measured
        false positives of the unanchored shape are asserted dead in the same
        breath -- a net this narrow is only useful if it is also this quiet.
        """
        self.assertEqual(
            LOWER_REASON.findall('logger.info("leg reason=w62-stale ...")'),
            [("62", "stale")],
            "the lowercase reason form is the one #1273's shadow emitted",
        )
        self.assertEqual(
            LOWER_REASON.findall('f"reason=w11b-not-rearmed"'),
            [("11b", "not-rearmed")],
        )
        # the two measured false positives of the unanchored form
        self.assertEqual(
            LOWER_REASON.findall(
                "# E.g., abhinavkulkarni/mosaicml-mpt-7b-instruct-w4-g128-awq"
            ), [], "a hyphen before the token means it is part of a slug",
        )
        self.assertEqual(
            LOWER_REASON.findall('keys = [f"w2-1298-m1-{kv}-{i:07d}"]'),
            [], "a digit after the hyphen is a ticket/shape id, not a reason",
        )
        # AND THE TREE READING, with its denominator named. The fixtures
        # above live in THIS file and the scan sees them, which is the proof
        # that the form is wired and not merely compiled; what must stay empty
        # is every OTHER file. A lowercase token appearing in a source module
        # is not a defect by itself -- it IS a site a renumber has to move,
        # and no uppercase grep will show it to whoever does the renumber.
        mine = "test/registered/unit/weg2/test_weg2_wcode_uniqueness_1263.py:"
        seen = lowercase_reasons()
        self.assertEqual(
            sorted(t for toks in seen.values() for t in toks),
            ["w11b-not-rearmed", "w62-stale"],
            "this file's own fixture tokens must be visible to the scan",
        )
        elsewhere = {
            code: {t: [l for l in locs if not l.startswith(mine)]
                   for t, locs in toks.items()}
            for code, toks in seen.items()
        }
        elsewhere = {c: {t: l for t, l in v.items() if l}
                     for c, v in elsewhere.items()}
        self.assertEqual(
            {c: v for c, v in elsewhere.items() if v}, {},
            "a lowercase W-code reason token appeared outside this file's own "
            "fixtures. Move it with the code at the next renumber.",
        )

    def test_prose_reservations_are_enumerated_with_their_holders(self):
        """#1306, the PROSE half -- the W57 shape from record SECTION 1bh: the
        xchg chain reserved a number in eight comments across four modules and
        never gave it an exception class, so the census could not see the
        claim and the serve line had already assigned it.

        Reported in two halves, because they are two different questions:

        * a code reserved in prose that DOES have a holder -- these are extra
          SITES a renumber must move, not a defect;
        * a code reserved in prose with NO holder anywhere -- a reservation
          that a census-driven renumber would spend. Pinned.
        """
        p = prose_reservations()
        c = census()
        self.assertTrue(p, "the reservation scan matched nothing at all")
        orphan = {
            code: sorted(files) for code, files in p.items() if code not in c
        }
        self.assertEqual(
            orphan, {k: sorted(v) for k, v in PROSE_ONLY_RESERVATIONS.items()},
            "a W-code is reserved in prose with no holder of any kind. It is "
            "invisible to the collision map by construction, so it is pinned "
            "here instead -- do not hand the number out.",
        )
        # the reservation vocabulary is anchored, not a bare-token sweep: the
        # bare shape matches ~1900 sites on this tree, nearly all of them
        # design-section markers and MoE weight names.
        self.assertLess(
            sum(len(ls) for f in p.values() for ls in f.values()), 100,
            "the reservation net went broad; it is a report, and a report "
            "nobody can read is not one",
        )

    def test_the_ignore_markers_are_bounded_and_named(self):
        """The suppression mechanism gets a denominator, or it becomes the
        allow-list this file's docstring refuses to keep. Every
        ``wcode-census: ignore`` line in the tree is counted per file and
        pinned; adding one is then a visible act in this file's diff, and a
        line that quietly acquires the marker fails here.

        Line numbers are deliberately NOT pinned -- this file already lost a
        day to a hard-pinned line that an unrelated merge moved by one.
        """
        self.assertEqual(
            _scan()["ignored"], IGNORED_LINES,
            "the wcode-census ignore markers changed. Each one must quote a "
            "HISTORICAL label rather than assign a live one; if it assigns "
            "one, the marker is wrong and the code is a real holder.",
        )

    def test_w52_belongs_to_the_no_route_refusal_alone(self):
        """#1257's regression, pinned. W52 is #1290's, claimed at the base
        commit 80de2d31d1 in two whitespace-free forms; the corridor pass was
        renumbered to W54 rather than sharing it."""
        c = census()
        self.assertEqual(set(c["W52"]), {"Weg2NoServiceableRoute"})

    def test_the_corridor_codes_are_w54_w55_w56_alone(self):
        """#1257c. W53 was claimed on the base by #1291's
        ``W53 Weg2StoreHandbackFailed`` between #1257's enumeration and its
        merge, so on the merged tree the two contradicted each other: this
        census sees both forms since the CONCAT hardening. The corridor pass
        renumbered to W55 (enumerated free against this same census) and the
        older claim keeps W53. W56 is the new verdict-only floor refusal."""
        c = census()
        self.assertEqual(set(c["W53"]), {"Weg2StoreHandbackFailed"})
        self.assertEqual(set(c["W54"]), {"Weg2CorridorBudgetUnpriced"})
        self.assertEqual(set(c["W55"]), {"Weg2CorridorBudgetWouldBind"})
        self.assertEqual(set(c["W56"]), {"Weg2CorridorFloorUnmeasured"})
        # and the label is not built by concatenation, because that is what
        # hid the previous claim from this very census.
        from sglang.srt.weg2 import corridor_budget as cb

        self.assertEqual(cb.UNPRICED_NAME, "W54 Weg2CorridorBudgetUnpriced")
        self.assertEqual(cb.WOULD_BIND_NAME, "W55 Weg2CorridorBudgetWouldBind")
        self.assertEqual(
            cb.UNMEASURED_FLOOR_NAME, "W56 Weg2CorridorFloorUnmeasured"
        )

    def test_the_serve_next5_train_renumbered_the_incoming_codes_by_census(self):
        """serve-next5 train (2026-09-09). Two input branches were cut on trees
        where W52-W56 were free: ``weg2/tp3-decode-dec2-0909`` (base
        4f762260ba, before #1236) claimed W52-W56 for the #1241/#1293
        operating-point refusals and the eager-decode arm, and
        ``weg2/fix-1298-0909`` claimed W55 for the #580 span STOP. On the
        merged tree every one of those collided with a serve-line holder
        (W52 no-route, W53 handback, W54-W56 corridor). Renumbered to the
        first free numbers above this census's maximum (W60 -> W61..W66),
        exception NAMES unchanged, so a boot log is still grepped by name."""
        c = census()
        self.assertEqual(set(c["W61"]), {"Weg2TpOperatingPointUnpriced"})
        self.assertEqual(set(c["W62"]), {"Weg2TpOperatingPointDisablesUnevenAxis"})
        self.assertEqual(set(c["W63"]), {"Weg2TpOperatingPointNeedsEnvPin"})
        self.assertEqual(set(c["W64"]), {"Weg2TpOperatingPointInfeasible"})
        self.assertEqual(set(c["W65"]), {"Weg2PrefetchSpanSplit"})
        self.assertEqual(set(c["W66"]), {"Weg2EagerDecodeArm"})
        # and the serve-line holders the incoming codes collided with kept
        # their numbers (the pushed serve-next4 tip's boot ticket names them).
        self.assertEqual(set(c["W52"]), {"Weg2NoServiceableRoute"})
        self.assertEqual(set(c["W53"]), {"Weg2StoreHandbackFailed"})
        self.assertEqual(set(c["W56"]), {"Weg2CorridorFloorUnmeasured"})

    def test_the_chosen_number_was_free_and_the_free_ones_are_named(self):
        """W50 is not 'the next one': it is the first free number above the
        highest assigned code, and the census can say which others are free.

        #1306 note: the free set is now computed over FIVE forms, so W19 and
        W27 have left it -- they were never free, the old scan just could not
        see their holders.
        """
        c = census()
        used = {int(m.group(1)) for code in c
                for m in [re.match(r"W(\d{1,2})", code)] if m}
        self.assertNotIn(50, used - {50}, "W50 must have been free before this")
        # #1257 renumbered to W54 by enumerating this same census. The numbers
        # below stayed free through both renumbers and are the next candidates.
        self.assertIn(54, used, "W54 is the corridor pass's number now")
        self.assertIn(55, used, "W55 is the would-bind refusal's number now")
        self.assertIn(56, used, "W56 is the unmeasured-floor refusal's number")
        for n in (14, 15, 23, 39):
            self.assertNotIn(n, used, f"W{n} was named free and is not")


if __name__ == "__main__":
    unittest.main()
