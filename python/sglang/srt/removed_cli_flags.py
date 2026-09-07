# SPDX-License-Identifier: Apache-2.0
"""The compatibility surface of a removed CLI flag: one loud refusal, no shim.

#1233 (WEG 2, slice S0).  Spec §7 item 4 decides the shape -- "hard-remove with
a loud 'flag removed, see WEG2' refusal at parse time -- no silent shim" -- and
this module is that refusal, kept OUT of ``server_args.py`` for one reason: the
message has to SPELL the removed flags, and the S0 gate holds ``server_args.py``
to zero executable flip vocabulary.  Naming the dead mechanism belongs in the
one file whose whole job is to say it is dead.

S7 deletes this module together with the rest of the flip vocabulary, once no
launch line in the field can still carry these spellings.
"""

from typing import Dict, List, Optional

#: The CLI spellings #1233 (WEG 2, slice S0) HARD-REMOVED, each mapped to its
#: replacement or to ``None`` when the mechanism itself is gone.
#:
#: ONE DEFINITION, and the gate reads it from here rather than keeping a second
#: list beside it: a refusal list that drifts from the flags it refuses is the
#: fork-owned second bookkeeping the upstream-minimal law forbids.
#:
#: WHY A REFUSAL AND NOT SILENCE.  Spec §7 item 4 decides this shape: "hard-
#: remove with a loud 'flag removed, see WEG2' refusal at parse time -- no
#: silent shim".  argparse's own answer to an unknown option is
#: ``unrecognized arguments: --enable-phase-flip``, which is exactly what a
#: TYPO produces too; it neither says the mechanism was removed nor names what
#: replaced it, and a stale launch line then reads as an operator mistake.
#: `--phase-flip-canonical-kv-page` makes the difference load-bearing: the flag
#: was RENAMED, not deleted, and the page it arms is Weg 2's only carrier
#: across a flip, so an operator who reads "unrecognized" and drops the flag
#: silently boots without the carrier.
REMOVED_FLIP_FLAGS: Dict[str, Optional[str]] = {
    "--enable-phase-flip": None,
    "--phase-flip-policy": None,
    "--phase-flip-purity": None,
    "--phase-flip-tp-vector": None,
    "--phase-flip-spill-depth": None,
    "--phase-flip-corridor-floor-mib": None,
    "--phase-flip-image-file-backed": None,
    "--phase-flip-canonical-kv-page": "--hicache-canonical-kv-page",
    "--phase-flip-writeback": None,
    "--phase-flip-writeback-deadline-s": None,
    "--phase-flip-rebind-hicache": None,
}


def refuse_removed_flip_flags(argv: List[str]) -> None:
    """Refuse, by name, any argv entry naming a flag #1233 removed.

    Runs BEFORE ``parser.parse_args`` -- and after the ``--config`` merge, so a
    flag injected by a config file is refused on the same terms as one typed on
    the command line.  Every hit is reported in one message: an operator whose
    launch line carries five of them should have to re-read it once, not five
    times.

    ``--flag=value`` is matched as well as ``--flag value``, because a launcher
    that writes the joined form would otherwise walk past the refusal.
    """
    hits = []
    for entry in argv:
        if not isinstance(entry, str) or not entry.startswith("--"):
            continue
        name = entry.split("=", 1)[0]
        if name in REMOVED_FLIP_FLAGS and name not in hits:
            hits.append(name)
    if not hits:
        return
    lines = [
        "This build has no in-process phase flip. #1233 (WEG 2) replaced the "
        "PP-prefill <-> TP-decode cutover inside one process with two "
        "independent process groups per card, so the flags that armed it are "
        "removed rather than ignored:",
    ]
    for name in hits:
        replacement = REMOVED_FLIP_FLAGS[name]
        if replacement:
            lines.append(
                "  %s -- REMOVED, renamed to %s (same mechanism, new name)."
                % (name, replacement)
            )
        else:
            lines.append(
                "  %s -- REMOVED with the mechanism it armed; there is no "
                "replacement flag." % name
            )
    lines.append(
        "See /spinning/gpu-arb/weg2/WEG2_DESIGN_SPEC_2026-09-06.md (§7 item 4) "
        "for the deletion surface, and boot the two groups from the WEG 2 "
        "launcher instead of arming a flip on one."
    )
    raise ValueError("\n".join(lines))
