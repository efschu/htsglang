"""Weg 2 (#1233): two process groups per card, one awake, the store as carrier.

Modules:

* ``host_ledger`` -- the #721 host ledger for the six-process shape at both
  moments (launch, run), the S/M arm ladder, the tmpfs store size, W20.
* ``launcher`` -- the sequenced group launcher of spec section 1.5: group P
  (PP=3 prefill, :30031) up, asleep with its measured dormant residue, group D
  (TP=3+NEXTN decode, :30032) up and awake, the front up, deadmen armed.
* ``front`` -- the phase router on :30030: request ledger, the sequential
  two-leg route, drain-and-flip on the #1011 work-exhaustion clocks plus the
  V1 fairness bound, the named refusals of spec section 5.

Nothing here duplicates an upstream mechanism: sleep/wake is the upstream
memory saver (slice S1), the carrier is the canonical page store (slice S5),
and the groups are two stock ``sglang.launch_server`` launches.
"""

#: THE TWO SCHEDULING BATCH SIZES, EACH WRITTEN EXACTLY ONCE.
#:
#: THE ORDER IN FORCE, user 2026-09-09, verbatim: "der decode bs6 soll mit bs6
#: (nicht mehr bs4) der standard werden."  D's default is therefore 6.
#:
#: SUPERSEDED, same day, and kept because it is the source of P's number and of
#: the re-measurability clause that still governs both -- user 2026-09-09,
#: verbatim: "nimm jetzt vorerst bs4 fuer decode und bs2 fuer prefill. wenn
#: alles fertig ist kann man das immernoch nachmessen wo da das optimum fuer
#: meinen anwendungsfall liegt."  P stays 2 from that order; D's 4 was replaced
#: by the order above before it ever reached metal.
#:
#: So these two numbers remain a PROVISIONAL operating point, explicitly
#: re-measurable once the strand is finished -- not a derived or proven
#: optimum, and nothing may treat them as one.  The supersession inside a
#: single day is the cheapest possible evidence that the pair moves: it moved
#: once already, and the only reason that cost one edit instead of twenty is
#: the single-site rule below.
#:
#: They live here, in the package root, because BOTH ends need them and neither
#: may own the other's copy: ``launcher`` owns ``--p-bs`` / ``--d-bs`` and every
#: pricing helper that reads a bs off the argv it builds, and ``front`` owns
#: ``--p-concurrency`` / ``--d-bs`` for the standalone case (the launcher always
#: writes both explicitly, R-6/R-12, so the front's own defaults bind only when
#: it is run by hand).  A second copy in either module would drift the day one
#: of the two numbers is re-measured -- exactly the class ``_max_running_requests``
#: and ``_p_page_size`` already avoid by reading the flag off the argv instead of
#: restating it.  Re-measuring the operating point must be an edit to THESE TWO
#: LINES and nowhere else.
#:
#: They stay INDEPENDENT (law 2 / C1-R-12): P's prefill concurrency and D's
#: decode seat count are two knobs, and their being unequal here is the point.

#: K1: group P's ``--max-running-requests`` AND the front's leg-1 concurrency.
DEFAULT_P_BS = 2

#: K2: group D's ``--max-running-requests`` AND the number of front D seats.
#: 6 by the order of 2026-09-09 quoted above ("nicht mehr bs4").
DEFAULT_D_BS = 6
