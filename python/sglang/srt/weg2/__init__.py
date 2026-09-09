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

#: K3 -- THE ORDERED P CUT, from which the solver's POOL FLOOR is DERIVED AT
#: BOOT.  Moved here from ``launcher.py`` by the serve-next5 train (2026-09-09)
#: as a pool-token CONSTANT (448,027) and REPLACED on the same train by the
#: serve-next5 fixer (#1305) with the CUT the order actually names: the three
#: shipped operating-point constants of the two user orders of that day live
#: in ONE place, and ``launcher`` imports this one exactly as it imports the bs
#: pair above (upstream-minimal: one home, not two).  Its only consumer is
#: ``launcher.resolve_pool_floor``, which hands it to the solver.
#:
#: THE ORDER IN FORCE, user 2026-09-09, verbatim: "39,13,12 mit bs2 im pp
#: layout soll standard werden vorerst."
#:
#: THE CUT IS STILL SOLVED, NOT PINNED, and that distinction is the whole
#: reason this is a floor's SOURCE and not a ``--pp-stage-ratio``.  A hand pin
#: would make ``39,13,12`` survive a card change, a checkpoint change or a
#: re-measure that made it wrong; the solver stays the authority.  What the
#: order changes is a CONSTRAINT under ``--pp-solve-objective`` (makespan,
#: itself a user default since 2026-09-08): the floor under the shipped pool.
#:
#: THE RULE (#1305, operator ruling of 2026-09-09 after boot weg2sn5pre):
#: the floor is NOT a number written here.  It is READ OFF THIS BOOT'S OWN
#: FRONTIER by the solver: ``floor = pool(ordered cut)`` where the pool is
#: the one the solver just priced for 39,13,12 on this boot's census budgets.
#: Under "fastest cut that clears F" with F equal to the ordered cut's own
#: pool, the ordered cut is selected exactly (every faster frontier point
#: holds strictly less pool, by the definition of the frontier), and the
#: floor can never drift away from the frontier it is judged against,
#: because both are computed from the same inputs in the same solve.
#:
#: WHY NOT A NUMBER, measured: the previous form was the midpoint 448,027 of
#: the #1286b interval (414654, 481400] read off boot 68308f7fae's frontier
#: (record SECTION 1bf).  On the merged tree the census budgets changed
#: (corridor law 7d69c132e2, MEASURED-P floors) and the whole frontier moved
#: up in pool -- the interval that selects 39,13,12 became (482768, 578199]
#: -- so boot weg2sn5pre (2026-09-09, tip 319e76d60b) printed the order on
#: its provenance line and SHIPPED 42,11,11.  A hand-derived constant against
#: a frontier that moves with every census change is the defect CLASS, not
#: an instance; a constant "corrected" to 530,483 would have failed the same
#: way on the next re-price.
#:
#: WHEN THE ORDERED CUT IS NOT ON THE FRONTIER (dominated by another cut, or
#: not priceable on this rig's budgets), the solver REFUSES BY NAME
#: (``W67 Weg2PPCutOrderedCutOffFrontier``, planner/pp_cut_launch.py) and
#: prints the frontier and the dominating cut, instead of silently shipping a
#: neighbour.  ``--pp-solve-pool-floor N`` stays the operator's explicit
#: override (a number outranks the derivation), and ``--pp-solve-pool-floor
#: 0`` turns the floor OFF and restores the unfloored makespan byte for byte.
#:
#: THE TRADE THE ORDER BOUGHT, on the weg2sn5pre frontier so nobody re-derives
#: it as a regression: 39,13,12 (578,199 tokens / 421.4 ms per chunk) against
#: the unfloored makespan winner 44,10,10 (387,411 / 367.2 ms) is +49.3 %%
#: pool for +14.8 %% ms/chunk; against 42,11,11 (463,763 / 378.2 ms), the cut
#: the constant shipped, +24.7 %% pool for +11.4 %% ms/chunk.  The user chose
#: the knee; the value fork is the user's (record SECTION 1bg).
#:
#: PROVISIONAL in the same sense the bs pair is -- "vorerst" is in the order
#: -- so re-ordering the cut must stay an edit to THIS ONE LINE.
DEFAULT_PP_ORDERED_CUT = (39, 13, 12)
