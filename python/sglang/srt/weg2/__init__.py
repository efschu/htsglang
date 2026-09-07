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
