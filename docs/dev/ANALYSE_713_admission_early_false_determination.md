# ANALYSE #713 — the admission early-false: candidate CONFIRMED, fix already landed elsewhere

Verdict: **the labeled candidate is correct, and the fix exists, tested, on
`train/0817-control`. This branch cannot host it — the code it patches is not
here. Do not rebuild.**

## 1. The candidate is right, and here is the mechanism

`recv_requests` evaluates the phase policy BEFORE the request reaches the
queue the policy reads:

```
recv_reqs = self._pull_raw_reqs()      # the new request is HERE
policy_req = self.phase_policy_hook()  # the policy evaluates NOW
                                       # -> later: waiting_queue.append(req)
```
(`scheduler_components/request_receiver.py:103`, append at `scheduler.py:4089`,
line numbers on `train/0817-control`.)

`_pending_prefill_tokens` sums `waiting_queue` plus the chunked remainder, so
on an idle box it reads **0** at exactly the moment the policy asks whether the
target layout can admit. `_layout_admits` (`scheduler.py:3320`) then early-
falses on its FIRST line:

```python
if int(pending_tokens) <= 0:
    return False
```

which is exactly why the specimen's diagnostic showed BOTH pp terms holding —
72033 rows >= need, 3 mamba slots free — while the verdict was still False:
**the function never reached them.** TP refuses too (nothing resident), so
`_idle_locked_inputs` returned `(nothing_can_run=True, target_can_admit=False)`
and the policy declined the flip. 31.64 s TTFT for a ten-token prompt.

**Which number is wrong: the 22 is the truth, the 0 is the defect.** The work
has been received and will be queued in this same round; the queue has simply
not been told yet. So this is the "formation-vs-queue accounting seam" of the
three hypotheses in the brief — not a queued-but-unformed request, and not
uncounted chunked commitment.

Note what it is NOT, because that candidate was already killed:
`_idle_locked_inputs` passes ONE pending value to both `_layout_admits` calls,
so pp and tp cannot disagree. They do not. Both read the same stale zero.

## 2. The fix exists, on `train/0817-control`

`3c1aa6f29b` "[#713] The flip policy asked an empty queue: count work that has
ARRIVED", live on that branch (verified present, not reverted — the reverted
#713 commit is `4a16043d1a`, a different one about post-cutover settle):

| piece | file:line |
| --- | --- |
| `_arriving_prefill_tokens(inflight)` | `scheduler.py:422` |
| `_pending_prefill_tokens(self, inflight=None)` | `scheduler.py:8600` |
| `maybe_arm_phase_policy(self, inflight_reqs=None)` | `scheduler.py:8686` |
| one reading used for verdict AND message | `scheduler.py:8836` |
| the receiver hands the policy its own batch | `scheduler.py:9156` |

It fixes the INPUT and not the rule — the rule was already proven correct on
replayed inputs — and it counts only items that actually carry a prompt,
because `recv_reqs` is heterogeneous (aborts, the policy's own flip arm,
control traffic) and counting a control message as work would arm a flip on
nothing: the opposite defect and a harder one to see.

Tests already there: `test_admission_intake_713.py`,
`test_idle_locked_terms_713.py`, `test_resident_prefill_visible_713.py`.

## 3. Why this branch stops

None of the #713 commits is in this branch's ancestry, and neither
`_layout_admits` nor `target_can_admit` nor `_idle_locked_inputs` exists in its
tree. `scheduler.py` is **8675 lines here against 9950 there** — the whole
idle-locked admission mechanism postdates this lineage. There is no early-false
here to fix; writing one would be inventing a second admission path to patch.

Items 2-4 of the task follow from that: the red-first test, the neighbour check
against #689 and #698/#699, and the F4-r4 boot validation all belong on the
branch that has the code. The boot item still stands and is named here:
**TTFT < 3 s for a short prompt on an idle box, after the fix, on
`train/0817-control`** — it has not been measured on metal.

## 4. Operator note: this is the second lineage-stop in a row

The previous slice (#329 per-session KV) stopped for the same structural
reason, and the two together are a signal rather than a coincidence: this
branch family is far enough behind the trains that recent work keeps landing
outside it. Re-basing the `feat/329-*` / `fix/713-*` family onto the current
train before the next refill would move the prior-art gate from the END of a
task to the beginning, where it costs minutes instead of an assignment.
