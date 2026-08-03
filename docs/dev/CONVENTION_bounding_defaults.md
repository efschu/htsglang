# Convention — every shipped bounding default carries a value pin

Status: adopted in #514, from audit #505 axis C. Applies to fork-added flags and
env entries; upstream ones are out of scope until we touch them.

## The rule

A numeric default that exists to BOUND something — a cap, budget, threshold,
limit, reserve, margin, watermark, quota, timeout — ships with a test that FAILS
when the value is changed.

Two lines, one place:

```python
# VALUE PINNING (#505-C-05). The literal is the contract.
self.assertEqual(envs.SGLANG_SOME_CAP.get(), 8, "...why 8, and where the argument lives...")
```

The message names the file:line of the argument behind the number. That is the
whole mechanism. It is deliberately cheap, because a convention that costs a
morning per posten will not be applied.

## Why the obvious version does not work

The pattern audit #505 found across the fork reads the default and derives the
test from it:

```python
max_retries = envs.SGLANG_RETRACT_SOLO_OOM_MAX_RETRIES.get()
for attempt in range(1, max_retries + 2):
    ...
```

`test/registered/unit/managers/test_retract_decode_fcfs.py`, before #514. It is
a good test of the guard — it proves the retry budget is enforced and that the
request fails cleanly past it. It is not a test of the VALUE: it passes for 8,
for 16, for 100000. The number is untested, and a number nobody tests is a
number nobody has to justify.

That is not a lapse in one file. Axis C enumerated 106 fork-added bounding
defaults and found **zero** with a test that fails when the value is doubled or
removed, while 71 of the 106 sit behind a gate that is off in the served
configuration and therefore cannot act at all. The strongest possible finding —
a knob with no consumer — was **absent**: the defect here is not dead knobs, it
is live knobs whose values nothing has ever tested.

## What this prevents

#449 is the worked example, and it is why CLAUDE.md carries the REACH INCLUDES
PARAMETERS law. `SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB` was a correct mechanism
with a correct implementation, shipped at a desk-picked 2048 MiB that sat above
the real peak. It bound nothing for weeks. Nothing was broken; nothing was
protected either. A value pin would not have supplied the measurement, but it
would have made the *absence* of one visible at every commit that touched the
file, instead of only to an audit two months later.

## Three tiers of evidence

Record which tier a posten is at, in the pin message. Ranked:

- **(a) A falsifier.** A test that fails when the default is raised, lowered or
  removed. For a value pin this is the pin itself; for a BINDS claim it is a
  test showing behaviour differs at the default and one step past it.
- **(b) Calibrated at boot.** The default is not a value at all but a
  measurement taken at startup. `SGLANG_BARLINK_PIPE_CHUNK_MIB` is the fork's
  one example (`barlink_device.py:1070-1140` sweeps candidates over the real
  all-reduce path and picks the summed minimum). This is the best tier and the
  right target for anything performance-shaped.
- **(c) A recorded measurement at the constant.** A comment naming the
  measurement the number came from, with the task number.
  `SGLANG_ADAPTIVE_SERVING_MARGIN_MIB` (`environ.py:1430-1432`) is the model:
  "148 MiB post-map free OOM'd at KV-full deep prefill, 1367 MiB survived; 512
  is the enforced floor between them". Note that even this one is an
  interpolation between an OOM point and a survival point, not a measured
  threshold — say so when it is.

Anything below (c) is a desk value. Desk values are allowed — someone has to
pick a first number — but they are labelled, and the pin makes the label
unavoidable.

## Scope, honestly

A value pin proves the default is DELIBERATE. It does not prove the default is
CORRECT, and it does not prove the default BINDS at the served geometry. Those
are separate claims needing separate evidence, and axis C's backlog is exactly
the list of postens that have neither. Do not read a green pin as a measured
number — that would be the same category error the SUCCESS-CLAIMS law is about.

## Applying it

New bounding default: add the pin in the same change. Touching an existing one:
add the pin then, rather than opening a project to retrofit all 106. The audit
backlog in `AUDIT_505_silent_wrongness.md` §"Axis C" is ranked by damage
potential and is the order to work in when there is dedicated time; the
`#449` GPU measurement arm (`NOTE_449_dsv4_indexer_query_chunk.md` §5) remains
BOOT-PENDING and needs a GPU window, not a desk pass.
