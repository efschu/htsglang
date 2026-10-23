"""barlink-race-1023: an interleaving model of the framed bar1 ring, slow receiver.

The 23.10. 10:50Z specimen (pair 0,2, card 2 under a foreign load, "1 of 216
timed calls wrong ... status=0") raised the question whether a slow rank can
make the bar1 ring hand over a slot that was not yet -- or no longer -- the one
it was waiting for: slot reuse before the slow receiver finished reading, flag
before payload over PCIe posted writes with Relaxed Ordering, an old round taken
for a new one. The root turned out to be the BENCH (its timed loop ended on a
rank-local clock; test_barlink_round_bench_lockstep_1023.py). This file is the
transport half of that proof, kept as a regression guard:

* the ring kernel's step structure for R = 2 (``bar1_ring_kernel``:
  reduce-scatter step 0 into slot 0, allgather step into slot 1, the payload
  flag wait on equality with the launch round, round advanced at the end), and
  the framed receive (``frRecvTile``: a tile is consumed only when its trailer
  carries THIS round and the hash of the bytes read, else re-read),
  using the bit-identical host reference of the hash (``barlink_bar1_frame``);
* an ADVERSARIAL fabric: every posted write of a sender -- payload packets,
  trailers and the flag -- becomes visible at the receiver in an arbitrary,
  seeded order (3080: GPU_DIRECT_RDMA_WRITES_ORDERING = NONE, RO on, a PEX
  switch may reorder), and a receiver's tile read is 32 separate packet loads
  that writes can land between (torn tiles);
* a SLOW rank: one rank is scheduled 1 in ``slow`` times (a time-sliced card);
* several back-to-back calls with a round term in the payload, so a slot of
  the previous call is always there to be mistaken for the current one.

The property: every completed call carries exactly the right sum, or the rank
took an abort path (status 1 payload flag, 3 frame) -- never a wrong number
with status 0. MUTANTS (the CUDA ``FRAME_MUTANT`` 1 and 2, mirrored here) must
produce silent wrong results under the same schedules, which is the evidence
that the schedules reach the danger direction at all.

This is a MODEL of the protocol, not of the silicon; what it rules out is a
protocol-level race. The silicon half is the GPU repro in
/spinning/gpu-arb/deskq/done/barlink-race-1023.md.
"""

import random
import unittest

from flliper.srt.distributed.device_communicators import barlink_bar1_frame as fr
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

R = 2
N4 = 70                 # packets per call: chunks of 35 -> 2 tiles each
ZERO = bytes(16)
SPIN_CAP = 20000        # scheduler steps a flag wait may spin (status 1)
RETRY_CAP = 4000        # tile re-reads before a framed receive gives up (3)
FLAG_W = 4.0          # scheduler weight: one pending FLAG write becomes visible
DATA_W = 0.3          # scheduler weight: one pending PAYLOAD packet becomes visible
HOLD_P = 0.5          # probability a slot write is held until the receiver read once
M64 = (1 << 64) - 1


def chunk_bounds(n4, j, r_):
    base = n4 // r_
    rest = n4 - base * r_
    ln = base + (1 if j < rest else 0)
    off = j * base + (j if j < rest else rest)
    return off, ln


def pkt(v):
    return (int(v) & ((1 << 128) - 1)).to_bytes(16, "little")


def val(b):
    return int.from_bytes(b, "little")


def payload(rank, call, i):
    # differs per rank, call (the round term) and element; never 0
    return 1 + 1000003 * rank + 7919 * call + 31 * i


def key_of(round_, salt_, tile, mutant):
    return fr.tile_key(0 if mutant == 1 else round_, salt_, tile)


def tile_image(vals, round_, salt_, mutant):
    """{packet index in slot: 16 bytes} a sender writes for ``vals``."""
    pk = [pkt(v) for v in vals]
    img = {}
    for t in range(-(-len(pk) // fr.FRAME_DATA)):
        data = pk[t * fr.FRAME_DATA:(t + 1) * fr.FRAME_DATA]
        h = fr.tile_hash(data, key_of(round_, salt_, t, mutant))
        for lane, p in enumerate(data):
            img[t * fr.FRAME_LANES + lane] = p
        img[t * fr.FRAME_LANES + fr.FRAME_DATA] = (
            h.to_bytes(8, "little") + (round_ & M64).to_bytes(8, "little"))
    return img


def tile_ok(read, n_data, round_, salt_, tile, mutant):
    if mutant == 2:
        return True
    tr = read[fr.FRAME_DATA]
    th = int.from_bytes(tr[0:8], "little")
    trr = int.from_bytes(tr[8:16], "little")
    h = fr.tile_hash(read[:n_data], key_of(round_, salt_, tile, mutant))
    if mutant == 1:
        return th == h
    return th == h and trr == (round_ & M64)


class Fabric:
    """Receiver-side memory plus every sender's not-yet-visible posted writes.

    Two adversaries on top of the random visibility order:

    * flags become visible FLAG_W / DATA_W times faster than payload packets
      (the small posted write overtaking the payload under Relaxed Ordering);
    * HOLD: with probability HOLD_P a whole slot write is held back in the
      fabric until the receiver has made one full read attempt of that slot
      after the flag landed -- "flag before data" at its purest. The previous
      call's tiles are then what the receiver reads first.
    """

    def __init__(self, rng):
        self.rng = rng
        self.mem = {}           # (dst, slot, packet) -> bytes
        self.flag = {}          # (dst, step) -> round
        self.pending = []       # [kind, key, value, held_tag]
        self.held = set()       # (dst, slot) whose current burst is held
        self.consumed = {}      # (dst, slot) -> last round the receiver read
        self.early = 0          # slot writes for round N before N-1 was read

    def begin_burst(self, tag):
        if self.rng.random() < HOLD_P:
            self.held.add(tag)

    def release(self, tag):
        self.held.discard(tag)

    def post(self, kind, key, value):
        tag = (key[0], key[1]) if kind == "mem" and (key[0], key[1]) in self.held else None
        self.pending.append([kind, key, value, tag])

    def deliver(self, want_flag):
        """Make one pending write visible: a flag (``want_flag``) or a payload
        packet, chosen at random among the deliverable ones of that kind."""
        idx = [k for k, w in enumerate(self.pending)
               if (w[0] == "flag") == want_flag
               and (w[3] is None or w[3] not in self.held)]
        if not idx:
            return False
        kind, key, value, _ = self.pending.pop(self.rng.choice(idx))
        if kind == "mem":
            self.mem[key] = value
        else:
            self.flag[key] = value
        return True


def ring_rank(fab, r, calls, results, status, mutant):
    """``bar1_ring_kernel`` for R = 2, one generator step per memory access."""
    peer = (r + 1) % R          # rgSend target
    src = (r - 1 + R) % R       # fromRank
    round_ = 0
    for c in range(calls):
        N = round_ + 1
        inp = [payload(r, c, i) for i in range(N4)]
        out = [0] * N4
        for phase in (0, 1):
            if phase == 0:          # reduce-scatter, s = 0
                cs, cr, sl = r % R, (r - 1 + R) % R, 0
                offS, lenS = chunk_bounds(N4, cs, R)
                send = inp[offS:offS + lenS]
            else:                   # allgather, s = 0, sl = R - 1
                cs, cr, sl = (r + 1) % R, r % R, R - 1
                offS, lenS = chunk_bounds(N4, cs, R)
                send = out[offS:offS + lenS]
            offE, lenE = chunk_bounds(N4, cr, R)
            # sendPhase: framed packet stores, then the flag; all posted
            img = tile_image(send, N, fr.salt(fr.KIND_RING, sl, r), mutant)
            # SLOT REUSE BEFORE THE SLOW RECEIVER READ IT (the suspicion of
            # 23.10.): writing round N while the peer has not consumed N-1.
            if fab.consumed.get((peer, sl), 0) < N - 1:
                fab.early += 1
            fab.begin_burst((peer, sl))
            for p, b in img.items():
                fab.post("mem", (peer, sl, p), b)
                yield
            fab.post("flag", (peer, sl), N)
            yield
            # payload flag wait: equality with this launch's round
            spins = 0
            while fab.flag.get((r, sl), 0) != N:
                spins += 1
                if spins > SPIN_CAP:
                    status[r] = 1
                    return
                yield
            # framed receive, tile by tile, packet by packet
            got = []
            n_t = -(-lenE // fr.FRAME_DATA)
            for t in range(n_t):
                n_data = min(fr.FRAME_DATA, lenE - t * fr.FRAME_DATA)
                tries = 0
                while True:
                    read = []
                    for lane in range(fr.FRAME_LANES):
                        read.append(fab.mem.get((r, sl, t * fr.FRAME_LANES + lane), ZERO))
                        yield
                    fab.release((r, sl))    # one full attempt made: un-hold
                    if tile_ok(read, n_data, N, fr.salt(fr.KIND_RING, sl, src), t, mutant):
                        got.extend(val(b) for b in read[:n_data])
                        break
                    tries += 1
                    if tries > RETRY_CAP:
                        status[r] = 3
                        return
                    yield
            fab.consumed[(r, sl)] = N
            for i in range(lenE):
                out[offE + i] = (inp[offE + i] + got[i]) if phase == 0 else got[i]
        round_ = N
        results[r].append(out)


def run(seed, calls=4, slow=8, mutant=0):
    """One schedule. Returns (completed calls per rank, silent wrongs, status)."""
    rng = random.Random(seed)
    fab = Fabric(rng)
    results = [[] for _ in range(R)]
    status = [0] * R
    gens = {r: ring_rank(fab, r, calls, results, status, mutant) for r in range(R)}
    slow_rank = seed % R
    steps = 0
    while gens and steps < 3_000_000:
        steps += 1
        choices, weights = [], []
        if fab.pending:
            # FLAG OVERTAKES PAYLOAD: flags become visible FLAG_W / DATA_W
            # times faster than payload packets -- the small posted write that
            # passes the payload under Relaxed Ordering, the adversary the
            # frame exists for. Payload packets land in random order.
            choices += ["f", "d"]
            weights += [FLAG_W, DATA_W]
        for r in gens:
            choices.append(r)
            weights.append(1 if r == slow_rank else slow)
        pick = rng.choices(choices, weights)[0]
        if pick in ("f", "d"):
            fab.deliver(pick == "f")
            continue
        try:
            next(gens[pick])
        except StopIteration:
            del gens[pick]
        if any(status):
            break
    want = [[sum(payload(q, c, i) for q in range(R)) for i in range(N4)]
            for c in range(calls)]
    wrong = 0
    for r in range(R):
        for c, out in enumerate(results[r]):
            if out != want[c]:
                wrong += 1
    run.early = fab.early
    return [len(x) for x in results], wrong, list(status)


SEEDS = range(24)


class TestFramedRingUnderASlowReceiver(CustomTestCase):

    def test_the_real_protocol_never_hands_over_a_wrong_number(self):
        silent, completed, aborted, early = [], 0, 0, 0
        for seed in SEEDS:
            done, wrong, st = run(seed)
            early += run.early
            completed += sum(done)
            aborted += bool(any(st))
            if wrong:
                silent.append((seed, done, wrong, st))
        self.assertEqual(silent, [], msg=f"silent wrong results: {silent}")
        # and the ring's step chain excludes slot reuse before the slow
        # receiver read the slot -- by structure, not by the frame
        self.assertEqual(early, 0, msg="a slot was rewritten before it was read")
        # the schedules must actually carry calls through, or the property
        # above would hold vacuously
        self.assertGreater(completed, len(SEEDS) * R * 4 * 3 // 4,
                           msg=f"completed {completed}, aborted {aborted}")

    def test_mutant_old_round_accepted_is_killed(self):
        # FRAME_MUTANT 1: round left out of the key and the trailer check --
        # the previous call's tile, same salt and tile index, passes.
        wrong = sum(run(seed, mutant=1)[1] for seed in SEEDS)
        self.assertGreater(wrong, 0, msg="mutant 1 survived every schedule")

    def test_mutant_no_check_is_killed(self):
        # FRAME_MUTANT 2: the pre-22.10. protocol, flag before data accepted.
        wrong = sum(run(seed, mutant=2)[1] for seed in SEEDS)
        self.assertGreater(wrong, 0, msg="mutant 2 survived every schedule")


if __name__ == "__main__":
    unittest.main()
