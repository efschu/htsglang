# SPDX-License-Identifier: Apache-2.0
"""Framed BAR1 slots: a payload protocol that does not rely on PCIe write order.

WHY THIS EXISTS (Fix A, 22.10., switch-p2p-wurzel-1022.md)
-----------------------------------------------------------
Every BAR1 kernel used to be "data, fence, flag; the receiver waits for the
flag and then reads the slot". That is correct only if the receiver's SMs see
the posted writes in the order the sender issued them. This rig's hardware
does not promise that:

* ``CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING`` is ``NONE`` on the
  RTX 3080 and ``OWNER`` on the RTX 5090 (measured 22.10.);
* the GPUs set Relaxed Ordering on their P2P posted writes, and a PCIe switch
  may let RO-marked writes pass each other (NVIDIA release notes 570.x,
  "regardless of the RO bit" for Ampere/Ada);
* with RO cleared the defect stays (E1, 22.10. 21:53Z: 3 of 8 runs wrong).

Measured symptom: the receiver read the END of its slot before the sender's
bytes were visible there, on the first rounds that touched a new slot area.

WHAT REPLACES THE ORDERING ASSUMPTION
-------------------------------------
Each slot is cut into TILES of 32 packets of 16 bytes (512 bytes). Packets
0..30 of a tile carry 31 data packets, packet 31 is the tile's TRAILER::

    trailer = { H.lo, H.hi, round.lo, round.hi }
    H       = mix(XOR_lanes mix2(packet_l, key(round, salt, tile, lane)) ^ key)

The receiver reads the tile (data and trailer in the same pass), recomputes H
from the data it actually read, and accepts the tile ONLY if the trailer
carries this collective's round AND the recomputed H. Otherwise it fences and
reads the tile again, until it matches or the deadline fires (status 3). No
statement about the ORDER in which the 512 bytes arrive is needed anywhere:

* trailer from an older round (stale) -> round mismatch -> rejected, always;
* trailer new, one data packet stale  -> the per-lane mix is a bijection of the
  packet, so the XOR differs -> rejected, always;
* trailer new, several packets stale   -> rejected unless a 64-bit hash
  collides (2^-64 per check);
* torn 16-byte packet (some dwords new) -> same as "packet stale".

Single-copy atomicity is not assumed either -- not for 8 bytes (NCCL LL), not
for 128 bytes (NCCL LL128, which NCCL itself only enables on NVLink).

PRICE: 1 of 32 packets is the trailer -> 96.9 % payload per wire byte (LL:
50 %), and the trailer goes out in the SAME warp store instruction as the
data, so it adds no TLP of its own. On the receiver: ~20 integer ops per 16
bytes and one warp XOR-reduction per 512 bytes, both on local VRAM data.

The flag lines and the #622 acknowledgement banks are unchanged: they still
carry back-pressure, the step structure, the rank agreement on the round and
the deadline/abort paths. They are simply no longer what makes a payload
byte VALID -- the trailer is.

``salt`` names the slot stream: ``(kind << 16) | (step << 8) | sender``, so a
tile written for a different step, topology or sender in the same round can
never pass (a routing defect becomes a status-3 abort, not a wrong number).
"""

from __future__ import annotations

#: Packets per tile and data packets per tile. The trailer is packet 31.
FRAME_LANES = 32
FRAME_DATA = 31
PACKET = 16
TILE_BYTES = FRAME_LANES * PACKET          # 512
TILE_DATA_BYTES = FRAME_DATA * PACKET      # 496

#: Slot-stream kinds, the high byte of the salt. Must match the CUDA side.
KIND_MESH = 1
KIND_ONESHOT = 2
KIND_RING = 3
KIND_A2A = 4

#: Status word value of a framed receive that never verified (deadline/abort
#: while waiting for a tile's trailer to match). 1 = payload barrier,
#: 2 = entry acknowledgement wait (#622), 3 = frame verification.
STATUS_FRAME = 3


def framed_packets(n4: int) -> int:
    """Slot packets that carry ``n4`` data packets (data + trailers)."""
    n4 = int(n4)
    if n4 <= 0:
        return 0
    return -(-n4 // FRAME_DATA) * FRAME_LANES


def framed_bytes(nbytes: int) -> int:
    """Slot bytes that carry ``nbytes`` of payload (rounded to 16 first)."""
    return framed_packets(-(-int(nbytes) // PACKET)) * PACKET


def capacity_of(stride_bytes: int) -> int:
    """Largest payload (bytes, multiple of 16) a slot of ``stride_bytes`` carries."""
    return (int(stride_bytes) // TILE_BYTES) * TILE_DATA_BYTES


def salt(kind: int, step: int, sender: int) -> int:
    return ((int(kind) & 0xFF) << 16) | ((int(step) & 0xFF) << 8) | (int(sender) & 0xFF)


# ---------------------------------------------------------------------------
# A host-side reference of the hash, bit-identical to the CUDA one. Used by the
# desk tests to pin the device code (a framing that silently changed on one side
# only would turn every collective into a status-3 abort) and by the mutant tests.
# ---------------------------------------------------------------------------
_M64 = (1 << 64) - 1


def mix64(v: int) -> int:
    v &= _M64
    v ^= v >> 30
    v = (v * 0xBF58476D1CE4E5B9) & _M64
    v ^= v >> 27
    v = (v * 0x94D049BB133111EB) & _M64
    v ^= v >> 31
    return v


def tile_key(round_: int, salt_: int, tile: int) -> int:
    return mix64(((int(round_) * 0x9E3779B97F4A7C15) ^ (int(salt_) << 40)
                  ^ (int(tile) * 0xD6E8FEB86659FD93)) & _M64)


def lane_hash(packet: bytes, key: int, lane: int) -> int:
    assert len(packet) == 16
    a = int.from_bytes(packet[0:8], "little")
    b = int.from_bytes(packet[8:16], "little")
    h = mix64(a ^ key ^ ((lane * 0xA24BAED4963EE407) & _M64))
    return mix64(h ^ b)


def tile_hash(packets, key: int) -> int:
    """``packets``: list of up to 31 16-byte data packets of ONE tile."""
    x = 0
    for lane, p in enumerate(packets):
        x ^= lane_hash(p, key, lane)
    return mix64(x ^ key)


def frame_slot(payload: bytes, round_: int, salt_: int) -> bytes:
    """The slot image a sender writes for ``payload`` (reference, host side)."""
    if len(payload) % PACKET:
        payload = payload + bytes(PACKET - len(payload) % PACKET)
    n4 = len(payload) // PACKET
    out = bytearray(framed_packets(n4) * PACKET)
    for t in range(-(-n4 // FRAME_DATA)):
        pk = [payload[(t * FRAME_DATA + l) * PACKET:(t * FRAME_DATA + l + 1) * PACKET]
              for l in range(FRAME_DATA) if t * FRAME_DATA + l < n4]
        key = tile_key(round_, salt_, t)
        h = tile_hash(pk, key)
        base = t * TILE_BYTES
        for l, p in enumerate(pk):
            out[base + l * PACKET:base + (l + 1) * PACKET] = p
        out[base + FRAME_DATA * PACKET:base + TILE_BYTES] = (
            h.to_bytes(8, "little") + (int(round_) & _M64).to_bytes(8, "little"))
    return bytes(out)


def verify_slot(slot: bytes, n4: int, round_: int, salt_: int) -> list:
    """Indices of tiles that a receiver would REJECT (reference, host side)."""
    bad = []
    for t in range(-(-int(n4) // FRAME_DATA)):
        base = t * TILE_BYTES
        pk = [slot[base + l * PACKET:base + (l + 1) * PACKET]
              for l in range(FRAME_DATA) if t * FRAME_DATA + l < n4]
        tr = slot[base + FRAME_DATA * PACKET:base + TILE_BYTES]
        key = tile_key(round_, salt_, t)
        if (int.from_bytes(tr[0:8], "little") != tile_hash(pk, key)
                or int.from_bytes(tr[8:16], "little") != (int(round_) & _M64)):
            bad.append(t)
    return bad


# ---------------------------------------------------------------------------
# The CUDA side. Spliced into barlink_bar1_ext._CUDA_SRC after the primitives
# (readV4/writeV4/u64 must already be defined). ONE copy of the source text, so
# the two kernels files cannot drift apart the way readV4/writeV4 did.
#
# FRAME_MUTANT (compile-time, tests only): 1 = accept a tile on the hash alone,
# ignoring the round in the trailer and leaving the round out of the key
# ("old generation accepted"); 2 = accept every tile without checking ("flag
# before data", the pre-22.10. protocol). Never set in a build that serves.
# ---------------------------------------------------------------------------
FRAME_CUDA_SRC = r"""
#ifndef FRAME_MUTANT
#define FRAME_MUTANT 0
#endif
#define FR_LANES 32
#define FR_DATA  31
#define FR_FULL  0xffffffffu
#define FR_KIND_MESH    1u
#define FR_KIND_ONESHOT 2u
#define FR_KIND_RING    3u
#define FR_KIND_A2A     4u
#define FR_STATUS 3u

__device__ __host__ __forceinline__ long long frPackets(long long n4)
{
    return n4 <= 0 ? 0LL : ((n4 + FR_DATA - 1) / FR_DATA) * FR_LANES;
}

__device__ __forceinline__ unsigned int frSalt(unsigned int kind, int step, int sender)
{
    return ((kind & 0xffu) << 16) | (((unsigned)step & 0xffu) << 8) | ((unsigned)sender & 0xffu);
}

__device__ __forceinline__ u64 frMix(u64 v)
{
    v ^= v >> 30; v *= 0xBF58476D1CE4E5B9ull;
    v ^= v >> 27; v *= 0x94D049BB133111EBull;
    v ^= v >> 31;
    return v;
}

__device__ __forceinline__ u64 frKey(u64 round, unsigned int salt, long long tile)
{
#if FRAME_MUTANT == 1
    round = 0ull;
#endif
    return frMix((round * 0x9E3779B97F4A7C15ull) ^ ((u64)salt << 40)
                 ^ ((u64)tile * 0xD6E8FEB86659FD93ull));
}

__device__ __forceinline__ u64 frLane(uint4 v, u64 key, int lane)
{
    const u64 a = ((u64)v.y << 32) | (u64)v.x;
    const u64 b = ((u64)v.w << 32) | (u64)v.z;
    const u64 h = frMix(a ^ key ^ ((u64)lane * 0xA24BAED4963EE407ull));
    return frMix(h ^ b);
}

__device__ __forceinline__ u64 frWarpXor(u64 h)
{
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) h ^= __shfl_xor_sync(FR_FULL, h, o);
    return h;
}

// SENDER, one tile, whole warp. Lanes 0..30 carry data packet `v` (if
// `valid`), lane 31 carries the trailer. ONE store instruction for all 32
// lanes, so data and trailer leave as one coalesced 512-byte access.
__device__ __forceinline__ void frSendTile(uint4 *slot, long long tile, int lane,
                                           bool valid, uint4 v, u64 key, u64 round)
{
    u64 h = (lane < FR_DATA && valid) ? frLane(v, key, lane) : 0ull;
    h = frMix(frWarpXor(h) ^ key);
    uint4 w = v;
    if (lane == FR_DATA)
        w = make_uint4((unsigned)h, (unsigned)(h >> 32),
                       (unsigned)round, (unsigned)(round >> 32));
    if (lane == FR_DATA || valid) writeV4(slot + tile * FR_LANES + lane, w);
}

// SENDER, one tile to SEVERAL targets (mesh allgather): the hash is formed
// once, the same 32-lane store goes to each target.
template<int MAXR>
__device__ __forceinline__ void frSendTileN(uint4 *const *dst, int R, int self,
                                            long long tile, int lane, bool valid,
                                            uint4 v, u64 key, u64 round)
{
    u64 h = (lane < FR_DATA && valid) ? frLane(v, key, lane) : 0ull;
    h = frMix(frWarpXor(h) ^ key);
    uint4 w = v;
    if (lane == FR_DATA)
        w = make_uint4((unsigned)h, (unsigned)(h >> 32),
                       (unsigned)round, (unsigned)(round >> 32));
    if (lane == FR_DATA || valid) {
        for (int z = 0; z < R; ++z) {
            if (z == self) continue;
            writeV4(dst[z] + tile * FR_LANES + lane, w);
        }
    }
}

// Cache-bypassing 16-byte read as two ld.mmio.relaxed.sys u64 loads (.mmio has
// no vector form). Used only on the RETRY path of frRecvTile.
__device__ __forceinline__ uint4 frReadMmio(const uint4 *p)
{
    u64 a, b;
    asm volatile("ld.mmio.relaxed.sys.global.u64 %0, [%1];" : "=l"(a) : "l"(p) : "memory");
    asm volatile("ld.mmio.relaxed.sys.global.u64 %0, [%1];"
                 : "=l"(b) : "l"((const char *)p + 8) : "memory");
    return make_uint4((unsigned)a, (unsigned)(a >> 32), (unsigned)b, (unsigned)(b >> 32));
}

// RECEIVER, one tile, whole warp. Reads data and trailer in one pass, recomputes
// the hash from what it READ and returns only a tile whose trailer carries this
// round and that hash. Re-reads (after a system fence -- incoming PCIe writes are
// not seen by ld.cv alone on this rig, BEFUND_L2_NICHT_KOHAERENT.md) until it
// matches. Returns false on the cycle deadline or the host abort word; the caller
// then takes the kernel's abort path with status FR_STATUS. The return value is
// warp-uniform (every lane holds the same reduced hash and the same trailer).
__device__ __forceinline__ bool frRecvTile(const uint4 *slot, long long tile, int lane,
                                           bool valid, u64 key, u64 round, uint4 *out,
                                           u64 capCycles, const unsigned int *abortHost)
{
    long long t0 = 0;
    unsigned int probe = 0u;
    bool started = false;          // lane 0 only (deadline bookkeeping)
    int attempt = 0;               // every lane: warp-uniform
    for (;;) {
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (lane == FR_DATA || valid) {
            const uint4 *q = slot + tile * FR_LANES + lane;
            // First read: the ordinary cached path (steady state: the tile is
            // already there). RETRIES bypass the caches (ld.mmio): measured
            // 22.10. on this rig, a tile line read before the peer's bytes had
            // landed stays STALE in the receiving card's L2 for the rest of the
            // kernel -- ld.cv after __threadfence_system re-read zeros for
            // 2.3 s (status-3 aborts in the dissect reproducer) while the bytes
            // were in memory. ld.mmio is the load the flag waits already use
            // (readFlag LA_MMIO) and sees BAR1 writes on both cards.
            if (attempt == 0) v = readV4(q);
            else              v = frReadMmio(q);
        }
        u64 h = (lane < FR_DATA && valid) ? frLane(v, key, lane) : 0ull;
        h = frMix(frWarpXor(h) ^ key);
        const unsigned tx = __shfl_sync(FR_FULL, v.x, FR_DATA);
        const unsigned ty = __shfl_sync(FR_FULL, v.y, FR_DATA);
        const unsigned tz = __shfl_sync(FR_FULL, v.z, FR_DATA);
        const unsigned tw = __shfl_sync(FR_FULL, v.w, FR_DATA);
        const u64 th = ((u64)ty << 32) | (u64)tx;
        const u64 tr = ((u64)tw << 32) | (u64)tz;
#if FRAME_MUTANT == 2
        (void)th; (void)tr;
        const bool ok = true;
#elif FRAME_MUTANT == 1
        (void)tr;
        const bool ok = (th == h);
#else
        const bool ok = (th == h) && (tr == round);
#endif
        if (ok) { *out = v; return true; }
        // Give-up decision on lane 0 only and broadcast: every lane must leave
        // together, a lane that stayed would sit in the next __shfl_sync alone.
        int stop = 0;
        if (lane == 0) {
            if (!started) { t0 = clock64(); started = true; }
            else if ((u64)(clock64() - t0) > capCycles) stop = 1;
            if (abortHost != nullptr && ((++probe & 1023u) == 0u)
                && *(const volatile unsigned int *)abortHost != 0u) stop = 1;
        }
        if (__shfl_sync(FR_FULL, stop, 0)) return false;
        ++attempt;
        // BACK OFF before the next read. Measured 22.10. (dissect reproducer,
        // status-3 aborts at 4.7 s with back-to-back ld.mmio retries): warps
        // re-reading a not-yet-visible tile without pause never saw the peer's
        // bytes land -- the retry traffic must leave the receiving card's
        // memory path to the incoming writes. Exponential, 64 ns .. 32 us.
        __nanosleep(attempt < 10 ? (64u << attempt) : 32768u);
    }
}
"""
