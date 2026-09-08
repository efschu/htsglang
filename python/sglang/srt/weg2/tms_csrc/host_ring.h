#pragma once
// WEG2 flip-cost C1: the per-card SHARED HOST GRANULE RING.
//
// WHAT IT REPLACES (upstream-minimal law, spec section 4 C1-C5): the
// per-allocation ``cudaMallocHost`` / ``cudaFreeHost`` pair inside
// ``TorchMemorySaver::pause`` / ``::resume``.  The ring does not sit BESIDE
// that path -- there is no second host-byte bookkeeping, no journal, no LRU
// (spec section 10.8).  A boot either publishes ``TMS_HOST_RING_*`` and every
// cpu-backed allocation of both co-located ranks draws its host bytes from the
// one per-card region, or it publishes nothing and the stock ``cudaMallocHost``
// path runs unchanged (spec section 10.4: the only fallback is boot-level and
// all-or-nothing).
//
// WHY A RING AT ALL (spec R5): the two process groups flip weights in opposite
// directions on the same card at the same time.  With private pinned images the
// host must hold BOTH full images at one instant (61.4 GiB, refused in record
// section 1h).  With one shared region of ``H(c) = max_g image_g(c)`` bytes the
// walk is provably deadlock-free as long as
//     H(c) >= image_W(c) - device_credit(c) + max_tag_S(c) + max_tag_W(c)
// which the launcher checks per card per direction BEFORE either group starts
// and refuses by name (W32, spec C20).  ``acquire`` is therefore BLOCKING: a
// rank that cannot get its granules yet waits for the peer's release rather
// than falling back to a second allocator that the ledger cannot price (R4).
//
// SECOND CONSUMER (WEG2_CARRIER_SPEC_0907.md Amendment 2, A2-1..A2-4): the same
// per-card region is also the L2 host tier of both process groups.  That is why
// every granule carries a TAG FAMILY and why ``acquire``/``stats`` are keyed by
// it: the carrier acquires from this region with ``TMS_RING_FAMILY_CARRIER``
// and is accounted separately without a second implementation.  This header is
// the API surface for that; the carrier read itself is NOT built here.
//
// THE FORM WAS NOT ASSUMED (spec R16 / C0(a)(b)): whether the driver accepts
// ``cudaHostRegister`` on a cross-process shared mapping is a METAL question,
// not a desk one, and it was settled on the metal -- record
// WEG2_BUILD_DECISIONS_0906 section 1p, 2026-09-07T23:13:23Z: two processes
// register the SAME 4 GiB ``/dev/shm`` ``MAP_SHARED`` range with
// ``cudaSuccess`` in both, the roundtrip is bit-exact in both directions,
// ``RLIMIT_MEMLOCK`` does not account it, and the verdict is
// ``BUILD-MAP_SHARED`` with "the memfd fallback is NOT NEEDED".  So there is
// ONE form: a ``MAP_SHARED`` file under a tmpfs directory, addressed by PATH.
// The launcher still publishes ``TMS_HOST_RING_FORM`` and anything but
// ``MAP_SHARED`` is refused by name (W33 Weg2RingFormUnproven) rather than
// guessed at.  An fd number is never part of the map: the ranks are
// ``spawn``-started scheduler processes and inherit no descriptor.
//
// The header stores NO absolute pointer (section 1p's one build caveat): the
// two mappings of a region are at unrelated virtual addresses, so a granule is
// per-process ``base_ + offset``.

#include <pthread.h>
#include <stdint.h>
#include <string>
#include <vector>

#include "macro.h"

//: 2 MiB, the driver's own allocation granularity on this rig
//: (``MEMCREATE_CHUNK_SIZE``) and a page-aligned size for ``cudaHostRegister``.
static const size_t TMS_RING_GRANULE_BYTES = 2u * 1024u * 1024u;
//: The header occupies exactly one granule so that granule 0 of the DATA area
//: is 2 MiB aligned -- registration spans must start on a page boundary.
static const size_t TMS_RING_HEADER_BYTES = TMS_RING_GRANULE_BYTES;
//: 16384 x 2 MiB = 32 GiB per card, above the largest card on this rig; the
//: per-granule arrays are sized statically so the header layout is a constant
//: both processes agree on without negotiating.
static const uint32_t TMS_RING_MAX_GRANULES = 16384u;
static const uint64_t TMS_RING_MAGIC = 0x5747325248524E47ULL;  // "WG2RHRNG"
static const uint32_t TMS_RING_VERSION = 1u;

//: Tag families.  FREE is not a family, it is the absence of one.
static const uint8_t TMS_RING_FAMILY_FREE = 0u;
//: This slice: the weights image a group parks while the other group wakes.
static const uint8_t TMS_RING_FAMILY_FLIP_BACKUP = 1u;
//: CARRIER_SPEC Amendment 2: canonical HiCache pages / parked requests.  No
//: code in this slice acquires with it; it exists so the second consumer lands
//: as a caller of this module and not as a second region.
static const uint8_t TMS_RING_FAMILY_CARRIER = 2u;

//: The bounded wait of a blocking ``acquire``, in seconds.  It is NOT a tuning
//: knob and not operator-facing: it is the budget of the fence the acquire
//: already sits inside -- ``_weg2_group_fence``'s ``monitored_barrier``
//: (weight_updater.py:527) and ``DEFAULT_PCIE_LOCK_TIMEOUT_S``
//: (weg2_memory_saver.py) are both 120 s.  A ring wait allowed to outlive them
//: would surface as a fence expiry naming NOBODY, which is exactly the
//: rank-local silent failure spec section 10.5 forbids.  Reaching it is W31,
//: and W31 is provably unreachable given the launch check (R15).
static const double TMS_RING_ACQUIRE_BUDGET_S = 120.0;

//: The shared control page.  Every field in it is read and written by BOTH
//: co-located rank processes; nothing about a granule lives anywhere else.
struct HostRingHeader {
    uint64_t magic;
    uint32_t version;
    uint32_t granule_bytes;
    //: The launcher's boot epoch.  An attach that finds a DIFFERENT epoch is
    //: attaching to a previous boot's leftovers and zeroes the header.
    uint64_t epoch;
    uint32_t granules_total;
    //: Span 1 = ``image_P(c)`` worth of granules, registered at the first
    //: pause; the remainder is span 2, registered when a request first reaches
    //: past span 1 (R7: registering everything at the launch moment costs the
    //: ladder 2.98 GiB it does not have).
    uint32_t span1_granules;
    uint32_t granules_free;
    uint32_t granules_free_min;
    uint32_t granules_peak_taken;
    uint64_t acquires;
    uint64_t releases;
    uint64_t blocked_us;
    uint64_t swept_stale;
    uint32_t initialised;
    uint32_t _pad0;
    pthread_mutex_t mutex;  // PROCESS_SHARED | ROBUST
    pthread_cond_t cond;    // PROCESS_SHARED
    //: 0 = free.  Any other value is the pid that holds the granule, which is
    //: what makes the stale sweep possible without a second index.
    int32_t owner_pid[TMS_RING_MAX_GRANULES];
    uint8_t family[TMS_RING_MAX_GRANULES];
    //: A cheap hash of the acquiring tag, for L3/L4 attribution only.  Never
    //: read back as an identity -- the granule's CONTENT is only ever read by
    //: the process that wrote it (R11: the ring is a byte POOL).
    uint32_t tag_hash[TMS_RING_MAX_GRANULES];
};

struct HostRingStats {
    uint32_t granules_total;
    uint32_t granules_free;
    uint32_t granules_free_min;
    uint32_t granules_peak_taken;
    uint64_t acquires;
    uint64_t releases;
    double blocked_ms;
    uint64_t swept_stale;
    uint32_t span1_granules;
    uint32_t spans_registered;
};

class HostBackupRing {
public:
    //: Attach this process to the region of the card ``device`` sits on, or
    //: return ``nullptr`` when the boot published no ring (the stock
    //: ``cudaMallocHost`` path then runs, spec section 10.4).  The card is
    //: resolved by ``cuDeviceGetUuid`` and NEVER by the CVD ordinal: the two
    //: groups run with different ``CUDA_VISIBLE_DEVICES`` orders and an ordinal
    //: would silently pair the wrong ranks.  Refuses by name (W33) when a ring
    //: directory is published without a form, with a form other than
    //: MAP_SHARED, or with a map entry that is not exactly
    //: ``<uuid>=<bytes>:<span1>``.
    static HostBackupRing* open_from_env(CUdevice device);

    //: Blocking.  Returns exactly ``ceil(bytes / granule)`` granule pointers,
    //: not necessarily contiguous -- callers copy per granule, so a scatter
    //: list covers a size no contiguous run could.  W31 + exit(1) on the
    //: bounded wait expiring (backstop, R15).
    std::vector<void*> acquire(size_t bytes, uint8_t family, const std::string& tag);
    void release(const std::vector<void*>& granules);

    //: Page-lock span ``index`` (0 or 1) for this process's contexts.
    //: Idempotent per process.  Emits L2.
    void register_span(int index);

    HostRingStats stats() const;
    const std::string& card_uuid() const { return card_uuid_; }
    size_t granule_bytes() const { return TMS_RING_GRANULE_BYTES; }
    size_t bytes_to_granules(size_t bytes) const {
        return (bytes + TMS_RING_GRANULE_BYTES - 1) / TMS_RING_GRANULE_BYTES;
    }

private:
    HostBackupRing() {}
    HostBackupRing(const HostBackupRing&) = delete;
    HostBackupRing& operator=(const HostBackupRing&) = delete;

    void lock() const;
    void unlock() const;
    //: Free every granule whose ``owner_pid`` is not in ``/proc``.  Called on
    //: attach and whenever the robust mutex reports EOWNERDEAD.  Caller holds
    //: the lock.
    uint32_t sweep_stale_locked() const;
    void ensure_span_for_locked(uint32_t granule_index);
    void* granule_ptr(uint32_t index) const {
        return static_cast<char*>(base_) + TMS_RING_HEADER_BYTES +
               static_cast<size_t>(index) * TMS_RING_GRANULE_BYTES;
    }
    uint32_t granule_index(void* ptr) const;

    void* base_ = nullptr;
    size_t map_bytes_ = 0;
    HostRingHeader* h_ = nullptr;
    std::string card_uuid_;
    std::string path_;
    std::string form_;
    int fd_ = -1;
    bool span_registered_[2] = {false, false};
    int peer_pid_hint_ = 0;
};
