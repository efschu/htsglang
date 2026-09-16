/* Shared-memory page ARENA for the HiCache file backend (BAUPLAN_SHM_ARENA_0916).
 *
 * ONE arena file in /dev/shm, mapped by every rank process of both groups.
 * Content-addressed: a slot holds one canonical page (or blob) of a fixed
 * width, keyed by a 128-bit key; the index is an open-addressing hash table
 * on the key's low 64 bits. No lock crosses processes: slots are claimed by
 * CAS, coverage is accumulated by atomic OR over a 256-byte granule bitmap,
 * completion is a CAS on the slot state, readers hold a refcount around
 * their copy. Everything here is plain memory over an mmap; the caller
 * (Python) owns the file, the mmap and the disk tier behind the arena.
 *
 * Layout (all offsets 64-byte aligned):
 *   ArenaHeader | index[2*slots] (u64 key_lo each; 0 = empty, ~0 = tombstone;
 *   the slot id lives in a parallel u32 array) | SlotHeader[slots] (each
 *   header_bytes, holding the bitmap) | data[slots * slot_bytes]
 *
 * Slot states: 0 FREE, 1 CLAIMED (being filled), 2 COMPLETE (readable),
 * 3 EVICTING (leaving; no new readers).
 */
#define _GNU_SOURCE
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

#define A_MAGIC 0x41524e4132363931ULL /* "ARNA2691" */
#define S_FREE 0u
#define S_CLAIMED 1u
#define S_COMPLETE 2u
#define S_EVICTING 3u
#define TOMB (~0ULL)
#define KV_IVALS 64      /* interval capacity for slots <= 1 MiB */
#define BLOB_IVALS 8192  /* interval capacity for larger slots (mamba blobs) */

typedef struct {
    uint64_t magic;
    uint64_t slots;
    uint64_t slot_bytes;
    uint64_t header_bytes;   /* per-slot header size incl. bitmap */
    uint64_t index_off;      /* byte offsets from file start */
    uint64_t index_slot_off; /* u32 array parallel to the key index */
    uint64_t headers_off;
    uint64_t data_off;
    uint64_t index_cap;      /* 2 * slots, power of two */
    _Atomic uint64_t clock_hand;
    _Atomic uint64_t n_complete;
    _Atomic uint64_t n_claimed;
    uint8_t pad[64 - 8 * 11 % 64];
} ArenaHeader;

typedef struct { uint64_t lo, hi; } Ival;

typedef struct {
    _Atomic uint32_t state;
    _Atomic uint32_t refcount;
    _Atomic uint32_t clock_bit;
    uint32_t generation;
    uint64_t key_lo;
    uint64_t key_hi;
    uint64_t total_bytes;
    _Atomic uint32_t lock;     /* spinlock over the interval list */
    uint32_t n_ivals;          /* merged intervals in use */
    uint64_t cap_ivals;        /* capacity of ivals[] */
    uint8_t pad[8];
    char stem[192];            /* the store stem, so ANY rank can evict this page to disk */
    Ival ivals[];              /* cap_ivals entries, sorted, disjoint */
} SlotHeader;

#define STEM_CAP 192

static inline ArenaHeader *hdr(uint8_t *base) { return (ArenaHeader *)base; }
static inline _Atomic uint64_t *index_keys(uint8_t *base) {
    return (_Atomic uint64_t *)(base + hdr(base)->index_off);
}
static inline _Atomic uint32_t *index_slots(uint8_t *base) {
    return (_Atomic uint32_t *)(base + hdr(base)->index_slot_off);
}
static inline SlotHeader *slot_hdr(uint8_t *base, uint64_t s) {
    return (SlotHeader *)(base + hdr(base)->headers_off + s * hdr(base)->header_bytes);
}
static inline uint8_t *slot_data(uint8_t *base, uint64_t s) {
    return base + hdr(base)->data_off + s * hdr(base)->slot_bytes;
}

/* Size the file for `slots` slots of `slot_bytes`; returns the total bytes and
 * fills the offsets into out[0..5] = header_bytes, index_off, index_slot_off,
 * headers_off, data_off, index_cap. */
static uint64_t ival_cap_for(int64_t slot_bytes) {
    return slot_bytes > (1 << 20) ? BLOB_IVALS : KV_IVALS;
}

int64_t arena_layout(int64_t slots, int64_t slot_bytes, int64_t *out) {
    uint64_t cap_iv = ival_cap_for(slot_bytes);
    uint64_t header_bytes = (sizeof(SlotHeader) + cap_iv * sizeof(Ival) + 63) & ~63ULL;
    uint64_t cap = 1;
    while (cap < (uint64_t)slots * 2) cap <<= 1;
    uint64_t off = 4096;
    uint64_t index_off = off;
    off += cap * 8;
    off = (off + 63) & ~63ULL;
    uint64_t index_slot_off = off;
    off += cap * 4;
    off = (off + 4095) & ~4095ULL;
    uint64_t headers_off = off;
    off += (uint64_t)slots * header_bytes;
    off = (off + 4095) & ~4095ULL;
    uint64_t data_off = off;
    off += (uint64_t)slots * (uint64_t)slot_bytes;
    out[0] = (int64_t)header_bytes;
    out[1] = (int64_t)index_off;
    out[2] = (int64_t)index_slot_off;
    out[3] = (int64_t)headers_off;
    out[4] = (int64_t)data_off;
    out[5] = (int64_t)cap;
    return (int64_t)off;
}

/* Initialise a fresh (zeroed) mapping. Idempotent on a mapping that already
 * carries the magic with the same geometry (returns 1); 0 = initialised now;
 * -1 = a different arena lives here. */
int arena_init(uint8_t *base, int64_t slots, int64_t slot_bytes) {
    int64_t o[6];
    arena_layout(slots, slot_bytes, o);
    ArenaHeader *h = hdr(base);
    if (h->magic == A_MAGIC) {
        return (h->slots == (uint64_t)slots && h->slot_bytes == (uint64_t)slot_bytes) ? 1 : -1;
    }
    h->slots = (uint64_t)slots;
    h->slot_bytes = (uint64_t)slot_bytes;
    h->header_bytes = (uint64_t)o[0];
    h->index_off = (uint64_t)o[1];
    h->index_slot_off = (uint64_t)o[2];
    h->headers_off = (uint64_t)o[3];
    h->data_off = (uint64_t)o[4];
    h->index_cap = (uint64_t)o[5];
    uint64_t cap_iv = ival_cap_for(slot_bytes);
    for (int64_t s = 0; s < slots; s++) {
        SlotHeader *sh = slot_hdr(base, (uint64_t)s);
        sh->cap_ivals = cap_iv;
        sh->n_ivals = 0;
    }
    atomic_thread_fence(memory_order_seq_cst);
    h->magic = A_MAGIC;
    return 0;
}

static uint64_t mix64(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
    return x;
}

/* find the slot of key; -1 when absent. Only COMPLETE/CLAIMED slots whose
 * header carries the same 128-bit key count. */
static int64_t find_slot(uint8_t *base, uint64_t klo, uint64_t khi) {
    ArenaHeader *h = hdr(base);
    _Atomic uint64_t *keys = index_keys(base);
    _Atomic uint32_t *slots = index_slots(base);
    uint64_t mask = h->index_cap - 1;
    uint64_t i = mix64(klo) & mask;
    for (uint64_t n = 0; n < h->index_cap; n++, i = (i + 1) & mask) {
        uint64_t k = atomic_load(&keys[i]);
        if (k == 0) return -1;
        if (k == klo) {
            uint32_t s = atomic_load(&slots[i]);
            SlotHeader *sh = slot_hdr(base, s);
            if (sh->key_lo == klo && sh->key_hi == khi) {
                uint32_t st = atomic_load(&sh->state);
                if (st == S_CLAIMED || st == S_COMPLETE) return (int64_t)s;
            }
            return -1;
        }
    }
    return -1;
}

/* claim a FREE slot for key (index entry published); -1 = arena full */
static int64_t claim_slot(uint8_t *base, uint64_t klo, uint64_t khi, uint64_t total) {
    ArenaHeader *h = hdr(base);
    /* #1431 (xsn193): a FULL arena refused every page only after walking all
     * 524288 slot headers -- per page, per sweep -- and P's main thread spent
     * its bubbles in here. The counters know the answer in O(1). They are
     * kept exact by claim/complete/evict/free below. */
    if (atomic_load(&h->n_complete) + atomic_load(&h->n_claimed) >= h->slots) return -1;
    uint64_t start = atomic_fetch_add(&h->clock_hand, 1) % h->slots;
    for (uint64_t n = 0; n < h->slots; n++) {
        uint64_t s = (start + n) % h->slots;
        SlotHeader *sh = slot_hdr(base, s);
        uint32_t expect = S_FREE;
        if (atomic_compare_exchange_strong(&sh->state, &expect, S_CLAIMED)) {
            sh->key_lo = klo;
            sh->key_hi = khi;
            sh->total_bytes = total;
            sh->generation++;
            sh->n_ivals = 0;
            sh->stem[0] = 0;
            atomic_store(&sh->lock, 0);
            atomic_store(&sh->clock_bit, 1);
            atomic_store(&sh->refcount, 0);
            /* publish in the index */
            _Atomic uint64_t *keys = index_keys(base);
            _Atomic uint32_t *slots = index_slots(base);
            uint64_t mask = h->index_cap - 1;
            uint64_t i = mix64(klo) & mask;
            for (uint64_t m = 0; m < h->index_cap; m++, i = (i + 1) & mask) {
                uint64_t k = atomic_load(&keys[i]);
                if (k == klo) {
                    /* another writer published this key concurrently: yield our slot */
                    atomic_store(&sh->state, S_FREE);
                    uint32_t other = atomic_load(&slots[i]);
                    return (int64_t)other;
                }
                if (k == 0 || k == TOMB) {
                    uint64_t expect_k = k;
                    if (atomic_compare_exchange_strong(&keys[i], &expect_k, klo)) {
                        atomic_store(&slots[i], (uint32_t)s);
                        atomic_fetch_add(&h->n_claimed, 1);
                        return (int64_t)s;
                    }
                    /* lost the race for this cell: re-read it */
                    m--; continue;
                }
            }
            atomic_store(&sh->state, S_FREE);
            return -1; /* index full */
        }
    }
    return -1;
}

/* status per page: 0 partial, 1 completed now, 2 already complete,
 * 3 refused (extent not granule-aligned or out of range), 4 arena full */
/* Merge k extents into the slot's sorted, disjoint interval list under the
 * slot lock. Returns 1 when the slot is fully covered, 0 when not, -1 on
 * interval-list overflow. Shared by arena_write (payload copy) and
 * arena_complete (#1427 direct DMA writes, no payload). */
static int merge_ivals(SlotHeader *sh, int64_t k, const int64_t *off_arr, const int64_t *len_arr,
                       uint64_t total) {
    int overflow = 0;
    while (atomic_exchange(&sh->lock, 1)) { /* spin */ }
    for (int64_t j = 0; j < k && !overflow; j++) {
        uint64_t lo = (uint64_t)off_arr[j], hi = lo + (uint64_t)len_arr[j];
        uint32_t n = sh->n_ivals;
        uint32_t a = 0;
        while (a < n && sh->ivals[a].hi < lo) a++;
        uint32_t b = a;
        while (b < n && sh->ivals[b].lo <= hi) {
            if (sh->ivals[b].lo < lo) lo = sh->ivals[b].lo;
            if (sh->ivals[b].hi > hi) hi = sh->ivals[b].hi;
            b++;
        }
        uint32_t removed = b - a;
        if (removed == 0) {
            if (n + 1 > sh->cap_ivals) { overflow = 1; break; }
            memmove(&sh->ivals[a + 1], &sh->ivals[a], (n - a) * sizeof(Ival));
            sh->ivals[a].lo = lo; sh->ivals[a].hi = hi;
            sh->n_ivals = n + 1;
        } else {
            sh->ivals[a].lo = lo; sh->ivals[a].hi = hi;
            if (removed > 1) memmove(&sh->ivals[a + 1], &sh->ivals[b], (n - b) * sizeof(Ival));
            sh->n_ivals = n - removed + 1;
        }
    }
    int full = (sh->n_ivals == 1 && sh->ivals[0].lo == 0 && sh->ivals[0].hi == total);
    atomic_store(&sh->lock, 0);
    return overflow ? -1 : full;
}

int64_t arena_write(uint8_t *base, int64_t n, const uint64_t *klo, const uint64_t *khi,
                    const int64_t *totals, const int64_t *n_ext, const int64_t *ext_off,
                    const int64_t *ext_len, const uint8_t **payload, const char **stems,
                    int8_t *status) {
    ArenaHeader *h = hdr(base);
    int64_t ok = 0, e = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t k = n_ext[i];
        int8_t s = 0;
        uint64_t total = (uint64_t)totals[i];
        if (total > h->slot_bytes) { status[i] = 3; e += k; continue; }
        for (int64_t j = 0; j < k; j++) {
            int64_t off = ext_off[e + j], len = ext_len[e + j];
            if (off < 0 || len <= 0 || (uint64_t)(off + len) > total) { s = 3; break; }
        }
        if (s) { status[i] = s; e += k; continue; }
        int64_t slot = find_slot(base, klo[i], khi[i]);
        if (slot < 0) slot = claim_slot(base, klo[i], khi[i], total);
        if (slot < 0) { status[i] = 4; e += k; continue; }
        SlotHeader *sh = slot_hdr(base, (uint64_t)slot);
        if (atomic_load(&sh->state) == S_COMPLETE) { status[i] = 2; e += k; ok++; continue; }
        if (stems && stems[i] && sh->stem[0] == 0) {
            size_t sl = strlen(stems[i]);
            if (sl >= STEM_CAP) sl = STEM_CAP - 1;
            memcpy(sh->stem, stems[i], sl);
            sh->stem[sl] = 0;
        }
        uint8_t *dst = slot_data(base, (uint64_t)slot);
        int64_t taken = 0;
        for (int64_t j = 0; j < k; j++) {
            int64_t off = ext_off[e + j], len = ext_len[e + j];
            memcpy(dst + off, payload[i] + taken, (size_t)len);
            taken += len;
        }
        atomic_thread_fence(memory_order_release);
        /* record coverage: merge the extents into the sorted interval list */
        int mr = merge_ivals(sh, k, ext_off + e, ext_len + e, total);
        int overflow = mr < 0, full = mr > 0;
        if (overflow) { status[i] = 3; e += k; continue; }
        if (full) {
            uint32_t expect = S_CLAIMED;
            if (atomic_compare_exchange_strong(&sh->state, &expect, S_COMPLETE)) {
                atomic_fetch_add(&h->n_complete, 1);
                atomic_fetch_sub(&h->n_claimed, 1);
                s = 1;
            } else {
                s = 2;
            }
        } else {
            s = 0;
        }
        status[i] = s;
        ok++;
        e += k;
    }
    return ok;
}

/* status per page: 0 ok, 1 absent/not complete, 2 width mismatch, 3 refused */
int64_t arena_read(uint8_t *base, int64_t n, const uint64_t *klo, const uint64_t *khi,
                   const int64_t *totals, const int64_t *n_ext, const int64_t *ext_off,
                   const int64_t *ext_len, uint8_t **out, int8_t *status) {
    int64_t ok = 0, e = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t k = n_ext[i];
        int64_t slot = find_slot(base, klo[i], khi[i]);
        if (slot < 0) { status[i] = 1; e += k; continue; }
        SlotHeader *sh = slot_hdr(base, (uint64_t)slot);
        atomic_fetch_add(&sh->refcount, 1);
        if (atomic_load(&sh->state) != S_COMPLETE || sh->key_lo != klo[i] || sh->key_hi != khi[i]) {
            atomic_fetch_sub(&sh->refcount, 1);
            status[i] = 1; e += k; continue;
        }
        if (sh->total_bytes != (uint64_t)totals[i]) {
            atomic_fetch_sub(&sh->refcount, 1);
            status[i] = 2; e += k; continue;
        }
        const uint8_t *src = slot_data(base, (uint64_t)slot);
        int64_t taken = 0;
        int8_t s = 0;
        for (int64_t j = 0; j < k; j++) {
            int64_t off = ext_off[e + j], len = ext_len[e + j];
            if (off < 0 || len < 0 || (uint64_t)(off + len) > sh->total_bytes) { s = 3; break; }
            memcpy(out[i] + taken, src + off, (size_t)len);
            taken += len;
        }
        atomic_store(&sh->clock_bit, 1);
        atomic_fetch_sub(&sh->refcount, 1);
        status[i] = s;
        if (s == 0) ok++;
        e += k;
    }
    return ok;
}

/* presence: 1 complete, 0 absent/partial */
int64_t arena_lookup(uint8_t *base, int64_t n, const uint64_t *klo, const uint64_t *khi,
                     int8_t *status) {
    int64_t found = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t slot = find_slot(base, klo[i], khi[i]);
        int8_t s = 0;
        if (slot >= 0) {
            SlotHeader *sh = slot_hdr(base, (uint64_t)slot);
            s = (atomic_load(&sh->state) == S_COMPLETE) ? 1 : 0;
        }
        status[i] = s;
        found += s;
    }
    return found;
}

/* Second-chance clock: pick up to `want` COMPLETE, unreferenced slots, move
 * them to EVICTING and unlink them from the index. Returns the count; the
 * slot ids land in `slots`, their keys in klo/khi, their widths in totals.
 * The caller copies the data out (arena_slot_ptr) and then calls
 * arena_free_slots. Slots whose key is listed in `keep` (pinned) are skipped. */
const char *arena_slot_stem(uint8_t *base, int64_t slot) { return slot_hdr(base, (uint64_t)slot)->stem; }

int64_t arena_evict_candidates(uint8_t *base, int64_t want, int64_t *slots, uint64_t *klo,
                               uint64_t *khi, int64_t *totals, const uint64_t *keep_lo,
                               int64_t n_keep) {
    ArenaHeader *h = hdr(base);
    int64_t got = 0;
    uint64_t scanned = 0;
    while (got < want && scanned < h->slots * 2) {
        uint64_t s = atomic_fetch_add(&h->clock_hand, 1) % h->slots;
        scanned++;
        SlotHeader *sh = slot_hdr(base, s);
        if (atomic_load(&sh->state) != S_COMPLETE) continue;
        if (atomic_exchange(&sh->clock_bit, 0)) continue; /* second chance */
        if (atomic_load(&sh->refcount) != 0) continue;
        int pinned = 0;
        for (int64_t p = 0; p < n_keep; p++) if (keep_lo[p] == sh->key_lo) { pinned = 1; break; }
        if (pinned) { atomic_store(&sh->clock_bit, 1); continue; }
        uint32_t expect = S_COMPLETE;
        if (!atomic_compare_exchange_strong(&sh->state, &expect, S_EVICTING)) continue;
        if (atomic_load(&sh->refcount) != 0) { atomic_store(&sh->state, S_COMPLETE); continue; }
        /* unlink from the index */
        _Atomic uint64_t *keys = index_keys(base);
        uint64_t mask = h->index_cap - 1;
        uint64_t i = mix64(sh->key_lo) & mask;
        for (uint64_t m = 0; m < h->index_cap; m++, i = (i + 1) & mask) {
            uint64_t k = atomic_load(&keys[i]);
            if (k == 0) break;
            if (k == sh->key_lo) { atomic_store(&keys[i], TOMB); break; }
        }
        atomic_fetch_sub(&h->n_complete, 1);
        slots[got] = (int64_t)s;
        klo[got] = sh->key_lo;
        khi[got] = sh->key_hi;
        totals[got] = (int64_t)sh->total_bytes;
        got++;
    }
    return got;
}

uint8_t *arena_slot_ptr(uint8_t *base, int64_t slot) { return slot_data(base, (uint64_t)slot); }

/* #1424 Stufe 3: the slot of each key (-1 absent) and its state, so a reader
 * can address the page IN PLACE instead of copying it out. */
int64_t arena_find_slots(uint8_t *base, int64_t n, const uint64_t *klo, const uint64_t *khi,
                         int64_t *slots, int8_t *states) {
    int64_t found = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t s = find_slot(base, klo[i], khi[i]);
        slots[i] = s;
        states[i] = s < 0 ? 0 : (int8_t)atomic_load(&slot_hdr(base, (uint64_t)s)->state);
        if (s >= 0) found++;
    }
    return found;
}
/* #1424 Stufe 3: reader references. +1 pins a COMPLETE slot against eviction
 * (arena_evict_candidates skips refcount != 0), -1 releases. Returns how many
 * slots took the delta; a +1 on a slot that is not COMPLETE is refused (0). */
/* #1427 DIRECT WRITES (Stufe 4): the writer DMAs its extents straight into
 * slot_data instead of handing a host payload to arena_write. claim gives it
 * the slot; complete merges its extents into the coverage list and flips the
 * slot COMPLETE when every extent of the page is in. Several ranks (PP layer
 * shards) claim the SAME key and each completes its own extents.
 * claim status per key: 0 = claimed by this call (fresh), 1 = CLAIMED by an
 * earlier writer (join: write your extents, then complete), 2 = already
 * COMPLETE (nothing to write), 3 = too large, 4 = no free slot. gen_out is
 * the slot generation to hand back to arena_complete. */
int64_t arena_claim(uint8_t *base, int64_t n, const uint64_t *klo, const uint64_t *khi,
                    const int64_t *totals, const char **stems, int64_t *slots_out,
                    int64_t *gen_out, int8_t *status) {
    ArenaHeader *h = hdr(base);
    int64_t ok = 0;
    for (int64_t i = 0; i < n; i++) {
        uint64_t total = (uint64_t)totals[i];
        slots_out[i] = -1; gen_out[i] = 0;
        if (total > h->slot_bytes) { status[i] = 3; continue; }
        int64_t slot = find_slot(base, klo[i], khi[i]);
        int fresh = 0;
        if (slot < 0) {
            slot = claim_slot(base, klo[i], khi[i], total);
            if (slot < 0) { status[i] = 4; continue; }
            fresh = 1;
        }
        SlotHeader *sh = slot_hdr(base, (uint64_t)slot);
        slots_out[i] = slot;
        gen_out[i] = (int64_t)sh->generation;
        if (atomic_load(&sh->state) == S_COMPLETE) { status[i] = 2; ok++; continue; }
        if (stems && stems[i] && sh->stem[0] == 0) {
            size_t sl = strlen(stems[i]);
            if (sl >= STEM_CAP) sl = STEM_CAP - 1;
            memcpy(sh->stem, stems[i], sl);
            sh->stem[sl] = 0;
        }
        status[i] = fresh ? 0 : 1;
        ok++;
    }
    return ok;
}

/* status per slot: 1 = completed by this call, 0 = extents merged, page not
 * yet full (other writers pending), 2 = already COMPLETE, 3 = slot recycled
 * (generation moved on) or interval overflow -- the writer's bytes are lost. */
int64_t arena_complete(uint8_t *base, int64_t n, const int64_t *slots, const int64_t *gens,
                       const int64_t *n_ext, const int64_t *ext_off, const int64_t *ext_len,
                       int8_t *status) {
    ArenaHeader *h = hdr(base);
    int64_t ok = 0, e = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t k = n_ext[i];
        SlotHeader *sh = slot_hdr(base, (uint64_t)slots[i]);
        if ((int64_t)sh->generation != gens[i]) { status[i] = 3; e += k; continue; }
        uint32_t st0 = atomic_load(&sh->state);
        if (st0 == S_COMPLETE) { status[i] = 2; e += k; ok++; continue; }
        if (st0 != S_CLAIMED) { status[i] = 3; e += k; continue; }  /* freed or evicting */
        atomic_thread_fence(memory_order_release);
        int mr = merge_ivals(sh, k, ext_off + e, ext_len + e, sh->total_bytes);
        e += k;
        if (mr < 0) { status[i] = 3; continue; }
        if (mr == 0) { status[i] = 0; ok++; continue; }
        uint32_t expect = S_CLAIMED;
        if (atomic_compare_exchange_strong(&sh->state, &expect, S_COMPLETE)) {
            atomic_fetch_add(&h->n_complete, 1);
            atomic_fetch_sub(&h->n_claimed, 1);
            status[i] = 1;
        } else {
            status[i] = 2;
        }
        ok++;
    }
    return ok;
}


/* ---- #1439: BLAKE2b (RFC 7693), unkeyed, 16-byte digest -- the same key128
 * the Python side computes with hashlib.blake2b(stem, digest_size=16), so the
 * arena can be asked by STEM from C without a Python hash + ctypes array per
 * key (xsn199 D profile: 6 % of the re-admission of a 100k prompt). */
static const uint64_t b2b_iv[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL, 0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL};
static const uint8_t b2b_sigma[12][16] = {
    {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}, {14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3},
    {11,8,12,0,5,2,15,13,10,14,3,6,7,1,9,4}, {7,9,3,1,13,12,11,14,2,6,5,10,4,0,15,8},
    {9,0,5,7,2,4,10,15,14,1,11,12,6,8,3,13}, {2,12,6,10,0,11,8,3,4,13,7,5,15,14,1,9},
    {12,5,1,15,14,13,4,10,0,7,6,3,9,2,8,11}, {13,11,7,14,12,1,3,9,5,0,15,4,8,6,2,10},
    {6,15,14,9,11,3,0,8,12,2,13,7,1,4,10,5}, {10,2,8,4,7,6,1,5,15,11,9,14,3,12,13,0},
    {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}, {14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3}};
static inline uint64_t b2b_rotr(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }
static inline uint64_t b2b_ld64(const uint8_t *p) {
    uint64_t v = 0; for (int i = 7; i >= 0; i--) v = (v << 8) | p[i]; return v;
}
#define B2B_G(a, b, c, d, x, y) do { \
    v[a] += v[b] + (x); v[d] = b2b_rotr(v[d] ^ v[a], 32); \
    v[c] += v[d];       v[b] = b2b_rotr(v[b] ^ v[c], 24); \
    v[a] += v[b] + (y); v[d] = b2b_rotr(v[d] ^ v[a], 16); \
    v[c] += v[d];       v[b] = b2b_rotr(v[b] ^ v[c], 63); } while (0)
static void b2b_compress(uint64_t h[8], const uint8_t blk[128], uint64_t t, int last) {
    uint64_t v[16], m[16];
    for (int i = 0; i < 8; i++) { v[i] = h[i]; v[i + 8] = b2b_iv[i]; }
    v[12] ^= t; /* t0 (messages here are < 2^64 bytes, t1 = 0) */
    if (last) v[14] = ~v[14];
    for (int i = 0; i < 16; i++) m[i] = b2b_ld64(blk + 8 * i);
    for (int r = 0; r < 12; r++) {
        const uint8_t *s = b2b_sigma[r];
        B2B_G(0, 4,  8, 12, m[s[0]],  m[s[1]]);  B2B_G(1, 5,  9, 13, m[s[2]],  m[s[3]]);
        B2B_G(2, 6, 10, 14, m[s[4]],  m[s[5]]);  B2B_G(3, 7, 11, 15, m[s[6]],  m[s[7]]);
        B2B_G(0, 5, 10, 15, m[s[8]],  m[s[9]]);  B2B_G(1, 6, 11, 12, m[s[10]], m[s[11]]);
        B2B_G(2, 7,  8, 13, m[s[12]], m[s[13]]); B2B_G(3, 4,  9, 14, m[s[14]], m[s[15]]);
    }
    for (int i = 0; i < 8; i++) h[i] ^= v[i] ^ v[i + 8];
}
static void blake2b_16(const uint8_t *in, size_t n, uint8_t out[16]) {
    uint64_t h[8];
    for (int i = 0; i < 8; i++) h[i] = b2b_iv[i];
    h[0] ^= 0x01010000ULL ^ (uint64_t)16; /* param block: digest 16, key 0, fanout 1, depth 1 */
    uint8_t blk[128];
    size_t off = 0;
    while (n - off > 128) { b2b_compress(h, in + off, (uint64_t)(off + 128), 0); off += 128; }
    size_t rem = n - off;
    memset(blk, 0, 128); memcpy(blk, in + off, rem);
    b2b_compress(h, blk, (uint64_t)n, 1);
    for (int i = 0; i < 2; i++) for (int j = 0; j < 8; j++) out[8 * i + j] = (uint8_t)(h[i] >> (8 * j));
}
/* key128(stem) exactly as hicache_arena.key128: lo/hi little-endian halves,
 * lo in {0, ~0} mapped to 1 (those are the index's empty/tomb markers). */
static void stem_key128(const char *stem, uint64_t *lo, uint64_t *hi) {
    uint8_t d[16];
    blake2b_16((const uint8_t *)stem, strlen(stem), d);
    uint64_t l = b2b_ld64(d), h = b2b_ld64(d + 8);
    if (l == 0 || l == ~0ULL) l = 1;
    *lo = l; *hi = h;
}
void arena_key128(const char *stem, uint64_t *lo, uint64_t *hi) { stem_key128(stem, lo, hi); }
/* find by STEM: hashing in C, one call for a whole prefix. */
int64_t arena_find_stems(uint8_t *base, int64_t n, const char **stems, int64_t *slots, int8_t *states) {
    int64_t found = 0;
    for (int64_t i = 0; i < n; i++) {
        uint64_t lo, hi;
        stem_key128(stems[i], &lo, &hi);
        int64_t s = find_slot(base, lo, hi);
        slots[i] = s;
        states[i] = s < 0 ? 0 : (int8_t)atomic_load(&slot_hdr(base, (uint64_t)s)->state);
        if (s >= 0) found++;
    }
    return found;
}

int64_t arena_ref_slots(uint8_t *base, int64_t n, const int64_t *slots, int32_t delta) {
    int64_t done = 0;
    for (int64_t i = 0; i < n; i++) {
        if (slots[i] < 0) continue;
        SlotHeader *sh = slot_hdr(base, (uint64_t)slots[i]);
        if (delta > 0) {
            atomic_fetch_add(&sh->refcount, 1);
            /* #1427: a writer's node references the slot from the claim on,
             * before the page is COMPLETE -- CLAIMED slots take refs too. */
            uint32_t st = atomic_load(&sh->state);
            if (st != S_COMPLETE && st != S_CLAIMED) { atomic_fetch_sub(&sh->refcount, 1); continue; }
            atomic_store(&sh->clock_bit, 1);
        } else if (delta < 0) {
            if (atomic_load(&sh->refcount) > 0) atomic_fetch_sub(&sh->refcount, 1); else continue;
        }
        done++;
    }
    return done;
}
/* #1424 Stufe 3: byte offset of slot 0's data from the mapping base, so a
 * torch view can address every slot as base + data_off + slot * slot_bytes. */
int64_t arena_data_offset(uint8_t *base) {
    ArenaHeader *h = hdr(base);
    int64_t o[6];
    arena_layout((int64_t)h->slots, (int64_t)h->slot_bytes, o);
    return o[4];
}
void arena_free_slots(uint8_t *base, int64_t n, const int64_t *slots) {
    ArenaHeader *h = hdr(base);
    for (int64_t i = 0; i < n; i++) {
        SlotHeader *sh = slot_hdr(base, (uint64_t)slots[i]);
        uint32_t prev = atomic_exchange(&sh->state, S_FREE);
        sh->key_lo = 0; sh->key_hi = 0;
        sh->generation++;  /* #1427: a late arena_complete on this slot is refused */
        /* #1431: keep the occupancy counters exact (EVICTING was already
         * taken out of n_complete by arena_evict_candidates). */
        if (prev == S_COMPLETE) atomic_fetch_sub(&h->n_complete, 1);
        else if (prev == S_CLAIMED) atomic_fetch_sub(&h->n_claimed, 1);
    }
}

/* Reap CLAIMED slots older than `max_generation_age` writes of the clock:
 * a rank that died mid-write leaves a CLAIMED slot behind. Cheap form: a
 * CLAIMED slot whose index entry is tombstoned or missing is freed. Returns
 * the number freed. Called by the sweeper. */
int64_t arena_reap_stale(uint8_t *base) {
    ArenaHeader *h = hdr(base);
    int64_t freed = 0;
    for (uint64_t s = 0; s < h->slots; s++) {
        SlotHeader *sh = slot_hdr(base, s);
        if (atomic_load(&sh->state) != S_EVICTING) continue;
        if (atomic_load(&sh->refcount) != 0) continue;
        /* an EVICTING slot nobody freed: its evictor died */
        sh->key_lo = 0; sh->key_hi = 0;
        atomic_store(&sh->state, S_FREE);
        freed++;
    }
    return freed;
}

void arena_stats(uint8_t *base, int64_t *out) {
    ArenaHeader *h = hdr(base);
    out[0] = (int64_t)h->slots;
    out[1] = (int64_t)atomic_load(&h->n_complete);
    out[2] = (int64_t)atomic_load(&h->n_claimed);
    out[3] = (int64_t)h->slot_bytes;
}
