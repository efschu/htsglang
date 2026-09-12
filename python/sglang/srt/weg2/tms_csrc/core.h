#pragma once
#include <sys/types.h>
#include <stdio.h>
#include <chrono>
#include <cstring>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <mutex>
#include <string>
#include <vector>
#include "utils.h"
#include "macro.h"
#include "host_ring.h"

#if defined(USE_ROCM)
#include "hardware_amd_support.h"
#endif

enum class AllocationState {
    // Memory is mapped and accessible
    ACTIVE,
    // Memory is unmapped and inaccessible
    PAUSED
};

struct AllocationMetadata {
    size_t size;
    CUdevice device;
    std::string tag;
    AllocationState state;
    bool enable_cpu_backup;
    // WEG2 flip-cost C2: the host backup is a SCATTER LIST of 2 MiB granules,
    // not one contiguous pinned block.  ``cpu_backup_from_ring`` says which
    // allocator owns them -- the shared per-card ring (the boot published
    // TMS_HOST_RING_*) or stock ``cudaMallocHost`` (it did not).  There is no
    // third state and no per-allocation fallback between the two: the choice is
    // made once per boot (spec section 10.4).
    std::vector<void*> cpu_backup_granules;
    bool cpu_backup_from_ring;

    // REMAP (#1352) -- THE PAGE-GRANULAR BACKING, and why it is not optional.
    //
    // The remap flip moves a PHYSICAL PAGE from one arena's VA to another's.
    // Two driver laws decide the shape of that, and both were read out of
    // cuda.h 12.9 rather than assumed:
    //
    //   * ``cuMemMap``'s ``offset`` parameter "currently must be zero"
    //     (cuda.h:13190-13192).  A handle can only ever be mapped at its OWN
    //     start, so a sub-range of a big handle cannot land at a page offset
    //     in a foreign VA.
    //   * ``cuMemUnmap`` "cannot unmap a sub-range of an address range mapped
    //     by cuMemCreate / cuMemMap" (cuda.h:13347-13351).  The unmap unit is
    //     exactly the map unit.
    //
    // Together: THE MOVABLE UNIT IS A HANDLE, and its size is frozen at
    // ``cuMemCreate``.  The legacy form creates ONE handle for the whole
    // allocation (core.cpp:27), so its movable unit is a whole PyTorch segment
    // -- and the two groups' segment size distributions do not match, so no
    // page could ever change owner.  ``page_handles`` is therefore the
    // enabling change, not a tuning: ``reserved_size / page_bytes`` handles,
    // each created and mapped by its own call.
    //
    // A ZERO entry means: this page's VA is UNMAPPED and this allocation owns
    // no physical page for it.  That single convention carries the whole
    // design -- ``pause`` zeroes, ``resume`` fills ONLY the zeroes, and a page
    // that arrived by ``remap_pages`` is already non-zero and is therefore
    // skipped by resume.  "How many pages did this wake have to allocate" is
    // then not an assertion but a counter, and the design's claim is that it
    // is ZERO.
    //
    // ``page_handles.empty()`` == the legacy one-handle form, unchanged.
    std::vector<CUmemGenericAllocationHandle> page_handles;
    //: ``size`` rounded up to the page granularity.  Equal to ``size`` on the
    //: legacy path; it is the extent that was RESERVED and must be unmapped
    //: and address-freed, which is not the extent the allocator was promised.
    size_t reserved_size;

#if defined(USE_CUDA)
    CUmemGenericAllocationHandle allocHandle;
#elif defined(USE_ROCM)
    size_t aligned_size;
    std::vector<hipMemGenericAllocationHandle_t> allocHandles;
    std::vector<size_t> chunk_sizes;
#else
    #error "USE_PLATFORM is not set"
#endif
};

class TorchMemorySaver {
public:
    static TorchMemorySaver& instance();

    cudaError_t malloc(void** ptr, CUdevice device, size_t size, const std::string& tag, bool enable_cpu_backup);
    cudaError_t free(void* ptr);

    void pause(const std::string& tag);
    void resume(const std::string& tag);

    //: C7: the planner's sizing input.  Sum of ``metadata.size`` over the
    //: allocations carrying ``tag``.  This REPLACES the RssShmem delta as the
    //: per-tag instrument: ring granules are shared pages mapped by both
    //: co-located processes, so RssShmem collapses to ~0 and a per-process sum
    //: double-counts (spec R8).
    uint64_t tag_bytes(const std::string& tag);
    //: C16 / A1-2, FIX 1 round 1: ``<tag>=<bytes>,...`` over EVERY tag that has
    //: at least one allocation with ``enable_cpu_backup``, summing only those
    //: allocations.  ``tag_bytes`` cannot answer this question -- it counts a
    //: tag's device bytes whether or not they are ever copied to the host, so
    //: summing it over the tag list would charge ``kv_cache`` (paused WITHOUT
    //: cpu backup, R20) into the host ring and refuse the boot.  The population
    //: is the saver's own metadata, which is the only place that fact exists.
    //: Returns the number of tags written, or -1 when ``len`` is too small.
    int backed_up_tag_bytes(char* out, size_t len);
    //: C7: the live ring counters of this rank's card, or false when this boot
    //: published no ring.
    bool ring_stats(HostRingStats* out, std::string* card_uuid);

    // ------------------------------------------------------------------ REMAP
    //: REMAP (#1352): how many pages this tag's allocations span, in ADDRESS
    //: ORDER.  The page plan addresses a page by its tag-relative index, so
    //: this is the denominator every plan is stated against.  0 for a tag with
    //: no page-granular allocation -- which is an ABSENCE (the tag was never
    //: armed for remap), never "the tag is empty".
    uint64_t tag_pages(const std::string& tag);

    //: REMAP (#1352): MOVE ``n_pages`` physical pages from ``src_tag``'s page
    //: sequence to ``dst_tag``'s, WITHOUT creating or releasing a single one.
    //: ``cuMemUnmap`` in the source VA, ``cuMemMap`` of the SAME handle in the
    //: destination VA, ``cuMemSetAccess`` on what moved.
    //:
    //: REFUSAL, NEVER FALLBACK.  Returns 0 on success and a negative code with
    //: a reason in ``err`` otherwise; it NEVER falls back to allocating the
    //: page it could not move, because that fallback is precisely the second
    //: bookkeeping of the physics this replaces -- a flip that quietly
    //: allocates is the VramCredit defect wearing a different name.
    //:
    //: ALL-OR-NOTHING.  Every page of the batch is validated BEFORE any page
    //: is unmapped: a half-performed remap leaves pages owned by neither side
    //: and is not recoverable from Python.
    int remap_pages(const std::string& src_tag, uint64_t src_page,
                    const std::string& dst_tag, uint64_t dst_page,
                    uint64_t n_pages, char* err, size_t errlen);

    //: REMAP (#1352): the per-tag page ledger of the LAST wake, read after
    //: ``resume``.  ``created`` is the number of pages this wake had to ask
    //: the driver for; the design's whole claim is that it is 0 once the plan
    //: funds the wake by remap, so it is MEASURED and never asserted.
    //: ``mapped_in``/``mapped_out`` are this process's remap traffic since the
    //: last reset.  Returns 0 when the tag has no page-granular allocation --
    //: an absence, distinguished from a measured zero by the return value.
    int page_stats(const std::string& tag, uint64_t* pages, uint64_t* created,
                   uint64_t* mapped_in, uint64_t* mapped_out);

    //: S7 (#1273): THE LAST ``resume`` CALL'S PASS-1 COST AND ITS DENOMINATOR.
    //:
    //: Resume pass 1 (``cu_mem_create`` + ``cuMemMap`` + ``cu_mem_set_access``,
    //: once per ALLOCATION of the tag) is proportional to an allocation count
    //: that is logged nowhere, so its wall has always been charged to the copy
    //: rate -- ADDENDUM 3 section 4's one unexplained remainder, and risk R2 of
    //: the #1273 spec.  The weight exchange does not change that cost in either
    //: arm, which is exactly why both arms must be able to measure it.
    //:
    //: ``tag_out`` is the tag the recorded numbers BELONG TO.  It is returned
    //: rather than assumed because the reader is a separate call: a caller that
    //: printed these numbers beside a tag it did not verify would be publishing
    //: a previous tag's cost under this tag's name, which is the instrument lie
    //: this fork catalogues as class A.  Returns the record sequence number, and
    //: **0 when no resume has ever been recorded** -- an absence, never a zero.
    inline uint64_t resume_stats(char* tag_out, size_t tag_len,
                                 uint64_t* allocations, double* map_ms, double* copy_ms) {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        if (tag_out != nullptr && tag_len > 0) {
            size_t n = last_resume_tag_.size() < (tag_len - 1) ? last_resume_tag_.size() : (tag_len - 1);
            memcpy(tag_out, last_resume_tag_.c_str(), n);
            tag_out[n] = '\0';
        }
        if (allocations != nullptr) *allocations = last_resume_allocations_;
        if (map_ms != nullptr) *map_ms = last_resume_map_ms_;
        if (copy_ms != nullptr) *copy_ms = last_resume_copy_ms_;
        return last_resume_seq_;
    }

private:
    //: The recorder S7 calls from ``resume``, between the passes it separates.
    //: ``t0`` is taken before pass 1 and ``t1`` between pass 1 and pass 2, so
    //: ``map_ms`` is the MAP phase alone and ``copy_ms`` is pass 2's issue plus
    //: pass 3's single synchronise -- pass 4's granule release is deliberately
    //: outside both, because it is host bookkeeping and not a device cost.
    //: Called under the metadata mutex, which ``resume`` already holds.
    inline void note_resume(const std::string& tag, size_t allocations,
                            const std::chrono::steady_clock::time_point& t0,
                            const std::chrono::steady_clock::time_point& t1) {
        last_resume_tag_ = tag;
        last_resume_allocations_ = (uint64_t) allocations;
        last_resume_map_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();
        last_resume_copy_ms_ =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
        ++last_resume_seq_;
    }

    // ---------------------------------------------------------- REMAP state
    //: The page size every page-granular allocation is cut to.  READ from the
    //: driver (``cu_mem_min_granularity``) at the first page-granular malloc,
    //: never assumed to be 2 MiB: the plan's arithmetic is stated in this unit
    //: and a wrong constant would mis-slice every tensor boundary silently.
    size_t page_bytes_ = 0;
    //: ``TMS_REMAP_TAGS``, parsed once: the tag PREFIXES whose allocations are
    //: page-granular.  Empty (the default) means the legacy one-handle form
    //: everywhere, so a boot that does not ask for the remap is byte-for-byte
    //: the old path.
    std::vector<std::string> remap_prefixes_;
    bool remap_prefixes_parsed_ = false;
    bool remap_exportable_ = false;
    //: Per tag: pages this process created at the last wake, and its remap
    //: traffic.  Counters, so "the wake allocated nothing" is measured.
    std::map<std::string, uint64_t> pages_created_;
    std::map<std::string, uint64_t> pages_mapped_in_;
    std::map<std::string, uint64_t> pages_mapped_out_;
    //: True when ``tag`` is page-granular under this boot's env.
    bool remap_tag(const std::string& tag);
    //: The tag's allocations in ADDRESS ORDER -- the order a page index means.
    //: Called under ``allocator_metadata_mutex_``.
    std::vector<void*> ordered_ptrs(const std::string& tag);
    //: page index -> (allocation, page within it), or false when out of range.
    bool locate_page(const std::vector<void*>& order, uint64_t index,
                     void** ptr_out, uint64_t* local_out);

    std::string last_resume_tag_;
    uint64_t last_resume_allocations_ = 0;
    double last_resume_map_ms_ = 0.0;
    double last_resume_copy_ms_ = 0.0;
    //: 0 means NO resume has been recorded in this process.  The reader turns
    //: that into "n/a", never into a zero cost.
    uint64_t last_resume_seq_ = 0;

    TorchMemorySaver();
    ~TorchMemorySaver() = default;
    TorchMemorySaver(const TorchMemorySaver&) = delete;
    TorchMemorySaver& operator=(const TorchMemorySaver&) = delete;

    //: R12 (NEW HAZARD): this stream is created with ``cudaStreamCreate``, i.e.
    //: DEFAULT (blocking) flags, and never with ``cudaStreamNonBlocking``.  The
    //: copies it carries replace ``cudaMemcpy`` on the legacy default stream,
    //: which implicitly ordered against PyTorch's default stream; a
    //: non-blocking stream silently drops that ordering and copies pages a
    //: pending kernel is still writing.  Pinned by T4.
    cudaStream_t backup_stream_ = nullptr;
    void ensure_backup_stream();
    //: The per-card shared host granule ring, or nullptr for the stock path.
    //: Opened lazily at the first cpu-backed pause, when a CUDA context (and
    //: therefore a resolvable device UUID) certainly exists.
    HostBackupRing* ring_ = nullptr;
    bool ring_open_attempted_ = false;
    HostBackupRing* ensure_ring(CUdevice device);
    //: C6 / R20: the ONE gate through which host bytes may be taken for an
    //: allocation.  kv_cache is paused WITHOUT enable_cpu_backup
    //: (weight_updater.py:686-687) and must therefore never reach either host
    //: allocator.
    static void assert_host_backup_eligible(const AllocationMetadata& metadata);

    std::mutex allocator_metadata_mutex_;
    std::unordered_map<void*, AllocationMetadata> allocation_metadata_;
};