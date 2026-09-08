#pragma once
#include <sys/types.h>
#include <stdio.h>
#include <chrono>
#include <cstring>
#include <unordered_map>
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